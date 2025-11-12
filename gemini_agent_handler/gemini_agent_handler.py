#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import base64
import logging
import threading
import traceback
import uuid
from decimal import Decimal
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Optional

import httpx
import pendulum
from google import genai
from google.genai import types

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


# ----------------------------
# Gemini Response Streaming with Function Handling and History
# ----------------------------
class GeminiEventHandler(AIAgentEventHandler):
    """
    A handler class for managing conversations and function calls with Google's Gemini API.

    Key capabilities:
    - Streams partial text responses in real-time
    - Detects and executes function calls embedded in responses
    - Maintains conversation history and context
    - Handles both streaming and non-streaming responses
    - Manages function execution state and results
    - Supports WebSocket streaming for real-time updates
    """

    @staticmethod
    def _sanitize_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively remove JSON Schema fields that aren't compatible with Gemini API.

        Args:
            tool: Tool definition dictionary

        Returns:
            Sanitized tool definition
        """
        if not isinstance(tool, dict):
            return tool

        # Fields to remove that are valid in JSON Schema but not in Gemini API
        fields_to_remove = ["additional_properties", "additionalProperties"]

        sanitized = {}
        for key, value in tool.items():
            if key in fields_to_remove:
                continue

            # Remove 'required' field if this is not an object type
            # Gemini only allows 'required' for objects, not arrays or other types
            if key == "required" and tool.get("type") != "object":
                continue

            if isinstance(value, dict):
                sanitized[key] = GeminiEventHandler._sanitize_tool_schema(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    (
                        GeminiEventHandler._sanitize_tool_schema(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initialize the Gemini event handler.

        Args:
            logger: Logger instance for debug/info messages
            agent: Configuration dictionary containing API keys, model settings and available tools
            setting: Additional handler settings and configurations
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        # Enable HTTP/2 for improved performance (multiplexing, header compression, better streaming)
        http_options = types.HttpOptions(
            client_args={"http2": True},  # Enable HTTP/2 for sync operations
            async_client_args={"http2": True},  # Enable HTTP/2 for async operations
        )

        if all(setting.get(k) for k in ["project", "location"]):
            vertex_credentials = {
                "vertexai": True,
                "project": setting["project"],
                "location": setting["location"],
                "http_options": http_options,
            }
            self.client = genai.Client(**vertex_credentials)
        else:
            self.client = genai.Client(
                api_key=self.agent["configuration"].get("api_key"),
                http_options=http_options,
            )

        self.model = self.agent["configuration"].get("model")

        tools = []

        # Always add function declarations if they exist (excluding google_search and code_execution)
        sanitized_tools = [
            self._sanitize_tool_schema(tool)
            for tool in self.agent["configuration"].get("tools", [])
            if tool["name"] not in ["code_execution", "google_search"]
        ]

        if sanitized_tools:
            tools.append(types.Tool(function_declarations=sanitized_tools))

        # Add Google Search if configured
        if any(
            tool["name"] == "google_search"
            for tool in self.agent["configuration"].get("tools", [])
        ):
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Add code execution if configured
        if any(
            tool["name"] == "code_execution"
            for tool in self.agent["configuration"].get("tools", [])
        ):
            tools.append(types.Tool(code_execution=types.ToolCodeExecution))

        # Convert Decimal to float once during initialization (performance optimization)
        # Exclude 'reasoning' from model_setting as it's not a Gemini API parameter
        self.model_setting = dict(
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in self.agent["configuration"].items()
                if k not in ["api_key", "tools", "model", "text", "reasoning"]
            },
            **{
                "system_instruction": self.agent["instructions"],
                "tools": tools,
            },
        )
        self.assistant_messages = []

        # Cache output format type for better performance (avoid repeated dict lookups)
        self.output_format_type = (
            self.agent["configuration"]
            .get("text", {})
            .get("format", {})
            .get("type", "text")
        )

        # Validate reasoning configuration if present
        if "reasoning" in self.agent["configuration"]:
            if not isinstance(self.agent["configuration"]["reasoning"], dict):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning configuration should be a dictionary. "
                        "Reasoning features may not work correctly."
                    )
            elif self.agent["configuration"]["reasoning"].get("enabled") is None:
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning is not explicitly enabled in configuration. "
                        "Reasoning events will be skipped during streaming."
                    )

        # Enable/disable timeline logging (default: disabled)
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the Gemini model with the provided configuration.

        Args:
            kwargs: Dictionary containing:
                - input: Messages to send to the model
                - stream: Boolean indicating if streaming response is desired

        Returns:
            Either a streaming or non-streaming model response

        Raises:
            Exception: If model invocation fails
        """
        try:
            invoke_start = pendulum.now("UTC")

            # Build config with reasoning/thinking if enabled
            config_params = dict(self.model_setting)

            # Add thinking configuration if reasoning is enabled
            reasoning_config = self.agent["configuration"].get("reasoning", {})
            if isinstance(reasoning_config, dict) and reasoning_config.get("enabled"):
                thinking_budget = reasoning_config.get(
                    "thinking_budget", -1
                )  # -1 = dynamic
                include_thoughts = reasoning_config.get(
                    "include_thoughts", True
                )  # True by default to show reasoning
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget, include_thoughts=include_thoughts
                )
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"[invoke_model] Reasoning enabled with thinking_budget={thinking_budget}, include_thoughts={include_thoughts}"
                    )

            config = types.GenerateContentConfig(**config_params)

            if kwargs.get("stream"):
                result = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=kwargs["input"],
                    config=config,
                )
            else:
                result = self.client.models.generate_content(
                    model=self.model,
                    contents=kwargs["input"],
                    config=config,
                )

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @Utility.performance_monitor.monitor_operation(operation_name="Gemini")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: List[str] = [],
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Sends a request to the Gemini API and handles the response.

        Args:
            input_messages: List of conversation messages including user queries
            queue: Queue for receiving streaming events (enables streaming mode if provided)
            stream_event: Event to signal when streaming is complete
            model_setting: Optional settings to override default model configuration

        Returns:
            str: Response ID for non-streaming requests
            None: For streaming requests

        Raises:
            Exception: If request processing fails
        """
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start

            # Reset reasoning_summary for new conversation turn
            # Recursive calls (function call loops) will continue accumulating
            if "reasoning_summary" in self.final_output:
                del self.final_output["reasoning_summary"]

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            if not self.client:
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error("No Gemini client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation - use .hex instead of str() conversion
            run_id = f"run-gemini-{self.model}-{timestamp}-{uuid.uuid4().hex[:8]}"

            _input_messages = self._process_input_messages(input_messages)

            # Process and append any input files to the last user message
            if input_files and _input_messages and _input_messages[-1].role == "user":
                for input_file in input_files:
                    # Upload and convert file to Gemini-compatible format
                    uploaded_file = self.insert_file(**input_file)
                    _input_messages[-1].parts.append(
                        types.Part(inline_data=uploaded_file),
                    )
                    self.uploaded_files.append({"file_name": uploaded_file.name})

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            response = self.invoke_model(
                **{
                    "input": _input_messages,
                    "stream": stream,
                }
            )

            # If streaming is enabled, process chunks
            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(
                    response,
                    _input_messages,
                    stream_event=stream_event,
                )
                return None

            self.handle_response(response, _input_messages)
            return run_id
        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def _process_input_messages(
        self, input_messages: List[Dict[str, Any]]
    ) -> List[types.Content]:
        _input_messages = []
        for msg in list(
            filter(
                lambda msg: msg["role"]
                in ["user", "assistant", self.agent["tool_call_role"]],
                input_messages,
            )
        ):
            # Check if content can be loaded as JSON and convert if needed
            try:
                contents = Utility.json_loads(msg["content"])
                parts = []
                for content in contents:
                    if content["type"] == "input_text":
                        parts.append(types.Part(text=content["text"]))
                    elif content["type"] == "input_file":
                        file = self.get_file(**{"file_name": content["file_name"]})
                        parts.append(types.Part(inline_data=file))
                    else:
                        raise Exception(f"Unsupported content type: {content['type']}")

                _input_messages.append(
                    types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=parts,
                    )
                )
            except Exception:
                _input_messages.append(
                    types.Content(
                        role="user" if msg["role"] == "user" else "model",
                        parts=[types.Part(text=msg["content"])],
                    )
                )
        return _input_messages

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Processes and executes function calls from the model response.

        The function:
        1. Extracts and validates function call details
        2. Records the function call in async storage
        3. Processes and validates function arguments
        4. Executes the requested function
        5. Updates conversation history with results
        6. Handles any errors during execution

        Args:
            tool_call: Function call information from model response
            input_messages: Current conversation history
            stream_event: Event to signal streaming completion

        Returns:
            Updated input_messages with function results

        Raises:
            Exception: If function execution fails
        """
        # Track function call timing
        function_call_start = pendulum.now("UTC")

        try:
            # Extract function call metadata
            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation
            tool_call_id = (
                f"tool_call-gemini-{self.model}-{timestamp}-{uuid.uuid4().hex[:8]}"
            )
            function_call_data = {
                "id": tool_call_id,
                "arguments": tool_call.args,
                "name": tool_call.name,
                "type": "function_call",
            }

            # Record initial function call
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
                )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
                )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
                )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{function_call_data['name']}] Updating conversation history"
                )
            self._update_conversation_history(
                tool_call, function_output, input_messages
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": function_call_data["id"],
                                        "tool_type": function_call_data["type"],
                                        "name": function_call_data["name"],
                                        "arguments": arguments,
                                    },
                                    "output": function_output,
                                }
                            ),
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Log function call execution time
                function_call_end = pendulum.now("UTC")
                function_call_time = (
                    function_call_end - function_call_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {function_call_time:.2f}ms)"
                )

            return input_messages

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """
        Records the initial function call details in async storage.

        Args:
            function_call_data: Dictionary containing function call metadata including:
                - id: Unique identifier for the function call
                - type: Type of tool call
                - name: Name of the function being called
        """
        self.invoke_async_funct(
            "async_insert_update_tool_call",
            **{
                "tool_call_id": function_call_data["id"],
                "tool_type": function_call_data["type"],
                "name": function_call_data["name"],
            },
        )

    def _process_function_arguments(
        self, function_call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes and validates function arguments.

        Args:
            function_call_data: Dictionary containing function metadata and arguments

        Returns:
            Dictionary of processed arguments with added endpoint_id

        Raises:
            ValueError: If argument parsing fails
        """
        try:
            arguments = function_call_data.get("arguments", {})

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        """
        Executes the requested function and handles execution state.

        Args:
            function_call_data: Dictionary containing function metadata
            arguments: Processed arguments to pass to the function

        Returns:
            Function execution result or error message

        Raises:
            ValueError: If requested function is not supported
        """
        agent_function = self.get_function(function_call_data["name"])
        if not agent_function:
            raise ValueError(
                f"Unsupported function requested: {function_call_data['name']}"
            )

        try:
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            # Track actual function execution time
            function_exec_start = pendulum.now("UTC")
            function_output = agent_function(**arguments)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                function_exec_end = pendulum.now("UTC")
                function_exec_time = (
                    function_exec_end - function_exec_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' executed (took {function_exec_time:.2f}ms)"
                )

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "content": Utility.json_dumps(function_output),
                    "status": "completed",
                },
            )
            return function_output

        except Exception as e:
            log = traceback.format_exc()
            # Reuse cached JSON serialization (performance optimization)
            if "arguments_json" not in locals():
                arguments_json = Utility.json_dumps(arguments)
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "failed",
                    "notes": log,
                },
            )
            return f"Function execution failed: {e}"

    def _update_conversation_history(
        self,
        tool_call: Any,
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call results.

        Args:
            tool_call: Original function call data from model
            function_output: Result from function execution
            input_messages: Conversation history to update

        The function:
        1. Creates a function response part with execution results
        2. Appends the original function call to history
        3. Appends the function response to history
        """
        # Create a function response part
        function_response_part = types.Part.from_function_response(
            name=tool_call.name,
            response={
                "result": Utility.json_normalize(function_output, parser_number=False)
            },
        )

        # Append function call and result of the function execution to contents
        input_messages.append(
            types.Content(role="model", parts=[types.Part(function_call=tool_call)])
        )  # Append the model's function call message
        input_messages.append(
            types.Content(role="user", parts=[function_response_part])
        )  # Append the function response

    def _check_retry_limit(self, retry_count: int) -> None:
        """
        Check if retry limit has been exceeded and raise exception if so.

        Args:
            retry_count: Current retry count

        Raises:
            Exception: If retry_count exceeds MAX_RETRIES
        """
        MAX_RETRIES = 5
        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _has_valid_content(self, text: str) -> bool:
        """
        Check if response text contains valid content.

        Args:
            text: Response text to check

        Returns:
            True if text is not None/empty/whitespace-only, False otherwise
        """
        return bool(text and text.strip())

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> None:
        """
        Processes a non-streaming model response.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty response → Retry up to 5 times
        3. Valid response → Set final_output

        Args:
            response: Complete model response object
            input_messages: Current conversation history
            retry_count: Current retry count (max 5 retries)
        """
        self._check_retry_limit(retry_count)

        candidate = response.candidates[0]
        has_function_call = any(
            hasattr(part, "function_call") and part.function_call
            for part in candidate.content.parts
        )

        # Extract and store reasoning/thinking if present
        reasoning_parts = []
        for part in candidate.content.parts:
            if hasattr(part, "thought") and part.thought:
                try:
                    reasoning_parts.append(part.text)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_response] Captured reasoning: {part.text[:100]}..."
                        )
                except Exception as e:
                    if self.logger.isEnabledFor(logging.ERROR):
                        self.logger.error(f"Failed to process reasoning: {e}")

        # Store reasoning summary if present
        if reasoning_parts:
            reasoning_text = "\n".join(reasoning_parts)
            if self.final_output.get("reasoning_summary"):
                # Accumulate reasoning from multiple rounds (e.g., function calls)
                self.final_output["reasoning_summary"] = (
                    self.final_output["reasoning_summary"] + "\n" + reasoning_text
                )
            else:
                self.final_output["reasoning_summary"] = reasoning_text

        # Scenario 1: Handle function calls
        if has_function_call:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    input_messages = self.handle_function_call(
                        part.function_call, input_messages
                    )
                elif part.text:
                    input_messages.append(
                        types.Content(role="model", parts=[types.Part(text=part.text)])
                    )

            # Recurse with fresh response (reset retry count)
            next_response = self.invoke_model(
                **{"input": input_messages, "stream": False}
            )
            self.handle_response(next_response, input_messages, retry_count=0)
            return

        # Scenario 2: Empty response - retry
        if not self._has_valid_content(response.text):
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(
                    f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
                )
            next_response = self.invoke_model(
                **{"input": input_messages, "stream": False}
            )
            self.handle_response(
                next_response, input_messages, retry_count=retry_count + 1
            )
            return

        # Scenario 3: Valid response - set final output
        timestamp = pendulum.now("UTC").int_timestamp
        # Optimized UUID generation
        message_id = f"msg-gemini-{self.model}-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.final_output.update(
            {
                "message_id": message_id,
                "role": "assistant",
                "content": response.text,
            }
        )

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> None:
        """
        Processes streaming responses from the model.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty stream → Retry up to 5 times
        3. Valid stream → Accumulate and set final_output

        Args:
            response_stream: Iterator of response chunks from model
            input_messages: Current conversation history
            stream_event: Event to signal streaming completion
            retry_count: Current retry count (max 5 retries)
        """
        self._check_retry_limit(retry_count)

        # Initialize state
        timestamp = pendulum.now("UTC").int_timestamp
        # Optimized UUID generation
        message_id = f"msg-gemini-{self.model}-{timestamp}-{uuid.uuid4().hex[:8]}"
        tool_call = None
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        received_any_content = False

        # Reasoning tracking variables (matching OpenAI and Ollama handler patterns)
        reasoning_no = 0
        reasoning_index = 0
        accumulated_reasoning_parts = []
        accumulated_partial_reasoning_text = ""
        reasoning_started = False

        # Use cached output format type (performance optimization)
        output_format = self.output_format_type
        index = 0
        message_started = False

        # Resume from previous assistant message if exists
        if self.assistant_messages:
            index = self.assistant_messages[-1]["index"]
            message_id = self.assistant_messages[-1]["message_id"]
            self.send_data_to_stream(
                index=index, data_format=output_format, chunk_delta=" "
            )
            index += 1
            message_started = True

        try:
            # Process stream chunks
            for chunk in response_stream:
                candidate = chunk.candidates[0]

                # Process each part in the candidate
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        # Check if this is a reasoning/thought part
                        if hasattr(part, "thought") and part.thought:
                            # Start reasoning block if not started
                            if not reasoning_started:
                                reasoning_started = True
                                self.send_data_to_stream(
                                    index=reasoning_index,
                                    data_format=output_format,
                                    chunk_delta=f"<ReasoningStart Id={reasoning_no}/>",
                                    suffix=f"rs#{reasoning_no}",
                                )
                                reasoning_index += 1

                                if (
                                    self.enable_timeline_log
                                    and self.logger.isEnabledFor(logging.INFO)
                                ):
                                    elapsed = self._get_elapsed_time()
                                    self.logger.info(
                                        f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning started"
                                    )

                            # Skip empty thought text
                            if not part.text:
                                continue

                            received_any_content = True
                            reasoning_text = part.text

                            # Accumulate reasoning text
                            print(reasoning_text, end="", flush=True)
                            accumulated_reasoning_parts.append(reasoning_text)
                            accumulated_partial_reasoning_text += reasoning_text

                            # Process and send reasoning text
                            reasoning_index, accumulated_partial_reasoning_text = (
                                self.process_text_content(
                                    reasoning_index,
                                    accumulated_partial_reasoning_text,
                                    output_format,
                                    suffix=f"rs#{reasoning_no}",
                                )
                            )

                        # Detect function calls
                        elif hasattr(part, "function_call") and part.function_call:
                            tool_call = part.function_call
                            received_any_content = True

                # Check if reasoning block has ended (regular content starts arriving)
                if (
                    reasoning_started
                    and chunk.text
                    and not (
                        candidate.content
                        and candidate.content.parts
                        and any(
                            hasattr(p, "thought") and p.thought
                            for p in candidate.content.parts
                        )
                    )
                ):
                    # End reasoning block
                    if len(accumulated_partial_reasoning_text) > 0:
                        self.send_data_to_stream(
                            index=reasoning_index,
                            data_format=output_format,
                            chunk_delta=accumulated_partial_reasoning_text,
                            suffix=f"rs#{reasoning_no}",
                        )
                        accumulated_partial_reasoning_text = ""
                        reasoning_index += 1

                    self.send_data_to_stream(
                        index=reasoning_index,
                        data_format=output_format,
                        chunk_delta=f"<ReasoningEnd Id={reasoning_no}/>",
                        suffix=f"rs#{reasoning_no}",
                    )
                    reasoning_no += 1
                    reasoning_started = False

                    if self.enable_timeline_log and self.logger.isEnabledFor(
                        logging.INFO
                    ):
                        elapsed = self._get_elapsed_time()
                        self.logger.info(
                            f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning ended"
                        )

                # Skip empty text chunks
                if not chunk.text:
                    continue

                # Skip thought chunks (already processed above)
                if (
                    candidate.content
                    and candidate.content.parts
                    and any(
                        hasattr(p, "thought") and p.thought
                        for p in candidate.content.parts
                    )
                ):
                    continue

                received_any_content = True

                # Start message on first text chunk
                if not message_started:
                    # Sync index with reasoning_index when starting content after reasoning
                    if index == 0 and reasoning_index > 0:
                        index = reasoning_index + 1

                    self.send_data_to_stream(index=index, data_format=output_format)
                    index += 1
                    message_started = True

                # Process text based on output format
                print(chunk.text, end="", flush=True)
                # Append to list instead of string concatenation (performance optimization)
                accumulated_text_parts.append(chunk.text)

                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json += chunk.text
                    # Temporarily build accumulated_text for processing
                    temp_accumulated_text = "".join(accumulated_text_parts)
                    index, temp_accumulated_text, accumulated_partial_json = (
                        self.process_and_send_json(
                            index,
                            temp_accumulated_text,
                            accumulated_partial_json,
                            output_format,
                        )
                    )
                else:
                    accumulated_partial_text += chunk.text
                    index, accumulated_partial_text = self.process_text_content(
                        index, accumulated_partial_text, output_format
                    )

            # Send any remaining partial text
            if accumulated_partial_text:
                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                    chunk_delta=accumulated_partial_text,
                )
                index += 1

            # Build final accumulated text from parts (performance optimization)
            final_accumulated_text = "".join(accumulated_text_parts)

            # Store accumulated reasoning summary if present
            if accumulated_reasoning_parts:
                final_reasoning_text = "".join(accumulated_reasoning_parts)
                if self.final_output.get("reasoning_summary"):
                    # Accumulate reasoning from multiple rounds (e.g., function calls)
                    self.final_output["reasoning_summary"] = (
                        self.final_output["reasoning_summary"]
                        + "\n"
                        + final_reasoning_text
                    )
                else:
                    self.final_output["reasoning_summary"] = final_reasoning_text

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"[handle_stream] Stored reasoning summary: {final_reasoning_text[:100]}..."
                    )

            # Scenario 1: Handle function call
            if tool_call:
                if final_accumulated_text:
                    input_messages.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=final_accumulated_text)],
                        )
                    )
                    self.assistant_messages.append(
                        {
                            "message_id": message_id,
                            "content": final_accumulated_text,
                            "index": index,
                        }
                    )
                input_messages = self.handle_function_call(tool_call, input_messages)
                next_response = self.invoke_model(
                    **{"input": input_messages, "stream": bool(stream_event)}
                )
                self.handle_stream(
                    next_response, input_messages, stream_event, retry_count=0
                )
                return

            # Scenario 2: Empty stream - retry
            if not received_any_content:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
                    )
                next_response = self.invoke_model(
                    **{"input": input_messages, "stream": bool(stream_event)}
                )
                self.handle_stream(
                    next_response, input_messages, stream_event, retry_count + 1
                )
                return

            # Scenario 3: Valid stream - finalize
            self.send_data_to_stream(
                index=index, data_format=output_format, is_message_end=True
            )

            # Merge assistant messages (use list for efficient concatenation)
            merged_parts = []
            while self.assistant_messages:
                assistant_message = self.assistant_messages.pop()
                merged_parts.insert(0, assistant_message["content"])
                merged_parts.insert(1, " ")

            if merged_parts:
                merged_parts.append(final_accumulated_text)
                final_accumulated_text = "".join(merged_parts)

            # Use update() instead of assignment to preserve reasoning_summary
            self.final_output.update(
                {
                    "message_id": message_id,
                    "role": "assistant",
                    "content": final_accumulated_text,
                }
            )

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
                self.logger.error(f"Error in handle_stream: {str(e)}")
            # Build final text from parts even on error
            final_text = (
                "".join(accumulated_text_parts) if accumulated_text_parts else ""
            )
            # Use update() instead of assignment to preserve reasoning_summary
            self.final_output.update(
                {
                    "message_id": message_id,
                    "role": "assistant",
                    "content": final_text,
                }
            )
            raise
        finally:
            if stream_event:
                stream_event.set()

    def insert_file(self, **kwargs: Dict[str, Any]) -> types.File:
        if "encoded_content" in kwargs:
            encoded_content = kwargs["encoded_content"]
            # Decode the Base64 string
            decoded_content = base64.b64decode(encoded_content)

            # Save the decoded content into a BytesIO object
            content_io = BytesIO(decoded_content)

            # Assign a filename to the BytesIO object
            content_io.name = kwargs["filename"]
        elif "file_uri" in kwargs:
            # Use HTTP/2 for better performance when downloading files
            with httpx.Client(http2=True, timeout=30.0) as http_client:
                content_io = BytesIO(http_client.get(kwargs["file_uri"]).content)
        else:
            raise Exception("No file content provided")

        file = self.client.files.upload(
            file=content_io,
            config=types.UploadFileConfig(mime_type=kwargs["mime_type"]),
        )
        return file

    def get_file(self, **kwargs: Dict[str, Any]) -> types.File:
        file = self.client.files.get(name=kwargs["file_name"])
        return file
