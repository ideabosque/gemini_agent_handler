#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import logging
import threading
import traceback
import uuid
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

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

        self.client = genai.Client(api_key=agent["configuration"].get("api_key"))
        self.model = agent["configuration"].get("model")

        if any(
            tool["name"] == "google_search"
            for tool in agent["configuration"].get("tools", [])
        ):
            tools = [types.Tool(google_search=types.GoogleSearch())]
        else:
            tools = [
                types.Tool(
                    function_declarations=agent["configuration"].get("tools", [])
                )
            ]

        self.model_setting = dict(
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in agent["configuration"].items()
                if k not in ["api_key", "tools", "model", "text"]
            },
            **{
                "system_instruction": agent["instructions"],
                "tools": tools,
            },
        )
        self.assistant_messages = []

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

            config = types.GenerateContentConfig(**self.model_setting)

            if kwargs.get("stream"):
                return self.client.models.generate_content_stream(
                    model=self.model,
                    contents=kwargs["input"],
                    config=config,
                )

            return self.client.models.generate_content(
                model=self.model,
                contents=kwargs["input"],
                config=config,
            )
        except Exception as e:
            self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
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
        try:
            if not self.client:
                self.logger.error("No Gemini client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            timestamp = pendulum.now("UTC").int_timestamp
            run_id = f"run-gemini-{self.model}-{timestamp}-{str(uuid.uuid4())[:8]}"

            _input_messages = [
                types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part(text=msg["content"])],
                )
                for msg in input_messages
                if msg["role"] in ["user", "assistant", self.agent["tool_call_role"]]
            ]
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
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
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
        try:
            # Extract function call metadata
            timestamp = pendulum.now("UTC").int_timestamp
            tool_call_id = (
                f"tool_call-gemini-{self.model}-{timestamp}-{str(uuid.uuid4())[:8]}"
            )
            function_call_data = {
                "id": tool_call_id,
                "arguments": tool_call.args,
                "name": tool_call.name,
                "type": "function_call",
            }

            # Record initial function call
            self.logger.info(
                f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
            )
            self._record_function_call_start(function_call_data)

            # Parse and process arguments
            self.logger.info(
                f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
            )
            arguments = self._process_function_arguments(function_call_data)

            # Execute function and handle result
            self.logger.info(
                f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
            )
            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
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

            return input_messages

        except Exception as e:
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
            arguments["endpoint_id"] = self._endpoint_id

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
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "in_progress",
                },
            )

            function_output = agent_function(**arguments)

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
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
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
                "result": Utility.json_loads(
                    Utility.json_dumps(function_output), parser_number=False
                )
            },
        )

        # Append function call and result of the function execution to contents
        input_messages.append(
            types.Content(role="model", parts=[types.Part(function_call=tool_call)])
        )  # Append the model's function call message
        input_messages.append(
            types.Content(role="user", parts=[function_response_part])
        )  # Append the function response

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes a non-streaming model response.

        Args:
            response: Complete model response object
            input_messages: Current conversation history

        The function:
        1. Checks for function calls in the response
        2. Handles any function calls found
        3. Updates conversation with text responses
        4. Stores final output message
        """

        candidate = response.candidates[0]

        if any(
            hasattr(part, "function_call") and part.function_call
            for part in candidate.content.parts
        ):
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_call = part.function_call

                    input_messages = self.handle_function_call(
                        tool_call, input_messages
                    )
                else:
                    input_messages.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=part.text)],
                        )
                    )

            response = self.invoke_model(**{"input": input_messages, "stream": False})
            self.handle_response(response, input_messages)
        else:
            timestamp = pendulum.now("UTC").int_timestamp
            message_id = f"msg-gemini-{self.model}-{timestamp}-{str(uuid.uuid4())[:8]}"
            self.final_output = {
                "message_id": message_id,
                "role": "assistant",
                "content": response.text,
            }

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming responses from the model.

        Args:
            response_stream: Iterator of response chunks from model
            input_messages: Current conversation history
            stream_event: Event to signal streaming completion

        The function:
        1. Accumulates text chunks from the stream
        2. Handles different output formats (text, JSON)
        3. Processes any function calls found in stream
        4. Sends incremental updates via WebSocket
        5. Maintains conversation state and history
        """
        message_id = None
        tool_call = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        output_format = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )
        index = 0
        if self.assistant_messages:
            index = self.assistant_messages[-1]["index"]
            message_id = self.assistant_messages[-1]["message_id"]
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=" ",
            )
            index += 1

        for chunk in response_stream:

            # if chunk.candidates:
            candidate = chunk.candidates[0]
            tool_call = (
                candidate.content.parts[0].function_call
                if candidate.content.parts
                else None
            )

            if not chunk.text:
                continue

            if not message_id:
                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                )
                index += 1

                timestamp = pendulum.now("UTC").int_timestamp
                message_id = (
                    f"msg-gemini-{self.model}-{timestamp}-{str(uuid.uuid4())[:8]}"
                )

            print(chunk.text, end="", flush=True)
            if output_format in ["json_object", "json_schema"]:
                accumulated_partial_json += chunk.text
                index, self.accumulated_text, accumulated_partial_json = (
                    self.process_and_send_json(
                        index,
                        self.accumulated_text,
                        accumulated_partial_json,
                        output_format,
                    )
                )
            else:
                self.accumulated_text += chunk.text
                accumulated_partial_text += chunk.text
                # Check if text contains XML-style tags and update format
                index, accumulated_partial_text = self.process_text_content(
                    index, accumulated_partial_text, output_format
                )

        if len(accumulated_partial_text) > 0:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=accumulated_partial_text,
            )
            accumulated_partial_text = ""
            index += 1

        # Handle tool usage if detected
        if tool_call:
            if self.accumulated_text:
                input_messages.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=self.accumulated_text)],
                    )
                )
                self.assistant_messages.append(
                    {
                        "message_id": message_id,
                        "content": self.accumulated_text,
                        "index": index,
                    }
                )
            input_messages = self.handle_function_call(tool_call, input_messages)
            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": bool(stream_event),
                }
            )
            self.handle_stream(
                response, input_messages=input_messages, stream_event=stream_event
            )
            return

        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        while self.assistant_messages:
            assistant_message = self.assistant_messages.pop()
            self.accumulated_text = (
                assistant_message["content"] + " " + self.accumulated_text
            )

        self.final_output = {
            "message_id": message_id,
            "role": "assistant",
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
