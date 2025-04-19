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
    Manages conversations and function calls in real-time with OpenAI's API:
      - Streams partial text deltas.
      - Detects function calls embedded in the response.
      - Executes local functions as needed.
      - Maintains the conversation history (input_messages).
      - Stores the final generated message or function call outputs.
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        :param logger: A logging instance for debug/info messages.
        :param client: An OpenAI client instance or a compatible object.
        :param model: Default model name to use for requests (defaults to "gpt-4o").
        :param tools: Optional list of tool definitions the model may call.
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        self.logger = logger
        self.client = genai.Client(api_key=agent["configuration"].get("api_key"))
        self.model = agent["configuration"].get("model")
        self.model_setting = dict(
            {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in agent["configuration"].items()
                if k not in ["api_key", "tools", "model", "text"]
            },
            **{
                "system_instruction": agent["instructions"],
                "tools": [
                    types.Tool(
                        function_declarations=agent["configuration"].get("tools", [])
                    )
                ],
            },
        )

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the Ollama model with provided messages and handles tool calls

        Args:
            kwargs: Contains input messages and streaming configuration

        Returns:
            Model response or streaming response

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
        Sends a request to the Gemini API. If a queue is provided, we switch to streaming mode,
        otherwise, a simple (non-streaming) request is made.

        :param input_messages: Conversation history, including the latest user question.
        :param queue: An optional queue to receive streaming events. If provided, streaming is used.
        :param stream_event: An optional threading.Event to signal streaming completion.
        :return: The response ID if non-streaming, otherwise None.
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

            self.handle_output(response, _input_messages)
            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(
        self,
        tool_call: Any,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
    ) -> None:
        """
        Handles function calls from the model by:
        1. Validating and extracting function call details
        2. Executing the requested function with provided arguments
        3. Recording function execution status and results
        4. Updating conversation history
        5. Continuing the conversation with function results

        Args:
            tool_call: Function call data from model response
            input_messages: Conversation history to update
            queue: Optional queue for streaming responses
            stream_event: Optional event to signal streaming completion

        Raises:
            ValueError: If tool_call is invalid or function not supported
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

            # Continue conversation
            self.logger.info(
                f"[handle_function_call][{function_call_data['name']}] Continuing conversation"
            )
            self._continue_conversation(input_messages, stream_event)

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
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

        except Exception as e:
            self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """Records the initial function call details"""
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
        """Parses and processes function arguments"""
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
        """Executes the requested function and handles the result"""
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
        """Updates conversation history with function call and output"""
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

    def _continue_conversation(
        self,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
    ) -> None:
        """Continues conversation with updated inputs"""
        response = self.invoke_model(
            **{
                "input": input_messages,
                "stream": bool(stream_event),
            }
        )

        if stream_event:
            self.handle_stream(response, input_messages, stream_event)
        else:
            self.handle_output(response, input_messages)

    def handle_output(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes a single output object. If it's a message, we store it as final output.
        If it's a function call, we route to handle_function_call.

        :param output: The model's output object.
        :param input_messages: Conversation history for potential updates.
        :param queue: Optional queue if streaming is in use.
        :param stream_event: Optional event to signal streaming completion.
        """
        self.logger.info("Processing output: %s", response)

        candidate = response.candidates[0]
        tool_call = (
            candidate.content.parts[0].function_call
            if candidate.content.parts
            else None
        )

        if tool_call:
            self.handle_function_call(tool_call, input_messages)
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
        response_stream,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Iterates over each chunk in a streaming response:
          - Logs chunk details for debugging.
          - Detects 'response.created' to store a run ID in the queue.
          - Detects 'response.completed' to handle final outputs.
          - Captures partial text from 'response.output_text.delta'.
          - Notifies completion via stream_event at the end.

        :param response_stream: The streaming response object.
        :param input_messages: Conversation history for updates.
        :param queue: Optional queue to push events like 'response_id'.
        :param stream_event: Optional event to signal streaming completion.
        """
        message_id = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        output_format = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )
        index = 0

        for chunk in response_stream:

            # if chunk.candidates:
            candidate = chunk.candidates[0]
            tool_call = (
                candidate.content.parts[0].function_call
                if candidate.content.parts
                else None
            )

            if tool_call:
                self.handle_function_call(
                    tool_call, input_messages, stream_event=stream_event
                )
                break

            if not chunk.text:
                continue

            if not message_id:
                self.send_data_to_websocket(
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
                # Send incremental text chunk to WebSocket server
                if len(accumulated_partial_text) >= int(
                    self.setting.get("accumulated_partial_text_buffer", "10")
                ):
                    self.send_data_to_websocket(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1

        if len(accumulated_partial_text) > 0:
            self.send_data_to_websocket(
                index=index,
                data_format=output_format,
                chunk_delta=accumulated_partial_text,
            )
            accumulated_partial_text = ""
            index += 1

        self.send_data_to_websocket(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        self.final_output = {
            "message_id": message_id,
            "role": "assistant",
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
