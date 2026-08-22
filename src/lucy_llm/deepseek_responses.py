from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.config_manager import ConfigManager
from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .openai_responses import _sleep_backoff


class DeepSeekApi(LLMApi):
    """DeepSeek API implementation using OpenAI-compatible endpoint.

    DeepSeek is text-only. For image support, the FCP layer delegates
    to a vision-capable agent via delegate_tasks.
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        *,
        client: Optional[OpenAI] = None,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
    ) -> None:
        self._client = client or self._build_default_client()
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        # Store conversation context by response_id
        self._conversation_context: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _build_default_client() -> OpenAI:
        config = ConfigManager("config.json")
        credential_path = config.get("credential_path")

        with open(os.path.join(credential_path, "deepseek_cred.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return OpenAI(
            api_key=config_data["deepseek_api_key"],
            base_url=DeepSeekApi.DEEPSEEK_BASE_URL,
        )

    # ------------------------------------------------------------------
    # supports_image_processing
    # ------------------------------------------------------------------

    def supports_image_processing(self, model: str) -> bool:
        """DeepSeek is text-only — no native image processing."""
        return False

    # ------------------------------------------------------------------
    # Tool format transform
    # ------------------------------------------------------------------

    def _transform_tools_for_deepseek(self, tools: Optional[list[dict]]) -> Optional[list[dict]]:
        """Transform OpenAI tool format to DeepSeek format."""
        if not tools:
            return None

        deepseek_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function_def = {}

                if "name" in tool:
                    function_def["name"] = tool["name"]

                if "description" in tool:
                    function_def["description"] = tool["description"]

                if "parameters" in tool:
                    params = tool["parameters"].copy() if isinstance(tool["parameters"], dict) else {}
                    params.pop("strict", None)
                    params.pop("additionalProperties", None)
                    function_def["parameters"] = params

                deepseek_tools.append({
                    "type": "function",
                    "function": function_def
                })
            else:
                deepseek_tools.append(tool)

        return deepseek_tools

    def _convert_tool_calls_to_assistant_message(self, tool_calls: List[ToolCall]) -> Dict[str, Any]:
        """Convert ToolCall objects to an assistant message with tool_calls."""
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append({
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments_json
                }
            })

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": formatted_tool_calls
        }

    @staticmethod
    def _reconstruct_tool_calls_from_outputs(
        items: List[Dict[str, Any]],
    ) -> List[ToolCall]:
        """Build minimal ToolCall stubs from function_call_output items.

        Used as a defensive fallback when _conversation_context is missing
        and no previous_tool_calls metadata was provided. The call_id is
        the critical field — the API uses it to match tool messages to
        the assistant's tool_calls. Tool name is a placeholder since it
        isn't validated for this purpose.
        """
        seen: set[str] = set()
        stubs: List[ToolCall] = []
        for item in items:
            if item.get("type") != "function_call_output":
                continue
            call_id = item.get("call_id", "")
            if not call_id or call_id in seen:
                continue
            seen.add(call_id)
            stubs.append(ToolCall(
                call_id=str(call_id),
                name="__reconstructed__",
                arguments_json="{}",
            ))
        return stubs

    def _normalize_input_to_messages(
        self,
        input: Any,
        previous_response_id: Optional[str] = None,
        previous_tool_calls: Optional[List[ToolCall]] = None
    ) -> List[Dict[str, Any]]:
        """Convert various input formats to a list of messages for DeepSeek."""

        # Case 1: Input is a list of tool outputs from the processor
        if isinstance(input, list) and input and isinstance(input[0], dict):
            if "type" in input[0] and input[0].get("type") == "function_call_output":
                # This is a tool response. We need to combine with previous context.
                context_messages = []
                if previous_response_id and previous_response_id in self._conversation_context:
                    context_messages = self._conversation_context[previous_response_id].copy()
                    logging.info(f"DeepSeekApi: retrieved {len(context_messages)} messages from context for response_id={previous_response_id}")

                # Add the assistant message with tool_calls if we have them
                if previous_tool_calls:
                    assistant_msg = self._convert_tool_calls_to_assistant_message(previous_tool_calls)
                    context_messages.append(assistant_msg)
                elif not context_messages:
                    # Defensive fallback: reconstruct tool_calls from the outputs themselves.
                    # This handles the case where _conversation_context is empty (e.g.
                    # response_id was None, or context was never stored for this ID) and
                    # the caller didn't pass previous_tool_calls in metadata.
                    reconstructed = self._reconstruct_tool_calls_from_outputs(input)
                    if reconstructed:
                        logging.warning(
                            "DeepSeekApi: _conversation_context missing for response_id=%s "
                            "and no previous_tool_calls in metadata — reconstructed %d tool_calls from function_call_output items",
                            previous_response_id,
                            len(reconstructed),
                        )
                        assistant_msg = self._convert_tool_calls_to_assistant_message(reconstructed)
                        context_messages.append(assistant_msg)

                # Convert tool outputs to tool response messages
                for item in input:
                    if item.get("type") == "function_call_output":
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": item.get("output", "")
                        }
                        context_messages.append(tool_message)

                logging.info(f"DeepSeekApi: built conversation with {len(context_messages)} total messages")
                return context_messages

        # Case 2: Input is already a list of messages — pass through as-is
        if isinstance(input, list):
            return input

        # Case 3: Input is a single message
        if isinstance(input, dict):
            return [input]

        # Case 4: Unknown format
        logging.warning(f"DeepSeekApi: unexpected input type: {type(input)}")
        return []

    def _validate_and_fix_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure all messages have required fields and proper format."""
        fixed_messages = []

        for i, msg in enumerate(messages):
            fixed_msg = dict(msg)

            # Ensure role exists
            if "role" not in fixed_msg:
                logging.error(f"Message at index {i} missing 'role' field: {fixed_msg}")
                continue

            # Ensure assistant messages with tool_calls have proper structure
            if fixed_msg.get("role") == "assistant":
                if "tool_calls" in fixed_msg and fixed_msg["tool_calls"]:
                    # Ensure each tool call has the proper structure
                    for tc in fixed_msg["tool_calls"]:
                        if isinstance(tc, dict) and "function" not in tc:
                            # Convert from flat format to nested format if needed
                            if "name" in tc and "arguments" in tc:
                                tc["function"] = {
                                    "name": tc.pop("name"),
                                    "arguments": tc.pop("arguments")
                                }

            fixed_messages.append(fixed_msg)

        return fixed_messages

    def create_response(
        self,
        *,
        model: str,
        input: Any,
        temperature: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_response_id: Optional[str] = None,
        text: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        # Transform tools to DeepSeek format
        deepseek_tools = self._transform_tools_for_deepseek(tools)

        # Extract previous tool_calls from metadata if available
        previous_tool_calls = None
        if metadata and "previous_tool_calls" in metadata:
            previous_tool_calls = metadata["previous_tool_calls"]

        # Normalize input to messages list
        messages = self._normalize_input_to_messages(input, previous_response_id, previous_tool_calls)

        # Validate and fix messages
        fixed_messages = self._validate_and_fix_messages(messages)

        logging.info("DeepSeekApi: model=%s temperature=%s tool_choice=%s tools_count=%d messages_count=%d previous_response_id=%s",
                    model, temperature, tool_choice, len(deepseek_tools) if deepseek_tools else 0,
                    len(fixed_messages), previous_response_id)

        # Check for empty messages
        if not fixed_messages:
            logging.error("DeepSeekApi: no messages to send after normalization")
            raise ValueError("No messages to send to DeepSeek API")

        for attempt in range(self._max_attempts):
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "messages": fixed_messages,
                    "temperature": temperature,
                }

                if deepseek_tools:
                    request_params["tools"] = deepseek_tools

                if tool_choice:
                    request_params["tool_choice"] = tool_choice

                resp = self._client.chat.completions.create(**request_params)

                # Extract from chat completion response format
                response_id = getattr(resp, "id", None)
                resp_model = getattr(resp, "model", None)

                # Extract text from the first choice
                output_text = ""
                if resp.choices and len(resp.choices) > 0:
                    message = resp.choices[0].message
                    output_text = message.content or ""

                # Extract tool calls from the response
                tool_calls_list = []
                if resp.choices and len(resp.choices) > 0:
                    message = resp.choices[0].message
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            tool_calls_list.append(ToolCall(
                                call_id=tc.id,
                                name=tc.function.name,
                                arguments_json=tc.function.arguments,
                            ))

                # Store the conversation context for future tool responses
                if response_id:
                    self._conversation_context[response_id] = fixed_messages.copy()
                    # Also store the assistant's response if it had tool_calls
                    if tool_calls_list:
                        assistant_msg = self._convert_tool_calls_to_assistant_message(tool_calls_list)
                        self._conversation_context[response_id].append(assistant_msg)
                    logging.debug(f"DeepSeekApi: stored context for response_id={response_id} with {len(self._conversation_context[response_id])} messages")

                # Extract usage
                usage = None
                if hasattr(resp, "usage") and resp.usage:
                    usage = LLMUsage(
                        input_tokens=getattr(resp.usage, "prompt_tokens", None),
                        output_tokens=getattr(resp.usage, "completion_tokens", None),
                        total_tokens=getattr(resp.usage, "total_tokens", None),
                        raw=resp.usage,
                    )

                logging.info(
                    "DeepSeekApi: response_id=%s model=%s output_text_len=%d tool_calls=%d",
                    response_id,
                    resp_model,
                    len(output_text),
                    len(tool_calls_list),
                )

                return LLMResponse(
                    response_id=response_id,
                    model=resp_model,
                    output_text=output_text,
                    tool_calls=tool_calls_list,
                    usage=usage,
                    raw=resp,
                )

            except Exception as e:
                logging.warning(
                    "DeepSeekApi: attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_attempts,
                    e,
                )
                if attempt == self._max_attempts - 1:
                    logging.error("DeepSeekApi: failed with messages count: %d", len(fixed_messages))
                    if fixed_messages:
                        for i, msg in enumerate(fixed_messages[:3]):  # Log first 3 messages
                            logging.error(f"DeepSeekApi: message[{i}]: role={msg.get('role')} keys={list(msg.keys())}")
                    raise
                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        raise RuntimeError("DeepSeekApi: exhausted retries")
