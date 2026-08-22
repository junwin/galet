from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config_manager import ConfigManager
from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .openai_responses import _sleep_backoff
import logging


class MistralApi(LLMApi):
    """Mistral API implementation using OpenAI-compatible endpoint."""

    MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

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

        with open(os.path.join(credential_path, "mistral_cred.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return OpenAI(
            api_key=config_data["mistral_api_key"],
            base_url=MistralApi.MISTRAL_BASE_URL,
        )

    def supports_image_processing(self, model: str) -> bool:
        """Mistral models (e.g. Pixtral) support native image processing."""
        return True

    # ------------------------------------------------------------------
    # Content-part normalization (image support — Step 6)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_content_parts(content: Any) -> Any:
        """Normalize provider-agnostic content parts to Mistral format.

        Intermediate format:
            {"type": "image", "source": {"data": "<base64>", "mime_type": "image/png"}}

        Mistral format:
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,<data>"}}
        """
        if not isinstance(content, list):
            return content

        normalized = []
        for part in content:
            if not isinstance(part, dict):
                normalized.append(part)
                continue

            ptype = part.get("type", "")

            if ptype == "image":
                source = part.get("source", {})
                data = source.get("data", "") if isinstance(source, dict) else ""
                mime = source.get("mime_type", "image/png") if isinstance(source, dict) else "image/png"
                normalized.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })
            elif ptype == "text":
                normalized.append(part)
            else:
                normalized.append(part)

        return normalized

    @staticmethod
    def _normalize_messages(messages: Any) -> Any:
        """Normalize content parts in all messages."""
        if not isinstance(messages, list):
            return messages

        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                result.append(msg)
                continue

            content = msg.get("content")
            if isinstance(content, list):
                msg_copy = dict(msg)
                msg_copy["content"] = MistralApi._normalize_content_parts(content)
                result.append(msg_copy)
            else:
                result.append(msg)

        return result

    # ------------------------------------------------------------------
    # Tool format transform
    # ------------------------------------------------------------------

    def _transform_tools_for_mistral(self, tools: Optional[list[dict]]) -> Optional[list[dict]]:
        """Transform OpenAI tool format to Mistral format.

        Mistral uses the same OpenAI-compatible tool format, but does not
        support:
          - 'strict' at the top level of the tool definition
          - 'strict' or 'additionalProperties' in the parameter schema
          - The flat format: {"type": "function", "name": "...", ...}
            Mistral expects: {"type": "function", "function": {"name": "...", ...}}
        """
        if not tools:
            return None

        mistral_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function_def = {}

                # Case 1: Nested format (OpenAI Responses API style)
                #   {"type": "function", "function": {"name": "...", ...}}
                func = tool.get("function", {})
                if func:
                    if "name" in func:
                        function_def["name"] = func["name"]
                    if "description" in func:
                        function_def["description"] = func["description"]
                    if "parameters" in func:
                        params = self._clean_parameters(func["parameters"])
                        function_def["parameters"] = params
                else:
                    # Case 2: Flat format (from handler registry)
                    #   {"type": "function", "name": "...", "parameters": {...}, "strict": true}
                    if "name" in tool:
                        function_def["name"] = tool["name"]
                    if "description" in tool:
                        function_def["description"] = tool["description"]
                    if "parameters" in tool:
                        params = self._clean_parameters(tool["parameters"])
                        function_def["parameters"] = params

                mistral_tools.append({
                    "type": "function",
                    "function": function_def
                })
            else:
                mistral_tools.append(tool)

        return mistral_tools

    def _clean_parameters(self, params: Any) -> dict:
        """Remove Mistral-unsupported keys from a parameter schema."""
        if not isinstance(params, dict):
            return params if isinstance(params, dict) else {}
        cleaned = params.copy()
        cleaned.pop("strict", None)
        cleaned.pop("additionalProperties", None)
        return cleaned

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
        and no previous_tool_calls metadata was provided.
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
        """Convert various input formats to a list of messages for Mistral."""

        # Case 1: Input is a list of tool outputs from the processor
        if isinstance(input, list) and input and isinstance(input[0], dict):
            if "type" in input[0] and input[0].get("type") == "function_call_output":
                # This is a tool response. Combine with previous context.
                context_messages = []
                if previous_response_id and previous_response_id in self._conversation_context:
                    context_messages = self._conversation_context[previous_response_id].copy()
                    logging.info(f"MistralApi: retrieved {len(context_messages)} messages from context for response_id={previous_response_id}")

                # Add the assistant message with tool_calls if we have them
                if previous_tool_calls:
                    assistant_msg = self._convert_tool_calls_to_assistant_message(previous_tool_calls)
                    context_messages.append(assistant_msg)
                elif not context_messages:
                    # Defensive fallback: reconstruct tool_calls from the outputs themselves.
                    reconstructed = self._reconstruct_tool_calls_from_outputs(input)
                    if reconstructed:
                        logging.warning(
                            "MistralApi: _conversation_context missing for response_id=%s "
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

                logging.info(f"MistralApi: built conversation with {len(context_messages)} total messages")
                return context_messages

        # Case 2: Input is already a list of messages — normalize content parts
        if isinstance(input, list):
            return self._normalize_messages(input)

        # Case 3: Input is a single message
        if isinstance(input, dict):
            return [input]

        # Case 4: Unknown format
        logging.warning(f"MistralApi: unexpected input type: {type(input)}")
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
                    for tc in fixed_msg["tool_calls"]:
                        if isinstance(tc, dict) and "function" not in tc:
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
        # Transform tools to Mistral format
        mistral_tools = self._transform_tools_for_mistral(tools)

        # Extract previous tool_calls from metadata if available
        previous_tool_calls = None
        if metadata and "previous_tool_calls" in metadata:
            previous_tool_calls = metadata["previous_tool_calls"]

        # Normalize input to messages list
        messages = self._normalize_input_to_messages(input, previous_response_id, previous_tool_calls)

        # Validate and fix messages
        fixed_messages = self._validate_and_fix_messages(messages)

        logging.info("MistralApi: model=%s temperature=%s tool_choice=%s tools_count=%d messages_count=%d previous_response_id=%s",
                    model, temperature, tool_choice, len(mistral_tools) if mistral_tools else 0,
                    len(fixed_messages), previous_response_id)

        # Check for empty messages
        if not fixed_messages:
            logging.error("MistralApi: no messages to send after normalization")
            raise ValueError("No messages to send to Mistral API")

        for attempt in range(self._max_attempts):
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "messages": fixed_messages,
                    "temperature": temperature,
                }

                if mistral_tools:
                    request_params["tools"] = mistral_tools

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
                    logging.debug(f"MistralApi: stored context for response_id={response_id} with {len(self._conversation_context[response_id])} messages")

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
                    "MistralApi: response_id=%s model=%s output_text_len=%d tool_calls=%d",
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
                    "MistralApi: attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_attempts,
                    e,
                )
                if attempt == self._max_attempts - 1:
                    logging.error("MistralApi: failed with messages count: %d", len(fixed_messages))
                    if fixed_messages:
                        for i, msg in enumerate(fixed_messages[:3]):
                            logging.error(f"MistralApi: message[{i}]: role={msg.get('role')} keys={list(msg.keys())}")
                    raise
                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        raise RuntimeError("MistralApi: exhausted retries")
