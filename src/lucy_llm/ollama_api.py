from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config_manager import ConfigManager
from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .openai_responses import _sleep_backoff


class OllamaApi(LLMApi):
    """Ollama API implementation using OpenAI-compatible endpoint.

    Ollama exposes an OpenAI-compatible Chat Completions API at
    http://localhost:11434/v1. Tool calling works with the standard
    {type: function, function: {...}} format.

    No API key needed — Ollama uses a placeholder key.
    No image processing support.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        *,
        client: Optional[OpenAI] = None,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        base_url: Optional[str] = None,
    ) -> None:
        self._client = client or self._build_default_client(base_url)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        # Store conversation context by response_id
        self._conversation_context: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _build_default_client(base_url: Optional[str] = None) -> OpenAI:
        url = base_url
        if url is None:
            url = OllamaApi._resolve_base_url()
        return OpenAI(
            api_key="ollama",  # placeholder — Ollama doesn't require a real key
            base_url=url,
        )

    @staticmethod
    def _resolve_base_url() -> str:
        """Resolve the Ollama base URL from config.json, falling back to the class constant.

        The URL is an endpoint, not a secret — it lives in config.json as
        "ollama_base_url". A config/local failure must never crash client
        construction (and therefore the stream), so any exception during
        config access falls back to OLLAMA_BASE_URL.
        """
        try:
            config = ConfigManager("config.json")
            url = config.get("ollama_base_url")
            if url:
                return url
        except Exception as e:
            logging.warning(
                "OllamaApi: failed to read ollama_base_url from config; "
                "falling back to %s (%s)",
                OllamaApi.OLLAMA_BASE_URL,
                e,
            )
        return OllamaApi.OLLAMA_BASE_URL

    # ------------------------------------------------------------------
    # supports_image_processing
    # ------------------------------------------------------------------

    def supports_image_processing(self, model: str) -> bool:
        """Ollama models generally don't support image processing via the Chat API."""
        return False

    # ------------------------------------------------------------------
    # Tool format — Ollama uses standard OpenAI format, no transform needed
    # ------------------------------------------------------------------

    def _convert_tool_calls_to_assistant_message(
        self, tool_calls: List[ToolCall]
    ) -> Dict[str, Any]:
        """Convert ToolCall objects to an assistant message with tool_calls."""
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append({
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments_json,
                },
            })

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": formatted_tool_calls,
        }

    @staticmethod
    def _reconstruct_tool_calls_from_outputs(
        items: List[Dict[str, Any]],
    ) -> List[ToolCall]:
        """Build minimal ToolCall stubs from function_call_output items."""
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
        previous_tool_calls: Optional[List[ToolCall]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert various input formats to a list of messages for Ollama."""

        # Case 1: Input is a list of tool outputs from the processor
        if isinstance(input, list) and input and isinstance(input[0], dict):
            if "type" in input[0] and input[0].get("type") == "function_call_output":
                context_messages = []
                if previous_response_id and previous_response_id in self._conversation_context:
                    context_messages = self._conversation_context[previous_response_id].copy()
                    logging.info(
                        "OllamaApi: retrieved %d messages from context for response_id=%s",
                        len(context_messages),
                        previous_response_id,
                    )

                if previous_tool_calls:
                    assistant_msg = self._convert_tool_calls_to_assistant_message(previous_tool_calls)
                    context_messages.append(assistant_msg)
                elif not context_messages:
                    reconstructed = self._reconstruct_tool_calls_from_outputs(input)
                    if reconstructed:
                        logging.warning(
                            "OllamaApi: _conversation_context missing for response_id=%s "
                            "and no previous_tool_calls in metadata — reconstructed %d tool_calls",
                            previous_response_id,
                            len(reconstructed),
                        )
                        assistant_msg = self._convert_tool_calls_to_assistant_message(reconstructed)
                        context_messages.append(assistant_msg)

                for item in input:
                    if item.get("type") == "function_call_output":
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": item.get("output", ""),
                        }
                        context_messages.append(tool_message)

                logging.info(
                    "OllamaApi: built conversation with %d total messages",
                    len(context_messages),
                )
                return context_messages

        # Case 2: Input is already a list of messages — pass through
        if isinstance(input, list):
            return input

        # Case 3: Input is a single message
        if isinstance(input, dict):
            return [input]

        # Case 4: Unknown format
        logging.warning("OllamaApi: unexpected input type: %s", type(input))
        return []

    def _validate_and_fix_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensure all messages have required fields and proper format."""
        fixed_messages = []

        for i, msg in enumerate(messages):
            fixed_msg = dict(msg)

            if "role" not in fixed_msg:
                logging.error("Message at index %d missing 'role' field: %s", i, fixed_msg)
                continue

            if fixed_msg.get("role") == "assistant":
                if "tool_calls" in fixed_msg and fixed_msg["tool_calls"]:
                    for tc in fixed_msg["tool_calls"]:
                        if isinstance(tc, dict) and "function" not in tc:
                            if "name" in tc and "arguments" in tc:
                                tc["function"] = {
                                    "name": tc.pop("name"),
                                    "arguments": tc.pop("arguments"),
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
        # Extract previous tool_calls from metadata if available
        previous_tool_calls = None
        if metadata and "previous_tool_calls" in metadata:
            previous_tool_calls = metadata["previous_tool_calls"]

        # Normalize input to messages list
        messages = self._normalize_input_to_messages(
            input, previous_response_id, previous_tool_calls
        )

        # Validate and fix messages
        fixed_messages = self._validate_and_fix_messages(messages)

        logging.info(
            "OllamaApi: model=%s temperature=%s tool_choice=%s tools_count=%d messages_count=%d previous_response_id=%s",
            model,
            temperature,
            tool_choice,
            len(tools) if tools else 0,
            len(fixed_messages),
            previous_response_id,
        )

        if not fixed_messages:
            logging.error("OllamaApi: no messages to send after normalization")
            raise ValueError("No messages to send to Ollama API")

        for attempt in range(self._max_attempts):
            try:
                request_params = {
                    "model": model,
                    "messages": fixed_messages,
                    "temperature": temperature,
                }

                if tools:
                    request_params["tools"] = tools

                if tool_choice:
                    request_params["tool_choice"] = tool_choice

                resp = self._client.chat.completions.create(**request_params)

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
                    if tool_calls_list:
                        assistant_msg = self._convert_tool_calls_to_assistant_message(tool_calls_list)
                        self._conversation_context[response_id].append(assistant_msg)
                    logging.debug(
                        "OllamaApi: stored context for response_id=%s with %d messages",
                        response_id,
                        len(self._conversation_context[response_id]),
                    )

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
                    "OllamaApi: response_id=%s model=%s output_text_len=%d tool_calls=%d",
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
                    "OllamaApi: attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_attempts,
                    e,
                )
                if attempt == self._max_attempts - 1:
                    logging.error(
                        "OllamaApi: failed with messages count: %d",
                        len(fixed_messages),
                    )
                    if fixed_messages:
                        for i, msg in enumerate(fixed_messages[:3]):
                            logging.error(
                                "OllamaApi: message[%d]: role=%s keys=%s",
                                i,
                                msg.get("role"),
                                list(msg.keys()),
                            )
                    raise
                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        raise RuntimeError("OllamaApi: exhausted retries")
