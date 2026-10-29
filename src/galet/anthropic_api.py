from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .openai_responses import _sleep_backoff
from .settings import Settings, default_settings

_MODERN_MODEL_ID_RE = re.compile(
    r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?=$|-)",
    re.IGNORECASE,
)
_LEGACY_MODEL_ID_RE = re.compile(r"^claude-(\d+)(?:[.-](\d{1,2}))?(?=$|-)", re.IGNORECASE)

_FIRST_VERSION_WITHOUT_SAMPLING_PARAMS = (4, 7)
_FIRST_VERSION_WITHOUT_FORCED_TOOL_CHOICE = (5, 5)
_FORCED_TOOL_CHOICE_FAMILIES = ("opus", "sonnet")
_ALWAYS_AUTO_TOOL_CHOICE_FAMILIES = ("fable", "mythos")
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_NON_RETRYABLE_ERROR_TYPES = (TypeError, ValueError, KeyError, AttributeError, AssertionError)


def _model_version(model: str) -> Optional[Tuple[Optional[str], Tuple[int, int]]]:
    text = (model or "").strip()
    match = _MODERN_MODEL_ID_RE.match(text)
    if match:
        return match.group(1).lower(), (int(match.group(2)), int(match.group(3) or 0))
    match = _LEGACY_MODEL_ID_RE.match(text)
    if match:
        return None, (int(match.group(1)), int(match.group(2) or 0))
    return None


def _sampling_params_supported(model: str) -> bool:
    parsed = _model_version(model)
    if parsed is None:
        return True
    return parsed[1] < _FIRST_VERSION_WITHOUT_SAMPLING_PARAMS


def _forced_tool_choice_supported(model: str) -> bool:
    parsed = _model_version(model)
    if parsed is None:
        return True
    family, version = parsed
    if family in _ALWAYS_AUTO_TOOL_CHOICE_FAMILIES:
        return False
    if family in _FORCED_TOOL_CHOICE_FAMILIES:
        return version < _FIRST_VERSION_WITHOUT_FORCED_TOOL_CHOICE
    return True


def _is_retryable(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES or status >= 500
    return not isinstance(error, _NON_RETRYABLE_ERROR_TYPES)


def _extract_usage(usage_obj: Any) -> Optional[LLMUsage]:
    if usage_obj is None:
        return None
    input_tokens = getattr(usage_obj, "input_tokens", None)
    output_tokens = getattr(usage_obj, "output_tokens", None)
    total_tokens = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total_tokens = input_tokens + output_tokens
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw=usage_obj,
    )


def _extract_tool_calls(content: Any) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for block in content or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        raw_input = getattr(block, "input", None)
        if raw_input is None:
            arguments_json = "{}"
        elif isinstance(raw_input, str):
            arguments_json = raw_input
        else:
            try:
                arguments_json = json.dumps(raw_input, ensure_ascii=False)
            except Exception:
                arguments_json = str(raw_input)
        calls.append(
            ToolCall(
                call_id=str(getattr(block, "id", "")),
                name=str(getattr(block, "name", "")),
                arguments_json=arguments_json,
            )
        )
    return calls


def _extract_text(content: Any) -> str:
    parts: List[str] = []
    for block in content or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


class AnthropicApi(LLMApi):
    """Anthropic Messages API implementation for Claude models.

    The Messages API is stateless. Tool-calling continuations are rebuilt from
    an in-memory conversation cache keyed by response id, so the assistant turn,
    including any thinking blocks, is replayed unmodified.
    """

    DEFAULT_MAX_TOKENS = 16000

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        cache_size: int = 1000,
        settings: Optional[Settings] = None,
    ) -> None:
        self._client = client
        self._settings = settings or default_settings
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._max_cache_size = cache_size
        self._conversation_context: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_default_client(self._settings)
        return self._client

    @staticmethod
    def _build_default_client(settings: Optional[Settings] = None) -> Any:
        if Anthropic is None:
            raise RuntimeError(
                "The anthropic SDK is not installed; install it with: "
                "pip install 'galet[anthropic]'"
            )
        return Anthropic(api_key=(settings or default_settings).api_key("anthropic"))

    def supports_image_processing(self, model: str) -> bool:
        return True

    @staticmethod
    def _convert_part(part: Any) -> Any:
        if not isinstance(part, dict) or part.get("type") != "image":
            return part
        source = part.get("source")
        if not isinstance(source, dict) or "type" in source:
            return part
        url = source.get("url") or source.get("uri")
        if url:
            return {"type": "image", "source": {"type": "url", "url": url}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": source.get("mime_type") or "image/png",
                "data": source.get("data", ""),
            },
        }

    @staticmethod
    def _convert_content(content: Any) -> Any:
        if isinstance(content, list):
            return [AnthropicApi._convert_part(part) for part in content]
        return content

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(texts)

    @staticmethod
    def _to_anthropic_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if "input_schema" in tool or tool.get("type") not in (None, "function"):
                converted.append(tool)
                continue
            func = tool.get("function")
            if not isinstance(func, dict):
                func = tool
            name = func.get("name")
            if not name:
                continue
            definition: Dict[str, Any] = {"name": name}
            if "description" in func:
                definition["description"] = func["description"]
            params = func.get("parameters")
            schema = dict(params) if isinstance(params, dict) else {"type": "object", "properties": {}}
            schema.pop("strict", None)
            definition["input_schema"] = schema
            converted.append(definition)
        return converted or None

    @staticmethod
    def _to_tool_choice(model: str, tool_choice: Any) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            choice = tool_choice
        elif tool_choice == "auto":
            choice = {"type": "auto"}
        elif tool_choice == "none":
            choice = {"type": "none"}
        elif tool_choice in ("required", "any"):
            choice = {"type": "any"}
        else:
            choice = {"type": "tool", "name": str(tool_choice)}
        if choice.get("type") in ("any", "tool") and not _forced_tool_choice_supported(model):
            logging.warning(
                "AnthropicApi: model %s rejects forced tool_choice %r; using auto",
                model,
                tool_choice,
            )
            return {"type": "auto"}
        return choice

    @staticmethod
    def _tool_calls_to_assistant_message(tool_calls: Any) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        for tc in tool_calls:
            try:
                arguments = json.loads(getattr(tc, "arguments_json", "") or "{}")
            except ValueError:
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(getattr(tc, "call_id", "")),
                    "name": str(getattr(tc, "name", "")),
                    "input": arguments if isinstance(arguments, dict) else {},
                }
            )
        return {"role": "assistant", "content": blocks}

    def _normalize_input(
        self,
        input: Any,
        previous_response_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        if isinstance(input, (str, dict)):
            items: List[Any] = [input]
        elif isinstance(input, list):
            items = input
        else:
            raise ValueError(f"unsupported input type for Anthropic API: {type(input).__name__}")

        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                raise ValueError(f"unsupported input item for Anthropic API: {type(item).__name__}")
            if item.get("type") == "function_call_output":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(item.get("call_id", "")),
                        "content": str(item.get("output", "")),
                    }
                )
                continue
            role = item.get("role")
            if role in ("system", "developer"):
                text = self._content_text(item.get("content"))
                if text:
                    system_parts.append(text)
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": self._convert_content(item.get("content"))})
            else:
                raise ValueError(f"unsupported input item for Anthropic API: role={role!r}")

        system = "\n\n".join(system_parts)
        if not tool_results:
            return messages, system

        prefix: List[Dict[str, Any]] = []
        entry = self._conversation_context.get(previous_response_id) if previous_response_id else None
        if entry is not None:
            self._conversation_context.move_to_end(previous_response_id)
            prefix = list(entry["messages"])
            system = system or entry["system"]
        else:
            previous_tool_calls = (metadata or {}).get("previous_tool_calls")
            if previous_tool_calls:
                prefix = [self._tool_calls_to_assistant_message(previous_tool_calls)]
        if not prefix and not any(m["role"] == "assistant" for m in messages):
            raise ValueError(
                "tool outputs were supplied but the assistant turn that requested them is "
                f"unknown (previous_response_id={previous_response_id!r}); pass "
                "metadata['previous_tool_calls'] or include the assistant turn in the input"
            )
        return prefix + [{"role": "user", "content": tool_results}] + messages, system

    def _remember(self, response_id: str, messages: List[Dict[str, Any]], system: str) -> None:
        self._conversation_context[response_id] = {"messages": messages, "system": system}
        self._conversation_context.move_to_end(response_id)
        while len(self._conversation_context) > self._max_cache_size:
            self._conversation_context.popitem(last=False)

    def _call_with_retry(self, params: Dict[str, Any]) -> Any:
        client = self._get_client()
        last_err: Optional[BaseException] = None
        for attempt in range(self._max_attempts):
            try:
                return client.messages.create(**params)
            except Exception as e:
                last_err = e
                logging.warning(
                    "AnthropicApi: attempt %d/%d failed with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    type(e).__name__,
                    e,
                )
                if not _is_retryable(e) or attempt == self._max_attempts - 1:
                    raise
                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)
        if last_err is not None:
            raise last_err
        raise RuntimeError("AnthropicApi: exhausted retries unexpectedly")

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
        messages, system = self._normalize_input(input, previous_response_id, metadata)
        if not messages:
            raise ValueError("No messages to send to Anthropic API")

        anthropic_tools = self._to_anthropic_tools(tools)

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if anthropic_tools:
            params["tools"] = anthropic_tools
            choice = self._to_tool_choice(model, tool_choice)
            if choice:
                params["tool_choice"] = choice
        if temperature is not None:
            if _sampling_params_supported(model):
                params["temperature"] = temperature
            else:
                logging.warning(
                    "AnthropicApi: dropping unsupported sampling param temperature for model %s",
                    model,
                )
        if text:
            logging.debug("AnthropicApi: ignoring text format options")

        logging.info(
            "AnthropicApi.create_response: model=%s tools=%d tool_choice=%s messages=%d previous_response_id=%s",
            model,
            len(anthropic_tools) if anthropic_tools else 0,
            tool_choice,
            len(messages),
            previous_response_id,
        )

        resp = self._call_with_retry(params)

        content = list(getattr(resp, "content", None) or [])
        response_id = getattr(resp, "id", None)
        output_text = _extract_text(content)
        tool_calls = _extract_tool_calls(content)
        stop_reason = getattr(resp, "stop_reason", None)

        if tool_calls and response_id:
            self._remember(
                response_id,
                list(messages) + [{"role": "assistant", "content": content}],
                system,
            )

        if stop_reason == "max_tokens" and not output_text and not tool_calls:
            logging.warning(
                "AnthropicApi: response ended at max_tokens=%d with no text; thinking tokens "
                "count toward max_tokens, so consider a larger max_tokens",
                self._max_tokens,
            )
        elif stop_reason == "refusal":
            logging.warning("AnthropicApi: model refused the request (model=%s)", model)

        logging.info(
            "AnthropicApi: response_id=%s model=%s stop_reason=%s output_text_len=%d tool_calls=%d",
            response_id,
            getattr(resp, "model", None),
            stop_reason,
            len(output_text),
            len(tool_calls),
        )

        return LLMResponse(
            response_id=response_id,
            model=getattr(resp, "model", None),
            output_text=output_text,
            tool_calls=tool_calls,
            usage=_extract_usage(getattr(resp, "usage", None)),
            raw=resp,
        )
