from __future__ import annotations

from collections import OrderedDict
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    from google import genai
except Exception:
    class _GenaiClientStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            class _Interactions:
                def create(self, *a: Any, **k: Any) -> Any:
                    return None

            self.interactions = _Interactions()

    class genai:  # type: ignore
        Client = _GenaiClientStub


from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .openai_responses import _sleep_backoff
from .settings import Settings, default_settings

_TOOL_CHOICE_MAP: Dict[str, str] = {"auto": "auto", "required": "any", "none": "none"}


def _extract_tool_calls(interaction: Any) -> List[ToolCall]:
    calls: List[ToolCall] = []
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        if getattr(step, "type", None) != "function_call":
            continue
        arguments = getattr(step, "arguments", None)
        if arguments is None:
            arguments_json = ""
        elif isinstance(arguments, str):
            arguments_json = arguments
        else:
            try:
                arguments_json = json.dumps(arguments)
            except Exception:
                arguments_json = str(arguments)
        calls.append(
            ToolCall(
                call_id=str(getattr(step, "id", "")),
                name=str(getattr(step, "name", "")),
                arguments_json=arguments_json,
            )
        )
    return calls


def _extract_usage(usage_obj: Any) -> Optional[LLMUsage]:
    if usage_obj is None:
        return None
    return LLMUsage(
        input_tokens=getattr(usage_obj, "total_input_tokens", None),
        output_tokens=getattr(usage_obj, "total_output_tokens", None),
        total_tokens=getattr(usage_obj, "total_tokens", None),
        raw=usage_obj,
    )


class GeminiApi(LLMApi):
    def __init__(
        self,
        *,
        client: Optional[genai.Client] = None,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        cache_size: int = 1000,
        settings: Optional[Settings] = None,
    ) -> None:
        self._client = client
        self._settings = settings or default_settings
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._max_cache_size = cache_size
        self._system_instruction_cache: OrderedDict[str, str] = OrderedDict()

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = self._build_default_client(self._settings)
        return self._client

    @staticmethod
    def _build_default_client(settings: Optional[Settings] = None) -> genai.Client:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return genai.Client(api_key=api_key)

        cred_env = os.environ.get("GEMINI_CREDENTIALS")
        if cred_env:
            if os.path.isfile(cred_env):
                try:
                    with open(cred_env, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        key = data.get("gemini_api_key") or data.get("api_key")
                        if key:
                            return genai.Client(api_key=key)
                except Exception:
                    pass
            else:
                try:
                    data = json.loads(cred_env)
                    key = data.get("gemini_api_key") or data.get("api_key")
                    if key:
                        return genai.Client(api_key=key)
                except Exception:
                    pass

        try:
            key = (settings or default_settings).api_key("gemini")
            if key:
                return genai.Client(api_key=key)
        except Exception:
            pass

        return genai.Client()

    def supports_image_processing(self, model: str) -> bool:
        return True

    @staticmethod
    def _extract_system_instruction(input: Any) -> str:
        if isinstance(input, dict):
            input = [input]
        if not isinstance(input, list):
            return ""
        sections: List[str] = []
        for item in input:
            if not isinstance(item, dict) or item.get("role") != "system":
                continue
            content = item.get("content")
            if isinstance(content, str):
                sections.append(content)
            elif isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                sections.append("\n".join(texts))
        return "\n\n".join(sections)

    @staticmethod
    def _content_to_parts(content: Any) -> List[Any]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return [content]
        parts: List[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source")
                if isinstance(source, dict):
                    url = source.get("url") or source.get("uri")
                    mime_type = source.get("mime_type", "image/png")
                    if url:
                        parts.append({"type": "image", "uri": url, "mime_type": mime_type})
                        continue
                    data = source.get("data", "")
                else:
                    data = ""
                    mime_type = "image/png"
                parts.append({"type": "image", "data": data, "mime_type": mime_type})
            else:
                parts.append(part)
        return parts

    @staticmethod
    def _to_gemini_input(input: Any) -> Any:
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            input = [input]
        if not isinstance(input, list):
            return input
        items: List[Any] = []
        for item in input:
            if not isinstance(item, dict):
                items.append(item)
                continue
            if item.get("type") == "function_result":
                items.append(item)
                continue
            role = item.get("role")
            if role == "system":
                continue
            if role == "user":
                items.append(
                    {"type": "user_input", "content": GeminiApi._content_to_parts(item.get("content"))}
                )
            elif role in ("assistant", "model"):
                items.append(
                    {"type": "model_output", "content": GeminiApi._content_to_parts(item.get("content"))}
                )
            else:
                items.append(item)
        return items

    @staticmethod
    def _to_gemini_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
        if not tools:
            return None
        declarations: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            func = tool.get("function")
            if not isinstance(func, dict):
                func = tool
            function_def: Dict[str, Any] = {"type": "function"}
            if "name" in func:
                function_def["name"] = func["name"]
            if "description" in func:
                function_def["description"] = func["description"]
            if "parameters" in func:
                params = func["parameters"]
                if isinstance(params, dict):
                    params = params.copy()
                    params.pop("strict", None)
                    params.pop("additionalProperties", None)
                function_def["parameters"] = params
            declarations.append(function_def)
        if not declarations:
            return None
        return declarations

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
        gemini_input = self._to_gemini_input(input)
        gemini_tools = self._to_gemini_tools(tools)

        conversation_id = None
        if metadata:
            conversation_id = metadata.get("conversation_id")

        system_instruction = self._extract_system_instruction(input)
        if system_instruction:
            if conversation_id:
                if conversation_id in self._system_instruction_cache:
                    self._system_instruction_cache.move_to_end(conversation_id)
                self._system_instruction_cache[conversation_id] = system_instruction
                if len(self._system_instruction_cache) > self._max_cache_size:
                    self._system_instruction_cache.popitem(last=False)
        elif conversation_id:
            if conversation_id in self._system_instruction_cache:
                self._system_instruction_cache.move_to_end(conversation_id)
                system_instruction = self._system_instruction_cache[conversation_id]

        generation_config: Dict[str, Any] = {}
        if tool_choice is not None:
            generation_config["tool_choice"] = _TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
        if temperature is not None:
            generation_config["temperature"] = temperature

        logging.info(
            "GeminiApi.create_response: enter model=%s prev_response_id=%s tools=%s tool_choice=%s store=%s",
            model,
            previous_response_id,
            len(gemini_tools) if gemini_tools else 0,
            tool_choice,
            store,
        )

        last_err: Optional[BaseException] = None
        for attempt in range(self._max_attempts):
            t0 = time.time()
            logging.info(
                "GeminiApi.create_response: attempt %d/%d starting",
                attempt + 1,
                self._max_attempts,
            )
            try:
                request_params: Dict[str, Any] = {
                    "model": model,
                    "input": gemini_input,
                    "system_instruction": system_instruction,
                    "tools": gemini_tools,
                    "store": store,
                    "previous_interaction_id": previous_response_id,
                }
                if generation_config:
                    request_params["generation_config"] = generation_config

                interaction = self._get_client().interactions.create(**request_params)

                elapsed = time.time() - t0
                response_id = getattr(interaction, "id", None)
                resp_model = getattr(interaction, "model", None)
                output_text = getattr(interaction, "output_text", "") or ""
                tool_calls = _extract_tool_calls(interaction)
                usage = _extract_usage(getattr(interaction, "usage", None))

                logging.info(
                    "GeminiApi.create_response: attempt %d succeeded in %.3fs "
                    "response_id=%s model=%s output_text_len=%d tool_calls=%d",
                    attempt + 1,
                    elapsed,
                    response_id,
                    resp_model,
                    len(output_text),
                    len(tool_calls),
                )

                return LLMResponse(
                    response_id=response_id,
                    model=resp_model,
                    output_text=output_text,
                    tool_calls=tool_calls,
                    usage=usage,
                    raw=interaction,
                )
            except Exception as e:
                elapsed = time.time() - t0
                last_err = e
                logging.warning(
                    "GeminiApi.create_response: attempt %d/%d failed after %.3fs with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                    type(e).__name__,
                    e,
                )
                if attempt < self._max_attempts - 1:
                    _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        logging.error(
            "GeminiApi.create_response: exhausted retries after %d attempts",
            self._max_attempts,
        )
        if last_err is not None:
            raise last_err
        raise RuntimeError("GeminiApi: exhausted retries unexpectedly")
