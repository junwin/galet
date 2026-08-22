from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .adapter_interface import LLMAdapter
from .interface import LLMApi
from .tool_output import format_tool_output


class MistralResponsesAdapter(LLMAdapter):
    """Adapter for Mistral API (OpenAI-compatible).

    This is the protocol glue between FunctionCallingProcessor and the Mistral
    API implementation. Follows the same pattern as OpenAIResponsesAdapter.
    """

    def __init__(self, api: LLMApi) -> None:
        self._api = api

    def supports_image_processing(self, model: str) -> bool:
        return self._api.supports_image_processing(model)

    def call_model(
        self,
        *,
        model: str,
        input: Any,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_response_id: Optional[str] = None,
        text: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self._api.create_response(
            model=model,
            input=input,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            previous_response_id=previous_response_id,
            text=text,
        )

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for tc in getattr(response, "tool_calls", []) or []:
            call_id = getattr(tc, "call_id", None)
            name = getattr(tc, "name", None)
            args = getattr(tc, "arguments_json", None)

            if not call_id or not name:
                continue

            # Ensure arguments is a JSON string.
            if isinstance(args, dict):
                args_str = json.dumps(args, ensure_ascii=False)
            else:
                args_str = str(args) if args is not None else "{}"

            calls.append({"id": str(call_id), "name": str(name), "arguments": args_str})

        return calls

    def format_tool_output(
        self,
        *,
        call_id: str,
        output: str,
        name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        return format_tool_output(call_id=call_id, output=output, name=name, provider=provider)

    def get_text(self, response: Any) -> str:
        return (getattr(response, "output_text", "") or "").strip()

    def get_response_id(self, response: Any) -> Optional[str]:
        rid = getattr(response, "response_id", None)
        return str(rid) if rid else None
