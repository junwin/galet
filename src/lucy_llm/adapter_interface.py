from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class LLMAdapter(Protocol):
    """Protocol glue between the FunctionCallingProcessor and a specific LLM API.

    The processor should remain LLM-agnostic. The adapter is responsible for:
    - calling the model
    - extracting tool calls in a normalized shape
    - formatting tool outputs in the model's expected protocol

    For now, we keep messages and tool definitions OpenAI-shaped.
    """

    def supports_image_processing(self, model: str, provider: Optional[str] = None) -> bool:
        """Return True if the selected model can natively process images."""
        ...

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
        provider: Optional[str] = None,
    ) -> Any: ...

    def extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]: ...

    def format_tool_output(
        self,
        *,
        call_id: str,
        output: str,
        name: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def get_text(self, response: Any) -> str: ...

    def get_response_id(self, response: Any) -> Optional[str]: ...
