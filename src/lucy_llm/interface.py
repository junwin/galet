from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from .dto import LLMResponse


class LLMApi(Protocol):
    """Interface for calling an LLM.

    We return a normalized DTO (LLMResponse) so the rest of the codebase does not
    depend on the OpenAI SDK response object shape.
    """

    def supports_image_processing(self, model: str) -> bool:
        """Return True if models from this provider natively handle images."""
        ...

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
    ) -> LLMResponse: ...
