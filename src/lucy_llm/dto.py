from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool call extracted from an LLM response."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class LLMUsage:
    """Normalized usage info (best-effort)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by LLMApi implementations."""

    response_id: Optional[str]
    model: Optional[str]
    output_text: str
    tool_calls: List[ToolCall]
    usage: Optional[LLMUsage] = None
    raw: Optional[Any] = None
