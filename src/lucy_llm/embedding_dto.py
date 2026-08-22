from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .dto import LLMUsage


@dataclass(frozen=True)
class EmbeddingResponse:
    """Normalized response from an embedding API call."""

    model: str
    embeddings: List[List[float]]
    usage: Optional[LLMUsage] = None
    raw: Optional[Any] = None
