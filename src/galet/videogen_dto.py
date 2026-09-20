from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class VideoResult:
    """A generated video exposed through a temporary provider URL."""

    url: Optional[str] = None
    mime_type: str = "video/mp4"
    duration_seconds: Optional[int] = None
    raw: Optional[Any] = None


@dataclass(frozen=True)
class VideoGenResponse:
    """Normalized response from a completed video generation request."""

    videos: List[VideoResult]
    model: Optional[str] = None
    operation_name: Optional[str] = None
    raw: Optional[Any] = None
