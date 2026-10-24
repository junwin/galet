from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class VideoResult:
    """A generated video with a temporary provider URL or inline bytes."""

    url: Optional[str] = None
    mime_type: str = "video/mp4"
    duration_seconds: Optional[int] = None
    raw: Optional[Any] = None
    video_bytes: Optional[bytes] = field(default=None, repr=False)


@dataclass(frozen=True)
class VideoGenResponse:
    """Normalized response from a completed video generation request."""

    videos: List[VideoResult]
    model: Optional[str] = None
    operation_name: Optional[str] = None
    raw: Optional[Any] = None
