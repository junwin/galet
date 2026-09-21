from __future__ import annotations

from typing import Optional, Protocol

from .videogen_dto import VideoGenResponse


class VideoGenApi(Protocol):
    """Interface for image-to-video generation."""

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 6,
        resolution: str = "720p",
        preserve_subject: bool = True,
    ) -> VideoGenResponse: ...
