from __future__ import annotations

from typing import Optional

from .gemini_videogen import GeminiVideoGenApi
from .videogen_dto import VideoGenResponse
from .videogen_interface import VideoGenApi


class VideoGenRouter(VideoGenApi):
    """Route video generation requests according to model name."""

    def __init__(self, *, gemini_api: Optional[GeminiVideoGenApi] = None) -> None:
        self._gemini = gemini_api or GeminiVideoGenApi()

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        image_url: str,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 6,
        resolution: str = "720p",
        preserve_subject: bool = True,
    ) -> VideoGenResponse:
        if model.startswith("veo-"):
            return self._gemini.generate_video(
                model=model,
                prompt=prompt,
                image_url=image_url,
                aspect_ratio=aspect_ratio,
                duration_seconds=duration_seconds,
                resolution=resolution,
                preserve_subject=preserve_subject,
            )
        raise ValueError(
            f"VideoGenRouter: no video generation provider for model '{model}'"
        )
