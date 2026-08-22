from __future__ import annotations

from typing import Protocol

from .imagegen_dto import ImageGenResponse


class ImageGenApi(Protocol):
    """Interface for calling an image generation model."""

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse: ...
