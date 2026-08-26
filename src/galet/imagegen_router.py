from __future__ import annotations

from typing import Optional

from .gemini_imagegen import GeminiImageGenApi
from .imagegen_dto import ImageGenResponse
from .imagegen_interface import ImageGenApi
from .openai_imagegen import OpenAIImageGenApi


class ImageGenRouter(ImageGenApi):
    """Routes image generation requests to the correct backend based on the model name."""

    def __init__(
        self,
        *,
        openai_api: Optional[OpenAIImageGenApi] = None,
        gemini_api: Optional[GeminiImageGenApi] = None,
    ) -> None:
        self._openai = openai_api or OpenAIImageGenApi()
        self._gemini = gemini_api or GeminiImageGenApi()

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        if (
            model.startswith("openai")
            or model.startswith("dall-e")
            or model.startswith("gpt-image")
        ):
            return self._openai.generate_image(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
        if model.startswith("gemini") or model.startswith("imagen"):
            return self._gemini.generate_image(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
        raise ValueError(
            f"ImageGenRouter: no image generation provider for model '{model}'"
        )
