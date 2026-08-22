from __future__ import annotations

from typing import Optional

from .imagegen_dto import ImageGenResponse
from .imagegen_interface import ImageGenApi
from .openai_imagegen import OpenAIImageGenApi


class ImageGenRouter(ImageGenApi):
    """Routes image generation requests to the correct backend based on the model name.

    - Currently only OpenAI is supported.  Other backends raise ``ValueError``.
    """

    def __init__(
        self,
        *,
        openai_api: Optional[OpenAIImageGenApi] = None,
    ) -> None:
        self._openai = openai_api or OpenAIImageGenApi()

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        if model.startswith("openai") or model.startswith("dall-e"):
            return self._openai.generate_image(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
            )
        raise ValueError(
            f"ImageGenRouter: no image generation provider for model '{model}'"
        )
