from __future__ import annotations

import base64
import logging
import time
from typing import Any, List, Optional

try:
    from google import genai
    from google.genai import types
except Exception:
    class _GenaiClientStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            class _Models:
                def generate_content(self, *a: Any, **k: Any) -> Any:
                    return None

            self.models = _Models()

    class genai:  # type: ignore
        Client = _GenaiClientStub

    class types:  # type: ignore
        class GenerateContentConfig:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class ImageConfig:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)


from .gemini_api import GeminiApi
from .imagegen_dto import ImageGenResponse, ImageResult
from .imagegen_interface import ImageGenApi
from .openai_responses import _sleep_backoff
from .settings import Settings, default_settings


class GeminiImageGenApi(ImageGenApi):
    """Gemini image generation via the google-genai generate_content endpoint.

    Uses the current image models (e.g. ``gemini-2.5-flash-image``), which
    return images inline through ``generate_content`` with
    ``response_modalities=["IMAGE"]``. ``quality`` is accepted for protocol
    compatibility with ``ImageGenApi`` but is ignored, since these models have
    no equivalent parameter. A single image is generated per call.
    """

    def __init__(
        self,
        *,
        client: Optional[genai.Client] = None,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        settings: Optional[Settings] = None,
    ) -> None:
        self._client = client
        self._settings = settings or default_settings
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = self._build_default_client(self._settings)
        return self._client

    @staticmethod
    def _build_default_client(settings: Optional[Settings] = None) -> genai.Client:
        return GeminiApi._build_default_client(settings)

    @staticmethod
    def _size_to_aspect_ratio(size: str) -> str:
        return {
            "1024x1024": "1:1",
            "512x512": "1:1",
            "256x256": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1536x1024": "3:2",
            "1024x1536": "2:3",
        }.get(size, "1:1")

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        if n != 1:
            logging.warning(
                "GeminiImageGenApi.generate_image: n=%d requested; a single image is generated per call",
                n,
            )

        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=self._size_to_aspect_ratio(size),
            ),
        )

        logging.info(
            "GeminiImageGenApi.generate_image: enter model=%s size=%s quality=%s n=%d",
            model,
            size,
            quality,
            n,
        )

        last_err: Optional[BaseException] = None
        for attempt in range(self._max_attempts):
            t0 = time.time()
            logging.info(
                "GeminiImageGenApi.generate_image: attempt %d/%d starting",
                attempt + 1,
                self._max_attempts,
            )
            try:
                response = self._get_client().models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )

                elapsed = time.time() - t0
                images = self._to_image_results(response)

                logging.info(
                    "GeminiImageGenApi.generate_image: attempt %d succeeded in %.3fs images=%d",
                    attempt + 1,
                    elapsed,
                    len(images),
                )

                return ImageGenResponse(images=images, model=model, raw=response)
            except Exception as e:
                elapsed = time.time() - t0
                last_err = e
                logging.warning(
                    "GeminiImageGenApi.generate_image: attempt %d/%d failed after %.3fs with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                    type(e).__name__,
                    e,
                )
                if attempt < self._max_attempts - 1:
                    _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        logging.error(
            "GeminiImageGenApi.generate_image: exhausted retries after %d attempts",
            self._max_attempts,
        )
        if last_err is not None:
            raise last_err
        raise RuntimeError("GeminiImageGenApi: exhausted retries unexpectedly")

    @staticmethod
    def _to_image_results(response: Any) -> List[ImageResult]:
        results: List[ImageResult] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None:
                    data = getattr(inline_data, "data", None)
                    if data is not None:
                        if isinstance(data, bytes):
                            b64_json = base64.b64encode(data).decode("utf-8")
                        else:
                            b64_json = str(data)
                        results.append(ImageResult(url=None, b64_json=b64_json, revised_prompt=None))
                        continue
                file_data = getattr(part, "file_data", None)
                if file_data is not None:
                    uri = getattr(file_data, "file_uri", None)
                    results.append(ImageResult(url=uri, b64_json=None, revised_prompt=None))
        return results
