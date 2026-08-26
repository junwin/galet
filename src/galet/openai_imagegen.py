from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

# The real 'openai' package may not be available in test environments. Provide
# lightweight fallbacks so this module can be imported without the real SDK.
try:
    from openai import OpenAI
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover - environment dependent
    class OpenAI:  # type: ignore
        def __init__(self, *args, **kwargs):
            class _Img:
                def generate(self, *a, **k):
                    return None

            self.images = _Img()

    class APIConnectionError(Exception):
        pass

    class APIError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass


from .imagegen_dto import ImageGenResponse, ImageResult
from .imagegen_interface import ImageGenApi
from .openai_responses import _sleep_backoff
from .settings import Settings, default_settings

_GPT_IMAGE_QUALITY_VALUES = ("low", "medium", "high", "auto")
_GPT_IMAGE_QUALITY_MAP = {"standard": "medium", "hd": "high"}


class OpenAIImageGenApi(ImageGenApi):
    """OpenAI Image Generation API implementation.

    Notes:
    - By default, this class loads credentials the same way as OpenAIResponsesApi.
    - For tests, pass a fake/mocked client via ``client=...``.
    - ``generate_image()`` calls ``client.images.generate(...)`` and normalizes
      the result into ``ImageGenResponse``.

    Model families:
    - ``dall-e-*`` uses ``quality`` of ``standard``/``hd`` and supports ``n>1``.
    - ``gpt-image-*`` uses ``quality`` of ``low``/``medium``/``high``/``auto``
      and only supports ``n=1``. The legacy ``standard``/``hd`` values are
      mapped to ``medium``/``high`` for these models.

    Retry/backoff:
    - Retries RateLimitError, APIError, APITimeoutError, APIConnectionError.
    - Backoff is exponential with jitter.
    """

    def __init__(
        self,
        *,
        client: Optional[OpenAI] = None,
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

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = self._build_default_client(self._settings)
        return self._client

    @staticmethod
    def _build_default_client(settings: Optional[Settings] = None) -> OpenAI:
        return OpenAI(api_key=(settings or default_settings).api_key("openai"))

    @staticmethod
    def _normalize_quality(model: str, quality: str) -> str:
        if model.startswith("gpt-image") and quality not in _GPT_IMAGE_QUALITY_VALUES:
            return _GPT_IMAGE_QUALITY_MAP.get(quality, "auto")
        return quality

    @staticmethod
    def _validate_n(model: str, n: int) -> None:
        if model.startswith("gpt-image") and n != 1:
            raise ValueError(
                f"OpenAIImageGenApi: model '{model}' only supports n=1, got n={n}"
            )

    @staticmethod
    def _to_image_results(response: Any) -> List[ImageResult]:
        results: List[ImageResult] = []
        data = getattr(response, "data", None) or []
        for item in data:
            results.append(
                ImageResult(
                    url=getattr(item, "url", None),
                    b64_json=getattr(item, "b64_json", None),
                    revised_prompt=getattr(item, "revised_prompt", None),
                )
            )
        return results

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        self._validate_n(model, n)
        quality = self._normalize_quality(model, quality)

        logging.info(
            "OpenAIImageGenApi.generate_image: enter model=%s size=%s quality=%s n=%d",
            model,
            size,
            quality,
            n,
        )

        last_err: Optional[BaseException] = None
        for attempt in range(self._max_attempts):
            t0 = time.time()
            logging.info(
                "OpenAIImageGenApi.generate_image: attempt %d/%d starting",
                attempt + 1,
                self._max_attempts,
            )
            try:
                resp = self._get_client().images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n,
                )

                elapsed = time.time() - t0
                images = self._to_image_results(resp)

                logging.info(
                    "OpenAIImageGenApi.generate_image: attempt %d succeeded in %.3fs images=%d",
                    attempt + 1,
                    elapsed,
                    len(images),
                )

                return ImageGenResponse(images=images, model=model, raw=resp)

            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
                elapsed = time.time() - t0
                last_err = e

                logging.warning(
                    "OpenAIImageGenApi.generate_image: attempt %d/%d failed after %.3fs "
                    "with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                    type(e).__name__,
                    e,
                )

                if attempt == self._max_attempts - 1:
                    logging.error(
                        "OpenAIImageGenApi.generate_image: exhausted retries after %d attempts",
                        self._max_attempts,
                    )
                    raise

                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

            except Exception as e:
                elapsed = time.time() - t0
                logging.exception(
                    "OpenAIImageGenApi.generate_image: unexpected error on attempt %d/%d after %.3fs",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                )
                raise

        raise RuntimeError("OpenAIImageGenApi: exhausted retries unexpectedly") from last_err
