from __future__ import annotations

import json
import logging
import os
from typing import Optional

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

from src.config_manager import ConfigManager

from .imagegen_dto import ImageGenResponse
from .imagegen_interface import ImageGenApi
from .openai_responses import _sleep_backoff


class OpenAIImageGenApi(ImageGenApi):
    """OpenAI Image Generation API implementation (stub).

    Notes:
    - By default, this class loads credentials the same way as OpenAIResponsesApi.
    - For tests, pass a fake/mocked client via ``client=...``.
    - ``generate_image()`` is currently a stub — it raises NotImplementedError
      on every call.  This lets wiring / imports work while the implementation
      is filled in later.

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
    ) -> None:
        self._client = client or self._build_default_client()
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap

    @staticmethod
    def _build_default_client() -> OpenAI:
        config = ConfigManager("config.json")
        credential_path = config.get("credential_path")
        with open(os.path.join(credential_path, "oaicred.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)
        return OpenAI(api_key=config_data["openai_api_key"])

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        logging.warning(
            "OpenAIImageGenApi.generate_image: not yet implemented "
            "(model=%s size=%s quality=%s n=%d)",
            model,
            size,
            quality,
            n,
        )
        raise NotImplementedError("Image generation not yet implemented")
