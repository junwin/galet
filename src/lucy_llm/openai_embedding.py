from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

# The real 'openai' package may not be available in test environments. Provide
# lightweight fallbacks so this module can be imported without the real SDK.
try:
    from openai import OpenAI
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover - environment dependent
    class OpenAI:  # type: ignore
        def __init__(self, *args, **kwargs):
            class _Emb:
                def create(self, *a, **k):
                    return None

            self.embeddings = _Emb()

    class APIConnectionError(Exception):
        pass

    class APIError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

from src.config_manager import ConfigManager

from .embedding_dto import EmbeddingResponse
from .embedding_interface import EmbeddingApi
from .openai_responses import _extract_usage, _sleep_backoff


class OpenAIEmbeddingApi(EmbeddingApi):
    """OpenAI Embeddings API implementation.

    Notes:
    - By default, this class loads credentials the same way as OpenAIResponsesApi.
    - For tests, pass a fake/mocked client via `client=...`.

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

    def embed(
        self,
        *,
        model: str,
        input: list[str],
    ) -> EmbeddingResponse:
        last_err: Optional[BaseException] = None

        # ---- entry log ----
        logging.info(
            "OpenAIEmbeddingApi.embed: enter model=%s input_count=%d",
            model,
            len(input),
        )

        for attempt in range(self._max_attempts):
            t0 = time.time()
            logging.info(
                "OpenAIEmbeddingApi.embed: attempt %d/%d starting",
                attempt + 1,
                self._max_attempts,
            )

            try:
                resp = self._client.embeddings.create(
                    model=model,
                    input=input,
                )

                elapsed = time.time() - t0

                resp_model = getattr(resp, "model", None) or model
                embeddings = [d.embedding for d in resp.data]
                usage = _extract_usage(getattr(resp, "usage", None))

                # ---- response summary ----
                logging.info(
                    "OpenAIEmbeddingApi.embed: attempt %d succeeded in %.3fs "
                    "model=%s embedding_count=%d dims=%d",
                    attempt + 1,
                    elapsed,
                    resp_model,
                    len(embeddings),
                    len(embeddings[0]) if embeddings else 0,
                )

                return EmbeddingResponse(
                    model=resp_model,
                    embeddings=embeddings,
                    usage=usage,
                    raw=resp,
                )

            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as e:
                elapsed = time.time() - t0
                last_err = e

                logging.warning(
                    "OpenAIEmbeddingApi.embed: attempt %d/%d failed after %.3fs "
                    "with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                    type(e).__name__,
                    e,
                )

                if attempt == self._max_attempts - 1:
                    logging.error(
                        "OpenAIEmbeddingApi.embed: exhausted retries after %d attempts",
                        self._max_attempts,
                    )
                    raise

                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

            except Exception as e:
                elapsed = time.time() - t0
                logging.exception(
                    "OpenAIEmbeddingApi.embed: unexpected error on attempt %d/%d after %.3fs",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                )
                raise

        # Should be unreachable
        raise RuntimeError("OpenAIEmbeddingApi: exhausted retries unexpectedly") from last_err
