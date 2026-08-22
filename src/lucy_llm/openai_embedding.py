from __future__ import annotations

import logging
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


from .embedding_dto import EmbeddingResponse
from .embedding_interface import EmbeddingApi
from .openai_responses import _extract_usage, _sleep_backoff
from .settings import Settings, default_settings


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
                resp = self._get_client().embeddings.create(
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
