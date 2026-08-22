from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from openai import OpenAI

from src.config_manager import ConfigManager

from .embedding_dto import EmbeddingResponse
from .embedding_interface import EmbeddingApi
from .openai_responses import _extract_usage, _sleep_backoff


class MistralEmbeddingApi(EmbeddingApi):
    """Mistral Embeddings API implementation using OpenAI-compatible endpoint.

    Notes:
    - By default, this class loads credentials from mistral_cred.json.
    - For tests, pass a fake/mocked client via `client=...`.

    Retry/backoff:
    - Retries on transient errors with exponential backoff + jitter.
    """

    MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

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
        with open(os.path.join(credential_path, "mistral_cred.json"), "r", encoding="utf-8") as f:
            config_data = json.load(f)
        return OpenAI(
            api_key=config_data["mistral_api_key"],
            base_url=MistralEmbeddingApi.MISTRAL_BASE_URL,
        )

    def embed(
        self,
        *,
        model: str,
        input: list[str],
    ) -> EmbeddingResponse:
        last_err: Optional[BaseException] = None

        # ---- entry log ----
        logging.info(
            "MistralEmbeddingApi.embed: enter model=%s input_count=%d",
            model,
            len(input),
        )

        for attempt in range(self._max_attempts):
            t0 = time.time()
            logging.info(
                "MistralEmbeddingApi.embed: attempt %d/%d starting",
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
                    "MistralEmbeddingApi.embed: attempt %d succeeded in %.3fs "
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

            except Exception as e:
                elapsed = time.time() - t0
                last_err = e

                logging.warning(
                    "MistralEmbeddingApi.embed: attempt %d/%d failed after %.3fs "
                    "with %s: %s",
                    attempt + 1,
                    self._max_attempts,
                    elapsed,
                    type(e).__name__,
                    e,
                )

                if attempt == self._max_attempts - 1:
                    logging.error(
                        "MistralEmbeddingApi.embed: exhausted retries after %d attempts",
                        self._max_attempts,
                    )
                    raise

                _sleep_backoff(attempt, self._backoff_base, self._backoff_cap)

        # Should be unreachable
        raise RuntimeError("MistralEmbeddingApi: exhausted retries unexpectedly") from last_err
