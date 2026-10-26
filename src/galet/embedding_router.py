from __future__ import annotations

from typing import Optional
import logging
from time import perf_counter

from .embedding_dto import EmbeddingResponse
from .embedding_interface import EmbeddingApi
from .openai_embedding import OpenAIEmbeddingApi
from .mistral_embedding import MistralEmbeddingApi

logger = logging.getLogger(__name__)


class EmbeddingRouter(EmbeddingApi):
    """Routes embedding requests to the correct backend based on the model name.

    - Model names starting with ``"mistral"`` → ``MistralEmbeddingApi``
    - All other model names → ``OpenAIEmbeddingApi``
    """

    def __init__(
        self,
        *,
        openai_api: Optional[OpenAIEmbeddingApi] = None,
        mistral_api: Optional[MistralEmbeddingApi] = None,
    ) -> None:
        self._openai = openai_api or OpenAIEmbeddingApi()
        self._mistral = mistral_api or MistralEmbeddingApi()

    def embed(
        self,
        *,
        model: str,
        input: list[str],
    ) -> EmbeddingResponse:
        provider = "mistral" if model.startswith("mistral") else "openai"
        api = self._mistral if provider == "mistral" else self._openai
        started = perf_counter()
        logger.info("embedding_request_start provider=%s model=%s input_count=%d",
                    provider, model, len(input))
        try:
            response = api.embed(model=model, input=input)
        except Exception as exc:
            logger.error(
                "embedding_request_failed provider=%s model=%s input_count=%d duration_ms=%.1f error_type=%s",
                provider, model, len(input), (perf_counter() - started) * 1000, type(exc).__name__,
            )
            raise
        logger.info(
            "embedding_request_done provider=%s model=%s input_count=%d duration_ms=%.1f "
            "embedding_count=%d dimensions=%d",
            provider, response.model, len(input), (perf_counter() - started) * 1000,
            len(response.embeddings), len(response.embeddings[0]) if response.embeddings else 0,
        )
        return response
