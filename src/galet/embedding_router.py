from __future__ import annotations

from typing import Optional

from .embedding_dto import EmbeddingResponse
from .embedding_interface import EmbeddingApi
from .openai_embedding import OpenAIEmbeddingApi
from .mistral_embedding import MistralEmbeddingApi


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
        if model.startswith("mistral"):
            return self._mistral.embed(model=model, input=input)
        return self._openai.embed(model=model, input=input)
