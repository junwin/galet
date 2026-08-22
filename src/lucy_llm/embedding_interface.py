from __future__ import annotations

from typing import Protocol

from .embedding_dto import EmbeddingResponse


class EmbeddingApi(Protocol):
    """Interface for calling an embeddings model."""

    def embed(
        self,
        *,
        model: str,
        input: list[str],
    ) -> EmbeddingResponse: ...
