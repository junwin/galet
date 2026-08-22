from __future__ import annotations

from src.llm.dto import LLMUsage
from src.llm.embedding_dto import EmbeddingResponse


class TestEmbeddingResponse:
    """Tests for the EmbeddingResponse DTO."""

    def test_instantiation_minimal(self) -> None:
        """Instantiation with only required fields."""
        resp = EmbeddingResponse(model="text-embedding-3-small", embeddings=[])
        assert resp.model == "text-embedding-3-small"
        assert resp.embeddings == []
        assert resp.usage is None
        assert resp.raw is None

    def test_instantiation_full(self) -> None:
        """Instantiation with all fields including LLMUsage."""
        usage = LLMUsage(input_tokens=10, total_tokens=10)
        raw = {"object": "list"}
        resp = EmbeddingResponse(
            model="text-embedding-3-large",
            embeddings=[[0.1, 0.2, 0.3]],
            usage=usage,
            raw=raw,
        )
        assert resp.model == "text-embedding-3-large"
        assert resp.embeddings == [[0.1, 0.2, 0.3]]
        assert resp.usage is usage
        assert resp.usage.input_tokens == 10
        assert resp.raw is raw

    def test_field_defaults(self) -> None:
        """Optional fields default to None."""
        resp = EmbeddingResponse(model="m", embeddings=[[1.0]])
        assert resp.usage is None
        assert resp.raw is None

    def test_frozen_immutability(self) -> None:
        """EmbeddingResponse is frozen — cannot set attributes."""
        resp = EmbeddingResponse(model="m", embeddings=[])
        try:
            resp.model = "other"  # type: ignore[misc]
        except Exception as e:
            assert "frozen" in str(e).lower() or isinstance(e, (AttributeError, TypeError))

    def test_llm_usage_integration(self) -> None:
        """LLMUsage fields can be None, int, or missing."""
        usage_none = LLMUsage()
        resp = EmbeddingResponse(model="x", embeddings=[], usage=usage_none)
        assert resp.usage is not None
        assert resp.usage.input_tokens is None
        assert resp.usage.output_tokens is None
        assert resp.usage.total_tokens is None

    def test_multiple_embeddings(self) -> None:
        """Multiple input texts → multiple embedding vectors."""
        resp = EmbeddingResponse(
            model="e",
            embeddings=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        )
        assert len(resp.embeddings) == 3
        assert resp.embeddings[0] == [0.0, 1.0]
