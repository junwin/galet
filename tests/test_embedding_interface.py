from __future__ import annotations

from src.llm.embedding_dto import EmbeddingResponse
from src.llm.embedding_interface import EmbeddingApi


class ValidEmbedder:
    """Implementation that satisfies EmbeddingApi."""

    def embed(self, *, model: str, input: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(model=model, embeddings=[])


class WrongShape:
    """Implementation whose embed() lacks keyword-only params (positional)."""

    def embed(self, model: str, input: list[str]) -> EmbeddingResponse:
        return EmbeddingResponse(model=model, embeddings=[])


class MissingMethod:
    """No 'embed' method at all."""
    pass


class TestEmbeddingInterface:
    """Structural subtyping tests for EmbeddingApi (no @runtime_checkable)."""

    def test_valid_implementation_has_correct_signature(self) -> None:
        """A class with keyword-only embed() accepts the right call."""
        v = ValidEmbedder()
        result = v.embed(model="m", input=["hi"])
        assert result.model == "m"

    def test_valid_implementation_used_as_api(self) -> None:
        """A function accepting EmbeddingApi works with ValidEmbedder."""

        def call_embed(api: EmbeddingApi, model: str) -> EmbeddingResponse:
            return api.embed(model=model, input=["x"])

        result = call_embed(ValidEmbedder(), "gpt")
        assert result.model == "gpt"

    def test_wrong_shape_still_callable_positionally(self) -> None:
        """WrongShape can be called positionally — structural check is
        compile-time only with protocols that aren't runtime_checkable."""
        ws = WrongShape()
        result = ws.embed("m", ["x"])  # type: ignore[call-arg]
        assert result.model == "m"

    def test_missing_method_raises_attribute_error(self) -> None:
        """A class without embed raises AttributeError at call time."""
        mm = MissingMethod()
        try:
            mm.embed(model="m", input=["x"])  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            pytest.fail("Expected AttributeError")
