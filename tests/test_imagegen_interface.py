from __future__ import annotations

import pytest

from src.llm.imagegen_dto import ImageGenResponse
from src.llm.imagegen_interface import ImageGenApi


class ValidImageGen:
    """Implementation that satisfies ImageGenApi."""

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenResponse:
        return ImageGenResponse(images=[])


class MissingMethod:
    """No 'generate_image' method."""
    pass


class TestImageGenInterface:
    """Structural subtyping tests for ImageGenApi (no @runtime_checkable)."""

    def test_valid_implementation_has_correct_signature(self) -> None:
        v = ValidImageGen()
        result = v.generate_image(model="dall-e", prompt="cat")
        assert isinstance(result, ImageGenResponse)

    def test_valid_implementation_used_as_api(self) -> None:
        def call_gen(api: ImageGenApi, model: str) -> ImageGenResponse:
            return api.generate_image(model=model, prompt="x")

        result = call_gen(ValidImageGen(), "dall-e-3")
        assert isinstance(result, ImageGenResponse)

    def test_missing_method_raises_attribute_error(self) -> None:
        mm = MissingMethod()
        try:
            mm.generate_image(model="x", prompt="y")  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            pytest.fail("Expected AttributeError")
