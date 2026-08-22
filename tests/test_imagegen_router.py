from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from src.llm.imagegen_dto import ImageGenResponse, ImageResult
from src.llm.imagegen_router import ImageGenRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_api(return_value: ImageGenResponse = None) -> Mock:
    """Return a Mock with a generate_image method."""
    api = Mock()
    api.generate_image.return_value = return_value or ImageGenResponse(images=[])
    return api


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageGenRouterDispatch:
    """Routing by model prefix."""

    def test_openai_model_dispatched_to_openai(self) -> None:
        openai = make_mock_api(ImageGenResponse(images=[ImageResult(url="http://x")]))
        router = ImageGenRouter(openai_api=openai)
        result = router.generate_image(
            model="openai/dall-e-3",
            prompt="cat",
            size="512x512",
            quality="hd",
            n=2,
        )

        assert result.images[0].url == "http://x"
        openai.generate_image.assert_called_once_with(
            model="openai/dall-e-3",
            prompt="cat",
            size="512x512",
            quality="hd",
            n=2,
        )

    def test_dalle_model_dispatched_to_openai(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)
        router.generate_image(model="dall-e-3", prompt="dog")
        openai.generate_image.assert_called_once_with(
            model="dall-e-3",
            prompt="dog",
            size="1024x1024",
            quality="standard",
            n=1,
        )

    def test_unknown_provider_raises_value_error(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)

        with pytest.raises(ValueError, match="no image generation provider"):
            router.generate_image(model="mistral-pixtral", prompt="test")

    def test_value_error_message_contains_model(self) -> None:
        router = ImageGenRouter(openai_api=make_mock_api())
        with pytest.raises(ValueError, match="unknown-model-xyz"):
            router.generate_image(model="unknown-model-xyz", prompt="test")


class TestImageGenRouterDI:
    """Dependency injection."""

    def test_injected_api_used(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)
        assert router._openai is openai

    def test_injected_api_called(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)
        router.generate_image(model="dall-e-3", prompt="x")
        openai.generate_image.assert_called_once()
