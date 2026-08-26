from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from galet.imagegen_dto import ImageGenResponse, ImageResult
from galet.imagegen_router import ImageGenRouter


def make_mock_api(return_value: ImageGenResponse = None) -> Mock:
    """Return a Mock with a generate_image method."""
    api = Mock()
    api.generate_image.return_value = return_value or ImageGenResponse(images=[])
    return api


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

    def test_gpt_image_model_dispatched_to_openai(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)
        router.generate_image(model="gpt-image-1", prompt="dog")
        openai.generate_image.assert_called_once_with(
            model="gpt-image-1",
            prompt="dog",
            size="1024x1024",
            quality="standard",
            n=1,
        )

    def test_gemini_model_dispatched_to_gemini(self) -> None:
        openai = make_mock_api()
        gemini = make_mock_api(ImageGenResponse(images=[ImageResult(url="http://g")]))
        router = ImageGenRouter(openai_api=openai, gemini_api=gemini)
        result = router.generate_image(
            model="gemini-2.0-flash-exp",
            prompt="cat",
            size="512x512",
            quality="standard",
            n=2,
        )

        assert result.images[0].url == "http://g"
        gemini.generate_image.assert_called_once_with(
            model="gemini-2.0-flash-exp",
            prompt="cat",
            size="512x512",
            quality="standard",
            n=2,
        )
        openai.generate_image.assert_not_called()

    def test_imagen_model_dispatched_to_gemini(self) -> None:
        openai = make_mock_api()
        gemini = make_mock_api()
        router = ImageGenRouter(openai_api=openai, gemini_api=gemini)
        router.generate_image(model="imagen-3.0-generate-002", prompt="dog")
        gemini.generate_image.assert_called_once_with(
            model="imagen-3.0-generate-002",
            prompt="dog",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        openai.generate_image.assert_not_called()

    def test_unknown_provider_raises_value_error(self) -> None:
        openai = make_mock_api()
        router = ImageGenRouter(openai_api=openai)

        with pytest.raises(ValueError, match="no image generation provider"):
            router.generate_image(model="mistral-pixtral", prompt="test")

    def test_unknown_provider_calls_no_backend(self) -> None:
        openai = make_mock_api()
        gemini = make_mock_api()
        router = ImageGenRouter(openai_api=openai, gemini_api=gemini)

        with pytest.raises(ValueError):
            router.generate_image(model="mistral-pixtral", prompt="test")

        openai.generate_image.assert_not_called()
        gemini.generate_image.assert_not_called()

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

    def test_injected_gemini_api_used(self) -> None:
        gemini = make_mock_api()
        router = ImageGenRouter(gemini_api=gemini)
        assert router._gemini is gemini

    def test_injected_gemini_api_called(self) -> None:
        gemini = make_mock_api()
        router = ImageGenRouter(gemini_api=gemini)
        router.generate_image(model="imagen-3.0-generate-002", prompt="x")
        gemini.generate_image.assert_called_once()
