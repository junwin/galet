from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from galet.openai_imagegen import OpenAIImageGenApi
from galet.settings import Settings


class _RetryableError(Exception):
    pass


def _stub_image(
    *,
    url: str | None = None,
    b64_json: str | None = None,
    revised_prompt: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(url=url, b64_json=b64_json, revised_prompt=revised_prompt)


def _stub_response(*images: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(data=list(images))


def _make_api(client: MagicMock, **kwargs) -> OpenAIImageGenApi:
    return OpenAIImageGenApi(client=client, **kwargs)


class TestGenerateImageCall:
    """Request construction for images.generate."""

    def test_passes_parameters_through(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(
            model="dall-e-3",
            prompt="a cat",
            size="1792x1024",
            quality="hd",
            n=1,
        )

        client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="a cat",
            size="1792x1024",
            quality="hd",
            n=1,
        )

    def test_defaults_used_when_omitted(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="dall-e-3", prompt="a cat")

        client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="a cat",
            size="1024x1024",
            quality="standard",
            n=1,
        )


class TestGptImageNormalization:
    def test_standard_quality_mapped_to_medium(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gpt-image-1", prompt="a cat", quality="standard")

        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="1024x1024",
            quality="medium",
            n=1,
        )

    def test_hd_quality_mapped_to_high(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gpt-image-1", prompt="a cat", quality="hd")

        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="1024x1024",
            quality="high",
            n=1,
        )

    def test_valid_quality_passed_through(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gpt-image-1", prompt="a cat", quality="high")

        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="1024x1024",
            quality="high",
            n=1,
        )

    def test_unknown_quality_maps_to_auto(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gpt-image-1", prompt="a cat", quality="bogus")

        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="1024x1024",
            quality="auto",
            n=1,
        )

    def test_dalle_quality_unchanged(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="dall-e-3", prompt="a cat", quality="standard")

        client.images.generate.assert_called_once_with(
            model="dall-e-3",
            prompt="a cat",
            size="1024x1024",
            quality="standard",
            n=1,
        )

    def test_n_greater_than_one_raises(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        with pytest.raises(ValueError, match="only supports n=1"):
            api.generate_image(model="gpt-image-1", prompt="a cat", n=2)

        client.images.generate.assert_not_called()

    def test_n_one_allowed(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gpt-image-1", prompt="a cat", n=1)

        client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="a cat",
            size="1024x1024",
            quality="medium",
            n=1,
        )


class TestGenerateImageMapping:
    def test_url_mapped(self) -> None:
        response = _stub_response(_stub_image(url="http://img.example/cat.png"))
        client = MagicMock()
        client.images.generate.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="dall-e-3", prompt="a cat")

        assert len(result.images) == 1
        assert result.images[0].url == "http://img.example/cat.png"
        assert result.images[0].b64_json is None
        assert result.images[0].revised_prompt is None
        assert result.model == "dall-e-3"
        assert result.raw is response

    def test_b64_json_mapped(self) -> None:
        response = _stub_response(_stub_image(b64_json="cG5nLWJ5dGVz"))
        client = MagicMock()
        client.images.generate.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="gpt-image-1", prompt="a cat")

        assert result.images[0].b64_json == "cG5nLWJ5dGVz"
        assert result.images[0].url is None

    def test_revised_prompt_mapped(self) -> None:
        response = _stub_response(
            _stub_image(url="http://img.example/cat.png", revised_prompt="A majestic cat")
        )
        client = MagicMock()
        client.images.generate.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="dall-e-3", prompt="a cat")

        assert result.images[0].revised_prompt == "A majestic cat"

    def test_multiple_images_mapped_in_order(self) -> None:
        response = _stub_response(
            _stub_image(url="http://img.example/1.png", revised_prompt="first"),
            _stub_image(b64_json="c2Vjb25k"),
        )
        client = MagicMock()
        client.images.generate.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="dall-e-3", prompt="a cat", n=2)

        assert len(result.images) == 2
        assert result.images[0].url == "http://img.example/1.png"
        assert result.images[0].revised_prompt == "first"
        assert result.images[1].b64_json == "c2Vjb25k"
        assert result.images[1].url is None

    def test_empty_data_yields_no_images(self) -> None:
        client = MagicMock()
        client.images.generate.return_value = _stub_response()
        api = _make_api(client)

        result = api.generate_image(model="dall-e-3", prompt="a cat")

        assert result.images == []


class TestGenerateImageRetry:
    def test_retries_then_succeeds(self) -> None:
        client = MagicMock()
        ok = _stub_response(_stub_image(url="http://img.example/cat.png"))
        client.images.generate.side_effect = [_RetryableError("slow down"), ok]
        api = _make_api(client, max_attempts=3, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.openai_imagegen.RateLimitError", _RetryableError):
            with patch("galet.openai_imagegen._sleep_backoff") as mock_sleep:
                result = api.generate_image(model="dall-e-3", prompt="a cat")

        assert client.images.generate.call_count == 2
        mock_sleep.assert_called_once_with(0, 0.01, 0.02)
        assert len(result.images) == 1

    def test_exhausted_retries_raise_last_error(self) -> None:
        client = MagicMock()
        client.images.generate.side_effect = _RetryableError("API failed")
        api = _make_api(client, max_attempts=2, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.openai_imagegen.RateLimitError", _RetryableError):
            with patch("galet.openai_imagegen._sleep_backoff"):
                with pytest.raises(_RetryableError, match="API failed"):
                    api.generate_image(model="dall-e-3", prompt="a cat")

        assert client.images.generate.call_count == 2

    def test_sleep_backoff_called_between_attempts(self) -> None:
        client = MagicMock()
        client.images.generate.side_effect = [
            _RetryableError("boom"),
            _RetryableError("boom again"),
            _stub_response(_stub_image(url="http://img.example/cat.png")),
        ]
        api = _make_api(client, max_attempts=3, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.openai_imagegen.RateLimitError", _RetryableError):
            with patch("galet.openai_imagegen._sleep_backoff") as mock_sleep:
                api.generate_image(model="dall-e-3", prompt="a cat")

        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args == (0, 0.01, 0.02)
        assert mock_sleep.call_args_list[1].args == (1, 0.01, 0.02)

    def test_unexpected_error_raises_immediately(self) -> None:
        client = MagicMock()
        client.images.generate.side_effect = ValueError("boom")
        api = _make_api(client, max_attempts=3, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.openai_imagegen._sleep_backoff") as mock_sleep:
            with pytest.raises(ValueError, match="boom"):
                api.generate_image(model="dall-e-3", prompt="a cat")

        assert client.images.generate.call_count == 1
        mock_sleep.assert_not_called()


class TestOpenAIImageGenClientBuilding:
    """_build_default_client tests."""

    def test_build_default_client_uses_oaicred(self) -> None:
        """_build_default_client loads api_key from oaicred.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = os.path.join(tmpdir, "oaicred.json")
            with open(cred_file, "w") as f:
                f.write('{"openai_api_key": "sk-img-test"}')

            with patch("galet.openai_imagegen.OpenAI") as MockOpenAI:
                OpenAIImageGenApi._build_default_client(
                    settings=Settings(credential_path=tmpdir)
                )

                MockOpenAI.assert_called_once_with(api_key="sk-img-test")


class TestOpenAIImageGenConstructor:
    """Constructor / DI behaviour."""

    def test_accepts_optional_client(self) -> None:
        client = MagicMock()
        api = OpenAIImageGenApi(client=client)
        assert api._client is client

    def test_accepts_retry_params(self) -> None:
        api = OpenAIImageGenApi(max_attempts=3, backoff_base=1.0, backoff_cap=5.0)
        assert api._max_attempts == 3
        assert api._backoff_base == 1.0
        assert api._backoff_cap == 5.0

    def test_get_client_builds_default_when_none(self) -> None:
        api = OpenAIImageGenApi()
        fake_client = MagicMock()

        with patch.object(
            OpenAIImageGenApi, "_build_default_client", return_value=fake_client
        ) as mock_build:
            client = api._get_client()

        assert client is fake_client
        assert api._client is fake_client
        mock_build.assert_called_once_with(api._settings)

    def test_get_client_uses_injected_client(self) -> None:
        fake_client = MagicMock()
        api = _make_api(fake_client)

        with patch.object(OpenAIImageGenApi, "_build_default_client") as mock_build:
            client = api._get_client()

        assert client is fake_client
        mock_build.assert_not_called()
