from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from galet.gemini_api import GeminiApi
from galet.gemini_imagegen import GeminiImageGenApi
from galet.settings import Settings


def _stub_inline_part(
    data: bytes | None = b"raw-bytes",
    mime_type: str = "image/png",
) -> SimpleNamespace:
    inline = SimpleNamespace(data=data, mime_type=mime_type)
    return SimpleNamespace(inline_data=inline, file_data=None)


def _stub_file_part(uri: str, mime_type: str = "image/png") -> SimpleNamespace:
    file_data = SimpleNamespace(file_uri=uri, mime_type=mime_type)
    return SimpleNamespace(inline_data=None, file_data=file_data)


def _stub_candidate(*parts: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(content=SimpleNamespace(parts=list(parts)))


def _stub_response(*candidates: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(candidates=list(candidates))


def _make_api(client: MagicMock, **kwargs) -> GeminiImageGenApi:
    return GeminiImageGenApi(client=client, **kwargs)


class TestSizeToAspectRatio:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            ("1024x1024", "1:1"),
            ("512x512", "1:1"),
            ("256x256", "1:1"),
            ("1792x1024", "16:9"),
            ("1024x1792", "9:16"),
            ("1536x1024", "3:2"),
            ("1024x1536", "2:3"),
            ("2048x2048", "1:1"),
        ],
    )
    def test_maps_size_to_aspect_ratio(self, size: str, expected: str) -> None:
        assert GeminiImageGenApi._size_to_aspect_ratio(size) == expected


class TestGenerateContentConfig:
    def test_config_requests_image_modality_and_aspect_ratio(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(
            model="gemini-2.5-flash-image",
            prompt="a cat",
            size="1792x1024",
        )

        kwargs = client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-flash-image"
        assert kwargs["contents"] == "a cat"
        config = kwargs["config"]
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config.aspect_ratio == "16:9"

    def test_default_config_uses_square_ratio(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config.aspect_ratio == "1:1"

    def test_quality_accepted_but_not_mapped(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _stub_response()
        api = _make_api(client)

        api.generate_image(
            model="gemini-2.5-flash-image",
            prompt="a cat",
            quality="hd",
        )

        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.response_modalities == ["IMAGE"]
        assert config.image_config.aspect_ratio == "1:1"

    def test_n_above_one_still_returns_single_image(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _stub_response(
            _stub_candidate(_stub_inline_part(data=b"raw-bytes"))
        )
        api = _make_api(client)

        result = api.generate_image(
            model="gemini-2.5-flash-image",
            prompt="a cat",
            n=3,
        )

        assert len(result.images) == 1


class TestGenerateImageMapping:
    def test_b64_json_from_inline_bytes(self) -> None:
        raw = b"\x89PNG\r\n\x1a\n"
        response = _stub_response(_stub_candidate(_stub_inline_part(data=raw)))
        client = MagicMock()
        client.models.generate_content.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert len(result.images) == 1
        assert result.images[0].b64_json == base64.b64encode(raw).decode("utf-8")
        assert result.images[0].url is None
        assert result.images[0].revised_prompt is None
        assert result.model == "gemini-2.5-flash-image"
        assert result.raw is response

    def test_url_from_file_data(self) -> None:
        response = _stub_response(_stub_candidate(_stub_file_part("gs://bucket/img.png")))
        client = MagicMock()
        client.models.generate_content.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert result.images[0].url == "gs://bucket/img.png"
        assert result.images[0].b64_json is None

    def test_multiple_parts_mapped_in_order(self) -> None:
        response = _stub_response(
            _stub_candidate(
                _stub_inline_part(data=b"a"),
                _stub_file_part("gs://bucket/1.png"),
            )
        )
        client = MagicMock()
        client.models.generate_content.return_value = response
        api = _make_api(client)

        result = api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert len(result.images) == 2
        assert result.images[0].b64_json == base64.b64encode(b"a").decode("utf-8")
        assert result.images[0].url is None
        assert result.images[1].url == "gs://bucket/1.png"
        assert result.images[1].b64_json is None

    def test_empty_candidates_yields_no_images(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = _stub_response()
        api = _make_api(client)

        result = api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert result.images == []


class TestGenerateImageRetry:
    def test_retries_then_succeeds(self) -> None:
        client = MagicMock()
        ok = _stub_response(_stub_candidate(_stub_inline_part()))
        client.models.generate_content.side_effect = [ValueError("boom"), ok]
        api = _make_api(client, max_attempts=3, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.gemini_imagegen._sleep_backoff") as mock_sleep:
            result = api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once_with(0, 0.01, 0.02)
        assert len(result.images) == 1

    def test_exhausted_retries_raise_last_error(self) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = ValueError("API failed")
        api = _make_api(client, max_attempts=2, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.gemini_imagegen._sleep_backoff"):
            with pytest.raises(ValueError, match="API failed"):
                api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert client.models.generate_content.call_count == 2

    def test_sleep_backoff_called_between_attempts(self) -> None:
        client = MagicMock()
        client.models.generate_content.side_effect = [
            ValueError("boom"),
            ValueError("boom again"),
            _stub_response(_stub_candidate(_stub_inline_part())),
        ]
        api = _make_api(client, max_attempts=3, backoff_base=0.01, backoff_cap=0.02)

        with patch("galet.gemini_imagegen._sleep_backoff") as mock_sleep:
            api.generate_image(model="gemini-2.5-flash-image", prompt="a cat")

        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args == (0, 0.01, 0.02)
        assert mock_sleep.call_args_list[1].args == (1, 0.01, 0.02)


class TestCredentialDelegation:
    def test_build_default_client_delegates_to_gemini_api(self) -> None:
        settings = Settings(credential_path="/tmp/creds")
        fake_client = MagicMock()

        with patch.object(
            GeminiApi, "_build_default_client", return_value=fake_client
        ) as mock_build:
            client = GeminiImageGenApi._build_default_client(settings=settings)

        mock_build.assert_called_once_with(settings)
        assert client is fake_client

    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key-123"}, clear=True)
    @patch("galet.gemini_api.genai.Client")
    def test_build_default_client_uses_env_api_key(
        self, mock_client_cls: MagicMock
    ) -> None:
        GeminiImageGenApi._build_default_client()
        mock_client_cls.assert_called_once_with(api_key="env-key-123")

    def test_get_client_builds_default_when_none(self) -> None:
        api = GeminiImageGenApi()
        fake_client = MagicMock()

        with patch.object(
            GeminiImageGenApi, "_build_default_client", return_value=fake_client
        ) as mock_build:
            client = api._get_client()

        assert client is fake_client
        assert api._client is fake_client
        mock_build.assert_called_once_with(api._settings)

    def test_get_client_uses_injected_client(self) -> None:
        fake_client = MagicMock()
        api = _make_api(fake_client)

        with patch.object(GeminiImageGenApi, "_build_default_client") as mock_build:
            client = api._get_client()

        assert client is fake_client
        mock_build.assert_not_called()


class TestGeminiImageGenConstructor:
    def test_accepts_optional_client(self) -> None:
        client = MagicMock()
        api = _make_api(client)
        assert api._client is client

    def test_accepts_retry_params(self) -> None:
        api = _make_api(MagicMock(), max_attempts=3, backoff_base=1.0, backoff_cap=5.0)
        assert api._max_attempts == 3
        assert api._backoff_base == 1.0
        assert api._backoff_cap == 5.0
