from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.llm.openai_imagegen import OpenAIImageGenApi


class TestOpenAIImageGenNotImplemented:
    """The stub raises NotImplementedError."""

    def test_generate_image_raises_not_implemented(self) -> None:
        client = MagicMock()
        api = OpenAIImageGenApi(client=client)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            api.generate_image(
                model="dall-e-3",
                prompt="a cat",
                size="1024x1024",
                quality="standard",
                n=1,
            )

    def test_generate_image_does_not_call_client(self) -> None:
        """Stub raises before making any API call."""
        client = MagicMock()
        api = OpenAIImageGenApi(client=client)
        with pytest.raises(NotImplementedError):
            api.generate_image(model="dall-e-3", prompt="test")
        client.images.generate.assert_not_called()

    def test_logs_warning_before_raising(self, caplog) -> None:
        """A warning is logged before NotImplementedError is raised."""
        import logging

        api = OpenAIImageGenApi(client=MagicMock())
        with caplog.at_level(logging.WARNING):
            with pytest.raises(NotImplementedError):
                api.generate_image(model="dall-e-3", prompt="test")
        assert any("not yet implemented" in r.message for r in caplog.records)


class TestOpenAIImageGenClientBuilding:
    """_build_default_client tests."""

    def test_build_default_client_uses_oaicred(self) -> None:
        """_build_default_client loads api_key from oaicred.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = os.path.join(tmpdir, "oaicred.json")
            with open(cred_file, "w") as f:
                f.write('{"openai_api_key": "sk-img-test"}')

            with patch("src.llm.openai_imagegen.ConfigManager") as MockConfig, \
                 patch("src.llm.openai_imagegen.OpenAI") as MockOpenAI:
                mock_cm = MockConfig.return_value
                mock_cm.get.return_value = tmpdir

                OpenAIImageGenApi._build_default_client()

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
