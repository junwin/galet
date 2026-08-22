from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.llm.embedding_dto import EmbeddingResponse
from src.llm.mistral_embedding import MistralEmbeddingApi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbeddingData:
    embedding: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


class FakeEmbeddingResponse:
    """Mimics openai.embeddings.create() return."""

    def __init__(
        self,
        *,
        data: Any = None,
        model: str = "mistral-embed",
        usage: Any = None,
    ) -> None:
        self.data = data or []
        self.model = model
        self.usage = usage


def make_mock_client(resp: Any = None) -> Mock:
    client = Mock()
    client.embeddings.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMistralEmbeddingHappyPath:
    """Happy-path tests."""

    def test_single_input(self) -> None:
        client = make_mock_client(
            FakeEmbeddingResponse(
                data=[FakeEmbeddingData([0.5, 0.6])],
                model="mistral-embed",
            )
        )
        api = MistralEmbeddingApi(client=client)
        result = api.embed(model="mistral-embed", input=["hello"])

        assert isinstance(result, EmbeddingResponse)
        assert result.model == "mistral-embed"
        assert result.embeddings == [[0.5, 0.6]]

    def test_multiple_inputs(self) -> None:
        client = make_mock_client(
            FakeEmbeddingResponse(
                data=[
                    FakeEmbeddingData([1.0, 2.0]),
                    FakeEmbeddingData([3.0, 4.0]),
                ],
            )
        )
        api = MistralEmbeddingApi(client=client)
        result = api.embed(model="mistral-embed", input=["a", "b"])

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [1.0, 2.0]


class TestMistralEmbeddingRetry:
    """Retry behaviour."""

    def test_retry_exhaustion(self) -> None:
        client = Mock()
        client.embeddings.create.side_effect = RuntimeError("fail")

        api = MistralEmbeddingApi(client=client, max_attempts=2, backoff_base=0.0)

        with pytest.raises(RuntimeError, match="fail"):
            api.embed(model="mistral-embed", input=["x"])

        assert client.embeddings.create.call_count == 2

    def test_succeeds_after_one_failure(self) -> None:
        client = Mock()
        client.embeddings.create.side_effect = [
            RuntimeError("transient"),
            FakeEmbeddingResponse(data=[FakeEmbeddingData([9.9])]),
        ]

        api = MistralEmbeddingApi(client=client, max_attempts=3, backoff_base=0.0)
        result = api.embed(model="mistral-embed", input=["x"])

        assert result.embeddings == [[9.9]]
        assert client.embeddings.create.call_count == 2


class TestMistralEmbeddingClientBuilding:
    """_build_default_client tests."""

    def test_build_default_client_sets_base_url(self) -> None:
        """Mistral client is built with the correct base_url."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = os.path.join(tmpdir, "mistral_cred.json")
            with open(cred_file, "w") as f:
                f.write('{"mistral_api_key": "sk-mistral-test"}')

            with patch("src.llm.mistral_embedding.ConfigManager") as MockConfig, \
                 patch("src.llm.mistral_embedding.OpenAI") as MockOpenAI:
                mock_cm = MockConfig.return_value
                mock_cm.get.return_value = tmpdir

                MistralEmbeddingApi._build_default_client()

                MockOpenAI.assert_called_once_with(
                    api_key="sk-mistral-test",
                    base_url=MistralEmbeddingApi.MISTRAL_BASE_URL,
                )

    def test_mistral_base_url_constant(self) -> None:
        """MISTRAL_BASE_URL is the expected value."""
        assert MistralEmbeddingApi.MISTRAL_BASE_URL == "https://api.mistral.ai/v1"


class TestMistralEmbeddingConstructor:
    """Constructor / DI behaviour."""

    def test_accepts_optional_client(self) -> None:
        client = MagicMock()
        api = MistralEmbeddingApi(client=client)
        assert api._client is client

    def test_accepts_retry_params(self) -> None:
        api = MistralEmbeddingApi(max_attempts=5, backoff_base=2.0, backoff_cap=16.0)
        assert api._max_attempts == 5
        assert api._backoff_base == 2.0
        assert api._backoff_cap == 16.0
