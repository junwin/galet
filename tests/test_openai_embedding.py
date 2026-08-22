from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.llm.dto import LLMUsage
from src.llm.embedding_dto import EmbeddingResponse
from src.llm.openai_embedding import OpenAIEmbeddingApi


# ---------------------------------------------------------------------------
# Helpers — minimal fakes for the OpenAI response shape
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbeddingData:
    embedding: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


class FakeUsage:
    """Usage object that supports to_dict()."""

    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def to_dict(self) -> dict:
        return self._data


class FakeEmbeddingResponse:
    """Mimics openai.embeddings.create() return."""

    def __init__(
        self,
        *,
        data: Any = None,
        model: str = "text-embedding-3-small",
        usage: Any = None,
    ) -> None:
        self.data = data or []
        self.model = model
        self.usage = usage


def make_mock_client(resp: Any = None) -> Mock:
    """Return a Mock whose .embeddings.create returns *resp*."""
    client = Mock()
    client.embeddings.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingHappyPath:
    """Happy-path: normal response with embeddings."""

    def test_single_input(self) -> None:
        client = make_mock_client(
            FakeEmbeddingResponse(
                data=[FakeEmbeddingData([0.1, 0.2])],
                model="text-embedding-3-small",
            )
        )
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="text-embedding-3-small", input=["hello"])

        assert isinstance(result, EmbeddingResponse)
        assert result.model == "text-embedding-3-small"
        assert result.embeddings == [[0.1, 0.2]]
        assert result.usage is None

    def test_multiple_inputs(self) -> None:
        client = make_mock_client(
            FakeEmbeddingResponse(
                data=[
                    FakeEmbeddingData([1.0, 2.0]),
                    FakeEmbeddingData([3.0, 4.0]),
                    FakeEmbeddingData([5.0, 6.0]),
                ],
                model="text-embedding-3-large",
            )
        )
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="text-embedding-3-large", input=["a", "b", "c"])

        assert len(result.embeddings) == 3
        assert result.embeddings[0] == [1.0, 2.0]
        assert result.embeddings[2] == [5.0, 6.0]

    def test_empty_input_list(self) -> None:
        """Empty input list — API call still made, no embeddings returned."""
        client = make_mock_client(
            FakeEmbeddingResponse(data=[], model="text-embedding-3-small")
        )
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="text-embedding-3-small", input=[])

        assert result.embeddings == []

    def test_usage_in_response(self) -> None:
        """When the API returns usage info, it's included."""
        fake_usage = FakeUsage(input_tokens=5, total_tokens=5)
        client = make_mock_client(
            FakeEmbeddingResponse(
                data=[FakeEmbeddingData([0.5])],
                usage=fake_usage,
            )
        )
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="x", input=["hi"])

        assert result.usage is not None
        assert result.usage.input_tokens == 5
        assert result.usage.total_tokens == 5

    def test_raw_response_preserved(self) -> None:
        """The raw API response is preserved on the DTO."""
        raw_resp = FakeEmbeddingResponse(data=[FakeEmbeddingData()])
        client = make_mock_client(raw_resp)
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="m", input=["x"])

        assert result.raw is raw_resp

    def test_model_falls_back_to_input(self) -> None:
        """If the response has no model attr, the input model is used."""
        resp_no_model = FakeEmbeddingResponse(data=[FakeEmbeddingData()])
        object.__setattr__(resp_no_model, "model", None)
        client = make_mock_client(resp_no_model)
        api = OpenAIEmbeddingApi(client=client)
        result = api.embed(model="fallback-model", input=["x"])

        assert result.model == "fallback-model"


class TestOpenAIEmbeddingRetries:
    """Retry / error handling."""

    def test_retry_exhaustion_raises_last_error(self) -> None:
        """After max_attempts failures the last error propagates."""
        from openai import RateLimitError

        client = Mock()
        client.embeddings.create.side_effect = RateLimitError(
            "boom", response=Mock(), body=None
        )

        api = OpenAIEmbeddingApi(client=client, max_attempts=3, backoff_base=0.0)

        with pytest.raises(RateLimitError, match="boom"):
            api.embed(model="m", input=["x"])

        assert client.embeddings.create.call_count == 3

    def test_unexpected_error_propagates_immediately(self) -> None:
        """Non-retryable errors propagate on first failure."""
        client = Mock()
        client.embeddings.create.side_effect = ValueError("unexpected")

        api = OpenAIEmbeddingApi(client=client, max_attempts=3)

        with pytest.raises(ValueError, match="unexpected"):
            api.embed(model="m", input=["x"])

        assert client.embeddings.create.call_count == 1

    def test_succeeds_after_transient_failures(self) -> None:
        """Succeeds on second attempt after one transient error."""
        from openai import APIConnectionError

        client = Mock()
        client.embeddings.create.side_effect = [
            APIConnectionError(message="timeout", request=Mock()),
            FakeEmbeddingResponse(data=[FakeEmbeddingData([7.0])]),
        ]

        api = OpenAIEmbeddingApi(client=client, max_attempts=3, backoff_base=0.0)
        result = api.embed(model="m", input=["x"])

        assert result.embeddings == [[7.0]]
        assert client.embeddings.create.call_count == 2


class TestOpenAIEmbeddingClientBuilding:
    """_build_default_client tests."""

    def test_build_default_client_uses_oaicred(self) -> None:
        """_build_default_client loads api_key from oaicred.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cred_file = os.path.join(tmpdir, "oaicred.json")
            with open(cred_file, "w") as f:
                f.write('{"openai_api_key": "sk-test-key"}')

            with patch("src.llm.openai_embedding.ConfigManager") as MockConfig, \
                 patch("src.llm.openai_embedding.OpenAI") as MockOpenAI:
                mock_cm = MockConfig.return_value
                mock_cm.get.return_value = tmpdir

                OpenAIEmbeddingApi._build_default_client()

                MockOpenAI.assert_called_once_with(api_key="sk-test-key")


class TestOpenAIImportFallback:
    """Import fallback when openai package is missing."""

    def test_module_imports_without_openai(self) -> None:
        """openai_embedding.py is importable even when openai is not installed."""
        from src.llm.openai_embedding import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

        assert OpenAI is not None
        assert APIConnectionError is not None
        assert APIError is not None
        assert APITimeoutError is not None
        assert RateLimitError is not None


class TestOpenAIEmbeddingConstructor:
    """Constructor / DI behaviour."""

    def test_accepts_optional_client(self) -> None:
        """Can inject a mock client via client=."""
        client = MagicMock()
        api = OpenAIEmbeddingApi(client=client)
        assert api._client is client

    def test_accepts_retry_params(self) -> None:
        """Custom retry params are stored."""
        api = OpenAIEmbeddingApi(
            max_attempts=5,
            backoff_base=1.0,
            backoff_cap=10.0,
        )
        assert api._max_attempts == 5
        assert api._backoff_base == 1.0
        assert api._backoff_cap == 10.0
