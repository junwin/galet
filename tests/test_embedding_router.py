from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from galet.embedding_dto import EmbeddingResponse
from galet.embedding_router import EmbeddingRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_api(return_value: EmbeddingResponse = None) -> Mock:
    """Return a Mock with an embed method that returns the given value."""
    api = Mock()
    api.embed.return_value = return_value or EmbeddingResponse(model="mock", embeddings=[])
    return api


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingRouterDispatch:
    """Routing by model prefix."""

    def test_openai_model_dispatched_to_openai(self) -> None:
        openai = make_mock_api(EmbeddingResponse(model="openai-model", embeddings=[[1.0]]))
        mistral = make_mock_api()

        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)
        result = router.embed(model="text-embedding-3-small", input=["hi"])

        assert result.model == "openai-model"
        openai.embed.assert_called_once_with(model="text-embedding-3-small", input=["hi"])
        mistral.embed.assert_not_called()

    def test_mistral_model_dispatched_to_mistral(self) -> None:
        openai = make_mock_api()
        mistral = make_mock_api(EmbeddingResponse(model="mistral-embed", embeddings=[[2.0]]))

        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)
        result = router.embed(model="mistral-embed", input=["hello"])

        assert result.model == "mistral-embed"
        mistral.embed.assert_called_once_with(model="mistral-embed", input=["hello"])
        openai.embed.assert_not_called()

    def test_mistral_prefix_any_match(self) -> None:
        """Any model starting with 'mistral' goes to Mistral."""
        openai = make_mock_api()
        mistral = make_mock_api()

        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)
        router.embed(model="mistral-large", input=["x"])

        mistral.embed.assert_called_once()
        openai.embed.assert_not_called()

    def test_non_mistral_defaults_to_openai(self) -> None:
        """Any model not starting with 'mistral' goes to OpenAI."""
        openai = make_mock_api()
        mistral = make_mock_api()

        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)

        for model in ("gpt-4", "custom-embedder", "ada", ""):
            router.embed(model=model, input=["x"])

        assert openai.embed.call_count == 4
        mistral.embed.assert_not_called()


class TestEmbeddingRouterDI:
    """Dependency injection via __init__."""

    def test_injected_apis_used(self) -> None:
        openai = make_mock_api()
        mistral = make_mock_api()

        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)

        router.embed(model="text-embedding-3-small", input=["x"])
        openai.embed.assert_called_once()

        router.embed(model="mistral-embed", input=["x"])
        mistral.embed.assert_called_once()

    def test_injected_apis_stored(self) -> None:
        openai = MagicMock()
        mistral = MagicMock()
        router = EmbeddingRouter(openai_api=openai, mistral_api=mistral)
        assert router._openai is openai
        assert router._mistral is mistral


@pytest.mark.parametrize('model,provider', [('text-embedding-3-small', 'openai'), ('mistral-embed', 'mistral')])
def test_request_logs_success_and_failure_without_input_text(model, provider, caplog):
    import logging
    api = make_mock_api(EmbeddingResponse(model=model, embeddings=[[1.0, 0.0]]))
    router = EmbeddingRouter(openai_api=api, mistral_api=api)
    with caplog.at_level(logging.INFO):
        router.embed(model=model, input=['private document content'])
    assert 'embedding_request_start' in caplog.text
    assert 'embedding_request_done' in caplog.text
    assert f'provider={provider}' in caplog.text
    assert 'dimensions=2' in caplog.text
    assert 'private document content' not in caplog.text

    caplog.clear()
    failure = RuntimeError('private provider response')
    api.embed.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        router.embed(model=model, input=['private document content'])
    assert caught.value is failure
    assert 'embedding_request_failed' in caplog.text
    assert 'error_type=RuntimeError' in caplog.text
    assert 'private provider response' not in caplog.text
    assert 'private document content' not in caplog.text


@pytest.mark.parametrize('provider', ['openai', 'mistral'])
def test_direct_provider_logs_without_router(provider, caplog):
    import importlib
    import logging
    from types import SimpleNamespace
    module = importlib.import_module(f'galet.{provider}_embedding')
    cls = module.OpenAIEmbeddingApi if provider == 'openai' else module.MistralEmbeddingApi
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        model='configured-model', data=[SimpleNamespace(embedding=[1.0, 0.0])], usage=None)
    with caplog.at_level(logging.INFO):
        cls(client=client).embed(model='configured-model', input=['private document content'])
    assert f'embedding_request_start provider={provider}' in caplog.text
    assert f'embedding_request_done provider={provider}' in caplog.text
    assert f'galet.{provider}_embedding' in caplog.text
    assert 'private document content' not in caplog.text
