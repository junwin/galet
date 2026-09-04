from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from galet.dto import LLMResponse
from galet.openai_responses import (
    OpenAIResponsesApi,
    _GPT6_UNSUPPORTED_SAMPLING_PARAMS,
    _is_gpt6_model,
    _sanitize_generation_params,
)


@dataclass
class FakeResponse:
    id: str = "resp_123"
    model: str = "gpt-4o"
    output_text: str = "hello world"
    output: List[Any] = field(default_factory=list)
    usage: Any = None


class FakeResponses:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(dict(kwargs))
        return FakeResponse(model=kwargs.get("model", "gpt-4o"))


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class TestGpt6ModelCapability:
    @pytest.mark.parametrize("model", ["gpt-6", "gpt-6-astra", "gpt-6-mini"])
    def test_gpt6_prefix_models_lack_sampling_params(self, model: str) -> None:
        assert _is_gpt6_model(model) is True

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "gpt-4.1", "o3"])
    def test_other_models_keep_sampling_params(self, model: str) -> None:
        assert _is_gpt6_model(model) is False

    def test_unsupported_sampling_params_are_known(self) -> None:
        assert set(_GPT6_UNSUPPORTED_SAMPLING_PARAMS) == {
            "temperature",
            "top_p",
            "top_logprobs",
        }


class TestSanitizeGenerationParams:
    def _params(self) -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_logprobs": 5,
            "store": True,
            "metadata": {"session": "abc"},
        }

    @pytest.mark.parametrize("model", ["gpt-6", "gpt-6-astra"])
    def test_drops_sampling_params_and_logs_for_gpt6(
        self, model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = _sanitize_generation_params(model, self._params())
        assert "temperature" not in result
        assert "top_p" not in result
        assert "top_logprobs" not in result
        assert result["store"] is True
        assert result["metadata"] == {"session": "abc"}
        for name in ("temperature", "top_p", "top_logprobs"):
            assert (
                f"dropping unsupported sampling param {name} for model {model}"
                in caplog.text
            )

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "gpt-4.1"])
    def test_keeps_sampling_params_and_logs_nothing_for_other_models(
        self, model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        params = self._params()
        with caplog.at_level(logging.WARNING):
            result = _sanitize_generation_params(model, params)
        assert result == params
        assert caplog.text == ""

    def test_does_not_mutate_caller_dict(self) -> None:
        params = self._params()
        original = dict(params)
        _sanitize_generation_params("gpt-6-astra", params)
        assert params == original


class TestCreateResponseGenerationParams:
    def test_gpt6_astra_request_omits_all_sampling_param_keys(self) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client, max_attempts=1)
        result = api.create_response(model="gpt-6-astra", input="hi", temperature=0.0)

        assert len(client.responses.calls) == 1
        kwargs = client.responses.calls[0]
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_logprobs" not in kwargs
        assert kwargs["model"] == "gpt-6-astra"
        assert kwargs["input"] == "hi"
        assert isinstance(result, LLMResponse)
        assert result.model == "gpt-6-astra"
        assert result.output_text == "hello world"

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-4o", "gpt-4.1"])
    def test_supported_models_still_carry_temperature(self, model: str) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client)
        result = api.create_response(model=model, input="hi", temperature=0.5)

        kwargs = client.responses.calls[0]
        assert kwargs["temperature"] == 0.5
        assert kwargs["model"] == model
        assert isinstance(result, LLMResponse)
        assert result.model == model
        assert result.output_text == "hello world"

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5", "gpt-4o"])
    def test_temperature_none_never_adds_key(self, model: str) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client)
        api.create_response(model=model, input="hi", temperature=None)

        kwargs = client.responses.calls[0]
        assert "temperature" not in kwargs
