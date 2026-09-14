from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from galet.dto import LLMResponse
from galet.openai_responses import (
    OpenAIResponsesApi,
    _UNSUPPORTED_SAMPLING_PARAMS,
    _gpt_major_generation,
    _sampling_params_supported,
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


class TestGptMajorGeneration:
    @pytest.mark.parametrize(
        ("model", "generation"),
        [
            ("gpt-4", 4),
            ("gpt-4o", 4),
            ("gpt-4o-mini", 4),
            ("gpt-4.5-preview", 4),
            ("gpt-5", 5),
            ("gpt-5-mini", 5),
            ("gpt-5.1", 5),
            ("gpt-6-astra", 6),
            ("GPT-5", 5),
            ("  gpt-7  ", 7),
        ],
    )
    def test_parses_well_formed_gpt_prefixes(
        self, model: str, generation: int
    ) -> None:
        assert _gpt_major_generation(model) == generation

    @pytest.mark.parametrize(
        "model",
        [
            # Malformed "gpt" prefixes: no hyphen, no generation number, or
            # junk where a well-formed suffix should start.
            "gpt5",
            "gpt-",
            "gpt--5",
            "gpt-x5",
            "gpt-5x",
            "gpt-5_1",
            "gpt-5 mini",
            # Not a leading "gpt-" token, or not a GPT id at all.
            "chatgpt-5",
            "o1",
            "o3",
            "custom-unknown-id",
        ],
    )
    def test_rejects_malformed_and_non_gpt_prefixes(self, model: str) -> None:
        assert _gpt_major_generation(model) is None


class TestSamplingParamsCapabilityRule:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1",
            "GPT-5",
            "gpt-6",
            "gpt-6-astra",
            "gpt-6-mini",
            "gpt-7",
            "gpt-9",
        ],
    )
    def test_gpt5_and_later_lack_sampling_params(self, model: str) -> None:
        assert _sampling_params_supported(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-4.5-preview",
            "chatgpt-4o-latest",
            "o1",
            "o3",
            "o4-mini",
            "custom-unknown-id",
        ],
    )
    def test_other_models_keep_sampling_params(self, model: str) -> None:
        assert _sampling_params_supported(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            # Malformed GPT prefixes are not classified as GPT ids, so they
            # keep the unknown-id pass-through behaviour.
            "gpt5",
            "gpt-",
            "gpt--5",
            "gpt-5x",
            "gpt-5_1",
        ],
    )
    def test_malformed_gpt_prefixes_keep_sampling_params(self, model: str) -> None:
        assert _sampling_params_supported(model) is True

    def test_unsupported_sampling_params_are_known(self) -> None:
        assert set(_UNSUPPORTED_SAMPLING_PARAMS) == {
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

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-6-astra"])
    def test_drops_sampling_params_and_logs_for_gpt5_and_later(
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

    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.5", "o3"])
    def test_keeps_sampling_params_and_logs_nothing_for_other_models(
        self, model: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        params = self._params()
        with caplog.at_level(logging.WARNING):
            result = _sanitize_generation_params(model, params)
        assert result == params
        assert caplog.text == ""

    def test_malformed_gpt_prefixes_pass_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        params = self._params()
        with caplog.at_level(logging.WARNING):
            result = _sanitize_generation_params("gpt-5x", params)
        assert result == params
        assert caplog.text == ""

    def test_does_not_mutate_caller_dict(self) -> None:
        params = self._params()
        original = dict(params)
        _sanitize_generation_params("gpt-6-astra", params)
        assert params == original


class TestCreateResponseGenerationParams:
    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-6-astra"])
    def test_gpt5_and_later_requests_omit_sampling_params(self, model: str) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client, max_attempts=1)
        result = api.create_response(model=model, input="hi", temperature=0.0)

        assert len(client.responses.calls) == 1
        kwargs = client.responses.calls[0]
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_logprobs" not in kwargs
        assert kwargs["model"] == model
        assert kwargs["input"] == "hi"
        assert isinstance(result, LLMResponse)
        assert result.model == model
        assert result.output_text == "hello world"

    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "gpt-4.1", "gpt-4.5", "o3"])
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

    def test_malformed_gpt_prefix_request_keeps_temperature(self) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client)
        api.create_response(model="gpt-5x", input="hi", temperature=0.5)

        kwargs = client.responses.calls[0]
        assert kwargs["temperature"] == 0.5
        assert kwargs["model"] == "gpt-5x"

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-6-astra", "gpt-4o", "o3"])
    def test_temperature_none_never_adds_key(self, model: str) -> None:
        client = FakeClient()
        api = OpenAIResponsesApi(client=client)
        api.create_response(model=model, input="hi", temperature=None)

        kwargs = client.responses.calls[0]
        assert "temperature" not in kwargs
