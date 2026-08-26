"""Tests for LLMAdapter.get_usage on the adapters lucy uses."""

from __future__ import annotations

from galet.dto import LLMResponse, LLMUsage
from galet.mistral_responses_adapter import MistralResponsesAdapter
from galet.openai_responses_adapter import OpenAIResponsesAdapter


class _FakeApi:
    def create_response(self, **kwargs):
        return None


def _response(usage):
    return LLMResponse(
        response_id="r1",
        model="m",
        output_text="hi",
        tool_calls=[],
        usage=usage,
    )


def test_openai_get_usage_returns_usage() -> None:
    adapter = OpenAIResponsesAdapter(_FakeApi())
    usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert adapter.get_usage(_response(usage)) == usage


def test_openai_get_usage_none_when_absent() -> None:
    adapter = OpenAIResponsesAdapter(_FakeApi())
    assert adapter.get_usage(_response(None)) is None


def test_openai_get_usage_none_for_unknown_object() -> None:
    adapter = OpenAIResponsesAdapter(_FakeApi())
    assert adapter.get_usage(object()) is None


def test_mistral_get_usage_returns_usage() -> None:
    adapter = MistralResponsesAdapter(_FakeApi())
    usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert adapter.get_usage(_response(usage)) == usage


def test_mistral_get_usage_none_when_absent() -> None:
    adapter = MistralResponsesAdapter(_FakeApi())
    assert adapter.get_usage(_response(None)) is None
