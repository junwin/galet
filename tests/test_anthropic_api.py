from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import pytest

import galet.anthropic_api as anthropic_module
from galet.anthropic_api import (
    AnthropicApi,
    _forced_tool_choice_supported,
    _sampling_params_supported,
)
from galet.dto import LLMResponse, ToolCall
from galet.provider_registry import ProviderRegistry
from galet.router_api import RouterApi
from galet.settings import Settings


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def _thinking(text: str = "hmm") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature="sig")


def _response(
    content: List[Any],
    *,
    response_id: str = "msg_1",
    model: str = "claude-sonnet-5-5",
    stop_reason: str = "end_turn",
    usage: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=usage if usage is not None else SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _api(*responses: Any, **kwargs: Any) -> tuple[AnthropicApi, MagicMock]:
    create = MagicMock(side_effect=list(responses) if len(responses) > 1 else None)
    if len(responses) == 1:
        create.return_value = responses[0]
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return AnthropicApi(client=client, **kwargs), create


def _sent(create: MagicMock, call: int = -1) -> dict:
    return create.call_args_list[call].kwargs


def test_create_response_returns_normalized_response() -> None:
    api, create = _api(_response([_text("Hello "), _text("there")]))

    result = api.create_response(
        model="claude-sonnet-5-5",
        input=[{"role": "user", "content": "Say hi"}],
    )

    assert isinstance(result, LLMResponse)
    assert result.response_id == "msg_1"
    assert result.model == "claude-sonnet-5-5"
    assert result.output_text == "Hello there"
    assert result.tool_calls == []
    assert result.usage is not None
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == (10, 5, 15)
    sent = _sent(create)
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["max_tokens"] == AnthropicApi.DEFAULT_MAX_TOKENS
    assert sent["messages"] == [{"role": "user", "content": "Say hi"}]
    assert "system" not in sent
    assert "tools" not in sent


def test_max_tokens_is_configurable() -> None:
    api, create = _api(_response([_text("ok")]), max_tokens=2048)
    api.create_response(model="claude-haiku-5-5", input="hi")
    assert _sent(create)["max_tokens"] == 2048


def test_string_and_dict_inputs_become_user_messages() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(model="claude-haiku-5-5", input="plain")
    assert _sent(create)["messages"] == [{"role": "user", "content": "plain"}]
    api.create_response(model="claude-haiku-5-5", input={"role": "user", "content": "one"})
    assert _sent(create)["messages"] == [{"role": "user", "content": "one"}]


def test_system_messages_move_to_top_level_system() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(
        model="claude-sonnet-5-5",
        input=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
            {"role": "developer", "content": [{"type": "text", "text": "Use British spelling."}]},
        ],
    )
    sent = _sent(create)
    assert sent["system"] == "You are terse.\n\nUse British spelling."
    assert sent["messages"] == [{"role": "user", "content": "Hi"}]


def test_empty_input_raises_before_calling_provider() -> None:
    api, create = _api(_response([_text("ok")]))
    with pytest.raises(ValueError, match="No messages"):
        api.create_response(model="claude-sonnet-5-5", input=[{"role": "system", "content": "only system"}])
    assert not create.called


@pytest.mark.parametrize("bad_input", [42, [123], [{"role": "tool", "content": "x"}]])
def test_unsupported_input_raises(bad_input: Any) -> None:
    api, create = _api(_response([_text("ok")]))
    with pytest.raises(ValueError):
        api.create_response(model="claude-sonnet-5-5", input=bad_input)
    assert not create.called


def test_image_parts_convert_to_anthropic_blocks() -> None:
    api, create = _api(_response([_text("a cat")]))
    api.create_response(
        model="claude-sonnet-5-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image", "source": {"data": "QUJD", "mime_type": "image/jpeg"}},
                    {"type": "image", "source": {"url": "https://example.com/cat.png"}},
                    {"type": "image", "source": {"data": "REVG"}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/gif", "data": "R0lG"}},
                ],
            }
        ],
    )
    assert _sent(create)["messages"][0]["content"] == [
        {"type": "text", "text": "Describe"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "REVG"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/gif", "data": "R0lG"}},
    ]


def test_supports_image_processing() -> None:
    api = AnthropicApi(client=MagicMock())
    assert api.supports_image_processing("claude-sonnet-5-5") is True
    assert api.supports_image_processing("claude-haiku-5-5") is True


def test_tools_convert_from_nested_openai_format() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "strict": True,
                },
            },
        }
    ]
    assert AnthropicApi._to_anthropic_tools(tools) == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_tools_convert_from_flat_format_and_drop_top_level_strict() -> None:
    tools = [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": True,
        }
    ]
    assert AnthropicApi._to_anthropic_tools(tools) == [
        {
            "name": "echo",
            "description": "Echo",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    ]


def test_tools_without_parameters_get_an_empty_object_schema() -> None:
    assert AnthropicApi._to_anthropic_tools([{"type": "function", "name": "ping"}]) == [
        {"name": "ping", "input_schema": {"type": "object", "properties": {}}}
    ]


def test_native_anthropic_tools_pass_through_and_junk_is_skipped() -> None:
    native_client_tool = {"name": "calc", "description": "d", "input_schema": {"type": "object"}}
    native_server_tool = {"type": "web_search_20250305", "name": "web_search"}
    tools = [native_client_tool, native_server_tool, "junk", None, {"type": "function"}]
    assert AnthropicApi._to_anthropic_tools(tools) == [native_client_tool, native_server_tool]
    assert AnthropicApi._to_anthropic_tools([]) is None
    assert AnthropicApi._to_anthropic_tools(None) is None
    assert AnthropicApi._to_anthropic_tools(["junk"]) is None


def test_tool_use_blocks_become_tool_calls_with_json_arguments() -> None:
    api, _ = _api(
        _response(
            [
                _text("Checking."),
                _tool_use("toolu_1", "get_weather", {"city": "Paris", "unit": "é"}),
                _tool_use("toolu_2", "noargs", None),
            ],
            stop_reason="tool_use",
        )
    )

    result = api.create_response(
        model="claude-sonnet-5-5",
        input="Weather?",
        tools=[{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
    )

    assert result.output_text == "Checking."
    assert [tc.call_id for tc in result.tool_calls] == ["toolu_1", "toolu_2"]
    assert [tc.name for tc in result.tool_calls] == ["get_weather", "noargs"]
    assert json.loads(result.tool_calls[0].arguments_json) == {"city": "Paris", "unit": "é"}
    assert "é" in result.tool_calls[0].arguments_json
    assert result.tool_calls[1].arguments_json == "{}"


def test_thinking_blocks_are_not_part_of_output_text() -> None:
    api, _ = _api(_response([_thinking("secret"), _text("answer")]))
    assert api.create_response(model="claude-opus-5-5", input="q").output_text == "answer"


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ("any", {"type": "any"}),
        ("get_weather", {"type": "tool", "name": "get_weather"}),
        ({"type": "tool", "name": "x"}, {"type": "tool", "name": "x"}),
    ],
)
def test_tool_choice_mapping_on_models_that_allow_forcing(tool_choice: Any, expected: dict) -> None:
    assert AnthropicApi._to_tool_choice("claude-haiku-5-5", tool_choice) == expected


def test_tool_choice_none_value() -> None:
    assert AnthropicApi._to_tool_choice("claude-sonnet-5-5", None) is None


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"])
def test_forced_tool_choice_falls_back_to_auto_on_models_that_reject_it(
    model: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert AnthropicApi._to_tool_choice(model, "required") == {"type": "auto"}
        assert AnthropicApi._to_tool_choice(model, "get_weather") == {"type": "auto"}
    assert "rejects forced tool_choice" in caplog.text
    assert AnthropicApi._to_tool_choice(model, "none") == {"type": "none"}
    assert AnthropicApi._to_tool_choice(model, "auto") == {"type": "auto"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-5-5", False),
        ("claude-opus-5-5", False),
        ("claude-fable-5-1", False),
        ("claude-mythos-5-1", False),
        ("claude-haiku-5-5", True),
        ("claude-sonnet-5", True),
        ("claude-opus-5", True),
        ("claude-opus-4-8", True),
        ("claude-haiku-4-5-20251001", True),
        ("claude-3-5-sonnet-20241022", True),
        ("something-else", True),
    ],
)
def test_forced_tool_choice_support_by_model(model: str, expected: bool) -> None:
    assert _forced_tool_choice_supported(model) is expected


def test_tool_choice_is_sent_with_tools_and_ignored_without() -> None:
    api, create = _api(_response([_text("ok")]))
    tools = [{"type": "function", "name": "t", "parameters": {"type": "object"}}]

    api.create_response(model="claude-haiku-5-5", input="x", tools=tools, tool_choice="required")
    assert _sent(create)["tool_choice"] == {"type": "any"}
    assert _sent(create)["tools"][0]["name"] == "t"

    api.create_response(model="claude-haiku-5-5", input="x", tool_choice="required")
    assert "tool_choice" not in _sent(create)
    assert "tools" not in _sent(create)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-3-haiku-20240307", True),
        ("claude-3-opus-20240229", True),
        ("claude-3-5-sonnet-20241022", True),
        ("claude-2.1", True),
        ("claude-opus-4-20250514", True),
        ("claude-sonnet-4-5", True),
        ("claude-haiku-4-5-20251001", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-6", True),
        ("claude-opus-4-7", False),
        ("claude-opus-4-8", False),
        ("claude-sonnet-5", False),
        ("claude-opus-5", False),
        ("claude-sonnet-5-5", False),
        ("claude-opus-5-5", False),
        ("claude-haiku-5-5", False),
        ("claude-fable-5-1", False),
        ("CLAUDE-SONNET-5-5", False),
        ("claude-instant-1.2", True),
        ("not-a-claude-id", True),
        ("", True),
    ],
)
def test_sampling_params_support_by_model(model: str, expected: bool) -> None:
    assert _sampling_params_supported(model) is expected


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"])
def test_temperature_is_dropped_for_models_that_reject_it(model: str, caplog: pytest.LogCaptureFixture) -> None:
    api, create = _api(_response([_text("ok")]))
    with caplog.at_level(logging.WARNING):
        api.create_response(model=model, input="x", temperature=0.2)
    assert "temperature" not in _sent(create)
    assert "dropping unsupported sampling param temperature" in caplog.text


@pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-3-5-sonnet-20241022", "claude-haiku-4-5-20251001"])
def test_temperature_is_sent_for_models_that_accept_it(model: str) -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(model=model, input="x", temperature=0.2)
    assert _sent(create)["temperature"] == 0.2


def test_temperature_zero_is_still_sent_when_supported() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(model="claude-sonnet-4-5", input="x", temperature=0.0)
    assert _sent(create)["temperature"] == 0.0


def test_unused_provider_options_are_not_forwarded() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(
        model="claude-sonnet-5-5",
        input="x",
        store=True,
        metadata={"conversation_id": "c1"},
        text={"format": {"type": "json_object"}},
    )
    assert set(_sent(create)) == {"model", "max_tokens", "messages"}


def test_tool_loop_replays_assistant_turn_and_system() -> None:
    first_content = [_thinking(), _text("Let me check."), _tool_use("toolu_1", "get_weather", {"city": "Paris"})]
    api, create = _api(
        _response(first_content, response_id="msg_1", stop_reason="tool_use"),
        _response([_text("It is sunny.")], response_id="msg_2"),
    )
    tools = [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]

    first = api.create_response(
        model="claude-sonnet-5-5",
        input=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Paris?"},
        ],
        tools=tools,
    )
    second = api.create_response(
        model="claude-sonnet-5-5",
        input=[{"type": "function_call_output", "call_id": first.tool_calls[0].call_id, "output": "sunny"}],
        tools=tools,
        previous_response_id=first.response_id,
    )

    assert second.output_text == "It is sunny."
    sent = _sent(create)
    assert sent["system"] == "Be brief."
    assert sent["messages"][0] == {"role": "user", "content": "Weather in Paris?"}
    assistant = sent["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == first_content
    assert assistant["content"][0] is first_content[0]
    assert sent["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}],
    }
    assert len(sent["messages"]) == 3


def test_parallel_tool_results_share_one_user_message_in_order() -> None:
    api, create = _api(
        _response(
            [_tool_use("a", "f", {}), _tool_use("b", "g", {})],
            response_id="msg_1",
            stop_reason="tool_use",
        ),
        _response([_text("done")], response_id="msg_2"),
    )
    first = api.create_response(model="claude-sonnet-5-5", input="go", tools=[{"name": "f", "input_schema": {}}])
    api.create_response(
        model="claude-sonnet-5-5",
        input=[
            {"type": "function_call_output", "call_id": "a", "output": "one"},
            {"type": "function_call_output", "call_id": "b", "output": "two"},
        ],
        previous_response_id=first.response_id,
    )
    results = _sent(create)["messages"][-1]
    assert results["role"] == "user"
    assert [(r["tool_use_id"], r["content"]) for r in results["content"]] == [("a", "one"), ("b", "two")]


def test_multi_round_tool_loop_keeps_full_history() -> None:
    api, create = _api(
        _response([_tool_use("t1", "f", {})], response_id="r1", stop_reason="tool_use"),
        _response([_tool_use("t2", "f", {})], response_id="r2", stop_reason="tool_use"),
        _response([_text("final")], response_id="r3"),
    )
    tools = [{"name": "f", "input_schema": {}}]
    r1 = api.create_response(model="claude-sonnet-5-5", input="start", tools=tools)
    r2 = api.create_response(
        model="claude-sonnet-5-5",
        input=[{"type": "function_call_output", "call_id": "t1", "output": "o1"}],
        tools=tools,
        previous_response_id=r1.response_id,
    )
    api.create_response(
        model="claude-sonnet-5-5",
        input=[{"type": "function_call_output", "call_id": "t2", "output": "o2"}],
        tools=tools,
        previous_response_id=r2.response_id,
    )
    roles = [m["role"] for m in _sent(create)["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_tool_outputs_fall_back_to_previous_tool_calls_metadata() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(
        model="claude-sonnet-5-5",
        input=[{"type": "function_call_output", "call_id": "toolu_9", "output": "42"}],
        metadata={"previous_tool_calls": [ToolCall("toolu_9", "calc", '{"x": 1}')]},
        previous_response_id="unknown",
    )
    messages = _sent(create)["messages"]
    assert messages[0] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_9", "name": "calc", "input": {"x": 1}}],
    }
    assert messages[1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "42"}],
    }


def test_metadata_tool_calls_tolerate_invalid_argument_json() -> None:
    message = AnthropicApi._tool_calls_to_assistant_message(
        [ToolCall("c1", "f", "not json"), ToolCall("c2", "g", "[1]"), ToolCall("c3", "h", "")]
    )
    assert [block["input"] for block in message["content"]] == [{}, {}, {}]


def test_tool_outputs_without_known_assistant_turn_raise() -> None:
    api, create = _api(_response([_text("ok")]))
    with pytest.raises(ValueError, match="assistant turn"):
        api.create_response(
            model="claude-sonnet-5-5",
            input=[{"type": "function_call_output", "call_id": "x", "output": "y"}],
            previous_response_id="missing",
        )
    assert not create.called


def test_tool_outputs_with_explicit_assistant_turn_in_input_are_accepted() -> None:
    api, create = _api(_response([_text("ok")]))
    api.create_response(
        model="claude-sonnet-5-5",
        input=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "z", "name": "f", "input": {}}]},
            {"type": "function_call_output", "call_id": "z", "output": "r"},
        ],
    )
    assert [m["role"] for m in _sent(create)["messages"]] == ["user", "user", "assistant"]


def test_plain_messages_with_previous_response_id_are_not_prefixed_with_history() -> None:
    api, create = _api(
        _response([_tool_use("t1", "f", {})], response_id="r1", stop_reason="tool_use"),
        _response([_text("ok")], response_id="r2"),
    )
    api.create_response(model="claude-sonnet-5-5", input="first", tools=[{"name": "f", "input_schema": {}}])
    api.create_response(
        model="claude-sonnet-5-5",
        input=[{"role": "user", "content": "fresh"}],
        previous_response_id="r1",
    )
    assert _sent(create)["messages"] == [{"role": "user", "content": "fresh"}]


def test_conversation_cache_is_bounded() -> None:
    responses = [
        _response([_tool_use(f"t{i}", "f", {})], response_id=f"r{i}", stop_reason="tool_use") for i in range(5)
    ]
    api, _ = _api(*responses, cache_size=2)
    for _ in range(5):
        api.create_response(model="claude-sonnet-5-5", input="go", tools=[{"name": "f", "input_schema": {}}])
    assert list(api._conversation_context) == ["r3", "r4"]


def test_responses_without_tool_calls_are_not_cached() -> None:
    api, _ = _api(_response([_text("hi")], response_id="r1"))
    api.create_response(model="claude-sonnet-5-5", input="hello")
    assert not api._conversation_context


def test_usage_is_none_when_provider_reports_none() -> None:
    response = _response([_text("hi")])
    response.usage = None
    api, _ = _api(response)
    assert api.create_response(model="claude-sonnet-5-5", input="hello").usage is None


def test_max_tokens_stop_without_text_logs_a_hint(caplog: pytest.LogCaptureFixture) -> None:
    api, _ = _api(_response([_thinking()], stop_reason="max_tokens"))
    with caplog.at_level(logging.WARNING):
        result = api.create_response(model="claude-sonnet-5-5", input="hello")
    assert result.output_text == ""
    assert "thinking tokens count toward max_tokens" in caplog.text


def test_refusal_stop_reason_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    api, _ = _api(_response([_text("")], stop_reason="refusal"))
    with caplog.at_level(logging.WARNING):
        api.create_response(model="claude-haiku-5-5", input="hello")
    assert "refused" in caplog.text


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    sleep = MagicMock()
    monkeypatch.setattr(anthropic_module, "_sleep_backoff", sleep)
    return sleep


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
def test_retryable_status_is_retried(status: int, no_sleep: MagicMock) -> None:
    api, create = _api(_StatusError(status), _response([_text("recovered")]))
    result = api.create_response(model="claude-sonnet-5-5", input="x")
    assert result.output_text == "recovered"
    assert create.call_count == 2
    assert no_sleep.call_count == 1


def test_connection_errors_without_status_are_retried(no_sleep: MagicMock) -> None:
    api, create = _api(ConnectionError("reset"), _response([_text("ok")]))
    assert api.create_response(model="claude-sonnet-5-5", input="x").output_text == "ok"
    assert create.call_count == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(status: int, no_sleep: MagicMock) -> None:
    api, create = _api(_StatusError(status))
    create.side_effect = _StatusError(status)
    with pytest.raises(_StatusError):
        api.create_response(model="claude-sonnet-5-5", input="x")
    assert create.call_count == 1
    assert not no_sleep.called


@pytest.mark.parametrize("error", [TypeError("no auth"), ValueError("bad"), KeyError("k"), AttributeError("a")])
def test_configuration_and_programming_errors_are_not_retried(error: Exception, no_sleep: MagicMock) -> None:
    api, create = _api(error)
    create.side_effect = error
    with pytest.raises(type(error)):
        api.create_response(model="claude-sonnet-5-5", input="x")
    assert create.call_count == 1
    assert not no_sleep.called


def test_retries_are_exhausted_and_last_error_is_raised(no_sleep: MagicMock) -> None:
    api, create = _api(_StatusError(500), max_attempts=3)
    create.side_effect = _StatusError(500)
    with pytest.raises(_StatusError):
        api.create_response(model="claude-sonnet-5-5", input="x")
    assert create.call_count == 3
    assert no_sleep.call_count == 2


def test_missing_sdk_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anthropic_module, "Anthropic", None)
    api = AnthropicApi(settings=Settings())
    with pytest.raises(RuntimeError, match=r"galet\[anthropic\]"):
        api.create_response(model="claude-sonnet-5-5", input="x")


def test_default_client_uses_settings_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    constructor = MagicMock()
    monkeypatch.setattr(anthropic_module, "Anthropic", constructor)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "anthropic_cred.json").write_text(json.dumps({"anthropic_api_key": "sk-ant-file"}), encoding="utf-8")

    AnthropicApi(settings=Settings(credential_path=str(tmp_path)))._get_client()

    constructor.assert_called_once_with(api_key="sk-ant-file")


def test_default_client_prefers_environment_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    constructor = MagicMock()
    monkeypatch.setattr(anthropic_module, "Anthropic", constructor)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    (tmp_path / "anthropic_cred.json").write_text(json.dumps({"anthropic_api_key": "sk-ant-file"}), encoding="utf-8")

    AnthropicApi(settings=Settings(credential_path=str(tmp_path)))._get_client()

    constructor.assert_called_once_with(api_key="sk-ant-env")


def test_default_client_is_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock()
    monkeypatch.setattr(anthropic_module, "Anthropic", constructor)
    api = AnthropicApi(settings=Settings())
    assert api._get_client() is api._get_client()
    assert constructor.call_count == 1


def test_real_sdk_client_builds_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GALET_CREDENTIAL_PATH", raising=False)
    client = AnthropicApi._build_default_client(Settings())
    assert hasattr(client, "messages")


def test_registry_resolves_claude_models_to_the_connector() -> None:
    name, instance = ProviderRegistry.resolve("claude-sonnet-5-5", settings=Settings())
    assert name == "anthropic"
    assert isinstance(instance, AnthropicApi)


def test_router_dispatches_claude_requests_to_the_connector() -> None:
    api, create = _api(_response([_text("routed")]))
    router = RouterApi(instances={"anthropic": api})
    result = router.create_response(model="claude-haiku-5-5", input="hi")
    assert result.output_text == "routed"
    assert _sent(create)["model"] == "claude-haiku-5-5"
    assert router.supports_image_processing("claude-haiku-5-5") is True


def test_package_exports_the_connector() -> None:
    import galet

    assert galet.AnthropicApi is AnthropicApi


def test_real_sdk_round_trip_replays_thinking_blocks_on_the_wire() -> None:
    httpx = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("anthropic")
    requests: list[dict] = []
    replies = [
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [
                {"type": "thinking", "thinking": "need weather", "signature": "abc123"},
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
            ],
            "usage": {"input_tokens": 21, "output_tokens": 34},
        },
        {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": [{"type": "text", "text": "Sunny in Paris."}],
            "usage": {"input_tokens": 60, "output_tokens": 7},
        },
    ]

    def handler(request: Any) -> Any:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(requests) - 1])

    client = sdk.Anthropic(
        api_key="sk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    api = AnthropicApi(client=client)
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "strict": True,
        }
    ]

    first = api.create_response(
        model="claude-sonnet-5-5",
        temperature=0.3,
        tool_choice="required",
        tools=tools,
        input=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather?"}],
    )
    second = api.create_response(
        model="claude-sonnet-5-5",
        tools=tools,
        previous_response_id=first.response_id,
        input=[{"type": "function_call_output", "call_id": first.tool_calls[0].call_id, "output": "sunny"}],
    )

    assert first.tool_calls[0].arguments_json == '{"city": "Paris"}'
    assert second.output_text == "Sunny in Paris."
    assert "temperature" not in requests[0]
    assert requests[0]["tool_choice"] == {"type": "auto"}
    assert "strict" not in requests[0]["tools"][0]
    assert requests[1]["system"] == "Be brief."
    assistant = requests[1]["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "need weather", "signature": "abc123"}
    assert assistant["content"][2]["id"] == "toolu_01"
    assert requests[1]["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "sunny"}
    ]
