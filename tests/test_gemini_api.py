from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lucy_llm.dto import LLMResponse
from lucy_llm.gemini_api import GeminiApi
from lucy_llm.settings import Settings


def _stub_interaction(
    *,
    interaction_id: str = "inter-1",
    model: str = "gemini-2.0-flash",
    output_text: str = "",
    steps: list | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=interaction_id,
        model=model,
        output_text=output_text,
        steps=steps or [],
        usage=usage,
    )


def _stub_function_call_step(step_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=step_id, name=name, arguments=arguments, type="function_call")


def _stub_usage(input_tokens: int = 10, output_tokens: int = 5, total_tokens: int = 15) -> SimpleNamespace:
    return SimpleNamespace(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def test_extract_system_instruction() -> None:
    input_items = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
    ]
    assert GeminiApi._extract_system_instruction(input_items) == (
        "You are a helpful assistant.\n\nBe concise."
    )


def test_extract_system_instruction_single_dict() -> None:
    assert GeminiApi._extract_system_instruction({"role": "system", "content": "Sys"}) == "Sys"


def test_extract_system_instruction_empty() -> None:
    assert GeminiApi._extract_system_instruction("not a list") == ""
    assert GeminiApi._extract_system_instruction([]) == ""


def test_to_gemini_tools_nested() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "strict": True,
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert GeminiApi._to_gemini_tools(tools) == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]


def test_to_gemini_tools_flat() -> None:
    tools = [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object", "strict": True, "additionalProperties": False},
        }
    ]
    assert GeminiApi._to_gemini_tools(tools) == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo",
            "parameters": {"type": "object"},
        }
    ]


def test_to_gemini_tools_skips_non_function_entries() -> None:
    tools = [
        {"type": "other", "name": "x"},
        "not a dict",
        None,
        {"type": "function", "function": {"name": "only_func"}},
    ]
    assert GeminiApi._to_gemini_tools(tools) == [
        {"type": "function", "name": "only_func"}
    ]


def test_to_gemini_tools_empty() -> None:
    assert GeminiApi._to_gemini_tools(None) is None
    assert GeminiApi._to_gemini_tools([]) is None
    assert GeminiApi._to_gemini_tools([{"type": "other"}]) is None


def test_to_gemini_input_passthrough_function_result() -> None:
    function_result = {
        "type": "function_result",
        "name": "get_weather",
        "call_id": "call-1",
        "result": [{"type": "text", "text": "sunny"}],
    }
    input_items = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Weather?"},
        function_result,
    ]
    assert GeminiApi._to_gemini_input(input_items) == [
        {"type": "user_input", "content": [{"type": "text", "text": "Weather?"}]},
        function_result,
    ]


def test_create_response_normalizes_interaction() -> None:
    fake_client = MagicMock()
    stub = _stub_interaction(
        output_text="hello",
        steps=[_stub_function_call_step("step-1", "get_weather", {"city": "London"})],
        usage=_stub_usage(),
    )
    fake_client.interactions.create.return_value = stub

    api = GeminiApi(client=fake_client)
    result = api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Hi"}],
    )

    kwargs = fake_client.interactions.create.call_args.kwargs
    assert "tool_choice" not in kwargs
    assert "generation_config" not in kwargs

    assert isinstance(result, LLMResponse)
    assert result.response_id == "inter-1"
    assert result.model == "gemini-2.0-flash"
    assert result.output_text == "hello"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == "step-1"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments_json == '{"city": "London"}'
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.raw is stub


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        ("auto", "auto"),
        ("required", "any"),
        ("none", "none"),
        ("validated", "validated"),
    ],
)
def test_create_response_tool_choice_in_generation_config(tool_choice: str, expected: str) -> None:
    fake_client = MagicMock()
    fake_client.interactions.create.return_value = _stub_interaction()

    api = GeminiApi(client=fake_client)
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Hi"}],
        tool_choice=tool_choice,
        temperature=0.7,
    )

    kwargs = fake_client.interactions.create.call_args.kwargs
    assert "tool_choice" not in kwargs
    assert "temperature" not in kwargs
    assert kwargs["generation_config"] == {"tool_choice": expected, "temperature": 0.7}


def test_create_response_temperature_only() -> None:
    fake_client = MagicMock()
    fake_client.interactions.create.return_value = _stub_interaction()

    api = GeminiApi(client=fake_client)
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Hi"}],
        temperature=0.2,
    )

    kwargs = fake_client.interactions.create.call_args.kwargs
    assert kwargs["generation_config"] == {"temperature": 0.2}


def test_create_response_passes_function_tools() -> None:
    fake_client = MagicMock()
    fake_client.interactions.create.return_value = _stub_interaction()

    api = GeminiApi(client=fake_client)
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Hi"}],
        tools=tools,
    )

    kwargs = fake_client.interactions.create.call_args.kwargs
    assert kwargs["tools"] == tools


def test_system_instruction_cached_by_conversation_id() -> None:
    fake_client = MagicMock()
    function_result = {
        "type": "function_result",
        "name": "get_weather",
        "call_id": "call-1",
        "result": [{"type": "text", "text": "sunny"}],
    }
    fake_client.interactions.create.side_effect = [
        _stub_interaction(interaction_id="i-1"),
        _stub_interaction(interaction_id="i-2"),
        _stub_interaction(interaction_id="i-3"),
    ]

    api = GeminiApi(client=fake_client)
    api.create_response(
        model="gemini-2.0-flash",
        input=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        metadata={"conversation_id": "conv-1"},
    )
    api.create_response(
        model="gemini-2.0-flash",
        input=[function_result],
        metadata={"conversation_id": "conv-1"},
    )
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "user", "content": "Another chat"}],
        metadata={"conversation_id": "conv-2"},
    )

    first_call = fake_client.interactions.create.call_args_list[0]
    second_call = fake_client.interactions.create.call_args_list[1]
    third_call = fake_client.interactions.create.call_args_list[2]

    assert first_call.kwargs["system_instruction"] == "You are helpful."
    assert second_call.kwargs["system_instruction"] == "You are helpful."
    assert second_call.kwargs["input"] == [function_result]
    assert third_call.kwargs["system_instruction"] == ""


def test_system_instruction_cache_eviction() -> None:
    fake_client = MagicMock()
    fake_client.interactions.create.return_value = _stub_interaction()

    api = GeminiApi(client=fake_client, cache_size=2)
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "system", "content": "Sys 1"}],
        metadata={"conversation_id": "conv-1"},
    )
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "system", "content": "Sys 2"}],
        metadata={"conversation_id": "conv-2"},
    )
    api.create_response(
        model="gemini-2.0-flash",
        input=[{"role": "system", "content": "Sys 3"}],
        metadata={"conversation_id": "conv-3"},
    )

    # conv-1 should be evicted because cache max size is 2
    assert "conv-1" not in api._system_instruction_cache
    assert api._system_instruction_cache["conv-2"] == "Sys 2"
    assert api._system_instruction_cache["conv-3"] == "Sys 3"


def test_exhausted_retries_raises_last_error() -> None:
    fake_client = MagicMock()
    fake_client.interactions.create.side_effect = ValueError("API failed")

    api = GeminiApi(client=fake_client, max_attempts=2, backoff_base=0.01)

    with pytest.raises(ValueError, match="API failed"):
        api.create_response(
            model="gemini-2.0-flash",
            input=[{"role": "user", "content": "Hi"}],
        )

    assert fake_client.interactions.create.call_count == 2


@patch.dict("os.environ", {"GEMINI_API_KEY": "env-key-123"}, clear=True)
@patch("lucy_llm.gemini_api.genai.Client")
def test_build_default_client_env_api_key(mock_client_cls: MagicMock) -> None:
    GeminiApi._build_default_client()
    mock_client_cls.assert_called_once_with(api_key="env-key-123")


@patch.dict("os.environ", {"GEMINI_CREDENTIALS": '{"gemini_api_key": "cred-json-key"}'}, clear=True)
@patch("lucy_llm.gemini_api.genai.Client")
def test_build_default_client_env_credentials_json(mock_client_cls: MagicMock) -> None:
    GeminiApi._build_default_client()
    mock_client_cls.assert_called_once_with(api_key="cred-json-key")


@patch.dict("os.environ", {}, clear=True)
@patch("lucy_llm.gemini_api.genai.Client")
@patch("builtins.open")
def test_build_default_client_settings_credential_file(
    mock_open: MagicMock,
    mock_client_cls: MagicMock,
) -> None:
    mock_open.return_value.__enter__.return_value = SimpleNamespace(
        read=lambda: '{"gemini_api_key": "file-key-456"}'
    )

    GeminiApi._build_default_client(settings=Settings(credential_path="/path/to/creds"))

    mock_client_cls.assert_called_once_with(api_key="file-key-456")
