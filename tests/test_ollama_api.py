"""Unit tests for OllamaApi.

Tests the Ollama LLM backend using mocked OpenAI client responses.
Follows the same pattern as other LLM provider tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.llm.ollama_api import OllamaApi
from src.llm.dto import LLMResponse, LLMUsage, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(
    *,
    response_id: str = "resp-123",
    model: str = "llama3.1",
    content: str = "Hello!",
    tool_calls: list | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Build a mock chat completion response matching OpenAI SDK shape."""
    mock = MagicMock()
    mock.id = response_id
    mock.model = model

    choice = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice.message = message
    mock.choices = [choice]

    mock.usage = usage
    return mock


def _make_mock_tool_call(call_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock tool call matching OpenAI SDK shape."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_mock_usage(prompt: int = 10, completion: int = 5, total: int = 15) -> MagicMock:
    """Build a mock usage object."""
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    return u


# ---------------------------------------------------------------------------
# supports_image_processing
# ---------------------------------------------------------------------------

def test_supports_image_processing_returns_false() -> None:
    api = OllamaApi()
    assert api.supports_image_processing("llama3.1") is False
    assert api.supports_image_processing("qwen3:14b") is False
    assert api.supports_image_processing("gemma3") is False


# ---------------------------------------------------------------------------
# Default client construction
# ---------------------------------------------------------------------------

def test_default_client_uses_ollama_base_url() -> None:
    api = OllamaApi()
    assert api._client.base_url.host == "localhost"
    assert api._client.base_url.port == 11434


def test_custom_base_url() -> None:
    api = OllamaApi(base_url="http://192.168.1.50:11434/v1")
    assert api._client.base_url.host == "192.168.1.50"
    assert api._client.base_url.port == 11434


def test_config_override_base_url() -> None:
    """ollama_base_url from config.json overrides the default."""
    with patch("src.llm.ollama_api.ConfigManager") as MockConfig:
        mock_cm = MockConfig.return_value
        mock_cm.get.return_value = "http://192.168.1.50:11434/v1"

        api = OllamaApi()

    assert api._client.base_url.host == "192.168.1.50"
    assert api._client.base_url.port == 11434


def test_missing_config_key_falls_back_to_default() -> None:
    """Missing ollama_base_url config key falls back to localhost:11434."""
    with patch("src.llm.ollama_api.ConfigManager") as MockConfig:
        mock_cm = MockConfig.return_value
        mock_cm.get.return_value = None

        api = OllamaApi()

    assert api._client.base_url.host == "localhost"
    assert api._client.base_url.port == 11434


# ---------------------------------------------------------------------------
# create_response — basic text response
# ---------------------------------------------------------------------------

def test_create_response_text_only() -> None:
    mock_client = MagicMock()
    mock_resp = _make_mock_response(
        content="Hello from Ollama!",
        usage=_make_mock_usage(),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    api = OllamaApi(client=mock_client)

    result = api.create_response(
        model="llama3.1",
        input=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(result, LLMResponse)
    assert result.output_text == "Hello from Ollama!"
    assert result.model == "llama3.1"
    assert result.response_id == "resp-123"
    assert result.tool_calls == []
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


# ---------------------------------------------------------------------------
# create_response — tool calls
# ---------------------------------------------------------------------------

def test_create_response_with_tool_calls() -> None:
    mock_client = MagicMock()
    mock_tc = _make_mock_tool_call("call_1", "get_weather", '{"city": "London"}')
    mock_resp = _make_mock_response(
        content="",
        tool_calls=[mock_tc],
    )
    mock_client.chat.completions.create.return_value = mock_resp

    api = OllamaApi(client=mock_client)

    result = api.create_response(
        model="llama3.1",
        input=[{"role": "user", "content": "What's the weather?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments_json == '{"city": "London"}'


# ---------------------------------------------------------------------------
# create_response — conversation context (tool response)
# ---------------------------------------------------------------------------

def test_create_response_with_tool_outputs() -> None:
    mock_client = MagicMock()
    # First call returns tool calls
    mock_tc = _make_mock_tool_call("call_abc", "echo", '{"msg": "test"}')
    mock_resp1 = _make_mock_response(
        response_id="resp-1",
        content="",
        tool_calls=[mock_tc],
    )
    mock_resp2 = _make_mock_response(
        response_id="resp-2",
        content="Done.",
    )
    mock_client.chat.completions.create.side_effect = [mock_resp1, mock_resp2]

    api = OllamaApi(client=mock_client)

    # First call — model requests a tool
    result1 = api.create_response(
        model="llama3.1",
        input=[{"role": "user", "content": "Echo test"}],
        tools=[{"type": "function", "function": {"name": "echo"}}],
    )
    assert len(result1.tool_calls) == 1
    assert result1.response_id == "resp-1"

    # Second call — tool outputs are provided
    tool_outputs = [
        {"type": "function_call_output", "call_id": "call_abc", "output": "OK"},
    ]
    result2 = api.create_response(
        model="llama3.1",
        input=tool_outputs,
        previous_response_id="resp-1",
        metadata={"previous_tool_calls": result1.tool_calls},
    )
    assert result2.output_text == "Done."

    # Verify the messages sent included the tool response
    call_args = mock_client.chat.completions.create.call_args_list[1]
    messages = call_args[1]["messages"]
    # Contains: user msg + assistant tool_calls (from context) +
    # assistant tool_calls (from metadata) + tool response msg = 4
    assert len(messages) == 4
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_abc"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_abc"
    assert messages[3]["content"] == "OK"


# ---------------------------------------------------------------------------
# create_response — empty messages raises
# ---------------------------------------------------------------------------

def test_create_response_empty_input_raises() -> None:
    api = OllamaApi(client=MagicMock())
    with pytest.raises(ValueError, match="No messages"):
        api.create_response(model="llama3.1", input=[])


# ---------------------------------------------------------------------------
# create_response — retry on failure
# ---------------------------------------------------------------------------

def test_create_response_retries_then_succeeds() -> None:
    mock_client = MagicMock()
    mock_resp = _make_mock_response(content="Eventual success")
    mock_client.chat.completions.create.side_effect = [
        Exception("transient error"),
        mock_resp,
    ]

    api = OllamaApi(client=mock_client, max_attempts=3)
    result = api.create_response(
        model="llama3.1",
        input=[{"role": "user", "content": "Hi"}],
    )
    assert result.output_text == "Eventual success"
    assert mock_client.chat.completions.create.call_count == 2


def test_create_response_exhausts_retries() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("persistent error")

    api = OllamaApi(client=mock_client, max_attempts=2)
    with pytest.raises(Exception, match="persistent error"):
        api.create_response(
            model="llama3.1",
            input=[{"role": "user", "content": "Hi"}],
        )
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# _normalize_input_to_messages — input as dict
# ---------------------------------------------------------------------------

def test_normalize_single_dict_message() -> None:
    api = OllamaApi(client=MagicMock())
    messages = api._normalize_input_to_messages({"role": "user", "content": "Hi"})
    assert messages == [{"role": "user", "content": "Hi"}]


# ---------------------------------------------------------------------------
# _normalize_input_to_messages — input as list of messages
# ---------------------------------------------------------------------------

def test_normalize_list_of_messages() -> None:
    api = OllamaApi(client=MagicMock())
    input_msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    messages = api._normalize_input_to_messages(input_msgs)
    assert messages == input_msgs


# ---------------------------------------------------------------------------
# _convert_tool_calls_to_assistant_message
# ---------------------------------------------------------------------------

def test_convert_tool_calls_to_assistant_message() -> None:
    api = OllamaApi(client=MagicMock())
    tool_calls = [
        ToolCall(call_id="id1", name="search", arguments_json='{"q":"test"}'),
    ]
    msg = api._convert_tool_calls_to_assistant_message(tool_calls)
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["id"] == "id1"
    assert msg["tool_calls"][0]["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# _reconstruct_tool_calls_from_outputs
# ---------------------------------------------------------------------------

def test_reconstruct_tool_calls_from_outputs() -> None:
    outputs = [
        {"type": "function_call_output", "call_id": "call_x", "output": "result"},
        {"type": "function_call_output", "call_id": "call_y", "output": "result2"},
        # duplicate — should be skipped
        {"type": "function_call_output", "call_id": "call_x", "output": "dup"},
    ]
    stubs = OllamaApi._reconstruct_tool_calls_from_outputs(outputs)
    assert len(stubs) == 2
    assert stubs[0].call_id == "call_x"
    assert stubs[1].call_id == "call_y"
    assert all(tc.name == "__reconstructed__" for tc in stubs)
