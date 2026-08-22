from __future__ import annotations

from src.llm.tool_output import format_tool_output


def test_default_shape() -> None:
    item = format_tool_output(call_id="call-1", output="result")
    assert item == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "result",
    }


def test_gemini_shape_includes_name() -> None:
    item = format_tool_output(call_id="call-1", output="result", name="get_weather", provider="gemini")
    assert item == {
        "type": "function_result",
        "name": "get_weather",
        "call_id": "call-1",
        "result": [{"type": "text", "text": "result"}],
    }


def test_backward_compatible_call_without_name_provider() -> None:
    item = format_tool_output(call_id="call-2", output="ok")
    assert item["type"] == "function_call_output"
    assert "name" not in item
    assert item["output"] == "ok"
