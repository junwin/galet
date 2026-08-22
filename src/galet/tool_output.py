from __future__ import annotations

from typing import Any, Dict, Optional


def format_tool_output(
    *,
    call_id: str,
    output: str,
    name: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    if provider == "gemini":
        return {
            "type": "function_result",
            "name": str(name),
            "call_id": str(call_id),
            "result": [{"type": "text", "text": str(output)}],
        }
    return {"type": "function_call_output", "call_id": str(call_id), "output": str(output)}
