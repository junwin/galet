"""Demonstrate galet tool calling with a bounded handler loop.

Run from the repo root:

    python samples/tool_handlers.py

    python samples/tool_handlers.py "Use the execute_command tool to run 'echo hello'"

    python samples/tool_handlers.py --provider ollama --model llama3.1 "Echo hello"

    python samples/tool_handlers.py --credential-path /path/to/credentials --provider openai "Echo hello"

    python samples/tool_handlers.py --ollama-base-url http://localhost:11434/v1 --provider ollama "Echo hello"

The default provider is Ollama, which needs no API key but does require a
local Ollama server at http://localhost:11434. Non-Ollama providers need an
API key, which galet reads from environment variables (OPENAI_API_KEY,
DEEPSEEK_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY), from credential files via
``Settings(credential_path=...)``, or from the ``GALET_CREDENTIAL_PATH``
environment variable (the directory holding the credential files). Pass
``--credential-path`` to point at that directory explicitly.

Ollama needs no API key, but galet must know the server address. It uses
``--ollama-base-url`` when given, otherwise ``OLLAMA_BASE_URL``, otherwise the
default ``http://localhost:11434/v1``.

Each tool is a handler that exposes name(), tool_def(), result_schema(), and
execute(). The loop sends the prompt with the tool definitions, executes any
tool calls the model returns, feeds the formatted results back, and repeats
until the model answers without requesting a tool or the iteration cap is
reached.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Protocol

from galet.dto import ToolCall
from galet.router_api import RouterApi
from galet.settings import Settings
from galet.tool_output import format_tool_output

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "mistral": "mistral-small-latest",
    "ollama": "llama3.1",
}

STUBBED_COMMAND_RESULT = {
    "status": "stubbed",
    "output": "command execution is disabled in this sample",
}


class ToolHandler(Protocol):
    def name(self) -> str: ...

    def tool_def(self) -> Dict[str, Any]: ...

    def result_schema(self) -> Dict[str, Any]: ...

    def execute(self, arguments: Dict[str, Any]) -> Any: ...


class ExecuteCommandTool:
    def name(self) -> str:
        return "execute_command"

    def tool_def(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name(),
            "description": "Run a shell command. This sample never executes the command and always returns a stubbed result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                },
                "required": ["command"],
            },
        }

    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "output": {"type": "string"},
            },
            "required": ["status", "output"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return dict(STUBBED_COMMAND_RESULT)


class EchoTool:
    def name(self) -> str:
        return "echo"

    def tool_def(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name(),
            "description": "Echo the given text back to the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to echo.",
                    },
                },
                "required": ["text"],
            },
        }

    def result_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "echoed": {"type": "string"},
            },
            "required": ["echoed"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {"echoed": str(arguments.get("text", ""))}


class ToolRegistry:
    def __init__(self, handlers: List[ToolHandler]) -> None:
        self._handlers = {handler.name(): handler for handler in handlers}

    def tool_defs(self) -> List[Dict[str, Any]]:
        return [handler.tool_def() for handler in self._handlers.values()]

    def get(self, name: str) -> ToolHandler:
        if name not in self._handlers:
            raise KeyError(f"unknown tool: {name}")
        return self._handlers[name]


class ToolCallLoop:
    def __init__(
        self,
        router: RouterApi,
        registry: ToolRegistry,
        provider: str,
        model: str,
        max_iterations: int = 10,
    ) -> None:
        self._router = router
        self._registry = registry
        self._provider = provider
        self._model = model
        self._max_iterations = max_iterations

    @staticmethod
    def _parse_arguments(arguments_json: str) -> Dict[str, Any]:
        if not arguments_json:
            return {}
        try:
            parsed = json.loads(arguments_json)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _tool_calls_metadata(calls: List[ToolCall]) -> Dict[str, str]:
        return {
            "previous_tool_calls": json.dumps(
                [
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments_json": call.arguments_json,
                    }
                    for call in calls
                ],
                ensure_ascii=False,
            )
        }

    def _execute_call(self, call: ToolCall) -> Any:
        try:
            handler = self._registry.get(call.name)
        except KeyError:
            return {"status": "error", "message": f"unknown tool: {call.name}"}
        return handler.execute(self._parse_arguments(call.arguments_json))

    def run(self, prompt: str) -> None:
        tools = self._registry.tool_defs()
        input_messages: Any = [{"role": "user", "content": prompt}]
        previous_response_id: Optional[str] = None
        previous_tool_calls: Optional[List[ToolCall]] = None

        for _ in range(self._max_iterations):
            metadata = (
                self._tool_calls_metadata(previous_tool_calls)
                if previous_tool_calls
                else None
            )
            response = self._router.create_response(
                model=self._model,
                input=input_messages,
                tools=tools,
                provider=self._provider,
                previous_response_id=previous_response_id,
                metadata=metadata,
            )

            if not response.tool_calls:
                print(response.output_text)
                return

            input_messages = [
                format_tool_output(
                    call_id=call.call_id,
                    output=json.dumps(self._execute_call(call), ensure_ascii=False),
                    name=call.name,
                    provider=self._provider,
                )
                for call in response.tool_calls
            ]
            previous_response_id = response.response_id
            previous_tool_calls = response.tool_calls

        print(f"Stopped after {self._max_iterations} iterations without a final answer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded tool-calling loop through galet.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Use the execute_command tool to run 'echo hello'.",
        help="The user prompt to send.",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        help="Provider source name (defaults to ollama, which needs no API key).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults to a sensible value for the chosen provider).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of tool-calling round trips.",
    )
    parser.add_argument(
        "--credential-path",
        default=None,
        help="Directory holding the galet credential files (e.g. oaicred.json). "
        "Alternative to the GALET_CREDENTIAL_PATH environment variable.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama server base URL (OpenAI-compatible endpoint). "
        "Defaults to OLLAMA_BASE_URL or http://localhost:11434/v1.",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS.get(args.provider, "llama3.1")

    registry = ToolRegistry([ExecuteCommandTool(), EchoTool()])
    loop = ToolCallLoop(
        router=RouterApi(
            settings=Settings(
                credential_path=args.credential_path,
                ollama_base_url=args.ollama_base_url,
            )
        ),
        registry=registry,
        provider=args.provider,
        model=model,
        max_iterations=args.max_iterations,
    )

    print()
    print(f"Provider : {args.provider}")
    print(f"Model    : {model}")
    print(f"Prompt   : {args.prompt}")
    print()
    loop.run(args.prompt)


if __name__ == "__main__":
    main()
