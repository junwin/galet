# galet samples

Small, runnable scripts that show how to use the `galet` library.

## Setup

Run the samples from the repo root with an installed `galet` package:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

## Configuration

See the root `README.md` for full details. The short version:

- **API keys** — non-Ollama providers need one. Use the provider env var
  (`OPENAI_API_KEY`, etc.), or `--credential-path` (or `GALET_CREDENTIAL_PATH`)
  pointing at a directory of credential files such as `oaicred.json`.
- **Ollama URL** — no key needed, but set the server address with
  `--ollama-base-url` (or `OLLAMA_BASE_URL`). The default is
  `http://localhost:11434/v1`.

## Scripts

### `list_sources.py`

Lists the available LLM sources (providers) and the model-name prefix routing
galet uses when no provider is given.

```bash
python samples/list_sources.py
```

### `send_request.py`

Sends a simple request and prints the answer. By default it targets a local
Ollama server (`llama3.1`), which needs no API key.

```bash
python samples/send_request.py
python samples/send_request.py "Tell me a one-sentence joke."
python samples/send_request.py --provider openai --model gpt-4o-mini "What is 2 + 2?"
python samples/send_request.py --provider ollama --ollama-base-url http://localhost:11434/v1 "hi"
python samples/send_request.py --credential-path /home/myname/credential --provider openai --model gpt-4o-mini "Tell me a one-sentence joke."
```

### `tool_handlers.py`

Demonstrates the tool-handler pattern: each tool owns `name()`, `tool_def()`,
`result_schema()`, and `execute()`, and a bounded loop feeds tool results back
to the model until it answers without calling a tool or the iteration cap is
reached. The `execute_command` tool is stubbed — it never runs a real shell
command. By default it targets a local Ollama server (`llama3.1`), which needs
no API key.

```bash
python samples/tool_handlers.py
python samples/tool_handlers.py "Run 'echo hello' using execute_command and tell me what it printed"
python samples/tool_handlers.py 
python samples/tool_handlers.py --provider ollama --model llama3.1
python samples/tool_handlers.py --model qwen2.5:3b --max-iterations 5 "Echo hello"
```

CLI flags: an optional positional `prompt`, `--provider` (source name, defaults
to `ollama`), `--model` (defaults to a sensible model for the chosen provider),
`--credential-path` (credential directory), `--ollama-base-url` (Ollama server
URL), and `--max-iterations` (tool-calling round trips, default 10).
