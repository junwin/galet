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
python samples/send_request.py --credential-path /home/myname/credential --model mistral-large-latest "Tell me a one-sentence joke."
```

### `tool_handlers.py`

Demonstrates the tool-handler pattern: each tool owns `name()`, `tool_def()`,
`result_schema()`, and `execute()`, and a bounded loop feeds tool results back
to the model until it answers without calling a tool or the iteration cap is
reached. Three tools are defined: `execute_command` (run a shell command),
`file_load` (read a file), and `file_save` (write a file) — all are stubbed
and never touch the real shell or filesystem. By default it targets a local
Ollama server (`llama3.1`), which needs no API key.

```bash
python samples/tool_handlers.py
python samples/tool_handlers.py "Run 'echo hello' using execute_command and tell me what it printed"
python samples/tool_handlers.py "Use file_load to read README.md and tell me what it says"
python samples/tool_handlers.py "Use file_save to write a greeting to greeting.txt"
python samples/tool_handlers.py --provider ollama --model llama3.1
python samples/tool_handlers.py --model qwen2.5:3b --max-iterations 5 "Echo hello"
```

CLI flags: an optional positional `prompt`, `--provider` (source name, defaults
to `ollama`), `--model` (defaults to a sensible model for the chosen provider),
`--credential-path` (credential directory), `--ollama-base-url` (Ollama server
URL), and `--max-iterations` (tool-calling round trips, default 10).

### `generate_image.py`

Asks an image generation model to create an image and prints the result. By
default it uses OpenAI `gpt-image-1`; the model name decides the backend
(`dall-e-*` / `gpt-image-*` / `openai/*` go to OpenAI, `gemini-*` / `imagen-*`
go to Gemini). Pass `--out` to save the first generated image to a file.

```bash
python samples/generate_image.py "a red panda in a spacesuit"
python samples/generate_image.py --size 1536x1024 --quality high "a wide landscape"
python samples/generate_image.py --model gpt-image-1 --out /tmp/panda.png "a red panda"
python samples/generate_image.py --model gemini-2.5-flash-image --out /tmp/panda.png "a red panda"
python samples/generate_image.py --credential-path /home/myname/credential "a red panda"
```

Quality values: dall-e models accept `standard`/`hd`; gpt-image models accept
`low`/`medium`/`high`/`auto` (legacy `standard`/`hd` are mapped for you). Gemini
image models ignore quality, and gpt-image models only support `n=1`.

### `describe_image.py`

Sends an image file to a vision-capable model and prints its description. The
image is base64-encoded and sent inline (no upload). By default it uses OpenAI
`gpt-4o-mini`.

```bash
python samples/describe_image.py path/to/image.png
python samples/describe_image.py --prompt "What animals are in this photo?" path/to/image.png
python samples/describe_image.py --model gpt-4o path/to/image.jpg
python samples/describe_image.py --provider gemini --model gemini-2.0-flash path/to/image.png
python samples/describe_image.py --credential-path /home/myname/credential path/to/image.png
```
