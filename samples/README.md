# galet samples

Small, runnable scripts that show how to use the `galet` library.

## Setup

Run the samples from the repo root with an installed `galet` package:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

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
```

Other providers require an API key. galet resolves keys from environment
variables (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`,
`MISTRAL_API_KEY`) or from credential files via `Settings(credential_path=...)`.
