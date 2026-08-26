# galet

Provider-agnostic LLM, embedding, and image generation stack.

## Lineage

This package is extracted from the Lucy monorepo (`src/llm`) as described in
the Lucy design doc `software/ai/lucy/design/llm-module-extraction.md`.

It is a fresh repository with no shared git history. The extracted module's
behaviour, public interface, optional-SDK import fallbacks, and
no-fail-on-missing-credential semantics are preserved as-is; the only
intentional change is the configuration boundary (see `settings.py`).

## Features

- **Chat / completions** — provider-agnostic `create_response` with
  temperature, tool calling, and response metadata. Providers: OpenAI,
  DeepSeek, Gemini, Mistral, Ollama.
- **Routing** — explicit `provider` argument, or automatic model-name prefix
  routing with OpenAI fallback (`ProviderRegistry`).
- **Tool calling** — bounded tool loop; tools own `name()`, `tool_def()`,
  `result_schema()`, and `execute()`.
- **Image generation** — OpenAI (`dall-e-*`, `gpt-image-*`) and Gemini
  (`gemini-*`, `imagen-*`) backends behind a common interface.
- **Image description** — vision-capable models describe images sent inline
  (base64, no upload).
- **Embeddings** — OpenAI and Mistral embedding adapters.
- **Optional SDKs** — imports degrade gracefully when a provider SDK is not
  installed; missing credentials never raise at import or call time.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Runnable examples live in `samples/`:

| Script | What it shows |
|---|---|
| `samples/send_request.py` | Simple chat request (defaults to local Ollama) |
| `samples/tool_handlers.py` | Tool-calling loop with a stubbed `execute_command` tool |
| `samples/generate_image.py` | Image generation via OpenAI or Gemini |
| `samples/describe_image.py` | Describe an image file with a vision model |
| `samples/list_sources.py` | List providers and model-prefix routing |

See `samples/README.md` for each script's full usage.

## Configuration

galet needs two pieces of configuration, and both can be set with an
environment variable or a command-line flag.

### Credentials (API keys)

Only non-Ollama providers need an API key. galet resolves a key in this order:

1. The provider's own environment variable (`OPENAI_API_KEY`,
   `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`).
2. A credential file in the directory named by `--credential-path`, or the
   `GALET_CREDENTIAL_PATH` environment variable when no flag is given.

Credential file names:

| Provider  | File                | Key(s) in the file            |
|-----------|---------------------|-------------------------------|
| openai    | `oaicred.json`      | `openai_api_key`              |
| deepseek  | `deepseek_cred.json`| `deepseek_api_key`            |
| gemini    | `gemini_cred.json`  | `gemini_api_key`, `api_key`   |
| mistral   | `mistral_cred.json` | `mistral_api_key`             |

Example credential file (`oaicred.json`):

```json
{"openai_api_key": "sk-..."}
```

```bash
python samples/send_request.py --credential-path /path/to/credentials --provider openai "hi"
GALET_CREDENTIAL_PATH=/path/to/credentials python samples/send_request.py --provider openai "hi"
```

### Ollama base URL

Ollama needs no API key, but galet must know where the Ollama server is. galet
resolves the address in this order:

1. `--ollama-base-url` (command-line flag)
2. `OLLAMA_BASE_URL` (environment variable)
3. `http://localhost:11434/v1` (default, Ollama's OpenAI-compatible endpoint)

```bash
# Default local Ollama
python samples/send_request.py --provider ollama "hi"

# Explicit URL
python samples/send_request.py --provider ollama --ollama-base-url http://localhost:11434/v1 "hi"

# Environment variable
OLLAMA_BASE_URL=http://192.168.87.40:11434/v1 python samples/send_request.py --provider ollama "hi"
```

Note the `/v1` suffix: galet talks to Ollama's OpenAI-compatible endpoint, not
the native Ollama API.
