# galet

Provider-agnostic LLM, embedding, and image generation stack.

## Lineage

This package is extracted from the Lucy monorepo (`src/llm` in
`/home/junwin/src/repos/lucy`) as described in
`software/ai/lucy/design/llm-module-extraction.md`.

It is a fresh repository with no shared git history. The extracted module's
behaviour, public interface, optional-SDK import fallbacks, and
no-fail-on-missing-credential semantics are preserved as-is; the only
intentional change is the configuration boundary (see `settings.py` in later
checkpoints).

## Status

Scaffold only. The `src/llm` modules and their tests are moved into this
package in subsequent checkpoints.

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
