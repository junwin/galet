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
  routing with OpenAI fallback. Source names, model-name prefixes, class paths,
  and default models are declarative; provider implementations load only when selected.
- **Model catalog** — inspect source/model metadata and resolve a model from a
  profile, required capabilities, and optional model/source preferences.
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

Galet owns provider credential resolution. Applications and delegated tasks
should refer to a named profile rather than handle an API key.

A named profile selects exactly one source. Galet does not silently fall back
to an environment variable or another file when that profile is selected.

```python
from galet.settings import Settings

settings = Settings(
    credential_profiles={
        "openai-service": {
            "provider": "openai",
            "source": "file",
            "path": "/etc/galet/credentials/oaicred.json",
        }
    },
    credential_profile="openai-service",
)

key = settings.api_key("openai")
```

Supported profile sources:

| Source | Required field | Behaviour |
|---|---|---|
| `file` | `path` | Reads the existing provider JSON credential format |
| `environment` | `variable` | Reads only that named environment variable |
| `systemd` | `credential` | Reads that file beneath `CREDENTIALS_DIRECTORY` |

Systemd credentials may contain the existing JSON object or the raw API key:

```ini
[Service]
User=lucy
LoadCredential=openai:/etc/galet/credentials/oaicred.json
```

```python
settings = Settings(
    credential_profiles={
        "openai-service": {
            "provider": "openai",
            "source": "systemd",
            "credential": "openai",
        }
    },
    credential_profile="openai-service",
)
```

For ordinary protected files, keep credentials outside the repository, make
the directory accessible only to the service account, and make each credential
file readable only by that account. The `GALET_CREDENTIAL_PATH` environment
variable contains only a directory name and is not itself a secret.

For backward compatibility, when no named profile is selected Galet retains
the original lookup order:

1. The provider environment variable (`OPENAI_API_KEY`,
   `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, or `MISTRAL_API_KEY`).
2. The provider file in `credential_path` or `GALET_CREDENTIAL_PATH`.

Credential file names and JSON keys:

| Provider  | File                 | Key(s) in the file |
|-----------|----------------------|--------------------|
| openai    | `oaicred.json`       | `openai_api_key` |
| deepseek  | `deepseek_cred.json` | `deepseek_api_key` |
| gemini    | `gemini_cred.json`   | `gemini_api_key`, `api_key` |
| mistral   | `mistral_cred.json`  | `mistral_api_key` |

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


## Model information and capability resolution

`ProviderRegistry` continues to route explicit model names to their source.
Source and model metadata can be inspected without importing provider
implementations, making an API call, or reading credentials. `ModelCatalog`
adds deterministic capability-based selection.

```python
from galet import ModelRequirements, default_model_catalog

for source in default_model_catalog.sources():
    print(source.name, source.default_model)

for model in default_model_catalog.models(source="openai"):
    print(model.name, sorted(model.profiles), sorted(model.capabilities))

resolved = default_model_catalog.resolve(
    ModelRequirements.create(
        profile="focused-development",
        required_capabilities=("tool-calling", "code-generation"),
        preferred_model="gpt-5-mini",
    )
)
print(resolved.source, resolved.model)
```

Model preferences are not agent identities. A caller may request a capability
profile and allow Galet to select an eligible fallback. Credential lookup
remains inside Galet's provider settings and is not exposed by the catalog.
