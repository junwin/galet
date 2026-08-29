# How to add a new LLM source

This guide is written for an AI agent that will implement a new LLM source
(provider) for galet. It tells you what to study, what to implement, how to
register the source, how credentials work, and how general configuration
works. Follow the steps in order; each step ends with a checkpoint you can
verify.

---

## 1. What you are building

A connector module that implements the `LLMApi` protocol, registers itself in
the provider table, and is then callable through `RouterApi` exactly like the
existing sources (openai, deepseek, gemini, mistral, ollama).

The rest of galet stays untouched: the router, the processor, and the samples
all read provider metadata from the registration table.

---

## 2. Files to study first

Read these before writing any code. They define the contract.

| File | What it gives you |
|---|---|
| `src/galet/interface.py` | `LLMApi` protocol — the contract every connector implements |
| `src/galet/dto.py` | `LLMResponse`, `ToolCall`, `LLMUsage` — the normalized shapes you return |
| `src/galet/adapter_interface.py` | `LLMAdapter` protocol — tool-call extraction and tool-output formatting (used by the tool-calling processor) |
| `src/galet/provider_info.py` | `ProviderInfo` dataclass and the registration table (`register_provider`, `get_provider`, `registered_providers`) |
| `src/galet/provider_registry.py` | How connectors are discovered (`CONNECTOR_MODULES`) and how requests resolve to a provider |
| `src/galet/settings.py` | Credential lookup (`Settings.api_key`) and general config (`Settings`, env vars) |
| `src/galet/router_api.py` | How provider instances are wired into `RouterApi` |

Reference connectors, from simplest to most involved:

| File | Pattern it shows |
|---|---|
| `src/galet/deepseek_responses.py` | OpenAI-compatible chat completions, minimal |
| `src/galet/mistral_api.py` | OpenAI-compatible endpoint with its own base URL constant |
| `src/galet/ollama_api.py` | Local server, no API key, configurable base URL |
| `src/galet/openai_responses.py` | Responses API, tool calls, retry/backoff, image content normalization |
| `src/galet/gemini_api.py` | A non-OpenAI SDK (google-genai) |

Also fetch the API specification of the new source (its request/response
reference) and keep it open while implementing. The connector is the only
place that knows the provider's protocol.

---

## 3. The contract you implement

Your connector class must satisfy `LLMApi` (from `src/galet/interface.py`):

```python
class LLMApi(Protocol):
    def supports_image_processing(self, model: str) -> bool: ...

    def create_response(
        self,
        *,
        model: str,
        input: Any,
        temperature: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_response_id: Optional[str] = None,
        text: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse: ...
```

Rules:

- `create_response` always returns a normalized `LLMResponse` (from
  `src/galet/dto.py`): `response_id`, `model`, `output_text`, `tool_calls`,
  `usage`, `raw`. Never return the SDK's response object directly.
- `tool_calls` is a list of `ToolCall(call_id, name, arguments_json)` where
  `arguments_json` is a JSON string.
- `usage` is best-effort `LLMUsage(input_tokens, output_tokens, total_tokens,
  raw)`; `None` is acceptable when the provider does not report usage.
- `supports_image_processing` returns `True` only for models that natively
  accept images.
- Constructor convention: accept `settings: Optional[Settings] = None` and
  default to `default_settings`. Build the SDK client lazily, and accept an
  injected/mocked client parameter for tests (see `OpenAIResponsesApi`).
- If the connector participates in tool calling, also implement the
  `LLMAdapter` protocol from `src/galet/adapter_interface.py`:
  `extract_tool_calls`, `format_tool_output`, `get_text`, `get_response_id`,
  `get_usage`.

---

## 4. Register the source

Every connector module ends with a self-registration call. At the bottom of
your new module:

```python
from .provider_info import ProviderInfo, register_provider

register_provider(
    ProviderInfo(
        name="yourprovider",
        display_name="Your Provider",
        description="One-line description of the API.",
        prefixes=("yourprefix",),
        class_path="galet.yourprovider_api.YourProviderApi",
        default_model="your-default-model",
    )
)
```

Field meaning:

- `name` — short stable identifier; used as the `--provider` value and in
  `Settings.api_key("yourprovider")`.
- `display_name` — human-friendly name shown by `samples/list_sources.py`.
- `description` — one line shown by `samples/list_sources.py`.
- `prefixes` — model-name prefixes used to infer the provider when no explicit
  provider is given (e.g. `("deepseek",)` routes `deepseek-chat`). Prefixes
  are matched in registration order; first match wins.
- `class_path` — dotted path to your connector class. The registry imports it
  lazily when the provider is resolved.
- `default_model` — model used when the caller gives no model
  (samples/send_request.py derives its defaults from this).

Then add your module to the discovery seed list in
`src/galet/provider_registry.py`:

```python
CONNECTOR_MODULES: Tuple[str, ...] = (
    "galet.openai_responses",
    "galet.deepseek_responses",
    "galet.gemini_api",
    "galet.mistral_api",
    "galet.ollama_api",
    "galet.yourprovider_api",
)
```

The registry imports each module once to trigger registration; `load_all` is
idempotent and failures are logged at debug level (a broken connector must not
break the other providers).

---

## 5. Credentials

galet resolves API keys through `Settings.api_key(provider)` in
`src/galet/settings.py`. Lookup order:

1. Environment variable — from the `_PROVIDER_ENV_VAR` table
   (e.g. `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).
2. Credential file — a JSON file in the credential directory, looked up via
   `_PROVIDER_CREDENTIAL_FILE` and `_PROVIDER_CREDENTIAL_KEY`.
3. `None` when nothing is configured — the connector then fails clearly at
   request time (or builds a client that the SDK will reject).

To add credentials for your provider, extend the three tables in
`src/galet/settings.py`:

```python
_PROVIDER_ENV_VAR["yourprovider"] = "YOURPROVIDER_API_KEY"
_PROVIDER_CREDENTIAL_FILE["yourprovider"] = "yourprovider_cred.json"
_PROVIDER_CREDENTIAL_KEY["yourprovider"] = ("yourprovider_api_key",)
```

The credential directory comes from `Settings(credential_path=...)` or the
`GALET_CREDENTIAL_PATH` environment variable.

In your connector, fetch the key the same way the other connectors do:

```python
self._settings = settings or default_settings
api_key = self._settings.api_key("yourprovider")
```

If the provider needs no key (like Ollama), skip the tables entirely and pass
a placeholder.

---

## 6. General configuration

`Settings` (in `src/galet/settings.py`) is the single configuration object.
It currently holds `credential_path` and `ollama_base_url`, and is passed to
every connector constructor and to `RouterApi(settings=...)`.

Patterns:

- Provider-specific base URL as a class constant on the connector (see
  `DEEPSEEK_BASE_URL`, `MISTRAL_BASE_URL`).
- Configurable endpoint via `Settings` with an environment fallback (see
  `OllamaApi._resolve_base_url`, which reads `Settings.ollama_base_url` or the
  `OLLAMA_BASE_URL` env var).
- If your provider needs a new knob, add a field to the `Settings` dataclass
  plus an env-var fallback, and keep `default_settings` working with no config
  present. A configuration failure must never crash at import time.

---

## 7. Tests

Add `tests/test_yourprovider_api.py` and make it pass without network access:

- Build the connector with an injected fake/mocked client.
- Test `create_response` returns a normalized `LLMResponse`.
- Test tool-call extraction and `arguments_json` when the source supports
  tools.
- Test `supports_image_processing` for known model families.
- Test credential resolution through `Settings` (env var and credential file).

Run the full suite before pushing:

```bash
python -m pytest
```

All existing tests must keep passing.

---

## 8. Verify end to end

From the repo root:

```bash
python samples/list_sources.py
```

Your source must appear in the registration table with the right display
name, description, prefixes, and default model.

```bash
python samples/send_request.py --provider yourprovider --model your-default-model "Say hello"
```

Must print the provider, model, and the answer (with the provider's API key
configured).

---

## 9. Acceptance checklist

- [ ] Connector module implements `LLMApi`; returns normalized `LLMResponse`.
- [ ] Tool-calling connectors also implement `LLMAdapter`.
- [ ] Module self-registers with `ProviderInfo` at the bottom of the file.
- [ ] Module added to `CONNECTOR_MODULES` in `src/galet/provider_registry.py`.
- [ ] Credentials wired through `Settings.api_key("yourprovider")` (env var
      and/or credential file) — unless the provider needs no key.
- [ ] Any new config knob added to `Settings` with an env-var fallback.
- [ ] `tests/test_yourprovider_api.py` added; full `pytest` suite green.
- [ ] `samples/list_sources.py` shows the new source; `send_request.py`
      works with it.
- [ ] No comments added to code.
