# List models available for a source

Issue: https://github.com/junwin/galet/issues/15

## Goal

Add a public method on the source interface that returns the model ids a source
makes available, with per-connector implementations for OpenAI, DeepSeek,
Mistral, Gemini and Ollama.

## Current state

- The `LLMApi` protocol in `src/galet/interface.py` exposes only
  `supports_image_processing()` and `create_response()`.
- Each connector builds its own client:
  - OpenAI / DeepSeek / Mistral / Ollama: OpenAI SDK, differing `base_url`.
  - Gemini: `google.genai.Client`.
- There is no public way to discover which model ids a source accepts; ids must
  be known in advance (config `default_model`, call-site constants).
- `ProviderRegistry.resolve()` returns a connector instance, or `_DummyApi` as a
  non-fatal fallback when a connector cannot be loaded.

## Provider research

All five vendored sources support model listing. No "unsupported" path is
needed.

| Source | galet connector (client) | List call | Identifier | Notes |
|---|---|---|---|---|
| OpenAI | `OpenAIResponsesApi` (OpenAI SDK) | `client.models.list()` | `.id` | All accessible models, incl. fine-tuned |
| DeepSeek | `DeepSeekApi` (OpenAI SDK @ `api.deepseek.com`) | `client.models.list()` | `.id` | Official `GET /models`; OpenAI-compatible; e.g. `deepseek-chat` |
| Mistral | `MistralApi` (OpenAI SDK @ `api.mistral.ai/v1`) | `client.models.list()` | `.id` | OpenAI-compatible `/v1/models`, OpenAI-shaped response |
| Ollama | `OllamaApi` (OpenAI SDK @ `localhost:11434/v1`) | `client.models.list()` | `.id` | OpenAI compatibility includes `/v1/models`; ids are `name:tag` (e.g. `llama3.1:latest`) |
| Gemini | `GeminiApi` (`google.genai.Client`) | `client.models.list()`, then filter | `.name` minus `models/` prefix | Filter keeps chat models only; `display_name` also available |

Details:

- **OpenAI / DeepSeek / Mistral / Ollama** already hold an OpenAI SDK client
  (`self._client`), so `client.models.list()` is the single shared call. The
  only difference is which `base_url` the client was built with, which is
  already resolved by each connector today.
- **Ollama** also has a native `GET /api/tags`. We deliberately use `/v1/models`
  instead so the listing goes through the already-constructed OpenAI client and
  the id format matches what `create_response(model=...)` accepts.
- **Gemini** lists many non-chat models. Google's own guidance is to keep
  entries whose `supported_actions` includes `generateContent`. The public id is
  `name` (`models/gemini-2.0-flash`); galet calls the API with the id without
  the `models/` prefix, so the returned ids strip that prefix to stay consistent
  with `create_response(model=...)`.

## Scope

### In scope

1. Add `list_models()` to the `LLMApi` protocol.
2. Implement it in the five connectors: `OpenAIResponsesApi`, `DeepSeekApi`,
   `MistralApi`, `OllamaApi`, `GeminiApi`.
3. Shared helper for the four OpenAI-compatible connectors (DRY).
4. `_DummyApi` in `provider_registry.py` gains `list_models()` returning `[]`
   so registry-resolved callers never crash.
5. Tests per connector (mocked clients) plus the full `pytest` suite green.

### Out of scope

- Richer return values (capabilities, availability, context window, aliases) —
  follow-up; the issue's display use case needs ids only.
- Runtime validation of a source's `default_model` against the list — flagged
  in the issue; separate decision (see Open questions).
- Surfacing the method through a CLI/sample — the issue asks for a public method
  on the source only; `samples/list_sources.py` untouched.
- Filtering OpenAI/DeepSeek/Mistral results to chat-only models — the plain
  models endpoint carries no such flag; revisit if a source needs it.
- Using Ollama's native `/api/tags` endpoint.
- Lucy changes.

## Design

### Return type

`list[str]` of model ids, de-duplicated and sorted, in exactly the form callers
pass to `create_response(model=...)`.

- OpenAI-compatible sources: `.id` as returned.
- Gemini: `name` with the `models/` prefix stripped (e.g. `gemini-2.0-flash`).

### Interface — `src/galet/interface.py`

Add to the `LLMApi` protocol:

```python
def list_models(self) -> List[str]:
    """Return the model ids available for this source."""
    ...
```

### Shared helper — new module `src/galet/model_listing.py`

```python
def list_openai_compatible_models(client: Any) -> List[str]:
    """Return sorted, de-duplicated model ids via the OpenAI SDK models endpoint."""
```

Handles both the iterable page and its `.data` attribute, collecting
`getattr(model, "id", "")` and dropping blanks. Kept provider-agnostic so
OpenAI, DeepSeek, Mistral and Ollama share one implementation (DRY; matches the
existing pattern of sharing `_sleep_backoff`).

### Per-connector implementation

- `OpenAIResponsesApi.list_models()` → `list_openai_compatible_models(self._client)`
- `DeepSeekApi.list_models()` → same helper
- `MistralApi.list_models()` → same helper
- `OllamaApi.list_models()` → same helper
- `GeminiApi.list_models()`:
  1. `resp = self._client.models.list()`
  2. Keep models whose `supported_actions` contains `generateContent`
     (treating missing/empty `supported_actions` as not matching).
  3. Map `name` → strip `models/` prefix.
  4. Sort and de-duplicate.

`GeminiApi` already lazy-imports `google.genai` with a stub fallback; in
environments without `genai` the stub raises, and the error policy below applies.

### Error handling

Follow the issue #13 precedent (log and continue, don't raise): on any
exception during listing, log a warning and return `[]`. Rationale: the primary
use is display; a transient listing failure should not take down a stream or
tool run. A future validation use case that needs hard failures can raise at
that call site.

### Registry fallback

`_DummyApi.list_models()` returns `[]` so a caller that resolved through
`ProviderRegistry` never hits `NotImplementedError`.

## Test plan

- `tests/test_model_listing.py` (new): shared helper against a fake OpenAI-style
  client — sorted/deduped ids, blank-id filtering, iterator vs `.data` page,
  exception → `[]` with logged warning.
- `tests/test_openai_embedding.py`-style per-connector cases (or one focused
  file): each of `OpenAIResponsesApi`, `DeepSeekApi`, `MistralApi`, `OllamaApi`
  delegates to the helper with its own client.
- `tests/test_gemini_api.py`: add `list_models` cases — `generateContent` filter
  drops embedding models, `models/` prefix stripped, missing `supported_actions`
  excluded, exception → `[]`.
- `tests/test_provider_info.py` / registry tests: `_DummyApi.list_models()` is
  `[]`.

Run with:

```
galet/.venv/bin/python -m pytest tests/ -q
```

## Open questions

1. Should the registry validate `default_model` against `list_models()` at
   startup, and if so warn or error? (Deferred; not needed for display.)
2. Should the return value later be enriched to structured model objects
   (id + display name + capabilities)? Gemini already exposes `display_name`.
3. Should OpenAI/DeepSeek/Mistral lists be restricted to chat-capable models?
   Their plain models endpoints do not expose that flag today.
