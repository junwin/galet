# OpenAI GPT-6 Astra: drop sampling params (temperature, top_p, top_logprobs)

Issue: https://github.com/junwin/galet/issues/13

## Context

OpenAI guidance for GPT-6 Astra on the Responses API: remove `temperature`,
`top_p`, and `top_logprobs` from requests — sending them returns a 400
(unsupported parameter). (For Chat Completions, `logprobs` is also
unsupported; not applicable here since the OpenAI connector uses the Responses
API.)

Today `OpenAIResponsesApi.create_response()` forwards `temperature` straight to
`responses.create(...)`, so a non-None `temperature` with a GPT-6 Astra model
fails with a 400. Only `temperature` is plumbed through the stack
(`LLMApi.create_response` -> `OpenAIResponsesAdapter.call_model` ->
`OpenAIResponsesApi.create_response`); `top_p` and `top_logprobs` are not yet
part of the `LLMApi` protocol but will be added later and must be covered
automatically.

## Scope

### In scope

- OpenAI connector only: `src/galet/openai_responses.py`.
- Sanitize generation params before they reach `responses.create(...)`.
- Regression tests in `tests/test_openai_responses.py`.

### Out of scope

- Lucy changes (sanitization lives in galet, which owns OpenAI protocol
  knowledge).
- deepseek / gemini / mistral / ollama connectors.
- Extending the `LLMApi` protocol or `OpenAIResponsesAdapter` with new params.
- Chat Completions handling.

## Confirmed decisions

1. Only galet's OpenAI Responses connector is changed.
2. Sampling params are dropped for model ids starting with `gpt-6` (prefix
   match, e.g. `gpt-6-astra`).
3. When a caller passes an unsupported sampling param, log it and drop it —
   never raise.
4. Existing models (`gpt-4o`, `gpt-5`, ...) keep current behaviour unchanged.

## Design

### Capability helper — `src/galet/openai_responses.py`

- Small pure function, OpenAI-scoped, no client/settings/network access.
- Generic over a params dict so future params are covered automatically:
  `_sanitize_generation_params(model: str, params: dict[str, Any]) ->
  dict[str, Any]`.
- Owns the knowledge: sampling params (`temperature`, `top_p`,
  `top_logprobs`) are removed when the model id starts with `gpt-6`; any other
  params and any other model pass through unchanged.
- Returns a new dict; never mutates the caller's dict.

### Sanitize step in `OpenAIResponsesApi.create_response`

1. Collect the caller-supplied generation params (non-None) into a dict —
   `temperature` today; `top_p` / `top_logprobs` later, covered automatically.
2. Run the dict through `_sanitize_generation_params(model, params)`.
3. Log a warning for each dropped param (model id, param name).
4. Call `responses.create(...)` with the sanitized dict; all other request
   kwargs are unchanged.

No change to retry/backoff logic or DTO mapping.

## Test plan

- `tests/test_openai_responses.py` (new):
  - Helper unit tests: `gpt-6-astra` drops `temperature`, `top_p`,
    `top_logprobs` and keeps unrelated params; `gpt-6` (exact) drops too;
    `gpt-5`, `gpt-4o`, `o3` pass through unchanged; caller's dict is not
    mutated.
  - `create_response` regression (fake client): `gpt-6-astra` with
    `temperature=0.0` reaches the client without a `temperature` kwarg;
    `gpt-5` with `temperature=0.0` still passes it through.
- Full suite:

```
.venv/bin/python -m pytest tests/ -q
```
