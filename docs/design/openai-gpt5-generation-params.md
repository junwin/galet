# OpenAI GPT-5 and later: drop sampling params (temperature, top_p, top_logprobs)

Issue: https://github.com/junwin/galet/issues/13

## Context

OpenAI guidance for GPT-5 and later on the Responses API: remove `temperature`,
`top_p`, and `top_logprobs` from requests — sending them returns a 400
(unsupported parameter). This applies to the whole GPT-5+ generation, e.g.
`gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-6-astra`. (For Chat Completions,
`logprobs` is also unsupported; not applicable here since the OpenAI connector
uses the Responses API.)

Today `OpenAIResponsesApi.create_response()` forwards `temperature` straight to
`responses.create(...)`, so a non-None `temperature` with a GPT-5+ model fails
with a 400. Only `temperature` is plumbed through the stack
(`LLMApi.create_response` -> `OpenAIResponsesAdapter.call_model` ->
`OpenAIResponsesApi.create_response`); `top_p` and `top_logprobs` are not yet
part of the `LLMApi` protocol but will be added later and must be covered
automatically.

This decision supersedes the earlier gpt-6-only handling: the unsupported
parameter applies to GPT-5 and later, so the rule is generation-aware instead
of an exact-prefix special case.

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
2. Capability rule is generation-aware: sampling params are dropped for any
   model id parsed as `gpt-<N>...` with `N >= 5` (`gpt-5`, `gpt-5-mini`,
   `gpt-5.1`, `gpt-6-astra`, ...). Future generations (`gpt-7`, ...) are
   covered automatically; there is no per-model prefix list to maintain.
3. When a caller passes an unsupported sampling param, log it and drop it —
   never raise.
4. Other models keep current behaviour unchanged: GPT-4 and earlier (`gpt-4o`,
   `gpt-4.1`, `gpt-4.5`, `chatgpt-4o-latest`), the o-series (`o1`, `o3`), and
   any unknown model id.

## Design

### Capability helpers — `src/galet/openai_responses.py`

- Small pure functions, OpenAI-scoped, no client/settings/network access.
- `_gpt_major_generation(model)`: parse the leading `gpt-<N>` token
  (case-insensitive) into the major generation int; `None` for non-GPT ids.
- `_sampling_params_supported(model)`: the capability rule — True when the
  model is not a GPT id or has major generation < 5, False for GPT-5+.
- `_sanitize_generation_params(model: str, params: dict[str, Any]) ->
  dict[str, Any]`: generic over a params dict so future params are covered
  automatically; removes `temperature`, `top_p`, `top_logprobs` when the model
  lacks the capability; returns a new dict, never mutates the caller's dict.

### Sanitize step in `OpenAIResponsesApi.create_response`

1. Collect the caller-supplied generation params (non-None) into a dict —
   `temperature` today; `top_p` / `top_logprobs` later, covered automatically.
2. Run the dict through `_sanitize_generation_params(model, params)`.
3. Log a warning for each dropped param (model id, param name).
4. Call `responses.create(...)` with the sanitized dict; all other request
   kwargs are unchanged.

No change to retry/backoff logic or DTO mapping.

## Test plan

- `tests/test_openai_responses.py`:
  - Capability rule: `gpt-5` / `gpt-5-mini` / `gpt-5.1` / `gpt-6-astra` /
    `gpt-7` / `gpt-9` -> unsupported (incl. case-insensitive `GPT-5`);
    `gpt-4` / `gpt-4o` / `gpt-4.1` / `gpt-4.5` / `chatgpt-4o-latest` / `o1` /
    `o3` / unknown ids -> supported.
  - `_sanitize_generation_params`: GPT-5+ models drop `temperature`, `top_p`,
    `top_logprobs` with a warning per param and keep unrelated params; other
    models pass through unchanged with no log output; the caller's dict is
    never mutated.
  - `create_response` regression (fake client): `gpt-5` / `gpt-5-mini` /
    `gpt-6-astra` with `temperature=0.0` reach the client without a
    `temperature` kwarg; `gpt-4o` / `gpt-4.1` / `gpt-4.5` / `o3` with
    `temperature=0.5` still pass it through; `temperature=None` never adds a
    key.
- Full suite:

```
.venv/bin/python -m pytest tests/ -q
```
