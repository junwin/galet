# lucy-llm

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
