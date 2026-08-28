"""Send a simple request through galet and print the answer.

Run from the repo root:

    # Local Ollama (no API key required, but a local Ollama server must run)
    python samples/send_request.py

    # Any provider, with an explicit model
    python samples/send_request.py --provider openai --model gpt-4o-mini "What is 2 + 2?"

    # Provider inferred from the model name (--provider omitted)
    python samples/send_request.py --model mistral-large-latest "Tell me a one-sentence joke."

    # Point galet at a directory holding credential files
    python samples/send_request.py --credential-path /path/to/credentials --provider openai "What is 2 + 2?"

    # Point galet at a non-default Ollama server
    python samples/send_request.py --ollama-base-url http://localhost:11434/v1 --provider ollama "hi"

    # Ask a custom question
    python samples/send_request.py "Tell me a one-sentence joke."

Provider sources and their default models:

    openai   -> gpt-4o-mini
    deepseek -> deepseek-chat
    gemini   -> gemini-2.0-flash
    mistral  -> mistral-small-latest
    ollama   -> llama3.1

When ``--provider`` is omitted it is inferred from the model name (mistral-*,
deepseek-*, gemini-*, gpt-*/o1*/o3* map to their providers). With no
``--model`` and no ``--provider`` the request defaults to local Ollama.

Non-Ollama providers need an API key. galet reads keys from environment
variables (OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY),
from credential files via ``Settings(credential_path=...)``, or from the
``GALET_CREDENTIAL_PATH`` environment variable (the directory holding the
credential files). Pass ``--credential-path`` to point at that directory
explicitly.

Ollama needs no API key, but galet must know the server address. It uses
``--ollama-base-url`` when given, otherwise ``OLLAMA_BASE_URL``, otherwise the
default ``http://localhost:11434/v1``.
"""

from __future__ import annotations

import argparse
from typing import Optional

from galet.provider_registry import ProviderRegistry
from galet.router_api import RouterApi
from galet.settings import Settings

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "mistral": "mistral-small-latest",
    "ollama": "llama3.1",
}


def resolve_provider(model: Optional[str], explicit_provider: Optional[str]) -> str:
    if explicit_provider:
        return explicit_provider
    if model:
        return ProviderRegistry.resolve_name(model, None)
    return "ollama"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a request via galet and print the answer.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello in one short sentence.",
        help="The user prompt to send.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider source name: openai, deepseek, gemini, mistral, or ollama. "
        "Inferred from the model name when omitted; defaults to ollama.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults to a sensible value for the chosen provider).",
    )
    parser.add_argument(
        "--credential-path",
        default=None,
        help="Directory holding the galet credential files (e.g. oaicred.json). "
        "Alternative to the GALET_CREDENTIAL_PATH environment variable.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama server base URL (OpenAI-compatible endpoint). "
        "Defaults to OLLAMA_BASE_URL or http://localhost:11434/v1.",
    )
    args = parser.parse_args()

    provider = resolve_provider(args.model, args.provider)
    model = args.model or DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    settings = Settings(
        credential_path=args.credential_path,
        ollama_base_url=args.ollama_base_url,
    )
    router = RouterApi(settings=settings)
    response = router.create_response(
        model=model,
        input=[{"role": "user", "content": args.prompt}],
        provider=provider,
    )

    print()
    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"Prompt   : {args.prompt}")
    print(f"Answer   : {response.output_text}")

    if response.usage is not None:
        print(
            f"Usage    : in={response.usage.input_tokens} "
            f"out={response.usage.output_tokens} "
            f"total={response.usage.total_tokens}"
        )


if __name__ == "__main__":
    main()
