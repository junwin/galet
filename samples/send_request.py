"""Send a simple request through galet and print the answer.

Run from the repo root:

    # Local Ollama (no API key required, but a local Ollama server must run)
    python samples/send_request.py

    # Any provider, with an explicit model
    python samples/send_request.py --provider openai --model gpt-4o-mini "What is 2 + 2?"

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

from galet.router_api import RouterApi
from galet.settings import Settings

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
    "mistral": "mistral-small-latest",
    "ollama": "llama3.1",
}


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
        default="ollama",
        help="Provider source name: openai, deepseek, gemini, mistral, or ollama.",
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

    model = args.model or DEFAULT_MODELS.get(args.provider, "gpt-4o-mini")

    settings = Settings(
        credential_path=args.credential_path,
        ollama_base_url=args.ollama_base_url,
    )
    router = RouterApi(settings=settings)
    response = router.create_response(
        model=model,
        input=[{"role": "user", "content": args.prompt}],
        provider=args.provider,
    )

    print()
    print(f"Provider : {args.provider}")
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
