"""Send an image to a vision-capable model and print its description.

Run from the repo root:

    # Default: OpenAI gpt-4o-mini
    python samples/describe_image.py path/to/image.png

    # Different prompt / model
    python samples/describe_image.py --prompt "What animals are in this photo?" path/to/image.png
    python samples/describe_image.py --model gpt-4o path/to/image.jpg

    # Gemini vision model
    python samples/describe_image.py --provider gemini --model gemini-2.0-flash path/to/image.png

    # Point galet at a directory holding credential files
    python samples/describe_image.py --credential-path /path/to/credentials path/to/image.png

The image is read from disk, base64-encoded, and sent inline as a content part
(no upload step). The intermediate part format is provider-agnostic:

    {"type": "image", "source": {"data": "<base64>", "mime_type": "image/png"}}

galet adapters translate that to each provider's native input format.

Non-Ollama providers need an API key. galet reads keys from environment
variables (OPENAI_API_KEY, GEMINI_API_KEY), from credential files via
``Settings(credential_path=...)``, or from the ``GALET_CREDENTIAL_PATH``
environment variable. Pass ``--credential-path`` to point at that directory
explicitly.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from galet.router_api import RouterApi
from galet.settings import Settings

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1",
}

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _mime_type(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed:
        return guessed
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def _build_input(prompt: str, path: Path) -> list:
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {"data": data, "mime_type": _mime_type(path)},
                },
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a vision-capable model to describe an image.",
    )
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Question or prompt to send along with the image.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Provider source name: openai, gemini, or ollama (default openai).",
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

    path = Path(args.image)
    if not path.is_file():
        parser.error(f"image file not found: {path}")

    model = args.model or DEFAULT_MODELS.get(args.provider, "gpt-4o-mini")
    settings = Settings(
        credential_path=args.credential_path,
        ollama_base_url=args.ollama_base_url,
    )
    router = RouterApi(settings=settings)

    print()
    print(f"Provider : {args.provider}")
    print(f"Model    : {model}")
    print(f"Image    : {path}")
    print(f"MIME     : {_mime_type(path)}")
    print()

    response = router.create_response(
        model=model,
        input=_build_input(args.prompt, path),
        provider=args.provider,
    )

    print(response.output_text)
    print()

    if response.usage is not None:
        print(
            f"Usage    : in={response.usage.input_tokens} "
            f"out={response.usage.output_tokens} "
            f"total={response.usage.total_tokens}"
        )


if __name__ == "__main__":
    main()
