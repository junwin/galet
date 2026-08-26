"""Generate an image through galet and optionally save it to disk.

Run from the repo root:

    # Default: OpenAI gpt-image-1
    python samples/generate_image.py "a red panda in a spacesuit"

    # Different size / quality (gpt-image-1 supports 1024x1024, 1536x1024,
    # 1024x1536 and quality low/medium/high/auto)
    python samples/generate_image.py --size 1536x1024 --quality high "a wide landscape"

    # Google Gemini image model (returns base64, saved with --out)
    python samples/generate_image.py --model gemini-2.5-flash-image --out /tmp/panda.png "a red panda"

    # Point galet at a directory holding credential files
    python samples/generate_image.py --credential-path /path/to/credentials "a red panda"

The model name decides the backend: dall-e-* / gpt-image-* / openai/* go to
OpenAI, gemini-* / imagen-* go to Gemini. The default model is gpt-image-1.

Quality values: dall-e models accept standard/hd; gpt-image models accept
low/medium/high/auto (standard/hd are mapped for you). Gemini image models
ignore quality. gpt-image models only generate one image per call (n=1).

Non-Ollama providers need an API key. galet reads keys from environment
variables (OPENAI_API_KEY, GEMINI_API_KEY), from credential files via
``Settings(credential_path=...)``, or from the ``GALET_CREDENTIAL_PATH``
environment variable. Pass ``--credential-path`` to point at that directory
explicitly.

If ``--out`` is given, the first generated image is written to that file,
either by decoding ``b64_json`` or by downloading the result ``url``.
"""

from __future__ import annotations

import argparse
import base64
import urllib.request

from galet.gemini_imagegen import GeminiImageGenApi
from galet.imagegen_dto import ImageResult
from galet.imagegen_router import ImageGenRouter
from galet.openai_imagegen import OpenAIImageGenApi
from galet.settings import Settings

DEFAULT_MODELS = {
    "openai": "gpt-image-1",
    "gemini": "gemini-2.5-flash-image",
}


def _save_image(result: ImageResult, out_path: str) -> None:
    if result.b64_json:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result.b64_json))
        return
    if result.url:
        with urllib.request.urlopen(result.url) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        return
    raise RuntimeError("image result has neither b64_json nor url")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an image via galet and print the result.",
    )
    parser.add_argument("prompt", help="The image prompt to send.")
    parser.add_argument(
        "--model",
        default=None,
        help="Image model (defaults to gpt-image-1 for OpenAI, gemini-2.5-flash-image for Gemini).",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="Image size, e.g. 1024x1024, 1536x1024, 1024x1536 (default 1024x1024).",
    )
    parser.add_argument(
        "--quality",
        default="standard",
        help="Image quality: standard or hd for dall-e; low, medium, high or auto for gpt-image (default standard).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of images to generate (default 1). gpt-image models only support n=1.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path for the first generated image.",
    )
    parser.add_argument(
        "--credential-path",
        default=None,
        help="Directory holding the galet credential files (e.g. oaicred.json). "
        "Alternative to the GALET_CREDENTIAL_PATH environment variable.",
    )
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS["openai"]

    settings = Settings(credential_path=args.credential_path)
    router = ImageGenRouter(
        openai_api=OpenAIImageGenApi(settings=settings),
        gemini_api=GeminiImageGenApi(settings=settings),
    )

    print()
    print(f"Model    : {model}")
    print(f"Prompt   : {args.prompt}")
    print(f"Size     : {args.size}")
    print()

    response = router.generate_image(
        model=model,
        prompt=args.prompt,
        size=args.size,
        quality=args.quality,
        n=args.n,
    )

    print(f"Generated {len(response.images)} image(s):")
    for i, image in enumerate(response.images, start=1):
        if image.url:
            print(f"  {i}. url: {image.url}")
        if image.b64_json:
            print(f"  {i}. b64_json: {len(image.b64_json)} characters")
        if image.revised_prompt:
            print(f"     revised_prompt: {image.revised_prompt}")

    if args.out:
        if not response.images:
            print("No images returned; nothing to save.")
        else:
            _save_image(response.images[0], args.out)
            print(f"Saved first image to: {args.out}")


if __name__ == "__main__":
    main()
