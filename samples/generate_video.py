"""Generate a Google Omni video from an image, or use '-' for text only.

    python samples/generate_video.py IMAGE_URL_OR_PATH "motion prompt" [OUTPUT.mp4]
    python samples/generate_video.py - "A fox walking through snow" fox.mp4
"""
from __future__ import annotations

import argparse

from galet.gemini_omni_videogen import GeminiOmniVideoGenApi
from galet.gemini_videogen import GeminiVideoGenApi
from galet.videogen_router import VideoGenRouter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Image URL, local path, or '-' for text only")
    parser.add_argument("prompt")
    parser.add_argument("output", nargs="?", default="omni-video.mp4")
    parser.add_argument("--model", default="gemini-omni-1.1-flash")
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--aspect-ratio", choices=("9:16", "16:9"), default="9:16")
    parser.add_argument("--resolution", choices=("360p", "720p", "1080p", "4k"), default="720p")
    parser.add_argument("--no-preserve-subject", action="store_true")
    args = parser.parse_args()

    omni = GeminiOmniVideoGenApi()
    veo = GeminiVideoGenApi()
    router = VideoGenRouter(omni_api=omni, gemini_api=veo)
    image_args = {}
    if args.image != "-":
        key = "image_url" if args.image.startswith(("http://", "https://")) else "image_path"
        image_args[key] = args.image
    response = router.generate_video(
        model=args.model,
        prompt=args.prompt,
        **image_args,
        aspect_ratio=args.aspect_ratio,
        duration_seconds=args.duration,
        resolution=args.resolution,
        preserve_subject=not args.no_preserve_subject,
    )
    result = response.videos[0]
    provider = omni if args.model.startswith("gemini-omni-") else veo
    provider.download_video(result, args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
