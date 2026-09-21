"""Generate a six-second portrait fashion reel with Google Veo.

Set GEMINI_API_KEY, then run:

    python samples/generate_video.py IMAGE_URL_OR_PATH "motion prompt" [OUTPUT.mp4]
"""

from __future__ import annotations

import sys

from galet.gemini_videogen import GeminiVideoGenApi


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: python samples/generate_video.py IMAGE_URL_OR_PATH "
            '"motion prompt" [OUTPUT.mp4]'
        )
        return 2

    image_source = sys.argv[1]
    prompt = sys.argv[2]
    destination = sys.argv[3] if len(sys.argv) > 3 else "veo-fashion-reel.mp4"

    api = GeminiVideoGenApi()
    image_args = (
        {"image_url": image_source}
        if image_source.startswith(("http://", "https://"))
        else {"image_path": image_source}
    )
    response = api.generate_video(
        model="veo-3.1-fast-generate-preview",
        prompt=prompt,
        **image_args,
        aspect_ratio="9:16",
        duration_seconds=6,
        resolution="720p",
        preserve_subject=True,
    )
    result = response.videos[0]
    if result.url:
        print(f"Temporary provider URL: {result.url}")
    api.download_video(result, destination)
    print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
