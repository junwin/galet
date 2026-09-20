from __future__ import annotations

import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from google import genai
    from google.genai import types
except Exception:
    class _GenaiClientStub:
        pass

    class genai:  # type: ignore
        Client = _GenaiClientStub

    class types:  # type: ignore
        class Image:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class GenerateVideosConfig:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)


from .gemini_api import GeminiApi
from .settings import Settings, default_settings
from .videogen_dto import VideoGenResponse, VideoResult
from .videogen_interface import VideoGenApi

_PRESERVE_SUBJECT = (
    "Preserve the person's identity and every visible garment exactly: unchanged "
    "color, pattern, fabric, fit, length, fastenings, accessories, and layering. "
    "Do not add, remove, restyle, or transform any clothing. "
)
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_ALLOWED_DURATIONS = {4, 6, 8}


class GeminiVideoGenApi(VideoGenApi):
    """Generate a short Veo video from an image URL and wait for completion."""

    def __init__(
        self,
        *,
        client: Optional[genai.Client] = None,
        poll_interval_seconds: float = 10.0,
        timeout_seconds: float = 360.0,
        image_timeout_seconds: float = 30.0,
        max_image_bytes: int = 20 * 1024 * 1024,
        settings: Optional[Settings] = None,
    ) -> None:
        self._client = client
        self._settings = settings or default_settings
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._image_timeout_seconds = image_timeout_seconds
        self._max_image_bytes = max_image_bytes

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = GeminiApi._build_default_client(self._settings)
        return self._client

    def _load_image(self, image_url: str) -> Any:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("image_url must use http or https")

        request = Request(image_url, headers={"User-Agent": "galet/0.1"})
        with urlopen(request, timeout=self._image_timeout_seconds) as response:
            mime_type = response.headers.get_content_type()
            if mime_type not in _ALLOWED_IMAGE_TYPES:
                raise ValueError(f"unsupported image content type: {mime_type}")
            data = response.read(self._max_image_bytes + 1)

        if len(data) > self._max_image_bytes:
            raise ValueError("input image exceeds maximum allowed size")
        if not data:
            raise ValueError("input image is empty")
        return types.Image(image_bytes=data, mime_type=mime_type)

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        image_url: str,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 6,
        resolution: str = "720p",
        preserve_subject: bool = True,
    ) -> VideoGenResponse:
        if duration_seconds not in _ALLOWED_DURATIONS:
            raise ValueError("duration_seconds must be one of 4, 6, or 8")
        if aspect_ratio not in {"9:16", "16:9"}:
            raise ValueError("aspect_ratio must be '9:16' or '16:9'")
        if resolution not in {"720p", "1080p", "4k"}:
            raise ValueError("resolution must be '720p', '1080p', or '4k'")
        if resolution in {"1080p", "4k"} and duration_seconds != 8:
            raise ValueError("1080p and 4k generation require an 8-second duration")

        image = self._load_image(image_url)
        effective_prompt = _PRESERVE_SUBJECT + prompt if preserve_subject else prompt
        config = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            resolution=resolution,
            person_generation="allow_adult",
        )

        client = self._get_client()
        logging.info(
            "GeminiVideoGenApi.generate_video: submitting model=%s aspect_ratio=%s duration=%s resolution=%s",
            model,
            aspect_ratio,
            duration_seconds,
            resolution,
        )
        operation = client.models.generate_videos(
            model=model,
            prompt=effective_prompt,
            image=image,
            config=config,
        )
        operation_name = getattr(operation, "name", None)
        deadline = time.monotonic() + self._timeout_seconds

        while not getattr(operation, "done", False):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"video generation did not complete within {self._timeout_seconds} seconds"
                )
            time.sleep(self._poll_interval_seconds)
            operation = client.operations.get(operation)

        error = getattr(operation, "error", None)
        if error:
            raise RuntimeError(f"video generation failed: {error}")

        response = getattr(operation, "response", None)
        generated_videos = getattr(response, "generated_videos", None) or []
        videos = []
        for generated in generated_videos:
            video = getattr(generated, "video", None)
            uri = getattr(video, "uri", None)
            mime_type = getattr(video, "mime_type", None) or "video/mp4"
            videos.append(
                VideoResult(
                    url=uri,
                    mime_type=mime_type,
                    duration_seconds=duration_seconds,
                    raw=video,
                )
            )

        if not videos:
            raise RuntimeError("video generation completed without a video result")

        return VideoGenResponse(
            videos=videos,
            model=model,
            operation_name=operation_name,
            raw=operation,
        )

    def download_video(self, video: VideoResult, destination: str) -> None:
        """Download a generated video while its provider file remains available."""

        if video.raw is None:
            raise ValueError("video result has no provider file to download")
        self._get_client().files.download(file=video.raw, destination=destination)
