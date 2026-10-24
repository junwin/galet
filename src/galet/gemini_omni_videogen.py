from __future__ import annotations

import base64
import binascii
import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

from .gemini_videogen import GeminiVideoGenApi, _PRESERVE_SUBJECT
from .videogen_dto import VideoGenResponse, VideoResult


class GeminiOmniVideoGenApi(GeminiVideoGenApi):
    """Generate text/image-to-video with Google's Interactions API.

    Reuses Gemini credential resolution, bounded image loading and downloads.
    URI delivery avoids large inline responses; inline SDK results are also
    accepted. Editing and extension are outside this generation interface.
    """

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 6,
        resolution: str = "720p",
        preserve_subject: bool = True,
    ) -> VideoGenResponse:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if type(duration_seconds) is not int or not 3 <= duration_seconds <= 10:
            raise ValueError("duration_seconds must be an integer between 3 and 10")
        if aspect_ratio not in {"9:16", "16:9"}:
            raise ValueError("aspect_ratio must be '9:16' or '16:9'")
        if resolution not in {"360p", "720p", "1080p", "4k"}:
            raise ValueError("resolution must be '360p', '720p', '1080p', or '4k'")
        if image_url and image_path:
            raise ValueError("provide at most one of image_url or image_path")

        input_parts: list[dict[str, Any]] = []
        if image_url or image_path:
            image = self._load_image(image_url=image_url, image_path=image_path)
            input_parts.append({
                "type": "image",
                "data": base64.b64encode(image.image_bytes).decode("ascii"),
                "mime_type": image.mime_type,
            })
            if preserve_subject:
                prompt = _PRESERVE_SUBJECT + prompt
        input_parts.append({"type": "text", "text": prompt})

        client = self._get_client()
        create = getattr(getattr(client, "interactions", None), "create", None)
        if not callable(create):
            raise RuntimeError(
                "Omni requires a google-genai SDK with Interactions support; "
                'upgrade with pip install --upgrade "galet[gemini]"'
            )
        interaction = create(
            model=model,
            input=input_parts,
            response_format={
                "type": "video",
                "aspect_ratio": aspect_ratio,
                "duration": f"{duration_seconds}s",
                "resolution": resolution,
                "delivery": "uri",
            },
            background=False,
            # Google requires stored interactions for URI-delivered videos.
            store=True,
            stream=False,
            timeout=self._timeout_seconds,
        )
        status = self._field(interaction, "status")
        if status and status != "completed":
            raise RuntimeError(f"Omni video generation did not complete (status={status})")

        output = self._video_output(interaction)
        if output is None:
            raise RuntimeError("Omni generation completed without a video result")
        uri = self._field(output, "uri")
        data = self._field(output, "data")
        video_bytes = None
        if data:
            try:
                video_bytes = base64.b64decode(data, validate=True)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise RuntimeError("Omni returned invalid base64 video data") from exc
        elif uri:
            self._wait_for_file(uri)
        else:
            raise RuntimeError("Omni video result contains neither data nor a URI")

        return VideoGenResponse(
            videos=[VideoResult(
                url=uri,
                mime_type=self._field(output, "mime_type") or "video/mp4",
                duration_seconds=duration_seconds,
                raw=uri if uri else output,
                video_bytes=video_bytes,
            )],
            model=model,
            operation_name=self._field(interaction, "id"),
            raw=interaction,
        )

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    @classmethod
    def _video_output(cls, interaction: Any) -> Any:
        output = cls._field(interaction, "output_video")
        if output is not None:
            return output
        # output_video is an SDK convenience property; REST-shaped responses
        # expose video content in model_output steps instead.
        for step in cls._field(interaction, "steps") or []:
            if cls._field(step, "type") == "model_output":
                for content in cls._field(step, "content") or []:
                    if cls._field(content, "type") == "video":
                        return content
        return None

    def _wait_for_file(self, uri: str) -> None:
        # Google may return files/ID or a complete .../files/ID:download URL.
        match = re.search(r"(?:^|/)files/([A-Za-z0-9_-]+)(?=[:/]|$)", urlparse(uri).path)
        if match is None:
            raise RuntimeError("Omni returned an unrecognized video file URI")
        name = f"files/{match.group(1)}"
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            remaining = max(0.001, deadline - time.monotonic())
            info = self._get_client().files.get(
                name=name,
                config={"http_options": {"timeout": max(1, int(remaining * 1000))}},
            )
            state = self._field(info, "state")
            state = getattr(state, "name", state)
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise RuntimeError("Omni video file processing failed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Omni video file processing timed out")
            time.sleep(min(self._poll_interval_seconds, remaining))
