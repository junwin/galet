from __future__ import annotations

from email.message import Message
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from galet.gemini_videogen import GeminiVideoGenApi
from galet.videogen_router import VideoGenRouter


class _ImageResponse:
    def __init__(self, data: bytes, mime_type: str = "image/jpeg") -> None:
        self._data = data
        self.headers = Message()
        self.headers["Content-Type"] = mime_type

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, _limit: int) -> bytes:
        return self._data


def _completed_operation():
    video = SimpleNamespace(uri="https://example.test/generated.mp4", mime_type="video/mp4")
    return SimpleNamespace(
        name="operations/123",
        done=True,
        error=None,
        response=SimpleNamespace(
            generated_videos=[SimpleNamespace(video=video)]
        ),
    )


def test_generate_six_second_portrait_video_and_download(tmp_path) -> None:
    initial = SimpleNamespace(name="operations/123", done=False)
    completed = _completed_operation()
    client = MagicMock()
    client.models.generate_videos.return_value = initial
    client.operations.get.return_value = completed

    api = GeminiVideoGenApi(
        client=client,
        poll_interval_seconds=0,
        timeout_seconds=10,
    )

    with (
        patch(
            "galet.gemini_videogen.urlopen",
            return_value=_ImageResponse(b"jpeg-data"),
        ),
        patch(
            "galet.gemini_videogen.types.Image",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch(
            "galet.gemini_videogen.types.GenerateVideosConfig",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch("galet.gemini_videogen.time.sleep"),
    ):
        response = api.generate_video(
            model="veo-3.1-fast-generate-preview",
            prompt="The model turns slowly to her left.",
            image_url="https://example.test/model.jpg",
        )

    call = client.models.generate_videos.call_args.kwargs
    assert call["model"] == "veo-3.1-fast-generate-preview"
    assert call["image"].image_bytes == b"jpeg-data"
    assert call["config"].aspect_ratio == "9:16"
    assert call["config"].duration_seconds == 6
    assert call["config"].resolution == "720p"
    assert call["config"].person_generation == "allow_adult"
    assert "every visible garment exactly" in call["prompt"]
    assert call["prompt"].endswith("The model turns slowly to her left.")

    assert response.operation_name == "operations/123"
    assert response.videos[0].url == "https://example.test/generated.mp4"
    assert response.videos[0].duration_seconds == 6

    client.files.download.return_value = b"video-data"
    destination = tmp_path / "result.mp4"
    api.download_video(response.videos[0], str(destination))
    client.files.download.assert_called_once_with(file=response.videos[0].raw)
    assert destination.read_bytes() == b"video-data"


@pytest.mark.parametrize("duration", [5, 7])
def test_rejects_unsupported_duration_before_fetch(duration: int) -> None:
    api = GeminiVideoGenApi(client=MagicMock())
    with patch("galet.gemini_videogen.urlopen") as fetch:
        with pytest.raises(ValueError, match="4, 6, or 8"):
            api.generate_video(
                model="veo-3.1-fast-generate-preview",
                prompt="Turn",
                image_url="https://example.test/model.jpg",
                duration_seconds=duration,
            )
    fetch.assert_not_called()


def test_subject_preservation_can_be_disabled() -> None:
    client = MagicMock()
    client.models.generate_videos.return_value = _completed_operation()
    api = GeminiVideoGenApi(client=client)

    with (
        patch(
            "galet.gemini_videogen.urlopen",
            return_value=_ImageResponse(b"jpeg-data"),
        ),
        patch(
            "galet.gemini_videogen.types.Image",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        patch(
            "galet.gemini_videogen.types.GenerateVideosConfig",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    ):
        api.generate_video(
            model="veo-3.1-fast-generate-preview",
            prompt="Turn",
            image_url="https://example.test/model.jpg",
            preserve_subject=False,
        )

    assert client.models.generate_videos.call_args.kwargs["prompt"] == "Turn"


def test_rejects_non_http_image_url() -> None:
    api = GeminiVideoGenApi(client=MagicMock())
    with pytest.raises(ValueError, match="http or https"):
        api.generate_video(
            model="veo-3.1-fast-generate-preview",
            prompt="Turn",
            image_url="file:///tmp/model.jpg",
        )


def test_router_dispatches_veo_model() -> None:
    provider = MagicMock()
    expected = _completed_operation()
    provider.generate_video.return_value = expected
    router = VideoGenRouter(gemini_api=provider)

    result = router.generate_video(
        model="veo-3.1-fast-generate-preview",
        prompt="Turn",
        image_url="https://example.test/model.jpg",
    )

    assert result is expected
    provider.generate_video.assert_called_once()


def test_router_rejects_unknown_model() -> None:
    router = VideoGenRouter(gemini_api=MagicMock())
    with pytest.raises(ValueError, match="no video generation provider"):
        router.generate_video(
            model="unknown-video-model",
            prompt="Turn",
            image_url="https://example.test/model.jpg",
        )
