from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from galet.gemini_omni_videogen import GeminiOmniVideoGenApi
from galet.videogen_router import VideoGenRouter

MODEL = "gemini-omni-1.1-flash"
URI = "https://generativelanguage.googleapis.com/v1beta/files/clip-123:download?alt=media"


def _client(output=None, status="completed"):
    client = MagicMock()
    client.interactions.create.return_value = SimpleNamespace(
        id="interaction-123", status=status, output_video=output
    )
    client.files.get.return_value = SimpleNamespace(state=SimpleNamespace(name="ACTIVE"))
    return client


def test_image_generation_uri_poll_and_download(tmp_path):
    image = tmp_path / "model.png"
    image.write_bytes(b"png-data")
    client = _client(SimpleNamespace(uri=URI, data=None, mime_type="video/mp4"))
    client.files.get.side_effect = [
        SimpleNamespace(state="PROCESSING"), SimpleNamespace(state="ACTIVE")
    ]
    api = GeminiOmniVideoGenApi(client=client, poll_interval_seconds=0)
    response = api.generate_video(model=MODEL, prompt="Turn slowly", image_path=str(image))
    call = client.interactions.create.call_args.kwargs
    assert call["input"][0] == {
        "type": "image", "data": base64.b64encode(b"png-data").decode(), "mime_type": "image/png"
    }
    assert "every visible garment exactly" in call["input"][1]["text"]
    assert call["response_format"] == {
        "type": "video", "delivery": "uri", "aspect_ratio": "9:16",
        "duration": "6s", "resolution": "720p",
    }
    assert call["model"] == MODEL
    assert call["background"] is False and call["stream"] is False and call["store"] is True
    assert call["timeout"] == 360
    client.models.generate_videos.assert_not_called()
    assert client.files.get.call_args.kwargs["name"] == "files/clip-123"
    assert client.files.get.call_count == 2
    assert response.operation_name == "interaction-123"
    assert response.videos[0].url == URI
    client.files.download.return_value = b"mp4-data"
    target = tmp_path / "clip.mp4"
    api.download_video(response.videos[0], str(target))
    assert target.read_bytes() == b"mp4-data"
    client.files.download.assert_called_once_with(file=URI)


def test_prompt_only_inline_response_download(tmp_path):
    client = _client(SimpleNamespace(data=base64.b64encode(b"mp4").decode()))
    api = GeminiOmniVideoGenApi(client=client)
    response = api.generate_video(model=MODEL, prompt="Snow falls", duration_seconds=5, resolution="1080p")
    assert client.interactions.create.call_args.kwargs["input"] == [{"type": "text", "text": "Snow falls"}]
    assert response.videos[0].video_bytes == b"mp4"
    assert response.videos[0].url is None
    api.download_video(response.videos[0], str(tmp_path / "snow.mp4"))
    assert (tmp_path / "snow.mp4").read_bytes() == b"mp4"
    client.files.get.assert_not_called()
    client.files.download.assert_not_called()


def test_rest_shaped_output():
    client = _client()
    client.interactions.create.return_value = {
        "id": "rest-1", "status": "completed", "steps": [
            {"type": "user_input", "content": [{"type": "video", "data": "ignore"}]},
            {"type": "model_output", "content": [{"type": "video", "data": "bXA0"}]},
        ]
    }
    response = GeminiOmniVideoGenApi(client=client).generate_video(model=MODEL, prompt="Turn")
    assert response.videos[0].video_bytes == b"mp4"


@pytest.mark.parametrize("kwargs", [
    {"duration_seconds": 2}, {"duration_seconds": 11}, {"duration_seconds": 6.5},
    {"duration_seconds": True}, {"aspect_ratio": "1:1"}, {"resolution": "8k"},
    {"image_url": "https://example.com/a.jpg", "image_path": "a.jpg"}, {"prompt": " "},
])
def test_invalid_options_fail_before_provider_call(kwargs):
    client = _client()
    options = {"model": MODEL, "prompt": "Turn", **kwargs}
    with pytest.raises(ValueError):
        GeminiOmniVideoGenApi(client=client).generate_video(**options)
    client.interactions.create.assert_not_called()


@pytest.mark.parametrize("duration", [3, 10])
def test_duration_boundaries(duration):
    client = _client(SimpleNamespace(data="bXA0"))
    GeminiOmniVideoGenApi(client=client).generate_video(model=MODEL, prompt="Turn", duration_seconds=duration)
    assert client.interactions.create.call_args.kwargs["response_format"]["duration"] == f"{duration}s"


@pytest.mark.parametrize("output, status, error", [
    (None, "completed", "without a video"),
    (None, "failed", "status=failed"),
    (SimpleNamespace(), "completed", "neither data nor a URI"),
    (SimpleNamespace(data="bad-base64"), "completed", "invalid base64"),
])
def test_provider_failures(output, status, error):
    with pytest.raises(RuntimeError, match=error):
        GeminiOmniVideoGenApi(client=_client(output, status)).generate_video(model=MODEL, prompt="Turn")


def test_file_processing_failure_and_timeout():
    client = _client(SimpleNamespace(uri=URI))
    api = GeminiOmniVideoGenApi(client=client, timeout_seconds=0, poll_interval_seconds=0)
    client.files.get.return_value = SimpleNamespace(state="FAILED")
    with pytest.raises(RuntimeError, match="processing failed"):
        api.generate_video(model=MODEL, prompt="Turn")
    client.files.get.return_value = SimpleNamespace(state="PROCESSING")
    with pytest.raises(TimeoutError, match="timed out"):
        api.generate_video(model=MODEL, prompt="Turn")


def test_google_short_uri_and_sdk_destination_download(tmp_path):
    client = _client(SimpleNamespace(uri="files/abc"))
    api = GeminiOmniVideoGenApi(client=client)
    result = api.generate_video(model=MODEL, prompt="Turn").videos[0]
    assert client.files.get.call_count == 1
    assert client.files.get.call_args.kwargs["name"] == "files/abc"
    def download(*, file, destination):
        assert file == "files/abc"
        from pathlib import Path
        Path(destination).write_bytes(b"sdk-download")
    client.files.download = download
    api.download_video(result, str(tmp_path / "clip.mp4"))
    assert (tmp_path / "clip.mp4").read_bytes() == b"sdk-download"


def test_image_url_reuses_loader_and_preservation_can_be_disabled():
    client = _client(SimpleNamespace(data="bXA0"))
    api = GeminiOmniVideoGenApi(client=client)
    with patch.object(api, "_load_image", return_value=SimpleNamespace(image_bytes=b"image", mime_type="image/jpeg")) as load:
        api.generate_video(model=MODEL, prompt="Turn", image_url="https://example.com/image.jpg", preserve_subject=False)
    load.assert_called_once_with(image_url="https://example.com/image.jpg", image_path=None)
    assert client.interactions.create.call_args.kwargs["input"][1]["text"] == "Turn"


def test_router_dispatches_omni_without_changing_veo():
    omni, veo = MagicMock(), MagicMock()
    router = VideoGenRouter(omni_api=omni, gemini_api=veo)
    assert router.generate_video(model=MODEL, prompt="Turn") is omni.generate_video.return_value
    veo.generate_video.assert_not_called()
    router.generate_video(model="veo-legacy", prompt="Turn", image_path="model.jpg")
    veo.generate_video.assert_called_once()


def test_old_sdk_has_actionable_error():
    api = GeminiOmniVideoGenApi(client=SimpleNamespace())
    with pytest.raises(RuntimeError, match="upgrade"):
        api.generate_video(model=MODEL, prompt="Turn")


def test_installed_sdk_serialization_and_response_parsing(tmp_path):
    """Exercise the real SDK over a mock transport, with no paid API calls."""
    import json
    httpx = pytest.importorskip("httpx")
    genai = pytest.importorskip("google.genai")
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["model"] == MODEL
            assert body["response_format"]["duration"] == "6s"
            assert body["response_format"]["delivery"] == "uri"
            # Reproduce the provider constraint that caused the live HTTP 400.
            if body.get("store") is not True:
                return httpx.Response(400, json={"error": {
                    "message": "store=true is required when response format has video delivery set to URI.",
                    "code": "invalid_request",
                }})
            return httpx.Response(200, json={
                "id": "sdk-check", "object": "interaction", "model": MODEL,
                "status": "completed", "steps": [{"type": "model_output", "content": [
                    {"type": "video", "mime_type": "video/mp4", "uri": URI}
                ]}],
            })
        if ":download" in request.url.path:
            return httpx.Response(200, content=b"sdk-mp4")
        return httpx.Response(200, json={"name": "files/clip-123", "uri": URI, "state": "ACTIVE"})

    from google.genai import types
    with genai.Client(api_key="test", http_options=types.HttpOptions(
        client_args={"trust_env": False, "transport": httpx.MockTransport(handle)},
        async_client_args={"trust_env": False},
    )) as client:
        api = GeminiOmniVideoGenApi(client=client)
        response = api.generate_video(model=MODEL, prompt="Snow falls")
        api.download_video(response.videos[0], str(tmp_path / "sdk.mp4"))
    assert (tmp_path / "sdk.mp4").read_bytes() == b"sdk-mp4"
    assert sum(r.method == "POST" for r in requests) == 1
