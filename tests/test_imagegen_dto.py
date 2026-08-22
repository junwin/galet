from __future__ import annotations

from src.llm.imagegen_dto import ImageGenResponse, ImageResult


class TestImageResult:
    """Tests for the ImageResult DTO."""

    def test_instantiation_defaults(self) -> None:
        """All fields default to None."""
        r = ImageResult()
        assert r.url is None
        assert r.b64_json is None
        assert r.revised_prompt is None

    def test_instantiation_with_url(self) -> None:
        """Create with just a URL."""
        r = ImageResult(url="https://example.com/img.png")
        assert r.url == "https://example.com/img.png"
        assert r.b64_json is None
        assert r.revised_prompt is None

    def test_instantiation_with_b64(self) -> None:
        """Create with just a b64_json string."""
        r = ImageResult(b64_json="aGVsbG8=")
        assert r.url is None
        assert r.b64_json == "aGVsbG8="
        assert r.revised_prompt is None

    def test_instantiation_full(self) -> None:
        """All fields populated."""
        r = ImageResult(
            url="https://example.com/img.png",
            b64_json="aGVsbG8=",
            revised_prompt="a cat",
        )
        assert r.url == "https://example.com/img.png"
        assert r.b64_json == "aGVsbG8="
        assert r.revised_prompt == "a cat"

    def test_frozen_immutability(self) -> None:
        """ImageResult is frozen."""
        r = ImageResult(url="x")
        try:
            r.url = "y"  # type: ignore[misc]
        except Exception as e:
            assert "frozen" in str(e).lower() or isinstance(e, (AttributeError, TypeError))


class TestImageGenResponse:
    """Tests for the ImageGenResponse DTO."""

    def test_instantiation_minimal(self) -> None:
        """Only required field: images."""
        resp = ImageGenResponse(images=[])
        assert resp.images == []
        assert resp.model is None
        assert resp.raw is None

    def test_instantiation_with_images(self) -> None:
        """Response wrapping multiple ImageResults."""
        imgs = [
            ImageResult(url="u1"),
            ImageResult(b64_json="b1"),
        ]
        resp = ImageGenResponse(images=imgs, model="dall-e-3")
        assert len(resp.images) == 2
        assert resp.images[0].url == "u1"
        assert resp.images[1].b64_json == "b1"
        assert resp.model == "dall-e-3"
        assert resp.raw is None

    def test_field_defaults(self) -> None:
        """Optional fields default to None."""
        resp = ImageGenResponse(images=[ImageResult()])
        assert resp.model is None
        assert resp.raw is None

    def test_frozen_immutability(self) -> None:
        """ImageGenResponse is frozen."""
        resp = ImageGenResponse(images=[])
        try:
            resp.images = []  # type: ignore[misc]
        except Exception as e:
            assert "frozen" in str(e).lower() or isinstance(e, (AttributeError, TypeError))
