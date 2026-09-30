from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class ImageResult:
    """A single generated image (url or b64_json)."""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    mime_type: str = "image/png"


@dataclass(frozen=True)
class ImageGenResponse:
    """Normalized response from an image generation API call."""

    images: List[ImageResult]
    model: Optional[str] = None
    raw: Optional[Any] = None


@dataclass(frozen=True)
class ImageInput:
    """Provider-neutral reference image; the caller resolves storage/URLs."""

    data: bytes
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("image data must be non-empty bytes")
        if self.mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("unsupported image MIME type")


@dataclass(frozen=True)
class ImageMessage:
    """Explicit image conversation turn. No implicit chat history is loaded."""

    role: str
    text: str = ""
    images: Tuple[ImageInput, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("image message role must be user or assistant")
        if not self.text.strip() and not self.images:
            raise ValueError("image message must contain text or images")
