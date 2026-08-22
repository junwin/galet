from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class ImageResult:
    """A single generated image (url or b64_json)."""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


@dataclass(frozen=True)
class ImageGenResponse:
    """Normalized response from an image generation API call."""

    images: List[ImageResult]
    model: Optional[str] = None
    raw: Optional[Any] = None
