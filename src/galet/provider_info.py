"""Self-registration table for LLM connector providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    display_name: str
    description: str
    prefixes: Tuple[str, ...]
    class_path: str
    default_model: str


_REGISTRY: Dict[str, ProviderInfo] = {}


def register_provider(info: ProviderInfo) -> None:
    _REGISTRY[info.name] = info


def get_provider(name: str) -> Optional[ProviderInfo]:
    return _REGISTRY.get(name)


def registered_providers() -> Tuple[ProviderInfo, ...]:
    return tuple(sorted(_REGISTRY.values(), key=lambda info: info.name))
