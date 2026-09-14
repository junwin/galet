"""Declarative metadata for LLM connector sources.

Source information is available without importing connector implementations.
Provider classes are imported lazily by ProviderRegistry only when selected.
"""

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


BUILTIN_PROVIDERS: Tuple[ProviderInfo, ...] = (
    ProviderInfo(
        name="openai",
        display_name="OpenAI",
        description="OpenAI Responses API for GPT and o-series models.",
        prefixes=("gpt", "o1", "o3"),
        class_path="galet.openai_responses.OpenAIResponsesApi",
        default_model="gpt-4o-mini",
    ),
    ProviderInfo(
        name="deepseek",
        display_name="DeepSeek",
        description="DeepSeek API over an OpenAI-compatible endpoint.",
        prefixes=("deepseek",),
        class_path="galet.deepseek_responses.DeepSeekApi",
        default_model="deepseek-chat",
    ),
    ProviderInfo(
        name="gemini",
        display_name="Google Gemini",
        description="Google Gemini API for text and multimodal models.",
        prefixes=("gemini",),
        class_path="galet.gemini_api.GeminiApi",
        default_model="gemini-2.0-flash",
    ),
    ProviderInfo(
        name="mistral",
        display_name="Mistral",
        description="Mistral API over an OpenAI-compatible endpoint.",
        prefixes=("mistral",),
        class_path="galet.mistral_api.MistralApi",
        default_model="mistral-small-latest",
    ),
    ProviderInfo(
        name="ollama",
        display_name="Ollama",
        description="Local Ollama server over its OpenAI-compatible endpoint.",
        prefixes=("ollama",),
        class_path="galet.ollama_api.OllamaApi",
        default_model="llama3.1",
    ),
)


_REGISTRY: Dict[str, ProviderInfo] = {
    info.name: info for info in BUILTIN_PROVIDERS
}


def register_provider(info: ProviderInfo) -> None:
    """Register or replace source metadata for an extension provider."""

    _REGISTRY[info.name] = info


def get_provider(name: str) -> Optional[ProviderInfo]:
    return _REGISTRY.get(name)


def registered_providers() -> Tuple[ProviderInfo, ...]:
    return tuple(sorted(_REGISTRY.values(), key=lambda info: info.name))
