"""Public Galet API with lazy provider implementation imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .dto import LLMResponse, LLMUsage, ToolCall
from .interface import LLMApi
from .adapter_interface import LLMAdapter
from .provider_info import ProviderInfo, get_provider, register_provider, registered_providers
from .settings import CredentialProfile, Settings, default_settings
from .model_catalog import (
    ModelCatalog,
    ModelInfo,
    ModelRequirements,
    ResolvedModel,
    default_model_catalog,
)


_LAZY_EXPORTS = {
    "OpenAIResponsesApi": ("galet.openai_responses", "OpenAIResponsesApi"),
    "OpenAIResponsesAdapter": ("galet.openai_responses_adapter", "OpenAIResponsesAdapter"),
    "MistralApi": ("galet.mistral_api", "MistralApi"),
    "MistralResponsesAdapter": ("galet.mistral_responses_adapter", "MistralResponsesAdapter"),
    "OllamaApi": ("galet.ollama_api", "OllamaApi"),
    "GeminiApi": ("galet.gemini_api", "GeminiApi"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    try:
        value = getattr(import_module(module_name), attribute_name)
    except Exception:
        # Preserve the previous optional-SDK behaviour of exposing None when a
        # provider implementation cannot be imported.
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "LLMApi",
    "LLMAdapter",
    "LLMResponse",
    "LLMUsage",
    "ToolCall",
    "ProviderInfo",
    "get_provider",
    "register_provider",
    "registered_providers",
    "CredentialProfile",
    "Settings",
    "default_settings",
    "ModelCatalog",
    "ModelInfo",
    "ModelRequirements",
    "ResolvedModel",
    "default_model_catalog",
    "OpenAIResponsesApi",
    "OpenAIResponsesAdapter",
    "MistralApi",
    "MistralResponsesAdapter",
    "OllamaApi",
    "GeminiApi",
]
