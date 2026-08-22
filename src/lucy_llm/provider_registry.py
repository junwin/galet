"""Provider registry for LLM backends.

Provides a simple resolution strategy:
- explicit provider name takes priority (must be one of the known providers)
- otherwise, try a prefix map on the model string
- otherwise, default to 'openai'

The registry attempts to lazily import provider classes so tests can run in
environments where optional SDKs (eg. `openai`) are not installed.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Tuple

from .interface import LLMApi
from .dto import LLMResponse


# Map of canonical provider name -> import path for the LLMApi class.
PROVIDERS: Dict[str, str] = {
    "openai": "src.llm.openai_responses.OpenAIResponsesApi",
    "deepseek": "src.llm.deepseek_responses.DeepSeekApi",
    "gemini": "src.llm.gemini_api.GeminiApi",
    "mistral": "src.llm.mistral_api.MistralApi",
    "ollama": "src.llm.ollama_api.OllamaApi",
}

# Prefix map: model-name prefix -> provider name (checked in order of the keys)
PREFIX_MAP: Dict[str, str] = {
    # explicit provider-style prefixes
    "deepseek": "deepseek",
    "gemini": "gemini",
    "mistral": "mistral",
    "ollama": "ollama",
    # common openai-style prefixes
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
}


class _DummyApi:
    """Lightweight fallback implementing the minimal LLMApi protocol."""

    def __init__(self, provider_name: str) -> None:
        self._provider_name = provider_name

    def supports_image_processing(self, model: str) -> bool:  # type: ignore
        return False

    def create_response(self, **kwargs: Any) -> LLMResponse:  # type: ignore
        raise NotImplementedError(f"DummyApi for provider={self._provider_name} does not implement create_response")


class ProviderRegistry:
    """Resolve a provider name and return an instance of its LLMApi implementation.

    Resolution order:
    1. If explicit `provider` is provided, use it (must be known or ValueError).
    2. Try matching model name prefixes (PREFIX_MAP).
    3. Fall back to 'openai'.
    """

    providers = PROVIDERS
    prefix_map = PREFIX_MAP

    @classmethod
    def _load_provider_class(cls, provider_name: str):
        path = cls.providers.get(provider_name)
        if not path:
            return None

        module_name, _, attr = path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception as e:
            logging.debug("ProviderRegistry: failed to import %s -> %s: %s", provider_name, path, e)
            return None

    @classmethod
    def resolve_name(cls, model: Optional[str], provider: Optional[str] = None) -> str:
        """Return the resolved provider name string.

        Raises ValueError if an explicit provider is supplied but is unknown.
        """
        if provider:
            if provider not in cls.providers:
                raise ValueError(f"unknown provider: {provider}")
            return provider

        model = (model or "").lower()
        for prefix, pname in cls.prefix_map.items():
            if model.startswith(prefix):
                return pname

        # default
        return "openai"

    @classmethod
    def resolve(cls, model: Optional[str], provider: Optional[str] = None) -> Tuple[str, LLMApi]:
        """Resolve to (provider_name, api_instance).

        If the provider implementation cannot be imported/instantiated, a
        lightweight DummyApi instance is returned as a fallback so callers can
        still rely on resolution without requiring optional SDKs at test time.
        """
        provider_name = cls.resolve_name(model, provider)

        impl_class = cls._load_provider_class(provider_name)
        if impl_class is None:
            # Return a dummy implementation rather than failing import-time.
            logging.debug("ProviderRegistry: using DummyApi for provider=%s", provider_name)
            return provider_name, _DummyApi(provider_name)

        try:
            instance = impl_class()
        except Exception as e:  # pragmatic: don't let provider instantiation break tests
            logging.debug("ProviderRegistry: failed to instantiate %s: %s", provider_name, e)
            return provider_name, _DummyApi(provider_name)

        return provider_name, instance
