"""Provider registry for LLM backends.

Connectors register themselves through `provider_info` when their module is
imported. The registry imports every connector module shipped with galet to
trigger registration, then resolves requests against that table.

Resolution order:
1. If explicit `provider` is provided, use it (must be known or ValueError).
2. Try matching model-name prefixes from the registered connectors.
3. Fall back to the default provider.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Tuple

from .dto import LLMResponse
from .interface import LLMApi
from .provider_info import get_provider, registered_providers
from .settings import Settings, default_settings


CONNECTOR_MODULES: Tuple[str, ...] = (
    "galet.openai_responses",
    "galet.deepseek_responses",
    "galet.gemini_api",
    "galet.mistral_api",
    "galet.ollama_api",
)

DEFAULT_PROVIDER_NAME = "openai"


class _DummyApi:
    """Lightweight fallback implementing the minimal LLMApi protocol."""

    def __init__(self, provider_name: str) -> None:
        self._provider_name = provider_name

    def supports_image_processing(self, model: str) -> bool:
        return False

    def create_response(self, **kwargs: Any) -> LLMResponse:
        raise NotImplementedError(f"DummyApi for provider={self._provider_name} does not implement create_response")


class ProviderRegistry:
    """Resolve a provider name and return an instance of its LLMApi implementation.

    Resolution order:
    1. If explicit `provider` is provided, use it (must be known or ValueError).
    2. Try matching model name prefixes from the registered connectors.
    3. Fall back to DEFAULT_PROVIDER_NAME.
    """

    _loaded = False

    @classmethod
    def load_all(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        for module_path in CONNECTOR_MODULES:
            try:
                importlib.import_module(module_path)
            except Exception as e:
                logging.debug("ProviderRegistry: failed to load %s: %s", module_path, e)

    @classmethod
    def providers(cls) -> Dict[str, str]:
        cls.load_all()
        return {info.name: info.class_path for info in registered_providers()}

    @classmethod
    def prefix_map(cls) -> Dict[str, str]:
        cls.load_all()
        mapping: Dict[str, str] = {}
        for info in registered_providers():
            for prefix in info.prefixes:
                if prefix not in mapping:
                    mapping[prefix] = info.name
        return mapping

    @classmethod
    def _load_provider_class(cls, provider_name: str):
        cls.load_all()
        info = get_provider(provider_name)
        if info is None:
            return None

        module_name, _, attr = info.class_path.rpartition(".")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception as e:
            logging.debug("ProviderRegistry: failed to import %s -> %s: %s", provider_name, info.class_path, e)
            return None

    @classmethod
    def resolve_name(cls, model: Optional[str], provider: Optional[str] = None) -> str:
        if provider:
            if provider not in cls.providers():
                raise ValueError(f"unknown provider: {provider}")
            return provider

        model = (model or "").lower()
        for prefix, pname in cls.prefix_map().items():
            if model.startswith(prefix):
                return pname

        return DEFAULT_PROVIDER_NAME

    @classmethod
    def resolve(
        cls,
        model: Optional[str],
        provider: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> Tuple[str, LLMApi]:
        provider_name = cls.resolve_name(model, provider)

        impl_class = cls._load_provider_class(provider_name)
        if impl_class is None:
            logging.debug("ProviderRegistry: using DummyApi for provider=%s", provider_name)
            return provider_name, _DummyApi(provider_name)

        try:
            instance = impl_class(settings=settings or default_settings)
        except Exception as e:
            logging.debug("ProviderRegistry: failed to instantiate %s: %s", provider_name, e)
            return provider_name, _DummyApi(provider_name)

        return provider_name, instance
