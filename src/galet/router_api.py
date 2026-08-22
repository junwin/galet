from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .dto import LLMResponse
from .interface import LLMApi
from .openai_responses import OpenAIResponsesApi
from .deepseek_responses import DeepSeekApi
from .gemini_api import GeminiApi
from .mistral_api import MistralApi
from .ollama_api import OllamaApi
from .provider_registry import ProviderRegistry


class RouterApi(LLMApi):
    """Routes LLM requests to the correct backend based on the model name.

    This implementation delegates provider resolution to ProviderRegistry and
    lazily caches provider API instances keyed by provider name.
    """

    def __init__(
        self,
        *,
        openai_api: Optional[OpenAIResponsesApi] = None,
        deepseek_api: Optional[DeepSeekApi] = None,
        mistral_api: Optional[MistralApi] = None,
        ollama_api: Optional[OllamaApi] = None,
        gemini_api: Optional[GeminiApi] = None,
        registry: ProviderRegistry = ProviderRegistry(),
    ) -> None:
        self._registry = registry
        # cache of provider_name -> LLMApi instance
        self._instances: Dict[str, LLMApi] = {}

        # If explicit instances were provided, seed the cache.
        if openai_api is not None:
            self._instances["openai"] = openai_api
        if deepseek_api is not None:
            self._instances["deepseek"] = deepseek_api
        if mistral_api is not None:
            self._instances["mistral"] = mistral_api
        if ollama_api is not None:
            self._instances["ollama"] = ollama_api
        if gemini_api is not None:
            self._instances["gemini"] = gemini_api

    def _get_provider_and_api(self, model: Optional[str], provider: Optional[str]) -> Tuple[str, LLMApi]:
        # Resolve provider name (may raise ValueError for unknown explicit provider)
        provider_name = self._registry.resolve_name(model, provider)

        if provider_name in self._instances:
            return provider_name, self._instances[provider_name]

        # Ask registry to resolve and instantiate the provider implementation.
        resolved_name, api = self._registry.resolve(model, provider)
        # Cache and return
        self._instances[resolved_name] = api
        return resolved_name, api

    def supports_image_processing(self, model: str, provider: Optional[str] = None) -> bool:
        """Check whether the selected model supports native image processing.

        Delegates provider resolution to ProviderRegistry and then calls the
        provider-specific implementation.
        """
        _, api = self._get_provider_and_api(model, provider)
        return api.supports_image_processing(model)

    def create_response(
        self,
        *,
        model: str,
        input: Any,
        temperature: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        store: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        previous_response_id: Optional[str] = None,
        text: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
    ) -> LLMResponse:
        provider_name, api = self._get_provider_and_api(model, provider)
        return api.create_response(
            model=model,
            input=input,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            previous_response_id=previous_response_id,
            text=text,
        )
