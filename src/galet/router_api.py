from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .dto import LLMResponse
from .interface import LLMApi
from .provider_registry import ProviderRegistry
from .settings import Settings, default_settings


class RouterApi(LLMApi):
    def __init__(
        self,
        *,
        instances: Optional[Mapping[str, LLMApi]] = None,
        registry: ProviderRegistry = ProviderRegistry(),
        settings: Optional[Settings] = None,
    ) -> None:
        self._registry = registry
        self._settings = settings or default_settings
        self._instances: Dict[str, LLMApi] = dict(instances or {})

    def _get_provider_and_api(self, model: Optional[str], provider: Optional[str]) -> Tuple[str, LLMApi]:
        provider_name = self._registry.resolve_name(model, provider)

        if provider_name in self._instances:
            return provider_name, self._instances[provider_name]

        resolved_name, api = self._registry.resolve(model, provider, settings=self._settings)
        self._instances[resolved_name] = api
        return resolved_name, api

    def supports_image_processing(self, model: str, provider: Optional[str] = None) -> bool:
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
