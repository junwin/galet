"""Model metadata and capability-based selection.

ProviderRegistry answers "which source handles this model name?".  ModelCatalog
answers "which known model can satisfy these requirements?".  It contains no
credentials and performs no provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from .provider_info import ProviderInfo, registered_providers
from .provider_registry import ProviderRegistry


@dataclass(frozen=True)
class ModelInfo:
    """Description of a model offered by a Galet source."""

    name: str
    source: str
    profiles: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    cost_tier: Optional[str] = None
    supports_temperature: Optional[bool] = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        source: str,
        profiles: Iterable[str] = (),
        capabilities: Iterable[str] = (),
        cost_tier: Optional[str] = None,
        supports_temperature: Optional[bool] = None,
    ) -> "ModelInfo":
        return cls(
            name=name,
            source=source,
            profiles=frozenset(profiles),
            capabilities=frozenset(capabilities),
            cost_tier=cost_tier,
            supports_temperature=supports_temperature,
        )


@dataclass(frozen=True)
class ModelRequirements:
    """Capabilities requested by a caller such as an agent runtime."""

    profile: Optional[str] = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    preferred_model: Optional[str] = None
    preferred_source: Optional[str] = None
    allow_fallback: bool = True

    @classmethod
    def create(
        cls,
        *,
        profile: Optional[str] = None,
        required_capabilities: Iterable[str] = (),
        preferred_model: Optional[str] = None,
        preferred_source: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> "ModelRequirements":
        return cls(
            profile=profile,
            required_capabilities=frozenset(required_capabilities),
            preferred_model=preferred_model,
            preferred_source=preferred_source,
            allow_fallback=allow_fallback,
        )


@dataclass(frozen=True)
class ResolvedModel:
    """The source/model pair selected for a set of requirements."""

    source: str
    model: str
    info: ModelInfo


class ModelCatalog:
    """An in-memory model catalog with deterministic capability resolution."""

    def __init__(self, models: Iterable[ModelInfo] = ()) -> None:
        self._models: dict[tuple[str, str], ModelInfo] = {}
        for model in models:
            self.register(model)

    def register(self, model: ModelInfo) -> None:
        if not model.name or not model.source:
            raise ValueError("model name and source are required")
        self._models[(model.source, model.name)] = model

    def sources(self) -> Tuple[ProviderInfo, ...]:
        """Return provider/source metadata already registered with Galet."""

        return registered_providers()

    def models(self, source: Optional[str] = None) -> Tuple[ModelInfo, ...]:
        values = self._models.values()
        if source is not None:
            values = (model for model in values if model.source == source)
        return tuple(sorted(values, key=lambda model: (model.source, model.name)))

    def find(self, name: str, source: Optional[str] = None) -> Optional[ModelInfo]:
        if source is not None:
            return self._models.get((source, name))

        matches = [model for model in self._models.values() if model.name == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            return None

        inferred_source = ProviderRegistry.resolve_name(name)
        return self._models.get((inferred_source, name))

    @staticmethod
    def _satisfies(model: ModelInfo, requirements: ModelRequirements) -> bool:
        if requirements.profile and requirements.profile not in model.profiles:
            return False
        return requirements.required_capabilities.issubset(model.capabilities)

    def resolve(self, requirements: ModelRequirements) -> ResolvedModel:
        eligible = [
            model
            for model in self._models.values()
            if self._satisfies(model, requirements)
        ]

        if requirements.preferred_model:
            preferred = [
                model
                for model in eligible
                if model.name == requirements.preferred_model
                and (
                    requirements.preferred_source is None
                    or model.source == requirements.preferred_source
                )
            ]
            if preferred:
                selected = sorted(preferred, key=lambda model: (model.source, model.name))[0]
                return ResolvedModel(selected.source, selected.name, selected)
            if not requirements.allow_fallback:
                raise LookupError(
                    f"preferred model does not satisfy requirements: "
                    f"{requirements.preferred_model}"
                )

        if requirements.preferred_source:
            source_matches = [
                model for model in eligible
                if model.source == requirements.preferred_source
            ]
            if source_matches:
                eligible = source_matches
            elif not requirements.allow_fallback:
                raise LookupError(
                    f"preferred source does not satisfy requirements: "
                    f"{requirements.preferred_source}"
                )

        if not eligible:
            raise LookupError("no model satisfies the requested capabilities")

        selected = sorted(eligible, key=lambda model: (model.source, model.name))[0]
        return ResolvedModel(selected.source, selected.name, selected)


DEFAULT_MODELS: Tuple[ModelInfo, ...] = (
    ModelInfo.create(
        name="gpt-5.6-sol",
        source="openai",
        profiles=("reasoning", "development"),
        capabilities=("tool-calling", "structured-output", "long-context", "code-generation"),
        cost_tier="standard",
        supports_temperature=False,
    ),
    ModelInfo.create(
        name="gpt-5.6-luna",
        source="openai",
        profiles=("focused-development", "development"),
        capabilities=("tool-calling", "structured-output", "code-generation"),
        cost_tier="low",
        supports_temperature=False,
    ),
    ModelInfo.create(
        name="gpt-5-mini",
        source="openai",
        profiles=("focused-development",),
        capabilities=("tool-calling", "structured-output", "code-generation"),
        cost_tier="low",
        supports_temperature=False,
    ),
    ModelInfo.create(
        name="deepseek-v4-pro",
        source="deepseek",
        profiles=("reasoning", "development"),
        capabilities=("tool-calling", "structured-output", "code-generation"),
        cost_tier="standard",
    ),
    ModelInfo.create(
        name="deepseek-v4-flash",
        source="deepseek",
        profiles=("focused-development",),
        capabilities=("tool-calling", "code-generation"),
        cost_tier="low",
    ),
    ModelInfo.create(
        name="mistral-medium-latest",
        source="mistral",
        profiles=("writing", "general"),
        capabilities=("tool-calling", "structured-output", "long-form-text"),
        cost_tier="standard",
    ),
    ModelInfo.create(
        name="qwen2.5:1.5b",
        source="ollama",
        profiles=("local", "general"),
        capabilities=("text-generation",),
        cost_tier="local",
    ),
)


default_model_catalog = ModelCatalog(DEFAULT_MODELS)
