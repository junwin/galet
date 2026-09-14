"""Focused offline tests for Galet model metadata and resolution."""

from __future__ import annotations

import pytest

from galet.model_catalog import (
    ModelCatalog,
    ModelInfo,
    ModelRequirements,
    default_model_catalog,
)


def model(
    name: str,
    source: str,
    *,
    profiles: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = (),
) -> ModelInfo:
    return ModelInfo.create(
        name=name,
        source=source,
        profiles=profiles,
        capabilities=capabilities,
    )


def test_catalog_exposes_registered_source_information() -> None:
    sources = {source.name: source for source in default_model_catalog.sources()}

    assert set(sources) == {"openai", "deepseek", "gemini", "mistral", "ollama"}
    assert sources["openai"].default_model == "gpt-4o-mini"


def test_default_catalog_exposes_model_information() -> None:
    info = default_model_catalog.find("gpt-5.6-sol")

    assert info is not None
    assert info.source == "openai"
    assert "reasoning" in info.profiles
    assert "tool-calling" in info.capabilities
    assert info.supports_temperature is False


def test_models_can_be_filtered_by_source() -> None:
    models = default_model_catalog.models(source="mistral")

    assert [entry.name for entry in models] == ["mistral-medium-latest"]


def test_resolve_prefers_requested_model_when_it_is_eligible() -> None:
    catalog = ModelCatalog(
        (
            model("reasoner", "source-a", profiles=("reasoning",), capabilities=("tools",)),
            model("other", "source-b", profiles=("reasoning",), capabilities=("tools",)),
        )
    )

    result = catalog.resolve(
        ModelRequirements.create(
            profile="reasoning",
            required_capabilities=("tools",),
            preferred_model="reasoner",
        )
    )

    assert result.source == "source-a"
    assert result.model == "reasoner"
    assert result.info.name == "reasoner"


def test_resolve_falls_back_when_preferred_model_is_ineligible() -> None:
    catalog = ModelCatalog(
        (
            model("writer", "source-a", profiles=("writing",)),
            model("coder", "source-b", profiles=("development",), capabilities=("tools",)),
        )
    )

    result = catalog.resolve(
        ModelRequirements.create(
            profile="development",
            required_capabilities=("tools",),
            preferred_model="writer",
        )
    )

    assert result.model == "coder"


def test_resolve_can_disallow_fallback() -> None:
    catalog = ModelCatalog(
        (
            model("writer", "source-a", profiles=("writing",)),
            model("coder", "source-b", profiles=("development",)),
        )
    )

    with pytest.raises(LookupError, match="preferred model"):
        catalog.resolve(
            ModelRequirements.create(
                profile="development",
                preferred_model="writer",
                allow_fallback=False,
            )
        )


def test_resolve_uses_preferred_source_when_possible() -> None:
    catalog = ModelCatalog(
        (
            model("alpha", "source-a", profiles=("general",)),
            model("beta", "source-b", profiles=("general",)),
        )
    )

    result = catalog.resolve(
        ModelRequirements.create(
            profile="general",
            preferred_source="source-b",
        )
    )

    assert result.source == "source-b"


def test_resolve_rejects_unsatisfied_requirements() -> None:
    catalog = ModelCatalog(
        (model("writer", "source-a", profiles=("writing",)),)
    )

    with pytest.raises(LookupError, match="no model"):
        catalog.resolve(
            ModelRequirements.create(required_capabilities=("tool-calling",))
        )


def test_resolution_is_deterministic_without_preferences() -> None:
    catalog = ModelCatalog(
        (
            model("zeta", "source-z", profiles=("general",)),
            model("alpha", "source-a", profiles=("general",)),
        )
    )

    result = catalog.resolve(ModelRequirements.create(profile="general"))

    assert (result.source, result.model) == ("source-a", "alpha")


def test_registration_replaces_same_source_and_model() -> None:
    catalog = ModelCatalog()
    catalog.register(model("alpha", "source-a", profiles=("old",)))
    catalog.register(model("alpha", "source-a", profiles=("new",)))

    assert catalog.find("alpha", "source-a").profiles == frozenset({"new"})


def test_registration_requires_name_and_source() -> None:
    catalog = ModelCatalog()

    with pytest.raises(ValueError, match="name and source"):
        catalog.register(model("", "source-a"))
