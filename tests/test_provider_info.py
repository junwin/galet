"""Tests for connector self-registration and registry-derived routing."""

from __future__ import annotations

import pytest

from galet.provider_info import ProviderInfo, get_provider, registered_providers
from galet.provider_registry import ProviderRegistry


def test_all_shipped_connectors_register() -> None:
    ProviderRegistry.load_all()
    names = {info.name for info in registered_providers()}
    assert names == {"openai", "deepseek", "gemini", "mistral", "ollama"}


def test_registered_provider_metadata() -> None:
    ProviderRegistry.load_all()
    info = get_provider("deepseek")
    assert info is not None
    assert info.display_name == "DeepSeek"
    assert info.prefixes == ("deepseek",)
    assert info.class_path == "galet.deepseek_responses.DeepSeekApi"
    assert info.default_model == "deepseek-chat"


def test_get_provider_unknown_returns_none() -> None:
    assert get_provider("not-a-provider") is None


def test_providers_map_has_all_sources() -> None:
    providers = ProviderRegistry.providers()
    assert set(providers) == {"openai", "deepseek", "gemini", "mistral", "ollama"}
    assert providers["openai"] == "galet.openai_responses.OpenAIResponsesApi"
    assert providers["ollama"] == "galet.ollama_api.OllamaApi"


def test_prefix_map_built_from_registration() -> None:
    mapping = ProviderRegistry.prefix_map()
    assert mapping["gpt"] == "openai"
    assert mapping["o1"] == "openai"
    assert mapping["o3"] == "openai"
    assert mapping["deepseek"] == "deepseek"
    assert mapping["gemini"] == "gemini"
    assert mapping["mistral"] == "mistral"
    assert mapping["ollama"] == "ollama"


def test_default_models_derive_from_registration() -> None:
    ProviderRegistry.load_all()
    models = {info.name: info.default_model for info in registered_providers()}
    assert models["openai"] == "gpt-4o-mini"
    assert models["deepseek"] == "deepseek-chat"
    assert models["gemini"] == "gemini-2.0-flash"
    assert models["mistral"] == "mistral-small-latest"
    assert models["ollama"] == "llama3.1"


def test_resolve_name_uses_registered_prefixes() -> None:
    assert ProviderRegistry.resolve_name("deepseek-chat", None) == "deepseek"
    assert ProviderRegistry.resolve_name("gemini-2.0-flash", None) == "gemini"
    assert ProviderRegistry.resolve_name("mistral-small", None) == "mistral"
    assert ProviderRegistry.resolve_name("ollama/llama3.1", None) == "ollama"
    assert ProviderRegistry.resolve_name("gpt-4o-mini", None) == "openai"
    assert ProviderRegistry.resolve_name("o3-mini", None) == "openai"


def test_resolve_name_unknown_model_defaults_to_openai() -> None:
    assert ProviderRegistry.resolve_name("claude-3-opus", None) == "openai"


def test_resolve_name_unknown_explicit_provider_raises() -> None:
    with pytest.raises(ValueError):
        ProviderRegistry.resolve_name("gpt-4o", provider="not-a-provider")


def test_resolve_explicit_provider_wins() -> None:
    name, inst = ProviderRegistry.resolve("deepseek-xyz", provider="openai")
    assert name == "openai"
    assert inst is not None


def test_provider_info_frozen() -> None:
    info = ProviderInfo(
        name="probe",
        display_name="Probe",
        description="probe",
        prefixes=("probe",),
        class_path="probe.ProbeApi",
        default_model="probe-1",
    )
    with pytest.raises(Exception):
        info.name = "other"  # type: ignore[misc]
