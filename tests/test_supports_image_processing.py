"""Tests for supports_image_processing across the LLM stack.

Covers:
  - OpenAIResponsesApi → True
  - MistralApi → True
  - DeepSeekApi → False
  - OllamaApi → False
  - GeminiApi → True
  - RouterApi → routes correctly to each provider (explicit provider param)
  - RouterApi → prefix fallback still works
  - RouterApi → unknown model with no provider defaults to openai
"""

from __future__ import annotations

import pytest

from src.llm.openai_responses import OpenAIResponsesApi
from src.llm.mistral_api import MistralApi
from src.llm.deepseek_responses import DeepSeekApi
from src.llm.ollama_api import OllamaApi
from src.llm.gemini_api import GeminiApi
from src.llm.router_api import RouterApi


# ---------------------------------------------------------------------------
# OpenAIResponsesApi
# ---------------------------------------------------------------------------

def test_supports_image_processing_openai() -> None:
    api = OpenAIResponsesApi()
    assert api.supports_image_processing("gpt-4o") is True
    assert api.supports_image_processing("gpt-5") is True
    assert api.supports_image_processing("o1") is True
    assert api.supports_image_processing("o3-mini") is True


# ---------------------------------------------------------------------------
# MistralApi
# ---------------------------------------------------------------------------

def test_supports_image_processing_mistral() -> None:
    api = MistralApi()
    assert api.supports_image_processing("mistral-large") is True
    assert api.supports_image_processing("mistral-small") is True
    assert api.supports_image_processing("pixtral-large-latest") is True


# ---------------------------------------------------------------------------
# DeepSeekApi
# ---------------------------------------------------------------------------

def test_supports_image_processing_deepseek() -> None:
    api = DeepSeekApi()
    assert api.supports_image_processing("deepseek-chat") is False
    assert api.supports_image_processing("deepseek-reasoner") is False


# ---------------------------------------------------------------------------
# OllamaApi
# ---------------------------------------------------------------------------

def test_supports_image_processing_ollama() -> None:
    api = OllamaApi()
    assert api.supports_image_processing("llama3.1") is False
    assert api.supports_image_processing("qwen3") is False
    assert api.supports_image_processing("gemma3") is False


# ---------------------------------------------------------------------------
# GeminiApi
# ---------------------------------------------------------------------------

def test_supports_image_processing_gemini() -> None:
    api = GeminiApi()
    assert api.supports_image_processing("gemini-2.0-flash") is True
    assert api.supports_image_processing("gemini-pro") is True


# ---------------------------------------------------------------------------
# RouterApi
# ---------------------------------------------------------------------------

def test_router_explicit_provider_routing() -> None:
    router = RouterApi()
    # explicit provider selection overrides prefix resolution
    assert router.supports_image_processing("anything", provider="mistral") is True
    assert router.supports_image_processing("anything", provider="deepseek") is False
    assert router.supports_image_processing("anything", provider="openai") is True
    assert router.supports_image_processing("anything", provider="ollama") is False
    assert router.supports_image_processing("anything", provider="gemini") is True


def test_router_prefix_fallback_still_works() -> None:
    router = RouterApi()
    assert router.supports_image_processing("deepseek-chat") is False
    assert router.supports_image_processing("mistral-large") is True
    assert router.supports_image_processing("gpt-4o") is True
    assert router.supports_image_processing("ollama/llama3.1") is False
    assert router.supports_image_processing("gemini-2.0-flash") is True
    assert router.supports_image_processing("gemini-pro") is True


def test_router_unknown_model_defaults_to_openai() -> None:
    router = RouterApi()
    # unknown model name but no explicit provider -> should resolve to openai
    assert router.supports_image_processing("claude-3-opus") is True
