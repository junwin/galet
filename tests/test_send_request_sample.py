from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "samples" / "send_request.py"


@lru_cache(maxsize=1)
def _sample():
    spec = importlib.util.spec_from_file_location("galet_sample_send_request", _SAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_provider_wins() -> None:
    sample = _sample()
    assert sample.resolve_provider("mistral-large-latest", "ollama") == "ollama"


def test_infers_mistral_from_model_name() -> None:
    sample = _sample()
    assert sample.resolve_provider("mistral-large-latest", None) == "mistral"


def test_infers_openai_from_model_name() -> None:
    sample = _sample()
    assert sample.resolve_provider("gpt-4o-mini", None) == "openai"


def test_infers_deepseek_from_model_name() -> None:
    sample = _sample()
    assert sample.resolve_provider("deepseek-chat", None) == "deepseek"


def test_infers_gemini_from_model_name() -> None:
    sample = _sample()
    assert sample.resolve_provider("gemini-2.0-flash", None) == "gemini"


def test_defaults_to_ollama_without_model() -> None:
    sample = _sample()
    assert sample.resolve_provider(None, None) == "ollama"
