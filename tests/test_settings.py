from __future__ import annotations

import json

from galet.settings import Settings


def test_env_var_fallback(tmp_path, monkeypatch) -> None:
    (tmp_path / "oaicred.json").write_text(
        json.dumps({"openai_api_key": "sk-test"}), encoding="utf-8"
    )
    monkeypatch.setenv("GALET_CREDENTIAL_PATH", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings().api_key("openai") == "sk-test"


def test_explicit_path_wins(tmp_path, monkeypatch) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "oaicred.json").write_text(
        json.dumps({"openai_api_key": "sk-env"}), encoding="utf-8"
    )
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    (explicit_dir / "oaicred.json").write_text(
        json.dumps({"openai_api_key": "sk-explicit"}), encoding="utf-8"
    )
    monkeypatch.setenv("GALET_CREDENTIAL_PATH", str(env_dir))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(credential_path=str(explicit_dir))
    assert settings.api_key("openai") == "sk-explicit"


def test_no_path_and_no_env_var(monkeypatch) -> None:
    monkeypatch.delenv("GALET_CREDENTIAL_PATH", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings().api_key("openai") is None
