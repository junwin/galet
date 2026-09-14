from __future__ import annotations

import json

import pytest

from galet.settings import CredentialProfile, Settings


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



def test_named_file_profile_ignores_provider_environment(tmp_path, monkeypatch) -> None:
    credential_file = tmp_path / "openai.json"
    credential_file.write_text(
        json.dumps({"openai_api_key": "file-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    settings = Settings(
        credential_profiles={
            "openai-service": {
                "provider": "openai",
                "source": "file",
                "path": str(credential_file),
            }
        },
        credential_profile="openai-service",
    )

    assert settings.api_key("openai") == "file-key"


def test_named_environment_profile_uses_only_named_variable(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "oaicred.json").write_text(
        json.dumps({"openai_api_key": "file-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-environment-key")
    monkeypatch.setenv("LUCY_OPENAI_KEY", "profile-key")

    settings = Settings(
        credential_path=str(tmp_path),
        credential_profiles={
            "openai-development": CredentialProfile(
                provider="openai",
                source="environment",
                variable="LUCY_OPENAI_KEY",
            )
        },
        credential_profile="openai-development",
    )

    assert settings.api_key("openai") == "profile-key"


def test_named_profile_does_not_fall_back_when_credential_is_missing(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "oaicred.json").write_text(
        json.dumps({"openai_api_key": "legacy-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    settings = Settings(
        credential_path=str(tmp_path),
        credential_profiles={
            "openai-service": {
                "provider": "openai",
                "source": "file",
                "path": str(tmp_path / "missing.json"),
            }
        },
        credential_profile="openai-service",
    )

    assert settings.api_key("openai") is None


def test_systemd_profile_reads_json_credential(tmp_path, monkeypatch) -> None:
    (tmp_path / "openai").write_text(
        json.dumps({"openai_api_key": "systemd-json-key"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))

    settings = Settings(
        credential_profiles={
            "openai-service": {
                "provider": "openai",
                "source": "systemd",
                "credential": "openai",
            }
        },
        credential_profile="openai-service",
    )

    assert settings.api_key("openai") == "systemd-json-key"


def test_systemd_profile_reads_raw_credential(tmp_path, monkeypatch) -> None:
    (tmp_path / "openai").write_text("systemd-raw-key\n", encoding="utf-8")
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))

    settings = Settings(
        credential_profiles={
            "openai-service": {
                "provider": "openai",
                "source": "systemd",
                "credential": "openai",
            }
        }
    )

    assert (
        settings.api_key("openai", credential_profile="openai-service")
        == "systemd-raw-key"
    )


def test_unknown_named_profile_is_configuration_error() -> None:
    settings = Settings(credential_profiles={})

    with pytest.raises(ValueError, match="unknown credential profile"):
        settings.api_key("openai", credential_profile="missing")


def test_profile_provider_must_match_requested_provider() -> None:
    settings = Settings(
        credential_profiles={
            "writer": {
                "provider": "mistral",
                "source": "environment",
                "variable": "MISTRAL_API_KEY",
            }
        }
    )

    with pytest.raises(ValueError, match="not 'openai'"):
        settings.api_key("openai", credential_profile="writer")


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"provider": "openai", "source": "vault"}, "unknown.*source"),
        ({"provider": "openai", "source": "file"}, "requires path"),
        ({"provider": "openai", "source": "environment"}, "requires variable"),
        ({"provider": "openai", "source": "systemd"}, "requires credential"),
    ],
)
def test_invalid_profile_configuration_is_rejected(profile, message) -> None:
    settings = Settings(
        credential_profiles={"invalid": profile},
        credential_profile="invalid",
    )

    with pytest.raises(ValueError, match=message):
        settings.api_key("openai")
