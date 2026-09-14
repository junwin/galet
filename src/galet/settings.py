from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

_PROVIDER_ENV_VAR: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_PROVIDER_CREDENTIAL_FILE: Dict[str, str] = {
    "openai": "oaicred.json",
    "deepseek": "deepseek_cred.json",
    "mistral": "mistral_cred.json",
    "gemini": "gemini_cred.json",
}

_PROVIDER_CREDENTIAL_KEY: Dict[str, Tuple[str, ...]] = {
    "openai": ("openai_api_key",),
    "deepseek": ("deepseek_api_key",),
    "mistral": ("mistral_api_key",),
    "gemini": ("gemini_api_key", "api_key"),
}

_PROFILE_SOURCES = frozenset({"file", "environment", "systemd"})


@dataclass(frozen=True)
class CredentialProfile:
    """Named instructions for resolving one provider credential.

    A profile selects exactly one source. It contains references to a secret,
    never the secret value itself.
    """

    provider: str
    source: str
    path: Optional[str] = None
    variable: Optional[str] = None
    credential: Optional[str] = None

    @classmethod
    def from_value(
        cls,
        value: "CredentialProfile | Mapping[str, Any]",
    ) -> "CredentialProfile":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("credential profile must be a mapping")
        try:
            return cls(
                provider=str(value["provider"]),
                source=str(value["source"]),
                path=_optional_string(value.get("path")),
                variable=_optional_string(value.get("variable")),
                credential=_optional_string(value.get("credential")),
            )
        except KeyError as exc:
            raise ValueError(
                f"credential profile missing required field: {exc.args[0]}"
            ) from exc

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("credential profile provider is required")
        if self.source not in _PROFILE_SOURCES:
            raise ValueError(f"unknown credential profile source: {self.source}")
        if self.source == "file" and not self.path:
            raise ValueError("file credential profile requires path")
        if self.source == "environment" and not self.variable:
            raise ValueError("environment credential profile requires variable")
        if self.source == "systemd" and not self.credential:
            raise ValueError("systemd credential profile requires credential")


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


@dataclass
class Settings:
    credential_path: Optional[str] = None
    ollama_base_url: Optional[str] = None
    credential_profiles: Optional[
        Mapping[str, CredentialProfile | Mapping[str, Any]]
    ] = None
    credential_profile: Optional[str] = None

    def _resolved_credential_path(self) -> Optional[str]:
        if self.credential_path:
            return self.credential_path
        return os.environ.get("GALET_CREDENTIAL_PATH")

    def _named_profile(
        self,
        name: str,
        provider: str,
    ) -> CredentialProfile:
        raw_profile = (self.credential_profiles or {}).get(name)
        if raw_profile is None:
            raise ValueError(f"unknown credential profile: {name}")

        profile = CredentialProfile.from_value(raw_profile)
        profile.validate()
        if profile.provider != provider:
            raise ValueError(
                f"credential profile {name!r} is for provider "
                f"{profile.provider!r}, not {provider!r}"
            )
        return profile

    @staticmethod
    def _key_from_text(text: str, provider: str) -> Optional[str]:
        """Read either the existing JSON format or a raw systemd credential."""

        try:
            data = json.loads(text)
        except ValueError:
            value = text.strip()
            return value or None

        if not isinstance(data, dict):
            return None
        for key in _PROVIDER_CREDENTIAL_KEY.get(provider, ()):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @classmethod
    def _key_from_file(cls, path: Path, provider: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as credential_file:
                text = credential_file.read()
        except OSError:
            return None
        return cls._key_from_text(text, provider)

    def _profile_api_key(
        self,
        profile: CredentialProfile,
    ) -> Optional[str]:
        if profile.source == "environment":
            return os.environ.get(profile.variable or "") or None

        if profile.source == "file":
            return self._key_from_file(Path(profile.path or ""), profile.provider)

        credentials_directory = os.environ.get("CREDENTIALS_DIRECTORY")
        if not credentials_directory:
            return None
        path = Path(credentials_directory) / (profile.credential or "")
        return self._key_from_file(path, profile.provider)

    def api_key(
        self,
        provider: str,
        credential_profile: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a provider key without exposing its source to callers.

        When a named profile is selected, only that profile's source is used.
        With no profile, the legacy environment-then-directory lookup remains
        for backward compatibility.
        """

        profile_name = credential_profile or self.credential_profile
        if profile_name:
            profile = self._named_profile(profile_name, provider)
            return self._profile_api_key(profile)

        env_var = _PROVIDER_ENV_VAR.get(provider)
        if env_var:
            value = os.environ.get(env_var)
            if value:
                return value

        credential_path = self._resolved_credential_path()
        if not credential_path:
            return None
        file_name = _PROVIDER_CREDENTIAL_FILE.get(provider)
        if not file_name:
            return None
        return self._key_from_file(Path(credential_path) / file_name, provider)

    def base_url(self) -> Optional[str]:
        if self.ollama_base_url:
            return self.ollama_base_url
        return os.environ.get("OLLAMA_BASE_URL")


default_settings = Settings()
