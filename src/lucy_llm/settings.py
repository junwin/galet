from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


@dataclass
class Settings:
    credential_path: Optional[str] = None
    ollama_base_url: Optional[str] = None

    def api_key(self, provider: str) -> Optional[str]:
        env_var = _PROVIDER_ENV_VAR.get(provider)
        if env_var:
            value = os.environ.get(env_var)
            if value:
                return value
        if not self.credential_path:
            return None
        file_name = _PROVIDER_CREDENTIAL_FILE.get(provider)
        keys = _PROVIDER_CREDENTIAL_KEY.get(provider)
        if not file_name or not keys:
            return None
        try:
            with open(os.path.join(self.credential_path, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def base_url(self) -> Optional[str]:
        url = os.environ.get("OLLAMA_BASE_URL")
        if url:
            return url
        return self.ollama_base_url


default_settings = Settings()
