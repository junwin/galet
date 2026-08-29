"""List the LLM sources (providers) registered with galet.

Run from the repo root:

    python samples/list_sources.py

Every connector module registers itself (name, display name, description,
model-name prefixes, default model) with the provider registry. This script
prints that registration table plus the prefix routing it drives.
"""

from __future__ import annotations

from galet.provider_info import registered_providers
from galet.provider_registry import ProviderRegistry


def main() -> None:
    ProviderRegistry.load_all()
    print("Available galet LLM sources (providers):\n")
    for info in registered_providers():
        print(f"  {info.name:<10} {info.display_name:<20} default model: {info.default_model}")
        print(f"              {info.description}")
        print(f"              model prefixes: {', '.join(info.prefixes) or '(none)'}")
        print()

    print("Model-name prefix routing (used when no provider is given):\n")
    for prefix, provider in ProviderRegistry.prefix_map().items():
        print(f"  {prefix:<10} -> {provider}")

    print(f"\nAny other model name falls back to '{ProviderRegistry.resolve_name(None, None)}'.")


if __name__ == "__main__":
    main()
