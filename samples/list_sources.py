"""List the LLM sources (providers) available in galet.

Run from the repo root:

    python samples/list_sources.py

This script prints the provider registry that galet uses to route a request
to the right backend, plus the model-name prefix map used for automatic
routing when no explicit provider is given.
"""

from __future__ import annotations

from galet.provider_registry import ProviderRegistry


def main() -> None:
    print("Available galet LLM sources (providers):\n")
    for name, import_path in ProviderRegistry.providers.items():
        print(f"  {name:<10} -> {import_path}")

    print("\nModel-name prefix routing (used when no provider is given):\n")
    for prefix, provider in ProviderRegistry.prefix_map.items():
        print(f"  {prefix:<10} -> {provider}")

    print("\nAny other model name falls back to 'openai'.")


if __name__ == "__main__":
    main()
