"""Provider-agnostic LLM client factory."""

from __future__ import annotations

import os
from typing import Any, Iterable, Protocol

from MaLO.utils.ollama_client import OllamaClient
from MaLO.utils.openrouter_client import OpenRouterClient


class LLMClient(Protocol):
    """LLM client interface used by graph runners."""

    def generate(
        self,
        model: str,
        messages: Iterable[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
    ) -> Any:
        """Generates a completion."""


def build_llm_client(
    provider: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Builds an LLM client for the configured provider.

    Supported providers: "openrouter" (default), "ollama".
    """
    provider_name = (provider or os.environ.get("LLM_PROVIDER") or "openrouter")
    provider_name = provider_name.strip().lower()

    if provider_name in ("openrouter", "openrouter.ai"):
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")
        resolved_base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or None
        return OpenRouterClient(api_key=api_key, base_url=resolved_base_url)

    if provider_name == "ollama":
        resolved_base_url = base_url or os.environ.get("OLLAMA_BASE_URL") or None
        return OllamaClient(base_url=resolved_base_url)

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}. "
        "Set LLM_PROVIDER=openrouter/ollama or implement a new provider."
    )
