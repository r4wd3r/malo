from __future__ import annotations

import pytest

from MaLO.utils.llm_client import build_llm_client
from MaLO.utils.ollama_client import OllamaClient
from MaLO.utils.openrouter_client import OpenRouterClient


def test_build_llm_client_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = build_llm_client()
    assert isinstance(client, OpenRouterClient)


def test_build_llm_client_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = build_llm_client()
    assert isinstance(client, OllamaClient)


def test_build_llm_client_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_llm_client()
