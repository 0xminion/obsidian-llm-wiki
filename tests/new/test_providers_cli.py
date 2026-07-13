"""Tests for provider preflight and model-discovery CLI commands."""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from obsidian_llm_wiki.config import LLMProviderConfig
from obsidian_llm_wiki.providers import llm

runner = CliRunner()


def test_list_provider_models_uses_openai_models_endpoint(monkeypatch):
    """OpenAI-compatible providers enumerate IDs from their standard models endpoint."""
    seen: dict[str, object] = {}

    def fake_get(self, url, *, headers=None):
        seen["url"] = url
        seen["headers"] = headers
        return httpx.Response(
            200,
            json={"data": [{"id": "model-b"}, {"id": "model-a"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    config = LLMProviderConfig(
        provider="openai", host="https://llm.example", model="model-a", api_key="not-output"
    )

    assert llm.list_provider_models(config) == ["model-a", "model-b"]
    assert seen["url"] == "https://llm.example/v1/models"
    assert seen["headers"] == {"Authorization": "Bearer not-output"}


def test_list_provider_models_falls_back_to_ollama_tags(monkeypatch):
    """Older Ollama endpoints without /v1/models still expose their native model tags."""
    urls: list[str] = []

    def fake_get(self, url, *, headers=None):
        urls.append(url)
        if url.endswith("/v1/models"):
            return httpx.Response(404, request=httpx.Request("GET", url))
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:8b"}, {"name": "gemma3:27b"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    config = LLMProviderConfig(provider="ollama", host="http://localhost:11434", model="gemma3:27b")

    assert llm.list_provider_models(config) == ["gemma3:27b", "qwen3:8b"]
    assert urls == ["http://localhost:11434/v1/models", "http://localhost:11434/api/tags"]


def test_providers_check_reports_safe_structured_preflight(monkeypatch):
    """Provider preflight prints endpoint/auth/model diagnostics without exposing an API key."""
    from obsidian_llm_wiki.cli import app, providers

    monkeypatch.setattr(
        providers,
        "check_provider",
        lambda config: {
            "ok": True,
            "provider": config.provider,
            "endpoint": {"url": "https://llm.example/v1/models", "reachable": True},
            "auth": {"configured": True, "status": "configured"},
            "models": {"default": {"name": config.model, "status": "available"}},
        },
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_HOST", "https://llm.example")
    monkeypatch.setenv("LLM_MODEL", "safe-model")
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key")

    result = runner.invoke(app, ["providers", "check"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["models"]["default"]["name"] == "safe-model"
    assert "super-secret-key" not in result.output


def test_providers_models_prints_structured_model_list(monkeypatch):
    """The models subcommand is registered and returns a deterministic JSON list."""
    from obsidian_llm_wiki.cli import app, providers

    monkeypatch.setattr(providers, "list_provider_models", lambda config: ["z-model", "a-model"])
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "a-model")

    result = runner.invoke(app, ["providers", "models"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "models": ["a-model", "z-model"],
        "provider": "openai",
    }


def test_provider_check_redacts_userinfo_query_and_fragment_from_endpoint_diagnostics(monkeypatch):
    """Diagnostic payloads must never expose credentials embedded in an endpoint URL."""
    host = "https://operator:embedded-secret@llm.example:8443/base?token=query-secret#fragment-secret"
    config = LLMProviderConfig(provider="openai", host=host, model="model-a", api_key="api-secret")

    monkeypatch.setattr(llm, "list_provider_models", lambda _config: ["model-a"])

    payload = llm.check_provider(config)
    serialized = json.dumps(payload)

    assert payload["endpoint"]["url"] == "https://llm.example:8443/base/v1/models"
    for secret in ("operator", "embedded-secret", "query-secret", "fragment-secret", "api-secret"):
        assert secret not in serialized
