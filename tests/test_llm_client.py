"""Tests for pipeline/llm_client.py — unified LLM client."""

import json
from unittest.mock import MagicMock, patch


from pipeline.llm_client import (
    LLMClient,
    LLMResponse,
    OllamaProvider,
    OpenRouterProvider,
    HermesProvider,
    get_llm_client,
)
from pipeline.config import Config


# ─── Provider Unit Tests ────────────────────────────────────────────────────

class TestOllamaProvider:
    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Hello world"}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        provider = OllamaProvider()
        resp = provider.generate("test", "llama3", 30)
        assert resp.success is True
        assert resp.text == "Hello world"

    @patch("urllib.request.urlopen")
    def test_generate_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        provider = OllamaProvider()
        resp = provider.generate("test", "llama3", 30)
        assert resp.success is False
        assert "Connection refused" in resp.error

    @patch("urllib.request.urlopen")
    def test_embed_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        provider = OllamaProvider()
        emb = provider.embed("hello", "nomic", 30)
        assert emb == [0.1, 0.2, 0.3]

    @patch("urllib.request.urlopen")
    def test_embed_batch_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        provider = OllamaProvider()
        batch = provider.embed_batch(["a", "b"], "nomic", 30)
        assert len(batch) == 2
        assert batch["a"] == [0.1, 0.2]
        assert batch["b"] == [0.3, 0.4]

    @patch("urllib.request.urlopen")
    def test_embed_batch_404_fallback(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost/api/embed", 404, "Not Found", {}, None
        )

        provider = OllamaProvider()
        batch = provider.embed_batch(["a", "b"], "nomic", 30)
        assert batch == {}


class TestOpenRouterProvider:
    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "OpenRouter says hi"}}]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        provider = OpenRouterProvider()
        resp = provider.generate("test", "qwen/test", 30, api_key="sk-test")
        assert resp.success is True
        assert resp.text == "OpenRouter says hi"

    def test_generate_missing_key(self):
        provider = OpenRouterProvider()
        resp = provider.generate("test", "qwen/test", 30)
        assert resp.success is False
        assert "API key not configured" in resp.error

    def test_generate_missing_model(self):
        provider = OpenRouterProvider()
        resp = provider.generate("test", "", 30, api_key="sk-test")
        assert resp.success is False
        assert "requires a model" in resp.error

    @patch("urllib.request.urlopen")
    def test_generate_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429, "Too Many Requests", {}, b'{"error":"rate limit"}'
        )

        provider = OpenRouterProvider()
        resp = provider.generate("test", "qwen/test", 30, api_key="sk-test")
        assert resp.success is False
        assert "429" in resp.error

    def test_embed_not_supported(self):
        provider = OpenRouterProvider()
        assert provider.embed("hello", "model", 30) is None
        assert provider.embed_batch(["a"], "model", 30) == {}


class TestHermesProvider:
    @patch("subprocess.run")
    def test_generate_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Hermes output", stderr="")

        provider = HermesProvider()
        resp = provider.generate("test", "", 30)
        assert resp.success is True
        assert resp.text == "Hermes output"

    @patch("subprocess.run")
    def test_generate_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        provider = HermesProvider()
        resp = provider.generate("test", "", 30)
        assert resp.success is False
        assert "not found" in resp.error.lower()

    @patch("subprocess.run")
    def test_generate_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("hermes", 30)

        provider = HermesProvider()
        resp = provider.generate("test", "", 30)
        assert resp.success is False
        assert "timed out" in resp.error.lower()


# ─── LLMClient Integration ──────────────────────────────────────────────────

class TestLLMClient:
    def test_defaults_to_ollama(self):
        client = LLMClient()
        assert client.provider == "ollama"
        assert isinstance(client._provider_impl, OllamaProvider)

    def test_openrouter_provider(self):
        client = LLMClient(provider="openrouter", api_key="sk-test")
        assert isinstance(client._provider_impl, OpenRouterProvider)

    def test_unknown_provider_fallback(self):
        client = LLMClient(provider="unknown")
        assert isinstance(client._provider_impl, OllamaProvider)

    @patch("pipeline.llm_client.OllamaProvider.generate")
    def test_generate_delegates(self, mock_gen):
        mock_gen.return_value = LLMResponse(text="result", success=True)
        client = LLMClient(provider="ollama", model="llama3")
        result = client.generate("prompt")
        assert result == "result"
        mock_gen.assert_called_once()

    @patch("pipeline.llm_client.OllamaProvider.generate")
    def test_generate_empty_on_failure(self, mock_gen):
        mock_gen.return_value = LLMResponse(error="fail", success=False)
        client = LLMClient(provider="ollama", model="llama3")
        result = client.generate("prompt")
        assert result == ""

    @patch("pipeline.llm_client.OllamaProvider.embed")
    def test_embed_delegates(self, mock_embed):
        mock_embed.return_value = [0.1, 0.2]
        client = LLMClient(provider="ollama", embed_model="nomic")
        result = client.embed("hello")
        assert result == [0.1, 0.2]

    @patch("pipeline.llm_client.OllamaProvider.embed_batch")
    def test_embed_batch_delegates(self, mock_batch):
        mock_batch.return_value = {"a": [0.1]}
        client = LLMClient(provider="ollama", embed_model="nomic")
        result = client.embed_batch(["a"])
        assert result == {"a": [0.1]}

    @patch("pipeline.llm_client.OllamaProvider.generate")
    def test_generate_parallel(self, mock_gen):
        mock_gen.return_value = LLMResponse(text="answer", success=True)
        client = LLMClient(provider="ollama", model="llama3")
        results = client.generate_parallel([("k1", "p1"), ("k2", "p2")], max_workers=2)
        assert len(results) == 2
        assert results["k1"] == "answer"
        assert results["k2"] == "answer"


# ─── get_llm_client from Config ─────────────────────────────────────────────

class TestGetLLMClient:
    def test_ollama_defaults(self):
        cfg = Config(llm_provider="ollama")
        client = get_llm_client(cfg)
        assert client.provider == "ollama"
        assert client.model == "minimax-m2.7:cloud"  # fallback from ollama_insight_model

    def test_openrouter_requires_model(self):
        cfg = Config(llm_provider="openrouter", llm_model="qwen/test", llm_api_key="sk-test")
        client = get_llm_client(cfg)
        assert client.provider == "openrouter"
        assert client.model == "qwen/test"
        assert client.api_key == "sk-test"

    def test_base_url_resolution(self):
        cfg = Config(llm_provider="ollama", ollama_host="http://192.168.1.10:11434")
        client = get_llm_client(cfg)
        assert client.base_url == "http://192.168.1.10:11434"

    def test_explicit_model_overrides_default(self):
        cfg = Config(
            llm_provider="ollama",
            llm_model="qwen3.6-plus",
            ollama_insight_model="minimax-m2.7:cloud",
        )
        client = get_llm_client(cfg)
        assert client.model == "qwen3.6-plus"

    def test_embed_url_fallback(self):
        cfg = Config(ollama_host="http://ollama.local:11434")
        client = get_llm_client(cfg)
        assert client.embed_base_url == "http://ollama.local:11434"
