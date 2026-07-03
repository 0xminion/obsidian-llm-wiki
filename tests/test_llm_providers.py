"""Tests for pipeline.llm.providers — factory, client types, ABC contract."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from pipeline.config import LLMProviderConfig
from pipeline.llm.providers import (
    LLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)

# ──────────────────────────────────────────────────────────────────────────────
# Factory tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCreateLLMClient:
    """create_llm_client returns correct type for each provider."""

    def test_ollama_returns_ollama_client(self) -> None:
        cfg = LLMProviderConfig(provider="ollama", host="http://localhost:11434")
        client = create_llm_client(cfg)
        assert isinstance(client, OllamaClient)

    def test_openai_returns_openai_compatible_client(self) -> None:
        cfg = LLMProviderConfig(
            provider="openai",
            host="https://api.openai.com",
            api_key="sk-test",
        )
        client = create_llm_client(cfg)
        assert isinstance(client, OpenAICompatibleClient)

    def test_invalid_provider_raises_value_error(self) -> None:
        cfg = LLMProviderConfig(provider="bogus", host="http://localhost:1")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(cfg)

    def test_factory_returns_llm_client_subclass(self) -> None:
        """Both branches should return an LLMClient subclass."""
        ollama_cfg = LLMProviderConfig(provider="ollama")
        openai_cfg = LLMProviderConfig(provider="openai", api_key="sk-x")
        assert isinstance(create_llm_client(ollama_cfg), LLMClient)
        assert isinstance(create_llm_client(openai_cfg), LLMClient)


# ──────────────────────────────────────────────────────────────────────────────
# OllamaClient chat tests (mocked httpx)
# ──────────────────────────────────────────────────────────────────────────────


class TestOllamaClientChat:
    """OllamaClient.chat calls /api/chat and parses message.content."""

    def test_chat_returns_message_content(self) -> None:
        cfg = LLMProviderConfig(provider="ollama", host="http://localhost:11434")
        client = OllamaClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hello from Ollama!"}
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.chat("You are helpful.", "Say hi")
        assert result == "Hello from Ollama!"

    def test_chat_sends_correct_url_and_body(self) -> None:
        cfg = LLMProviderConfig(
            provider="ollama",
            host="http://my-host:1234",
            model="gemma:2b",
        )
        client = OllamaClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp) as m_post:
            client.chat("sys", "usr")

        call_args = m_post.call_args
        url = call_args.args[0]
        body = call_args.kwargs["json"]
        assert url == "http://my-host:1234/api/chat"
        assert body["model"] == "gemma:2b"
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        assert body["stream"] is False


# ──────────────────────────────────────────────────────────────────────────────
# OllamaClient embed tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOllamaClientEmbed:
    """OllamaClient.embed calls /api/embed and parses response."""

    def test_embed_newer_format(self) -> None:
        """Newer Ollama: {'embeddings': [[0.1, 0.2, ...]]}."""
        cfg = LLMProviderConfig(provider="ollama")
        client = OllamaClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            vec = client.embed("hello")
        assert vec == [0.1, 0.2, 0.3]

    def test_embed_older_format(self) -> None:
        """Older Ollama: {'embedding': [0.1, 0.2, ...]}."""
        cfg = LLMProviderConfig(provider="ollama")
        client = OllamaClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"embedding": [0.4, 0.5]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            vec = client.embed("world")
        assert vec == [0.4, 0.5]


# ──────────────────────────────────────────────────────────────────────────────
# OpenAICompatibleClient chat tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenAICompatibleClientChat:
    """OpenAICompatibleClient.chat calls /v1/chat/completions."""

    def test_chat_returns_choices_message_content(self) -> None:
        cfg = LLMProviderConfig(
            provider="openai",
            host="https://api.openai.com",
            api_key="sk-test",
        )
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hi from OpenAI!"}}
            ]
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.chat("sys", "usr")
        assert result == "Hi from OpenAI!"

    def test_chat_sends_bearer_auth(self) -> None:
        cfg = LLMProviderConfig(
            provider="openai",
            host="https://api.openai.com",
            api_key="sk-secret",
        )
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp) as m_post:
            client.chat("sys", "usr")

        headers = m_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"
        url = m_post.call_args.args[0]
        assert url == "https://api.openai.com/v1/chat/completions"

    def test_chat_no_api_key_no_auth_header(self) -> None:
        """When api_key is None, no Authorization header is sent."""
        cfg = LLMProviderConfig(provider="openai", host="http://localhost:8080")
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp) as m_post:
            client.chat("s", "u")

        headers = m_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_chat_no_choices_raises(self) -> None:
        cfg = LLMProviderConfig(provider="openai", api_key="sk-x")
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            with pytest.raises(ValueError, match="no choices"):
                client.chat("s", "u")


# ──────────────────────────────────────────────────────────────────────────────
# OpenAICompatibleClient embed tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOpenAICompatibleClientEmbed:
    """OpenAICompatibleClient.embed calls /v1/embeddings."""

    def test_embed_returns_data_embedding(self) -> None:
        cfg = LLMProviderConfig(provider="openai", api_key="sk-x")
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp):
            vec = client.embed("text")
        assert vec == [0.1, 0.2, 0.3]

    def test_embed_url(self) -> None:
        cfg = LLMProviderConfig(
            provider="openai", host="https://api.example.com", api_key="sk"
        )
        client = OpenAICompatibleClient(cfg)

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_resp) as m_post:
            client.embed("text")
        assert m_post.call_args.args[0] == "https://api.example.com/v1/embeddings"
