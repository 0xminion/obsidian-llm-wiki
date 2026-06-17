"""LLM provider abstraction layer.

Concrete implementations for Ollama's native API and any OpenAI-compatible
endpoint.  Use ``create_llm_client`` to get the right client for a given
:class:`~pipeline.config.LLMProviderConfig`.

This module is additive to the existing ``pipeline.llm_client`` — the old
async helper is preserved for backward compatibility while new code should
prefer the ``LLMClient`` interface defined here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from pipeline.config import LLMProviderConfig

logger = logging.getLogger("llmwiki.llm.providers")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────


class LLMClient(ABC):
    """Abstract LLM client with chat + embed capabilities.

    Implementations should be synchronous (httpx blocking calls).  Callers
    that need async can wrap calls in ``asyncio.to_thread``.
    """

    @abstractmethod
    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Send a chat-completion request and return the response text.

        Args:
            system: System-level instruction.
            user: User message content.
            **kwargs: Provider-specific overrides (e.g. ``max_tokens``).

        Returns:
            The full text content of the model response.
        """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """


# ──────────────────────────────────────────────────────────────────────────────
# Ollama client (native /api/chat + /api/embed)
# ──────────────────────────────────────────────────────────────────────────────


class OllamaClient(LLMClient):
    """Synchronous client for Ollama's native API.

    Uses ``/api/chat`` for chat completions and ``/api/embed`` for embeddings.
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self._timeout = httpx.Timeout(config.timeout_ms / 1000.0)

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Call Ollama /api/chat and return ``message.content``."""
        url = f"{self.config.host.rstrip('/')}/api/chat"
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        # Merge remaining kwargs (e.g. options)
        body.update(kwargs)

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        return data["message"]["content"]

    def embed(self, text: str) -> list[float]:
        """Call Ollama /api/embed and return the embedding vector.

        Handles both ``embeddings[0]`` (newer Ollama) and ``embedding`` (older)
        response keys.
        """
        url = f"{self.config.host.rstrip('/')}/api/embed"
        body: dict[str, Any] = {
            "model": self.config.embed_model,
            "input": text,
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        # Newer Ollama: {"embeddings": [[...]]}; older: {"embedding": [...]}
        if "embeddings" in data:
            embeddings = data["embeddings"]
            if embeddings and isinstance(embeddings, list):
                return list(embeddings[0])
            return []
        if "embedding" in data:
            return list(data["embedding"])

        raise ValueError(f"Unexpected Ollama embed response shape: {list(data.keys())}")


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client
# ──────────────────────────────────────────────────────────────────────────────


class OpenAICompatibleClient(LLMClient):
    """Synchronous client for any OpenAI-compatible chat/embeddings API.

    Uses ``/v1/chat/completions`` and ``/v1/embeddings``.  Authenticates via
    ``Authorization: Bearer {api_key}``.
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self._timeout = httpx.Timeout(config.timeout_ms / 1000.0)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Call /v1/chat/completions and return ``choices[0].message.content``."""
        url = f"{self.config.host.rstrip('/')}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": False,
        }
        body.update(kwargs)

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI-compatible response contained no choices")
        return choices[0]["message"]["content"]

    def embed(self, text: str) -> list[float]:
        """Call /v1/embeddings and return the embedding vector."""
        url = f"{self.config.host.rstrip('/')}/v1/embeddings"
        body: dict[str, Any] = {
            "model": self.config.embed_model,
            "input": text,
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=body, headers=self._timeout_headers())
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("data", [])
        if not embeddings:
            raise ValueError("OpenAI-compatible embeddings response had no data")
        return list(embeddings[0]["embedding"])

    def _timeout_headers(self) -> dict[str, str]:
        """Headers for embed requests (same auth as chat)."""
        return self._headers()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def create_llm_client(config: LLMProviderConfig) -> LLMClient:
    """Factory: return the right LLMClient for ``config.provider``.

    Args:
        config: LLM provider configuration.

    Returns:
        An ``OllamaClient`` or ``OpenAICompatibleClient`` instance.

    Raises:
        ValueError: If ``config.provider`` is not 'ollama' or 'openai'.
    """
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(config)
    if provider in ("openai", "openai-compatible", "openai_compatible"):
        return OpenAICompatibleClient(config)
    raise ValueError(
        f"Unknown LLM provider: {config.provider!r}. "
        "Supported: 'ollama', 'openai'."
    )
