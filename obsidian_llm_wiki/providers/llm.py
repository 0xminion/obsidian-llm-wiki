"""LLM provider abstraction — Ollama and OpenAI-compatible clients.

Clean port of the legacy ``pipeline.llm.providers`` module.  Provides
``call_llm`` (sync) and ``acall_llm`` (async) with exponential-backoff retry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from obsidian_llm_wiki.config import LLMProviderConfig

logger = logging.getLogger("obswiki.providers.llm")


# ── Abstract base ───────────────────────────────────────────────────────


class LLMClient(ABC):
    """Abstract LLM client with chat capability."""

    @abstractmethod
    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Send a chat-completion request and return the response text."""


# ── Ollama ──────────────────────────────────────────────────────────────


class OllamaClient(LLMClient):
    """Synchronous client for Ollama's native ``/api/chat`` API."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self._timeout = httpx.Timeout(config.timeout_ms / 1000.0)

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        url = f"{self.config.host.rstrip('/')}/api/chat"
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        # Pass context window to Ollama as num_ctx if not explicitly overridden.
        if "num_ctx" not in kwargs and self.config.context_window:
            kwargs["num_ctx"] = self.config.context_window
        body.update(kwargs)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]


# ── OpenAI-compatible ───────────────────────────────────────────────────


class OpenAICompatibleClient(LLMClient):
    """Synchronous client for any OpenAI-compatible ``/v1/chat/completions`` API."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self._timeout = httpx.Timeout(config.timeout_ms / 1000.0)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        url = f"{self.config.host.rstrip('/')}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": kwargs.pop("max_tokens", 8192),
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


# ── Factory ─────────────────────────────────────────────────────────────


def create_llm_client(config: LLMProviderConfig) -> LLMClient:
    """Return the right LLMClient for ``config.provider``."""
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(config)
    if provider in ("openai", "openai-compatible", "openai_compatible"):
        return OpenAICompatibleClient(config)
    raise ValueError(
        f"Unknown LLM provider: {config.provider!r}. Supported: 'ollama', 'openai'."
    )


# ── Convenience wrappers with retry ─────────────────────────────────────


def _resolve_llm_config(config: Any) -> LLMProviderConfig:
    """Extract LLMProviderConfig from a Config or pass through."""
    llm = getattr(config, "llm", None)
    if llm is not None and hasattr(llm, "provider"):
        return llm
    if hasattr(config, "provider"):
        return config  # type: ignore[return-value]
    raise TypeError(f"Cannot derive LLMProviderConfig from {type(config)!r}")


def _extract_user_content(messages: list[dict] | str) -> str:
    """Extract user message content from a messages list or plain string."""
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "user":
            parts.append(msg.get("content", ""))
    if not parts:
        parts = [msg.get("content", "") for msg in messages]
    return "\n\n".join(parts)


def call_llm(
    system: str,
    messages: list[dict] | str,
    config: Any,
    **kwargs: Any,
) -> str:
    """Synchronous LLM call with exponential-backoff retry."""
    llm_config = _resolve_llm_config(config)
    client = create_llm_client(llm_config)
    max_retries = getattr(config, "retry_count", 3)
    user = _extract_user_content(messages)

    # Filter kwargs the sync clients don't understand.
    chat_kwargs = {k: v for k, v in kwargs.items() if k not in ("tools", "on_token")}

    for attempt in range(max_retries):
        try:
            return client.chat(system, user, **chat_kwargs)
        except Exception:
            if attempt == max_retries - 1:
                raise
            delay = (
                getattr(config, "retry_base_ms", 1000)
                * (getattr(config, "retry_multiplier", 4) ** attempt)
            ) / 1000.0
            logger.warning(
                "LLM call attempt %d/%d failed. Retrying in %.1fs…",
                attempt + 1, max_retries, delay,
            )
            time.sleep(delay)

    raise RuntimeError("LLM call exhausted retries")  # pragma: no cover


async def acall_llm(
    system: str,
    messages: list[dict] | str,
    config: Any,
    **kwargs: Any,
) -> str:
    """Async wrapper around ``call_llm`` via ``asyncio.to_thread``."""
    return await asyncio.to_thread(call_llm, system, messages, config, **kwargs)
