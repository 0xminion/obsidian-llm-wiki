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
from urllib.parse import urlsplit, urlunsplit

import httpx

from obsidian_llm_wiki.config import LLMProviderConfig
from obsidian_llm_wiki.core.task_models import resolve_task_model

logger = logging.getLogger("obswiki.providers.llm")


def _endpoint_url(host: str, suffix: str) -> str:
    """Build a request/diagnostic URL without propagating URL credentials.

    Provider hosts are configuration, not a secret transport.  Deliberately
    discard userinfo, query parameters, and fragments so diagnostics and
    raised HTTP errors cannot expose credentials embedded in a copied URL.
    """
    parsed = urlsplit(host)
    if not parsed.scheme or not parsed.hostname:
        return f"{host.rstrip('/')}{suffix}"

    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname
    path = f"{parsed.path.rstrip('/')}{suffix}"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


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
        url = _endpoint_url(self.config.host, "/api/chat")
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        # Pass context window and output limit to Ollama via the "options" object.
        # Ollama only reads runtime parameters like num_ctx from the "options"
        # object — a top-level num_ctx is silently ignored.
        #
        # num_predict controls the maximum number of output tokens. Without it,
        # Ollama uses a model-specific default (often as low as 128 or 4096),
        # which truncates large synthesis JSON responses mid-object. For a
        # 200K-char source, the synthesis JSON can easily exceed 20K tokens.
        # Set num_predict to -1 (unlimited) so the model generates until it
        # naturally stops — the timeout is the real safety net.
        options: dict[str, Any] = kwargs.pop("options", {})
        num_ctx = kwargs.pop("num_ctx", None)
        if num_ctx is None and self.config.context_window:
            num_ctx = self.config.context_window
        if num_ctx:
            options.setdefault("num_ctx", num_ctx)
        # Allow generous output tokens — the timeout is the real ceiling.
        # Some Ollama proxies reject num_predict=-1, so use a large positive value.
        options.setdefault("num_predict", 65536)
        if options:
            body["options"] = options
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
        url = _endpoint_url(self.config.host, "/v1/chat/completions")
        body: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": kwargs.pop(
                "max_tokens", 65536,  # 64K output — enough for full synthesis JSON
            ),
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


# ── Provider discovery and preflight ──────────────────────────────────────


def _models_url(config: LLMProviderConfig) -> str:
    return _endpoint_url(config.host, "/v1/models")


def _ollama_tags_url(config: LLMProviderConfig) -> str:
    return _endpoint_url(config.host, "/api/tags")


def _auth_headers(config: LLMProviderConfig) -> dict[str, str]:
    if config.api_key:
        return {"Authorization": f"Bearer {config.api_key}"}
    return {}


def _request_timeout(config: LLMProviderConfig) -> httpx.Timeout:
    return httpx.Timeout(config.timeout_ms / 1000.0)


def _openai_model_ids(payload: Any) -> list[str]:
    """Extract sorted model identifiers from an OpenAI-compatible response."""
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return sorted(
        {
            item["id"].strip()
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
        }
    )


def _ollama_model_names(payload: Any) -> list[str]:
    """Extract sorted model names from an Ollama ``/api/tags`` response."""
    models = payload.get("models", []) if isinstance(payload, dict) else []
    return sorted(
        {
            item["name"].strip()
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()
        }
    )


def list_provider_models(config: LLMProviderConfig) -> list[str]:
    """List models from ``/v1/models``, falling back to Ollama's native tags API.

    Older Ollama installations may not implement the OpenAI-compatible endpoint;
    their ``/api/tags`` response remains a supported, read-only fallback.
    """
    try:
        with httpx.Client(timeout=_request_timeout(config)) as client:
            response = client.get(_models_url(config), headers=_auth_headers(config))
            response.raise_for_status()
            return _openai_model_ids(response.json())
    except httpx.HTTPError:
        if config.provider.lower().strip() != "ollama":
            raise

    with httpx.Client(timeout=_request_timeout(config)) as client:
        response = client.get(_ollama_tags_url(config))
        response.raise_for_status()
        return _ollama_model_names(response.json())


def _configured_task_models(config: LLMProviderConfig) -> dict[str, str]:
    """Return the effective model for the default and each declared task."""
    return {
        "default": config.model,
        "ingest": resolve_task_model(config, "ingest"),
        "maintenance": resolve_task_model(config, "maintenance"),
        "query": resolve_task_model(config, "query"),
        "expand": resolve_task_model(config, "expand"),
    }


def check_provider(config: LLMProviderConfig) -> dict[str, Any]:
    """Return a secret-free, structured provider endpoint/auth/model diagnostic."""
    configured = _configured_task_models(config)
    provider = config.provider.lower().strip()
    endpoint: dict[str, Any] = {"url": _models_url(config), "reachable": False}
    auth_status = "not_required" if provider == "ollama" else (
        "configured" if config.api_key else "missing"
    )
    available: list[str] | None = None

    try:
        available = list_provider_models(config)
        endpoint["reachable"] = True
        endpoint["status_code"] = 200
    except httpx.HTTPStatusError as exc:
        endpoint["reachable"] = True
        endpoint["status_code"] = exc.response.status_code
        endpoint["error"] = f"HTTP {exc.response.status_code}"
        if exc.response.status_code in (401, 403):
            auth_status = "rejected" if config.api_key else "missing"
    except httpx.HTTPError as exc:
        endpoint["error"] = type(exc).__name__
    except (TypeError, ValueError) as exc:
        endpoint["error"] = type(exc).__name__

    model_diagnostics: dict[str, dict[str, Any]] = {}
    for task, model in configured.items():
        if available is None:
            status = "unknown"
            is_available: bool | None = None
        elif model in available:
            status = "available"
            is_available = True
        else:
            status = "not_listed"
            is_available = False
        model_diagnostics[task] = {"name": model, "status": status, "available": is_available}

    ok = (
        endpoint["reachable"]
        and auth_status not in {"missing", "rejected"}
        and all(item["available"] is not False for item in model_diagnostics.values())
    )
    return {
        "ok": ok,
        "provider": provider,
        "endpoint": endpoint,
        "auth": {"configured": bool(config.api_key), "status": auth_status},
        "models": model_diagnostics,
    }


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
    *,
    task: str | None = None,
    **kwargs: Any,
) -> str:
    """Synchronous LLM call with exponential-backoff retry.

    When *messages* is a list of structured message dicts (with role/content),
    the messages are passed directly to the LLM client. When *messages* is a
    string or a flat list without role info, *system* is used as the system
    prompt and *messages* as the user content (legacy behaviour). Pass an
    optional *task* (``ingest``, ``maintenance``, or ``query``) to use its
    configured model override; omitted *task* preserves the unified model.
    """
    llm_config = _resolve_llm_config(config)
    client = create_llm_client(llm_config)
    max_retries = getattr(config, "retry_count", 3)

    # A task-specific model is opt-in, preserving the existing no-task call
    # contract. An explicit ``model=`` still takes precedence for advanced callers.
    resolved_model = (
        resolve_task_model(llm_config, task) if task is not None else llm_config.model
    )

    # Filter kwargs the sync clients don't understand.
    chat_kwargs = {k: v for k, v in kwargs.items() if k not in ("tools", "on_token")}
    chat_kwargs.setdefault("model", resolved_model)

    # Check if messages is a structured list with role dicts.
    is_structured = (
        isinstance(messages, list)
        and messages
        and isinstance(messages[0], dict)
        and "role" in messages[0]
    )
    if is_structured:
        # Structured messages — pass directly to client.
        # The OllamaClient/OpenAICompatibleClient.chat() expects (system, user)
        # but we bypass that and build the request body directly.
        structured_messages = messages
        # Extract system from the structured messages if present.
        sys_content = ""
        user_content = ""
        for msg in structured_messages:
            if msg.get("role") == "system":
                sys_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")
        # If no system in messages, fall back to the system arg (but only if
        # it's not the same as user_content — which happens when the pipeline
        # passes the prompt as both system and in messages).
        if not sys_content and system and system != user_content:
            sys_content = system
        # Use a short system prompt if none was provided.
        if not sys_content:
            sys_content = "You are a helpful assistant."

        for attempt in range(max_retries):
            try:
                return client.chat(sys_content, user_content, **chat_kwargs)
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

    # Legacy path: system + user content
    user = _extract_user_content(messages)

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
