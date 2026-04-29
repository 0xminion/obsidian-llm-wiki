"""Unified LLM client supporting multiple providers: Ollama, OpenRouter, Hermes.

Usage:
    from pipeline.llm_client import get_llm_client
    client = get_llm_client(cfg)
    text = client.generate("Summarize this...")
    embedding = client.embed("concept text")

Providers:
  - ollama:    Local Ollama server (default). Supports generate + embed.
  - openrouter: Cloud API (OpenAI-compatible). Supports generate only.
  - hermes:     Hermes subprocess (agentic). Supports generate only; slow.

Environment overrides (used when cfg values are empty):
  LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TIMEOUT
  EMBED_MODEL, EMBED_BASE_URL, OLLAMA_HOST
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class LLMGenerationError(Exception):
    """Raised when LLM generation fails and raise_on_error=True."""
    pass


@dataclass
class LLMResponse:
    text: str = ""
    error: str = ""
    success: bool = False


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object or array from raw text.

    Handles markdown code blocks (```json ... ```) and bare JSON.
    """
    if not text:
        return None

    # Strip markdown code fences with optional language tag
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove first fence line
        stripped = re.sub(r"^```(?:\w+)?\n?", "", stripped, count=1)
        # Remove trailing fence
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        stripped = stripped.strip()

    # Fast path: try the stripped text first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Find the first JSON array (greedy, anchored from first '[' to last ']')
    arr_start = stripped.find("[")
    if arr_start >= 0:
        # Greedy scan: find matching ']' by bracket depth
        depth = 0
        for i, ch in enumerate(stripped[arr_start:], start=arr_start):
            if ch == "[" and (i == 0 or stripped[i - 1] != "\\"):
                depth += 1
            elif ch == "]" and (i == 0 or stripped[i - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[arr_start : i + 1])
                    except json.JSONDecodeError:
                        break
                    break

    # Find the first JSON object (greedy, from first '{' to matching '}')
    obj_start = stripped.find("{")
    if obj_start >= 0:
        depth = 0
        for i, ch in enumerate(stripped[obj_start:], start=obj_start):
            if ch == "{" and (i == 0 or stripped[i - 1] != "\\"):
                depth += 1
            elif ch == "}" and (i == 0 or stripped[i - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[obj_start : i + 1])
                    except json.JSONDecodeError:
                        break
                    break

    return None


def _validate_schema(data, schema: type[T]) -> T | None:
    """Validate data against a dataclass schema (or Pydantic if available).

    Handles special cases:
    - Raw list -> wraps into dict with the schema's list field name
    - Recursive validation of list items against their element type annotations.
    """
    if not data:
        return None
    # Auto-wrap raw list into dict for list-bearing schemas
    if isinstance(data, list):
        if hasattr(schema, "__dataclass_fields__"):
            fields = schema.__dataclass_fields__
            list_fields = [
                f for f, v in fields.items()
                if (v.default_factory is list or "list" in str(v.type).lower())
            ]
            if len(fields) == 1 and len(list_fields) == 1:
                data = {list_fields[0]: data}
            elif "plans" in fields:
                data = {"plans": data}
        else:
            return None
    if not isinstance(data, dict):
        return None
    # Pydantic fallback
    try:
        import pydantic
        if issubclass(schema, pydantic.BaseModel):
            return schema(**data)  # type: ignore[return-value]
    except Exception:
        pass
    # Dataclass path (preferred — recursive)
    if not hasattr(schema, "__dataclass_fields__"):
        return None
    try:
        sig = inspect.signature(schema)
        kwargs: dict = {}
        import typing
        hints = typing.get_type_hints(schema)
        for param in sig.parameters.values():
            if param.name in data:
                raw_val = data[param.name]
                # Recursively validate list items if annotation is List[X] where X is a dataclass
                if isinstance(raw_val, list) and param.name in hints:
                    list_type = hints[param.name]
                    args = typing.get_args(list_type)
                    if args and len(args) == 1:
                        inner_type = args[0]
                        if hasattr(inner_type, "__dataclass_fields__"):
                            validated_items = []
                            for item in raw_val:
                                if isinstance(item, inner_type):
                                    validated_items.append(item)
                                elif isinstance(item, dict):
                                    validated_item = _validate_schema(item, inner_type)
                                    if validated_item is not None:
                                        validated_items.append(validated_item)
                            kwargs[param.name] = validated_items
                        else:
                            kwargs[param.name] = raw_val
                    else:
                        kwargs[param.name] = raw_val
                else:
                    kwargs[param.name] = raw_val
            elif param.default is not inspect.Parameter.empty:
                continue
            else:
                log.debug("Missing required field '%s' for %s", param.name, schema.__name__)
                return None
        return schema(**kwargs)  # type: ignore[return-value]
    except Exception as e:
        log.debug("Dataclass validation failed for %s: %s", schema.__name__, e)
        return None


class BaseProvider:
    """Abstract base for LLM providers."""

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> LLMResponse:
        raise NotImplementedError

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> T | None:
        """Generate structured output validated against a dataclass or Pydantic schema.

        Subclasses should override this for native structured-output support.
        The default implementation falls back to plain text with JSON extraction.
        """
        resp = self.generate(prompt, model, timeout, api_key, base_url)
        if not resp.success or not resp.text:
            return None
        data = _extract_json(resp.text)
        if data is None:
            return None
        return _validate_schema(data, schema)

    def embed(
        self,
        text: str,
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> list[float] | None:
        raise NotImplementedError

    def embed_batch(
        self,
        texts: list[str],
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> dict[str, list[float]]:
        raise NotImplementedError


class OllamaProvider(BaseProvider):
    """Local Ollama server. Supports generation and embeddings."""

    def _url(self, base_url: str) -> str:
        return (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> LLMResponse:
        url = self._url(base_url)
        try:
            req = urllib.request.Request(
                f"{url}/api/generate",
                data=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 131072,  # 128K context window default
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return LLMResponse(text=data.get("response", "").strip(), success=True)
        except Exception as e:
            log.debug("Ollama generate failed: %s", e)
            return LLMResponse(error=str(e))

    def embed(
        self,
        text: str,
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> list[float] | None:
        url = self._url(base_url)
        try:
            req = urllib.request.Request(
                f"{url}/api/embeddings",
                data=json.dumps({
                    "model": model,
                    "prompt": text[:4000],
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embedding = data.get("embedding")
                if embedding and len(embedding) > 0:
                    return embedding
        except Exception as e:
            log.debug("Ollama embed failed: %s", e)
        return None

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> T | None:
        url = self._url(base_url)
        try:
            req = urllib.request.Request(
                f"{url}/api/generate",
                data=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 131072,  # 128K context window default
                    },
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw = data.get("response", "").strip()
                parsed = _extract_json(raw)
                if parsed is None:
                    return None
                return _validate_schema(parsed, schema)
        except Exception as e:
            log.debug("Ollama structured generate failed: %s", e)
            return None

    def embed_batch(
        self,
        texts: list[str],
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> dict[str, list[float]]:
        url = self._url(base_url)
        if not texts:
            return {}

        payload = {
            "model": model,
            "input": [t[:4000] for t in texts],
        }
        try:
            req = urllib.request.Request(
                f"{url}/api/embed",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) == len(texts):
                    return {text: emb for text, emb in zip(texts, embeddings) if emb}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.warning(
                    "Ollama /api/embed not available (404). Your Ollama version may be too old. "
                    "Upgrade Ollama to >=0.1.48 or switch to QMD MCP by setting USE_QMD_MCP=true."
                )
            else:
                log.warning("Ollama batch embed HTTP error: %s", e)
        except Exception as e:
            log.warning("Ollama batch embed error: %s", e)
        return {}


class OpenRouterProvider(BaseProvider):
    """OpenRouter cloud API (OpenAI-compatible). Generation only."""

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> LLMResponse:
        if not api_key:
            return LLMResponse(error="OpenRouter API key not configured (set LLM_API_KEY)")
        if not model:
            return LLMResponse(error="OpenRouter requires a model (set LLM_MODEL)")

        url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        try:
            req = urllib.request.Request(
                f"{url}/chat/completions",
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/0xminion/obsidian-llm-wiki",
                    "X-Title": "Obsidian LLM Wiki",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    return LLMResponse(text=text, success=True)
                return LLMResponse(error="Empty response from OpenRouter")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(e)
            return LLMResponse(error=f"OpenRouter HTTP {e.code}: {body[:200]}")
        except Exception as e:
            log.debug("OpenRouter generate failed: %s", e)
            return LLMResponse(error=str(e))

    def embed(
        self,
        text: str,
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> list[float] | None:
        log.warning("OpenRouter embedding not supported; use Ollama for embeddings or set embed_provider=ollama")
        return None

    def embed_batch(
        self,
        texts: list[str],
        model: str,
        timeout: int,
        base_url: str = "",
    ) -> dict[str, list[float]]:
        log.warning("OpenRouter embedding not supported; use Ollama for embeddings or set embed_provider=ollama")
        return {}

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> T | None:
        if not api_key:
            log.debug("OpenRouter API key not configured")
            return None
        if not model:
            log.debug("OpenRouter requires a model")
            return None

        url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        try:
            req = urllib.request.Request(
                f"{url}/chat/completions",
                data=json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/0xminion/obsidian-llm-wiki",
                    "X-Title": "Obsidian LLM Wiki",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    parsed = _extract_json(text)
                    if parsed is None:
                        return None
                    return _validate_schema(parsed, schema)
                return None
        except Exception as e:
            log.debug("OpenRouter structured generate failed: %s", e)
            return None


class HermesProvider(BaseProvider):
    """Hermes subprocess — agentic, supports tool use. Slow."""

    def generate(
        self,
        prompt: str,
        model: str,
        timeout: int,
        api_key: str = "",
        base_url: str = "",
    ) -> LLMResponse:
        try:
            result = subprocess.run(
                [(base_url or "hermes"), "chat", "-q", prompt, "-Q"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return LLMResponse(text=result.stdout.strip(), success=True)
            return LLMResponse(error=f"Hermes exit {result.returncode}: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            return LLMResponse(error="Hermes timed out")
        except FileNotFoundError:
            return LLMResponse(error="Hermes command not found")
        except Exception as e:
            return LLMResponse(error=str(e))

    def embed(self, text: str, model: str, timeout: int, base_url: str = "") -> list[float] | None:
        return None

    def embed_batch(self, texts: list[str], model: str, timeout: int, base_url: str = "") -> dict[str, list[float]]:
        return {}


@dataclass
class LLMClient:
    """Provider-agnostic LLM client.

    Attributes:
        provider:   Provider name — "ollama", "openrouter", "hermes".
        model:      Generation model name. If empty, provider default is used.
        api_key:    API key for cloud providers.
        base_url:   Override base URL for the provider.
        timeout:    Default request timeout in seconds.
        embed_model: Embedding model (primarily for Ollama).
        embed_base_url: Override base URL for embeddings.
    """

    provider: str = "ollama"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: int = 60
    embed_model: str = "qwen3-embedding:0.6b"
    embed_base_url: str = ""
    agent_cmd: str = "hermes"

    def __post_init__(self):
        self._provider_impl = self._resolve_provider()

    def _resolve_provider(self) -> BaseProvider:
        p = (self.provider or "ollama").lower()
        if p == "ollama":
            return OllamaProvider()
        if p == "openrouter":
            return OpenRouterProvider()
        if p == "hermes":
            return HermesProvider()
        log.warning("Unknown provider '%s', falling back to ollama", p)
        return OllamaProvider()

    def generate(self, prompt: str, model: Optional[str] = None, timeout: Optional[int] = None, raise_on_error: bool = False) -> str:
        """Generate text. Returns empty string on any failure.

        Set raise_on_error=True to raise LLMGenerationError instead of returning empty string.
        """
        m = model or self.model
        t = timeout or self.timeout
        if not m and self.provider == "ollama":
            m = os.environ.get("OLLAMA_INSIGHT_MODEL", "minimax-m2.7:cloud")
        provider_base = self.agent_cmd if self.provider == "hermes" else self.base_url
        resp = self._provider_impl.generate(prompt, m, t, self.api_key, provider_base)
        if resp.success:
            return resp.text
        if raise_on_error:
            raise LLMGenerationError(f"{self.provider} generation failed: {resp.error}")
        log.warning("LLM generate failed (%s): %s", self.provider, resp.error)
        return ""

    def generate_or_raise(self, prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """Generate text, raising LLMGenerationError on failure."""
        return self.generate(prompt, model=model, timeout=timeout, raise_on_error=True)

    def embed(self, text: str, model: Optional[str] = None, timeout: Optional[int] = None) -> list[float] | None:
        """Embed a single text. Returns None on failure."""
        m = model or self.embed_model
        t = timeout or self.timeout
        url = self.embed_base_url or self.base_url
        return self._provider_impl.embed(text, m, t, url)

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> T | None:
        """Generate structured output and validate against a dataclass schema.

        Falls back to plain-text JSON extraction for providers without native support.
        """
        m = model or self.model
        t = timeout or self.timeout
        provider_base = self.agent_cmd if self.provider == "hermes" else self.base_url
        return self._provider_impl.generate_structured(
            prompt, schema, m, t, self.api_key, provider_base
        )

    def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, list[float]]:
        """Batch embed texts. Returns {text: embedding} for successes."""
        m = model or self.embed_model
        t = timeout or self.timeout
        url = self.embed_base_url or self.base_url
        return self._provider_impl.embed_batch(texts, m, t, url)

    def generate_parallel(
        self,
        prompts: list[tuple[str, str]],
        max_workers: int = 4,
        timeout: Optional[int] = None,
    ) -> dict[str, str]:
        """Generate for multiple prompts in parallel.

        Args:
            prompts: List of (key, prompt) tuples.

        Returns:
            Dict mapping key -> generated text (empty values omitted).
        """
        results: dict[str, str] = {}

        def _gen_one(key: str, prompt: str) -> tuple[str, str]:
            return key, self.generate(prompt, timeout=timeout)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_gen_one, k, p): k for k, p in prompts}
            for future in as_completed(futures):
                key, text = future.result()
                if text:
                    results[key] = text
        return results


def get_llm_client(cfg) -> LLMClient:
    """Create an LLMClient from a Config object.

    Falls back to legacy Ollama env vars when new LLM_* settings are empty.
    """
    provider = cfg.llm_provider or "ollama"

    # Model resolution
    model = cfg.llm_model
    if not model and provider == "ollama":
        model = cfg.ollama_insight_model or "minimax-m2.7:cloud"
    # For openrouter, user MUST set LLM_MODEL; we don't guess.

    # Base URL resolution
    base_url = cfg.llm_base_url
    if not base_url and provider == "ollama":
        base_url = cfg.ollama_host

    # Embedding base URL
    embed_url = cfg.embed_base_url
    if not embed_url:
        embed_url = cfg.ollama_host

    return LLMClient(
        provider=provider,
        model=model,
        api_key=cfg.llm_api_key,
        base_url=base_url,
        timeout=cfg.llm_timeout,
        embed_model=cfg.embed_model or "qwen3-embedding:0.6b",
        embed_base_url=embed_url,
        agent_cmd=getattr(cfg, "agent_cmd", "hermes") or "hermes",
    )
