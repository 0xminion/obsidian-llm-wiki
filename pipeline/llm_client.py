"""LLM provider abstraction layer.

Async client for Ollama's OpenAI-compatible endpoint with streaming,
tool-calling, and exponential-backoff retry.

Ported from llm-wiki-compiler/src/utils/llm.ts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx

from pipeline.config import Config

logger = logging.getLogger("llmwiki.llm")


async def call_llm(
    system: str,
    messages: list[dict],
    config: Config,
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    on_token: Callable[[str], Any] | None = None,
) -> str:
    """Send a chat-completion request and return the response text.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint on the
    configured Ollama host.

    Args:
        system: System-level instruction for the model.
        messages: Conversation messages as ``[{"role":"user","content":...},...]``.
        config: Pipeline configuration (host, model, retry params).
        max_tokens: Maximum tokens in the response.
        tools: Optional tool definitions for tool-calling mode.
        on_token: When provided, streaming mode is used and this callback is
            invoked with each content delta token as it arrives.

    Returns:
        The full text content of the response (or accumulated from stream).

    Raises:
        httpx.HTTPError: When all retry attempts are exhausted.
    """
    url = f"{config.ollama_host}/v1/chat/completions"

    # ── Build payload ──────────────────────────────────────────────────
    payload_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    payload_messages.extend(messages)

    body: dict[str, Any] = {
        "model": config.ollama_model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
        "stream": on_token is not None,
    }

    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    timeout = httpx.Timeout(config.ollama_timeout_ms / 1000.0)  # ms → seconds

    # ── Retry loop ─────────────────────────────────────────────────────
    last_error: Exception | None = None

    for attempt in range(1, config.retry_count + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if on_token is not None:
                    # ── Streaming path ─────────────────────────────────
                    body["stream"] = True
                    accumulator: list[str] = []

                    async with client.stream("POST", url, json=body) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            chunk_data = line.removeprefix("data: ").strip()
                            if chunk_data == "[DONE]":
                                break
                            try:
                                parsed = json.loads(chunk_data)
                            except json.JSONDecodeError:
                                continue
                            choices = parsed.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulator.append(content)
                                try:
                                    on_token(content)
                                except Exception:
                                    # User callback errors must not abort streaming.
                                    logger.warning(
                                        "on_token callback raised an exception",
                                        exc_info=True,
                                    )

                    full_response = "".join(accumulator)
                else:
                    # ── Non-streaming path ─────────────────────────────
                    body["stream"] = False
                    resp = await client.post(url, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("LLM response contained no choices")
                    full_response = choices[0]["message"]["content"]

                logger.debug(
                    "LLM call succeeded on attempt %d (model=%s, tokens=%d)",
                    attempt,
                    config.ollama_model,
                    len(full_response.split()),
                )
                return full_response

        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as exc:
            last_error = exc
            if attempt < config.retry_count:
                delay = config.retry_base_ms / 1000.0 * (config.retry_multiplier ** (attempt - 1))
                logger.warning(
                    "LLM call attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt,
                    config.retry_count,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    config.retry_count,
                    exc,
                )

    # All retries exhausted.
    raise httpx.HTTPError(
        f"LLM call failed after {config.retry_count} attempts"
    ) from last_error
