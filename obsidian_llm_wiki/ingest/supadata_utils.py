"""Supadata API shared utilities — rate limiting, usage tracking, key validation.

This module centralises Supadata API management used by both the YouTube
and podcast extractors:

  - **Rate limiter**: 3-second delay between API calls (module-level state).
  - **Usage tracking**: persists call counts to ``.llmwiki/supadata_usage.json``.
  - **Key validation**: lightweight API call to verify the key is valid.
  - **429 handling**: logs warnings when approaching rate limits.

Usage::

    from obsidian_llm_wiki.ingest.supadata_utils import (
        supadata_rate_limit,
        track_supadata_call,
        validate_supadata_key,
    )

    supadata_rate_limit()
    resp = client.get(url, headers=...)
    track_supadata_call(resp)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("obswiki.ingest.supadata_utils")

__all__ = [
    "supadata_rate_limit",
    "track_supadata_call",
    "validate_supadata_key",
    "get_supadata_usage",
    "reset_rate_limiter",
    "SUPADATA_RATE_LIMIT_SECONDS",
]

# ── Rate limiter ───────────────────────────────────────────────────────────

SUPADATA_RATE_LIMIT_SECONDS = 3.0

# Module-level last call timestamp — shared across all callers.
_last_call_time: float = 0.0


def reset_rate_limiter() -> None:
    """Reset the rate limiter state (for testing)."""
    global _last_call_time
    _last_call_time = 0.0


def supadata_rate_limit() -> None:
    """Enforce a minimum delay between Supadata API calls.

    Sleeps if the last call was less than ``SUPADATA_RATE_LIMIT_SECONDS`` ago.
    Updates ``_last_call_time`` after sleeping (or immediately if enough time
    has elapsed).
    """
    global _last_call_time
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < SUPADATA_RATE_LIMIT_SECONDS:
        sleep_duration = SUPADATA_RATE_LIMIT_SECONDS - elapsed
        logger.debug(
            "Supadata rate limit: sleeping %.1fs", sleep_duration,
        )
        time.sleep(sleep_duration)
    _last_call_time = time.monotonic()


# ── Usage tracking ─────────────────────────────────────────────────────────


def _get_usage_file() -> Path:
    """Resolve the supadata usage JSON file path from the vault."""
    vault_path = os.environ.get("VAULT_PATH", "")
    if vault_path:
        return (
            Path(vault_path).expanduser().resolve()
            / "04-Wiki" / ".llmwiki" / "supadata_usage.json"
        )
    # Fallback: home directory
    return Path.home() / ".llmwiki" / "supadata_usage.json"


def _load_usage() -> dict[str, Any]:
    """Load the current usage data from disk."""
    usage_file = _get_usage_file()
    if not usage_file.exists():
        return {}
    try:
        return json.loads(usage_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_usage(data: dict[str, Any]) -> None:
    """Persist usage data to disk."""
    usage_file = _get_usage_file()
    usage_file.parent.mkdir(parents=True, exist_ok=True)
    usage_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def track_supadata_call(response: httpx.Response) -> None:
    """Track a Supadata API response — updates call counts and handles 429s.

    Args:
        response: The httpx.Response from a Supadata API call.
    """
    today = date.today().strftime("%Y-%m-%d")
    data = _load_usage()

    # Reset if date changed
    if data.get("date") != today:
        data = {
            "date": today,
            "calls_made": 0,
            "calls_remaining": None,
            "last_reset": today,
        }

    data["calls_made"] = data.get("calls_made", 0) + 1

    # Try to extract remaining calls from response headers or body
    remaining = response.headers.get("x-ratelimit-remaining")
    if remaining is not None:
        with contextlib.suppress(ValueError, TypeError):
            data["calls_remaining"] = int(remaining)

    # If response body has usage info, extract it
    if response.status_code == 200:
        try:
            body = response.json()
            if isinstance(body, dict) and "credits_remaining" in body:
                data["calls_remaining"] = body.get("credits_remaining")
        except (json.JSONDecodeError, ValueError):
            pass

    # Handle 429 rate limit
    if response.status_code == 429:
        logger.warning(
            "Supadata API rate limit hit (429). "
            "Consider reducing call frequency or upgrading your plan."
        )
        retry_after = response.headers.get("retry-after")
        if retry_after:
            with contextlib.suppress(ValueError):
                data["retry_after_seconds"] = int(retry_after)

    _save_usage(data)


def get_supadata_usage() -> dict[str, Any]:
    """Get the current Supadata usage data.

    Returns:
        Dict with keys: date, calls_made, calls_remaining, last_reset.
    """
    return _load_usage()


# ── Key validation ─────────────────────────────────────────────────────────


def validate_supadata_key(api_key: str | None = None) -> bool:
    """Validate a Supadata API key with a lightweight API call.

    Makes a minimal request to the Supadata API. If the key is valid,
    returns True. If invalid (401/403) or the call fails, returns False.

    Args:
        api_key: The API key to validate. If None, reads from SUPADATA_API_KEY env.

    Returns:
        True if the key is valid, False otherwise.
    """
    if api_key is None:
        api_key = os.environ.get("SUPADATA_API_KEY", "").strip()
    if not api_key:
        logger.warning("No SUPADATA_API_KEY set — cannot validate.")
        return False

    supadata_rate_limit()
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                headers={
                    "x-api-key": api_key,
                    "Accept": "application/json",
                },
            )
        track_supadata_call(resp)
        # 200, 202, 206 = key is valid (transcript may or may not be available)
        # 401, 403 = key is invalid
        # 429 = key is valid but rate-limited
        if resp.status_code in (200, 202, 206, 429):
            return True
        if resp.status_code in (401, 403):
            logger.warning(
                "Supadata API key is invalid (HTTP %d). "
                "Check your key at https://dash.supadata.ai/organizations/api-key",
                resp.status_code,
            )
            return False
        logger.debug("Supadata key validation returned HTTP %d", resp.status_code)
        return False
    except Exception as exc:
        logger.warning("Supadata key validation failed: %s", exc)
        return False
