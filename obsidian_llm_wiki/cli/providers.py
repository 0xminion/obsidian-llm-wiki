"""``olw providers`` — read-only LLM provider preflight and model discovery."""

from __future__ import annotations

import json
from typing import Any

import httpx
import typer

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.providers.llm import check_provider, list_provider_models

providers_app = typer.Typer(
    help="Inspect the configured LLM provider without sending a chat completion.",
    no_args_is_help=True,
)


def _emit(payload: dict[str, Any]) -> None:
    """Emit deterministic machine-readable provider diagnostics."""
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


@providers_app.command("check")
def check() -> None:
    """Check configured endpoint, authentication state, and effective task models."""
    config = load_config().llm
    diagnostic = check_provider(config)
    _emit(diagnostic)
    if not diagnostic["ok"]:
        raise typer.Exit(code=1)


@providers_app.command("models")
def models() -> None:
    """List models exposed by the configured provider's models endpoint."""
    config = load_config().llm
    try:
        available = list_provider_models(config)
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        _emit(
            {
                "error": type(exc).__name__,
                "models": [],
                "provider": config.provider.lower().strip(),
            }
        )
        raise typer.Exit(code=1) from exc
    _emit({"models": sorted(set(available)), "provider": config.provider.lower().strip()})
