"""CLI subpackage — Typer app and command modules."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="olw",
    help="obsidian-llm-wiki — LLM-powered knowledge compiler for Obsidian vaults.",
    no_args_is_help=True,
)

# Register commands by importing their modules (each decorates `app`).
from obsidian_llm_wiki.cli import build as _build  # noqa: E402, F401
from obsidian_llm_wiki.cli import health as _health  # noqa: E402, F401
from obsidian_llm_wiki.cli import ingest as _ingest  # noqa: E402, F401
from obsidian_llm_wiki.cli import ops as _ops  # noqa: E402, F401
from obsidian_llm_wiki.cli import query as _query  # noqa: E402, F401
from obsidian_llm_wiki.cli import setup as _setup  # noqa: E402, F401
from obsidian_llm_wiki.cli import validate as _validate  # noqa: E402, F401

__all__ = ["app"]
