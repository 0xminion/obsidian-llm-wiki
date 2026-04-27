"""CLI package — Typer commands split by domain.

All public symbols are re-exported here for backward compatibility.
Tests that ``from pipeline.cli import app`` or ``patch("pipeline.cli.extract_all")``
continue to work unchanged.
"""

from __future__ import annotations

# ── Shared helpers (must load first — defines ``app``) ──────────────────────
from pipeline.cli._helpers import (  # noqa: F401
    PipelineLock,
    _auto_setup,
    _build_query_prompt,
    _collect_clipping_files,
    _collect_url_files,
    _collision_safe_path,
    _gather_query_note_context,
    _load_cfg,
    _query_keywords,
    _resolve_vault,
    _setup_logging,
    _validate_clipping_quality,
    app,
    check_dependencies,
    query_vault_fast,
)

# Re-export top-level pipeline functions so ``patch("pipeline.cli.extract_all")`` works.
from pipeline.create import create_all, create_file_templates  # noqa: F401
from pipeline.extract import extract_all  # noqa: F401
from pipeline.models import ExtractedSource, Manifest, Plans, SourceType  # noqa: F401
from pipeline.plan import plan_sources  # noqa: F401
from pipeline.vault import reindex as vault_reindex  # noqa: F401

# Also re-export shutil so ``patch("pipeline.cli.shutil.which")`` keeps working.
import shutil  # noqa: F401

# ── Register command modules (side-effect imports) ──────────────────────────
from pipeline.cli import ingest as ingest  # noqa: F401
from pipeline.cli import compile_cmd as compile_cmd  # noqa: F401
from pipeline.cli import review_cmd as review_cmd  # noqa: F401
from pipeline.cli import quality as quality  # noqa: F401
from pipeline.cli import manage as manage  # noqa: F401


__all__ = [
    "ingest",
    "compile_cmd",
    "review_cmd",
    "quality",
    "manage",
    "app",
    "main",
    "PipelineLock",
    "_auto_setup",
    "_build_query_prompt",
    "_collect_clipping_files",
    "_collect_url_files",
    "_collision_safe_path",
    "_gather_query_note_context",
    "_load_cfg",
    "_query_keywords",
    "_resolve_vault",
    "_setup_logging",
    "_validate_clipping_quality",
    "check_dependencies",
    "query_vault_fast",
    "create_all",
    "create_file_templates",
    "extract_all",
    "plan_sources",
    "vault_reindex",
    "shutil",
    "ExtractedSource",
    "Manifest",
    "Plans",
    "SourceType",
]


def main():
    app()
