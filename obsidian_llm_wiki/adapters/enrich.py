"""Enrichment adapter — re-exports the legacy enrich module."""

from __future__ import annotations

from pipeline.enrich import EnrichOptions, run_enrichment  # noqa: F401

__all__ = ["EnrichOptions", "run_enrichment"]
