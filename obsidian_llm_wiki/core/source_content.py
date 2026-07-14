"""Canonical UTF-8 source-content bounds and provenance normalization."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from obsidian_llm_wiki.core.models import SourceDoc


def bound_source_content(source: SourceDoc, maximum_bytes: int) -> tuple[SourceDoc, bool]:
    """Return a UTF-8 byte-bounded source with provenance matching its body."""
    encoded = source.content.encode("utf-8")
    if maximum_bytes >= len(encoded):
        return source, False
    bounded = encoded[:max(maximum_bytes, 0)].decode("utf-8", errors="ignore")
    provenance = replace(
        source.provenance,
        content_sha256=hashlib.sha256(bounded.encode("utf-8")).hexdigest(),
        diagnostics=(
            *source.provenance.diagnostics,
            f"content truncated to {len(bounded.encode('utf-8'))} UTF-8 bytes "
            f"(limit {maximum_bytes})",
        ),
    )
    return replace(source, content=bounded, provenance=provenance), True
