"""Stable, immutable provenance stamping at ingestion boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance

__all__ = ["stamp_source"]


def stamp_source(
    source: SourceDoc,
    *,
    requested_url: str,
    extractor: str,
    resolved_url: str = "",
    extracted_url: str = "",
    content_type: str = "",
    document_format: str = "",
    diagnostics: tuple[str, ...] = (),
    retrieved_at: str = "",
) -> SourceDoc:
    """Return ``source`` with complete immutable retrieval provenance.

    Existing source-specific facts always win: a specialized extractor knows
    more than a generic registry wrapper.  The wrapper only fills absent fields
    and appends a distinct extractor stage to the chain.
    """
    current = source.provenance
    chain = current.extractor_chain
    if extractor and extractor not in chain:
        chain = (*chain, extractor)
    final_url = source.url or ""
    provenance = SourceProvenance(
        requested_url=current.requested_url or requested_url,
        resolved_url=current.resolved_url or resolved_url or final_url or requested_url,
        extracted_url=current.extracted_url or extracted_url or final_url or requested_url,
        extractor_chain=chain,
        content_type=current.content_type or content_type,
        document_format=current.document_format or document_format,
        retrieved_at=current.retrieved_at
        or retrieved_at
        or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        content_sha256=current.content_sha256
        or hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
        diagnostics=(*current.diagnostics, *diagnostics),
    )
    return replace(source, provenance=provenance)
