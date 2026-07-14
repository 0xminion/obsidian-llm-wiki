"""Deterministic source-quote verification for claim evidence."""

from __future__ import annotations

import hashlib

from obsidian_llm_wiki.core.models import (
    EvidenceSpan,
    EvidenceVerification,
    SourceDoc,
    SourceSynthesis,
)

__all__ = ["resolve_quote", "resolve_synthesis_evidence"]


def resolve_quote(quote: str, source_content: str, source_file: str = "") -> EvidenceSpan:
    """Resolve an exact quote in one source without guessing a location.

    A quote is verified only if it occurs exactly once.  Ambiguous and unmatched
    quotes retain their source identity and hash for auditability, but carry no
    offsets so downstream rendering cannot present invented evidence anchors.
    Offsets are Unicode character indexes in the decoded UTF-8 source string.
    """
    content = source_content if isinstance(source_content, str) else ""
    safe_quote = quote if isinstance(quote, str) else ""
    source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    base = {
        "quote": safe_quote,
        "source_file": source_file if isinstance(source_file, str) else "",
        "source_hash": source_hash,
    }
    if not safe_quote:
        return EvidenceSpan(**base, verification=EvidenceVerification.UNMATCHED)

    first = content.find(safe_quote)
    if first < 0:
        return EvidenceSpan(**base, verification=EvidenceVerification.UNMATCHED)
    # Advance one character to count overlapping occurrences too.
    if content.find(safe_quote, first + 1) >= 0:
        return EvidenceSpan(**base, verification=EvidenceVerification.AMBIGUOUS)
    return EvidenceSpan(
        **base,
        start_offset=first,
        end_offset=first + len(safe_quote),
        verification=EvidenceVerification.VERIFIED,
    )


def resolve_synthesis_evidence(
    synthesis: SourceSynthesis,
    source: SourceDoc,
    source_file: str = "",
) -> SourceSynthesis:
    """Resolve every quoted claim in ``synthesis`` against its actual source.

    Claims created before claim-level evidence remain untouched.  The synthesis
    itself is returned for convenient use at the parse/cache pipeline boundary.
    """
    filename = source_file or source.source_file or synthesis.source_file
    for concept in synthesis.concepts:
        for claim in concept.claims:
            if claim.evidence is not None and claim.evidence.quote:
                claim.evidence = resolve_quote(claim.evidence.quote, source.content, filename)
    return synthesis
