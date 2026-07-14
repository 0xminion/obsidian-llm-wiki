"""Tests for claim preservation and evidence rendering fixes.

Fix 1: Pass 2 expansion unions claims with Pass 1 instead of replacing.
Fix 2: Unverified claims with quotes render in the evidence section.
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import (
    Claim,
    ConceptNote,
    EvidenceSpan,
    EvidenceVerification,
)
from obsidian_llm_wiki.render.obsidian import render_concept_page
from obsidian_llm_wiki.synth.quality import _merge_claims

# ── Fix 1: _merge_claims ────────────────────────────────────────────────


def test_merge_preserves_pass1_evidence_when_pass2_omits_quotes():
    """Pass 1 claim with evidence survives when Pass 2 has same claim without."""
    pass1 = [
        Claim(
            text="Jump risk eliminates leverage benefits",
            evidence=EvidenceSpan(
                quote="Properly price the risk of jump losses",
                source_file="some-source.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=100,
                end_offset=140,
            ),
        ),
    ]
    pass2 = [
        Claim(text="Jump risk eliminates leverage benefits"),
    ]

    merged = _merge_claims(pass1, pass2)
    assert len(merged) == 1
    assert merged[0].evidence is not None
    assert merged[0].evidence.quote == "Properly price the risk of jump losses"
    assert merged[0].evidence.source_file == "some-source.md"


def test_merge_prefers_pass2_evidence_when_both_have_it():
    """When both passes have evidence, Pass 2 (expanded) version wins."""
    pass1 = [
        Claim(
            text="Some claim",
            evidence=EvidenceSpan(
                quote="pass1 quote",
                source_file="source1.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=10,
            ),
        ),
    ]
    pass2 = [
        Claim(
            text="Some claim",
            evidence=EvidenceSpan(
                quote="pass2 quote",
                source_file="source2.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=10,
            ),
        ),
    ]

    merged = _merge_claims(pass1, pass2)
    assert len(merged) == 1
    assert merged[0].evidence.quote == "pass2 quote"


def test_merge_unions_unique_claims_from_both_passes():
    """Claims unique to each pass are all preserved."""
    pass1 = [
        Claim(text="Claim A from pass 1"),
        Claim(
            text="Claim B from pass 1",
            evidence=EvidenceSpan(
                quote="evidence B",
                source_file="source.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=10,
            ),
        ),
    ]
    pass2 = [
        Claim(text="Claim C from pass 2"),
        Claim(text="Claim B from pass 1"),  # duplicate, no evidence
    ]

    merged = _merge_claims(pass1, pass2)
    texts = [c.text for c in merged]
    assert "Claim A from pass 1" in texts
    assert "Claim B from pass 1" in texts
    assert "Claim C from pass 2" in texts
    # Claim B should retain evidence from Pass 1
    claim_b = [c for c in merged if "Claim B" in c.text][0]
    assert claim_b.evidence is not None
    assert claim_b.evidence.quote == "evidence B"


def test_merge_deduplicates_case_insensitively():
    """Claims with different casing but same text are deduplicated."""
    pass1 = [Claim(text="Prediction markets are useful")]
    pass2 = [Claim(text="prediction markets are useful")]

    merged = _merge_claims(pass1, pass2)
    assert len(merged) == 1


def test_merge_empty_claims():
    """Empty claim lists produce empty merged list."""
    assert _merge_claims([], []) == []


def test_merge_preserves_evidence_when_pass2_empty():
    """When Pass 2 produces no claims, all Pass 1 claims survive."""
    pass1 = [
        Claim(
            text="Important claim",
            evidence=EvidenceSpan(
                quote="important evidence",
                source_file="source.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=20,
            ),
        ),
    ]
    pass2: list[Claim] = []

    merged = _merge_claims(pass1, pass2)
    assert len(merged) == 1
    assert merged[0].evidence is not None
    assert merged[0].evidence.quote == "important evidence"


# ── Fix 2: Unverified evidence rendering ────────────────────────────────


def _make_concept(claims: list[Claim]) -> ConceptNote:
    return ConceptNote(
        title="Test Concept",
        slug="test-concept",
        summary="A test concept.",
        tags=["test"],
        sections=[],
        claims=claims,
    )


def test_verified_claim_gets_marker_and_evidence_entry():
    """A verified claim gets [1] marker and a clean evidence entry."""
    concept = _make_concept([
        Claim(
            text="Verified claim text",
            evidence=EvidenceSpan(
                quote="exact quote",
                source_file="source.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=11,
            ),
        ),
    ])

    page = render_concept_page(concept)
    assert "[1]" in page
    assert "1. \u201cexact quote\u201d" in page
    assert "*(unverified)*" not in page


def test_unverified_claim_with_quote_gets_star_marker_and_evidence():
    """An unverified claim with a quote gets [1*] marker and evidence with (unverified)."""
    concept = _make_concept([
        Claim(
            text="Unverified claim text",
            evidence=EvidenceSpan(
                quote="some quote",
                source_file="source.md",
                verification=EvidenceVerification.UNVERIFIED,
            ),
        ),
    ])

    page = render_concept_page(concept)
    assert "[1*]" in page
    assert "1. \u201csome quote\u201d *(unverified)*" in page


def test_claim_without_evidence_gets_no_marker():
    """A claim with no evidence at all gets no marker."""
    concept = _make_concept([
        Claim(text="Bare claim without evidence"),
    ])

    page = render_concept_page(concept)
    assert "[" not in page.split("## Claims")[1].split("## Evidence")[0]
    # No Evidence section should appear
    assert "## Evidence" not in page


def test_claim_with_quote_but_no_source_file_gets_no_marker():
    """A claim with a quote but no source_file doesn't render evidence."""
    concept = _make_concept([
        Claim(
            text="Claim with orphan quote",
            evidence=EvidenceSpan(
                quote="orphan quote",
                source_file="",
                verification=EvidenceVerification.UNVERIFIED,
            ),
        ),
    ])

    page = render_concept_page(concept)
    # No marker because source_file is empty
    assert "[" not in page.split("## Claims")[1].split("## Evidence")[0]
    assert "## Evidence" not in page


def test_mixed_verified_and_unverified_claims_both_render():
    """Both verified and unverified claims render with appropriate markers."""
    concept = _make_concept([
        Claim(
            text="Verified claim",
            evidence=EvidenceSpan(
                quote="verified quote",
                source_file="source.md",
                verification=EvidenceVerification.VERIFIED,
                start_offset=0,
                end_offset=14,
            ),
        ),
        Claim(
            text="Unverified claim",
            evidence=EvidenceSpan(
                quote="unverified quote",
                source_file="source.md",
                verification=EvidenceVerification.UNVERIFIED,
            ),
        ),
        Claim(text="Bare claim"),
    ])

    page = render_concept_page(concept)
    assert "[1]" in page  # verified
    assert "[2*]" in page  # unverified
    assert "*(unverified)*" in page
    # Bare claim should have no marker
    lines = page.split("## Claims")[1].split("## Evidence")[0]
    bare_line = [line for line in lines.split("\n") if "Bare claim" in line]
    assert bare_line and "[" not in bare_line[0]
