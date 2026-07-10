"""Tests for single-pass concept body gate + frontmatter rendering.

Tests:
  1. concept_body_chars helper — empty/thin/fat concepts
  2. Hard gate — thin concept filtered from SourceSynthesis; MOC refs pruned
  3. render_concept_page always emits confidence + provenance
  4. Regression: existing tests not broken by frontmatter changes
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptNote,
    ConceptLink,
    ConceptType,
    MapOfContent,
    SourceSynthesis,
)
from obsidian_llm_wiki.render.obsidian import render_concept_page
from obsidian_llm_wiki.synth.quality import concept_body_chars, filter_thin_concepts
import yaml


# ── concept_body_chars ────────────────────────────────────────────────────


def test_body_chars_empty_concept():
    """Empty concept → 0 body chars."""
    c = ConceptNote(title="Empty", slug="empty", summary="", sections=[])
    assert concept_body_chars(c) == 0


def test_body_chars_thin_concept():
    """Thin concept with short sections → small body char count."""
    c = ConceptNote(
        title="Thin",
        slug="thin",
        summary="ABCDEFGHIJ",  # 10 chars — NOT counted by _concept_body_chars
        sections=[
            BodySection(heading="Core", points=["12345"]),  # 5 chars
        ],
    )
    # _concept_body_chars counts sections only, not summary → 5
    assert concept_body_chars(c) == 5


def test_body_chars_fat_concept():
    """Fat concept with long sections → large body char count."""
    long_prose = "This is a very long prose paragraph. " * 30  # ~960 chars
    c = ConceptNote(
        title="Fat",
        slug="fat",
        summary="A comprehensive summary about the concept.",
        sections=[
            BodySection(heading="Core", prose=long_prose),
            BodySection(heading="Context", points=["Point one with evidence.", "Point two with more evidence."]),
        ],
    )
    assert concept_body_chars(c) > 800


def test_body_chars_excludes_summary():
    """Body char count does NOT include the summary field (matches _concept_body_chars semantics)."""
    c1 = ConceptNote(title="T", slug="t", summary="", sections=[])
    c2 = ConceptNote(title="T", slug="t", summary="Extra summary text here.", sections=[])
    # Both have zero sections → both 0, summary not counted
    assert concept_body_chars(c2) == concept_body_chars(c1)


# ── filter_thin_concepts (hard gate) ─────────────────────────────────────


def test_filter_thin_concepts_drops_short():
    """Concepts below threshold are dropped from synthesis."""
    thin = ConceptNote(
        title="Thin",
        slug="thin",
        summary="Short.",
        sections=[BodySection(heading="Core", points=["x"])],
    )
    fat1 = ConceptNote(
        title="Fat1",
        slug="fat1",
        summary="A good summary.",
        sections=[BodySection(heading="Core", prose="x" * 900)],
    )
    fat2 = ConceptNote(
        title="Fat2",
        slug="fat2",
        summary="Another good summary.",
        sections=[BodySection(heading="Core", prose="x" * 900)],
    )
    synth = SourceSynthesis(
        source_title="Test",
        source_summary="A test synthesis.",
        concepts=[thin, fat1, fat2],
        maps=[MapOfContent(title="MOC", slug="moc", summary="", concept_slugs=["thin", "fat1", "fat2"])],
    )

    result = filter_thin_concepts(synth, min_body_chars=800)

    assert len(result.concepts) == 2
    assert {c.slug for c in result.concepts} == {"fat1", "fat2"}
    # MOC references to dropped slug must be pruned
    moc = result.maps[0]
    assert "thin" not in moc.concept_slugs
    assert set(moc.concept_slugs) == {"fat1", "fat2"}


def test_filter_thin_concepts_prunes_related_links():
    """Related links to dropped slugs are pruned from surviving concepts."""
    thin = ConceptNote(
        title="Thin",
        slug="thin",
        summary="Short.",
        sections=[BodySection(heading="Core", points=["x"])],
    )
    fat = ConceptNote(
        title="Fat",
        slug="fat",
        summary="A good summary.",
        sections=[BodySection(heading="Core", prose="x" * 900)],
        related=[
            ConceptLink(slug="thin", relation="related_to"),
            ConceptLink(slug="other", relation="depends_on"),
        ],
    )
    synth = SourceSynthesis(
        source_title="Test",
        source_summary="A test synthesis.",
        concepts=[thin, fat],
        maps=[],
    )

    result = filter_thin_concepts(synth, min_body_chars=800)

    assert len(result.concepts) == 1
    survived = result.concepts[0]
    # "thin" link pruned, "other" retained
    slugs = [r.slug for r in survived.related]
    assert "thin" not in slugs
    assert "other" in slugs


def test_filter_thin_concepts_keeps_all_when_above_threshold():
    """All concepts above threshold are kept."""
    c1 = ConceptNote(
        title="C1", slug="c1", summary="Summary.",
        sections=[BodySection(heading="Core", prose="x" * 850)],
    )
    c2 = ConceptNote(
        title="C2", slug="c2", summary="Summary.",
        sections=[BodySection(heading="Core", prose="x" * 900)],
    )
    synth = SourceSynthesis(
        source_title="Test",
        source_summary="A test synthesis.",
        concepts=[c1, c2],
        maps=[],
    )

    result = filter_thin_concepts(synth, min_body_chars=800)
    assert len(result.concepts) == 2


def test_filter_thin_concepts_empty_mocs_pruned():
    """MOCs with no remaining concept slugs after filtering are dropped."""
    thin = ConceptNote(
        title="Thin", slug="thin", summary="S.",
        sections=[BodySection(heading="Core", points=["x"])],
    )
    synth = SourceSynthesis(
        source_title="Test",
        source_summary="A test synthesis.",
        concepts=[thin],
        maps=[MapOfContent(title="MOC", slug="moc", summary="", concept_slugs=["thin"])],
    )

    result = filter_thin_concepts(synth, min_body_chars=800)
    assert len(result.concepts) == 0
    # MOC with only dropped concepts should be pruned entirely
    assert result.maps == []


# ── render_concept_page frontmatter ──────────────────────────────────────


def test_render_concept_always_emits_confidence():
    """confidence is always present in frontmatter, even when 1.0."""
    c = ConceptNote(
        title="Test Concept",
        slug="test-concept",
        summary="A test concept.",
        tags=["test"],
        sections=[BodySection(heading="Core", prose="x" * 200)],
        confidence=1.0,
        provenance="extracted",
    )
    page = render_concept_page(c)
    fm, body = _split_fm(page)
    assert "confidence" in fm
    assert fm["confidence"] == 1.0


def test_render_concept_emits_provenance():
    """provenance is always present in frontmatter."""
    c = ConceptNote(
        title="Test Concept",
        slug="test-concept",
        summary="A test concept.",
        sections=[BodySection(heading="Core", prose="x" * 200)],
        confidence=0.5,
        provenance="merged",
    )
    page = render_concept_page(c)
    fm, _ = _split_fm(page)
    assert "provenance" in fm
    assert fm["provenance"] == "merged"


def test_render_concept_low_confidence():
    """Low confidence concept still renders with confidence in frontmatter."""
    c = ConceptNote(
        title="Low Conf",
        slug="low-conf",
        summary="A low confidence concept.",
        sections=[BodySection(heading="Core", prose="x" * 200)],
        confidence=0.3,
        provenance="ambiguous",
    )
    page = render_concept_page(c)
    fm, _ = _split_fm(page)
    assert fm["confidence"] == 0.3
    assert fm["provenance"] == "ambiguous"


# ── Helpers ──────────────────────────────────────────────────────────────


def _split_fm(page: str) -> tuple[dict, str]:
    """Split a rendered page into (frontmatter_dict, body)."""
    assert page.startswith("---\n"), f"Expected frontmatter, got: {page[:50]}"
    parts = page.split("---\n", 2)
    assert len(parts) >= 3
    fm = yaml.safe_load(parts[1])
    body = parts[2]
    return fm, body