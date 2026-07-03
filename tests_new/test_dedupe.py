"""Tests for obsidian_llm_wiki.synth.dedupe — corpus-level reconciliation."""

from __future__ import annotations

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
)
from obsidian_llm_wiki.synth.dedupe import (
    merge_bundle,
    merge_concepts,
    normalise_tags,
    slugify,
)


# ── normalise_tags ───────────────────────────────────────────────────────


def test_normalise_tags_basic():
    assert normalise_tags(["Machine Learning", "  optimization  "]) == [
        "machine-learning", "optimization"
    ]


def test_normalise_tags_dedup():
    assert normalise_tags(["ml", "ML", "Ml"]) == ["machine-learning"]


def test_normalise_tags_alias():
    assert normalise_tags(["ml"]) == ["machine-learning"]
    assert normalise_tags(["nlp"]) == ["natural-language-processing"]


def test_normalise_tags_empty_filtered():
    assert normalise_tags(["", "  ", "valid"]) == ["valid"]


def test_normalise_tags_special_chars():
    assert normalise_tags(["c++", "node.js"]) == ["c", "nodejs"]


# ── slugify ──────────────────────────────────────────────────────────────


def test_slugify_basic():
    assert slugify("Gradient Descent") == "gradient-descent"


def test_slugify_special_chars():
    assert slugify("C++ Programming!") == "c-programming"


def test_slugify_empty():
    assert slugify("") == "untitled"
    assert slugify("!!!") == "untitled"


# ── merge_concepts ───────────────────────────────────────────────────────


def test_merge_concepts_same_slug():
    c1 = ConceptNote(
        title="Gradient Descent", slug="gradient-descent", summary="S1",
        tags=["ml", "optimization"],
        sections=[BodySection(heading="Core", points=["P1"])],
    )
    c2 = ConceptNote(
        title="Gradient Descent", slug="gradient-descent", summary="S2",
        tags=["ai", "optimization"],
        sections=[BodySection(heading="Context", points=["P2"])],
    )
    merged = merge_concepts([c1, c2])
    assert len(merged) == 1
    m = merged[0]
    assert set(m.tags) == {"machine-learning", "optimization", "artificial-intelligence"}
    assert len(m.sections) == 2


def test_merge_concepts_different_slugs():
    c1 = ConceptNote(title="A", slug="a", summary="S")
    c2 = ConceptNote(title="B", slug="b", summary="S")
    merged = merge_concepts([c1, c2])
    assert len(merged) == 2


def test_merge_concepts_related_union():
    c1 = ConceptNote(
        title="A", slug="a", summary="S",
        related=[ConceptLink(slug="b"), ConceptLink(slug="c")],
    )
    c2 = ConceptNote(
        title="A", slug="a", summary="S",
        related=[ConceptLink(slug="c"), ConceptLink(slug="d")],
    )
    merged = merge_concepts([c1, c2])
    assert len(merged) == 1
    related_slugs = {r.slug for r in merged[0].related}
    assert related_slugs == {"b", "c", "d"}


def test_merge_concepts_auto_slug():
    c = ConceptNote(title="Gradient Descent", slug="", summary="S")
    merged = merge_concepts([c])
    assert merged[0].slug == "gradient-descent"


def test_merge_concepts_is_new_anded():
    c1 = ConceptNote(title="A", slug="a", summary="S", is_new=True)
    c2 = ConceptNote(title="A", slug="a", summary="S", is_new=False)
    merged = merge_concepts([c1, c2])
    assert merged[0].is_new is False


# ── merge_bundle ─────────────────────────────────────────────────────────


def test_merge_bundle_basic():
    s1 = SourceSynthesis(
        source_title="Paper A", source_summary="SA",
        concepts=[ConceptNote(title="CA", slug="ca", summary="s", tags=["ml"])],
        maps=[MapOfContent(title="Topic", slug="topic", summary="MOC",
                           concept_slugs=["ca"])],
    )
    s2 = SourceSynthesis(
        source_title="Paper B", source_summary="SB",
        concepts=[ConceptNote(title="CB", slug="cb", summary="s", tags=["ai"])],
    )
    bundle = merge_bundle([s1, s2])
    assert len(bundle.concepts) == 2
    assert len(bundle.maps) == 1


def test_merge_bundle_dedup_concepts():
    s1 = SourceSynthesis(
        source_title="Paper A", source_summary="SA",
        concepts=[ConceptNote(title="GD", slug="gradient-descent",
                              summary="s", tags=["ml"])],
    )
    s2 = SourceSynthesis(
        source_title="Paper B", source_summary="SB",
        concepts=[ConceptNote(title="GD", slug="gradient-descent",
                              summary="s2", tags=["optimization"])],
    )
    bundle = merge_bundle([s1, s2])
    assert len(bundle.concepts) == 1
    assert set(bundle.concepts[0].tags) == {"machine-learning", "optimization"}


def test_merge_bundle_moc_merge():
    s1 = SourceSynthesis(
        source_title="A", source_summary="S",
        maps=[MapOfContent(title="Topic", slug="topic", summary="MOC",
                           concept_slugs=["a", "b"])],
    )
    s2 = SourceSynthesis(
        source_title="B", source_summary="S",
        maps=[MapOfContent(title="Topic", slug="topic", summary="MOC",
                           concept_slugs=["b", "c"])],
    )
    bundle = merge_bundle([s1, s2])
    assert len(bundle.maps) == 1
    assert set(bundle.maps[0].concept_slugs) == {"a", "b", "c"}