"""Tests for obsidian_llm_wiki.core.models — SynthesisBundle schema."""

from __future__ import annotations

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    ConceptType,
    ProvenanceState,
    RelationType,
    SourceDoc,
    source_synthesis_from_dict,
)

# ── SourceDoc ────────────────────────────────────────────────────────────


def test_source_doc_minimal():
    doc = SourceDoc(title="Test", content="Some content")
    assert doc.title == "Test"
    assert doc.content == "Some content"
    assert doc.url is None
    assert doc.source_file is None


def test_source_doc_with_url():
    doc = SourceDoc(title="Test", content="Content", url="https://example.com")
    assert doc.url == "https://example.com"


# ── ConceptNote ──────────────────────────────────────────────────────────


def test_concept_note_defaults():
    c = ConceptNote(title="Gradient Descent", slug="gradient-descent",
                    summary="Optimization algorithm")
    assert c.tags == []
    assert c.aliases == []
    assert c.sections == []
    assert c.claims == []
    assert c.related == []
    assert c.confidence == 1.0
    assert c.provenance == "extracted"
    assert c.is_new is True


def test_concept_note_with_sections_and_links():
    c = ConceptNote(
        title="Transformer",
        slug="transformer",
        summary="Attention-based architecture",
        tags=["deep-learning", "attention"],
        sections=[BodySection(heading="Core", points=["Self-attention mechanism"])],
        related=[ConceptLink(slug="attention", relation="depends_on")],
    )
    assert len(c.sections) == 1
    assert c.sections[0].heading == "Core"
    assert c.related[0].relation == "depends_on"


# ── source_synthesis_from_dict ───────────────────────────────────────────


def test_from_dict_minimal():
    data = {
        "source_title": "Paper",
        "source_summary": "A summary",
    }
    synth = source_synthesis_from_dict(data)
    assert synth.source_title == "Paper"
    assert synth.source_summary == "A summary"
    assert synth.concepts == []
    assert synth.maps == []


def test_from_dict_with_concepts():
    data = {
        "source_title": "Paper",
        "source_summary": "Summary",
        "source_tags": ["ml", "ai"],
        "concepts": [
            {
                "title": "Gradient Descent",
                "slug": "gradient-descent",
                "summary": "Opt algo",
                "tags": ["optimization"],
                "sections": [
                    {"heading": "Core", "points": ["Step size"]},
                ],
                "related": [
                    {"slug": "sgd", "relation": "variant_of"},
                ],
            },
        ],
        "maps": [
            {
                "title": "Optimization",
                "slug": "optimization",
                "summary": "Overview",
                "concept_slugs": ["gradient-descent"],
            },
        ],
    }
    synth = source_synthesis_from_dict(data)
    assert len(synth.concepts) == 1
    assert synth.concepts[0].title == "Gradient Descent"
    assert synth.concepts[0].sections[0].heading == "Core"
    assert synth.concepts[0].related[0].slug == "sgd"
    assert len(synth.maps) == 1
    assert synth.maps[0].concept_slugs == ["gradient-descent"]


def test_from_dict_tolerates_missing_fields():
    data = {"title": "Paper"}  # no summary, no concepts
    synth = source_synthesis_from_dict(data)
    assert synth.source_title == "Paper"
    assert synth.source_summary == ""
    assert synth.concepts == []


def test_from_dict_with_claims():
    data = {
        "source_title": "Paper",
        "source_summary": "Summary",
        "concepts": [
            {
                "title": "C",
                "slug": "c",
                "summary": "S",
                "claims": [
                    {"text": "Claim 1", "source_ref": "para3"},
                ],
            },
        ],
    }
    synth = source_synthesis_from_dict(data)
    assert len(synth.concepts[0].claims) == 1
    assert synth.concepts[0].claims[0].text == "Claim 1"


# ── Enums ────────────────────────────────────────────────────────────────


def test_concept_type_values():
    assert ConceptType.SOURCE.value == "Source"
    assert ConceptType.CONCEPT.value == "Concept"
    assert ConceptType.MOC.value == "Map of Content"


def test_relation_type_values():
    assert RelationType.VARIANT_OF.value == "variant_of"
    assert RelationType.DEPENDS_ON.value == "depends_on"


def test_provenance_state_values():
    assert ProvenanceState.EXTRACTED.value == "extracted"
    assert ProvenanceState.MERGED.value == "merged"
