"""Tests for obsidian_llm_wiki.core.models — Phase 4 regression tests.

Covers:
  - normalize_relation() (added in Phase 1, never tested)
  - Empty section rejection in _concept_from_dict (Phase 4)
  - RelationType enum expanded values (Phase 1)
"""

from __future__ import annotations

from obsidian_llm_wiki.core.models import (
    RelationType,
    normalize_relation,
    source_synthesis_from_dict,
)

# ── normalize_relation ───────────────────────────────────────────────────


def test_normalize_relation_valid():
    """Valid relation strings pass through (with hyphen→underscore)."""
    assert normalize_relation("variant_of") == "variant_of"
    assert normalize_relation("variant-of") == "variant_of"
    assert normalize_relation("DEPENDS_ON") == "depends_on"
    assert normalize_relation("depends on") == "depends_on"


def test_normalize_relation_invalid_falls_back():
    """Invalid relation strings fall back to 'related_to'."""
    assert normalize_relation("nonsense") == "related_to"
    assert normalize_relation("is_a_type_of") == "related_to"
    assert normalize_relation("") == "related_to"
    assert normalize_relation(None) == "related_to"  # type: ignore[arg-type]


def test_normalize_relation_all_enum_values():
    """All RelationType enum values are valid relations."""
    for rt in RelationType:
        assert normalize_relation(rt.value) == rt.value


# ── RelationType enum ────────────────────────────────────────────────────


def test_relation_type_expanded_values():
    """Phase 1 added new relation types — verify they exist."""
    assert RelationType.COMPONENT_OF.value == "component_of"
    assert RelationType.CAUSES.value == "causes"
    assert RelationType.ENABLES.value == "enables"
    assert RelationType.PART_OF.value == "part_of"
    assert RelationType.EXPLAINS.value == "explains"


# ── Empty section rejection ──────────────────────────────────────────────


def test_empty_sections_are_dropped():
    """Sections with empty points AND empty prose are filtered out."""
    data = {
        "source_title": "Test",
        "source_summary": "Summary.",
        "concepts": [
            {
                "title": "C",
                "slug": "c",
                "summary": "S",
                "sections": [
                    {"heading": "Good", "points": ["real point"]},
                    {"heading": "Empty points", "points": []},
                    {"heading": "Empty prose", "prose": ""},
                    {"heading": "Both empty", "points": [], "prose": ""},
                    {"heading": "Prose only", "prose": "Some prose here."},
                ],
            },
        ],
    }
    synth = source_synthesis_from_dict(data)
    assert len(synth.concepts) == 1
    concept = synth.concepts[0]
    # 2 sections survive: Good (has points) and Prose only (has prose).
    assert len(concept.sections) == 2
    headings = [s.heading for s in concept.sections]
    assert "Good" in headings
    assert "Prose only" in headings
    assert "Empty points" not in headings
    assert "Empty prose" not in headings
    assert "Both empty" not in headings


def test_all_empty_sections_yields_no_sections():
    """When all sections are empty, the concept has zero sections."""
    data = {
        "source_title": "T",
        "source_summary": "S",
        "concepts": [
            {
                "title": "C",
                "slug": "c",
                "summary": "S",
                "sections": [
                    {"heading": "A", "points": [], "prose": ""},
                    {"heading": "B", "points": []},
                ],
            },
        ],
    }
    synth = source_synthesis_from_dict(data)
    assert len(synth.concepts[0].sections) == 0


def test_concept_with_no_sections_key():
    """Missing 'sections' key yields empty list."""
    data = {
        "source_title": "T",
        "source_summary": "S",
        "concepts": [{"title": "C", "slug": "c", "summary": "S"}],
    }
    synth = source_synthesis_from_dict(data)
    assert synth.concepts[0].sections == []
