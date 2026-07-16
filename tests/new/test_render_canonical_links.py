"""Renderer links must point only to canonical emitted concept pages."""

from __future__ import annotations

from obsidian_llm_wiki.core.models import ConceptLink, ConceptNote, MapOfContent
from obsidian_llm_wiki.render.obsidian import render_concept_page, render_moc_page


def test_render_moc_omits_concepts_absent_from_canonical_map():
    present = ConceptNote(title="Present", slug="present", summary="Canonical concept")
    moc = MapOfContent(
        title="Map",
        slug="map",
        summary="",
        concept_slugs=["present", "stale-local-slug"],
    )

    page = render_moc_page(moc, all_concepts={"present": present})

    assert "[[present]]" in page
    assert "stale-local-slug" not in page


def test_render_concept_omits_relations_absent_from_canonical_map():
    present = ConceptNote(title="Present", slug="present", summary="Canonical concept")
    concept = ConceptNote(
        title="Current",
        slug="current",
        summary="",
        related=[
            ConceptLink(slug="present", relation="related_to"),
            ConceptLink(slug="stale-local-slug", relation="related_to"),
        ],
    )

    page = render_concept_page(concept, all_concepts={"present": present})

    assert "present|related_to|present" in page
    assert "stale-local-slug" not in page
