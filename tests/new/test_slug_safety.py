"""Regression coverage for untrusted synthesis slugs at the vault boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SynthesisBundle,
    normalize_slug,
    source_synthesis_from_dict,
)
from obsidian_llm_wiki.render.obsidian import render_vault
from obsidian_llm_wiki.synth.dedupe import merge_bundle


def _is_safe_slug(slug: str) -> bool:
    return bool(slug) and len(slug) <= 80 and slug[0].isalnum() and all(
        char.isalnum() or char in "_-" for char in slug
    )


@pytest.mark.parametrize(
    "unsafe_slug",
    [
        "../escape",
        "/absolute/escape",
        r"..\escape",
        "café\x00",
        "line\x00break",
        "x" * 81,
    ],
)
def test_parser_regenerates_unsafe_concept_moc_and_relation_slugs(unsafe_slug: str) -> None:
    synthesis = source_synthesis_from_dict(
        {
            "source_title": "Source",
            "source_summary": "Summary",
            "concepts": [
                {
                    "title": "Safe Concept",
                    "slug": unsafe_slug,
                    "summary": "Summary",
                    "related": [{"slug": unsafe_slug}],
                }
            ],
            "maps": [
                {
                    "title": "Safe Map",
                    "slug": unsafe_slug,
                    "summary": "Summary",
                    "concept_slugs": [unsafe_slug],
                }
            ],
        }
    )

    concept = synthesis.concepts[0]
    moc = synthesis.maps[0]
    assert concept.slug == "safe-concept"
    assert moc.slug == "safe-map"
    assert concept.related[0].slug == normalize_slug(unsafe_slug)
    assert moc.concept_slugs == [normalize_slug(unsafe_slug)]
    assert all(
        _is_safe_slug(slug)
        for slug in (concept.slug, moc.slug, concept.related[0].slug, moc.concept_slugs[0])
    )


def test_parser_preserves_an_existing_safe_slug() -> None:
    synthesis = source_synthesis_from_dict(
        {
            "concepts": [
                {
                    "title": "Different title",
                    "slug": "existing-有效-slug",
                    "summary": "Summary",
                }
            ],
            "maps": [
                {
                    "title": "Different map",
                    "slug": "existing-有效-map",
                    "summary": "Summary",
                }
            ],
        }
    )

    assert synthesis.concepts[0].slug == "existing-有效-slug"
    assert synthesis.maps[0].slug == "existing-有效-map"


def test_parse_dedupe_and_render_cannot_write_outside_the_vault(tmp_path: Path) -> None:
    escaped = tmp_path.parent / "escaped-from-vault.md"
    synthesis = source_synthesis_from_dict(
        {
            "source_title": "Source",
            "source_summary": "Summary",
            "concepts": [
                {
                    "title": "Safe Concept",
                    "slug": "../escaped-from-vault",
                    "summary": "Summary",
                    "related": [{"slug": "../related-target"}],
                }
            ],
            "maps": [
                {
                    "title": "Safe Map",
                    "slug": "../escaped-map",
                    "summary": "Summary",
                    "concept_slugs": ["../related-target"],
                }
            ],
        }
    )

    bundle = merge_bundle([synthesis])
    render_vault(tmp_path, bundle, {"source.md": SourceDoc(title="Source", content="Body")})

    assert not escaped.exists()
    assert (tmp_path / "concepts" / "safe-concept.md").is_file()
    assert (tmp_path / "mocs" / "safe-map.md").is_file()


def test_renderer_normalizes_directly_constructed_unsafe_output_slugs(tmp_path: Path) -> None:
    bundle = SynthesisBundle(
        concepts=[
            ConceptNote(
                title="Safe Concept",
                slug="../escaped-from-vault",
                summary="Summary",
                related=[ConceptLink(slug=r"..\related-target")],
            )
        ],
        maps=[
            MapOfContent(
                title="Safe Map",
                slug="../escaped-map",
                summary="Summary",
                concept_slugs=[r"..\related-target"],
            )
        ],
    )

    render_vault(tmp_path, bundle, {"source.md": SourceDoc(title="Source", content="Body")})

    assert not (tmp_path / "escaped-from-vault.md").exists()
    assert not (tmp_path / "escaped-map.md").exists()
    assert (tmp_path / "concepts" / "safe-concept.md").is_file()
    assert (tmp_path / "mocs" / "safe-map.md").is_file()
