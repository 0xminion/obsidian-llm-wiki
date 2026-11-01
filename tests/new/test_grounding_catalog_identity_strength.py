"""Regression for a reproduced collision in 32-bit source-span catalog tags."""
import pytest

from obsidian_llm_wiki.core.span_catalog import CitationError, build_span_catalog


def test_catalog_rejects_foreign_id_from_reproduced_v1_collision():
    first = build_span_catalog("Record 8404 has an independent measurement.", "source-8404.md")
    second = build_span_catalog("Record 78481 has an independent measurement.", "source-78481.md")
    assert first.revision != second.revision
    with pytest.raises(CitationError):
        first.cite([second.spans[0].id])


def test_catalog_tag_has_at_least_128_bits_of_revision_identity():
    catalog = build_span_catalog("A measured source statement.", "source.md")
    assert len(catalog.tag) >= 32
    assert all(char in "0123456789abcdef" for char in catalog.tag)


@pytest.mark.parametrize("mode", ["required", "off"])
def test_catalog_version_invalidates_artifacts_in_every_review_mode(monkeypatch, mode):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import span_catalog
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config

    config = Config(grounding_review=mode)
    before = _artifact_versions_for_config(config)
    monkeypatch.setattr(span_catalog, "CATALOG_VERSION", "span-catalog-next")
    after = _artifact_versions_for_config(config)
    assert after != before
