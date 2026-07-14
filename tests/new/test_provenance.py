"""Regression tests for immutable source retrieval provenance persistence."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
from obsidian_llm_wiki.ingest.sources import load_source_file
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, render_source_page


def test_source_provenance_defaults_are_immutable():
    """Existing SourceDoc callers retain an empty immutable provenance value."""
    provenance = SourceProvenance()
    doc = SourceDoc(title="Article", content="Body")

    assert provenance.requested_url == ""
    assert provenance.extractor_chain == ()
    assert provenance.diagnostics == ()
    assert doc.provenance == provenance

    bounded = SourceProvenance(diagnostics=tuple(str(index) for index in range(21)))
    assert bounded.diagnostics == tuple(str(index) for index in range(20))

    with pytest.raises(FrozenInstanceError):
        provenance.requested_url = "https://example.com/changed"  # type: ignore[misc]


def test_source_provenance_round_trips_through_source_frontmatter(tmp_path: Path):
    """Nonempty retrieval metadata is serialized and reconstructed exactly."""
    provenance = SourceProvenance(
        requested_url="https://example.com/requested",
        resolved_url="https://example.com/resolved",
        extracted_url="https://cdn.example.com/full-text.pdf",
        extractor_chain=("scientific-candidate", "pdf"),
        content_type="application/pdf",
        document_format="pdf",
        retrieved_at="2026-07-13T12:00:00Z",
        content_sha256="a" * 64,
        diagnostics=("candidate selected",),
    )
    source = SourceDoc(
        title="Article",
        content="Source body.",
        url="https://example.com/legacy-url",
        provenance=provenance,
    )

    page = render_source_page(source, "2026-07-13T12:30:00Z")
    meta, _ = parse_frontmatter(page)
    assert meta["url"] == "https://example.com/legacy-url"
    assert meta["provenance"] == [
        "requested_url: https://example.com/requested",
        "resolved_url: https://example.com/resolved",
        "extracted_url: https://cdn.example.com/full-text.pdf",
        "content_type: application/pdf",
        "document_format: pdf",
        "retrieved_at: 2026-07-13T12:00:00Z",
        f"content_sha256: {'a' * 64}",
        "extractor_chain: scientific-candidate",
        "extractor_chain: pdf",
        "diagnostics: candidate selected",
    ]

    source_path = tmp_path / "article.md"
    source_path.write_text(page, encoding="utf-8")
    reloaded = load_source_file(source_path)

    assert reloaded is not None
    assert reloaded.provenance == provenance
    assert reloaded.source_file == "article.md"

    empty_meta, _ = parse_frontmatter(
        render_source_page(SourceDoc(title="Legacy", content="Body"), "2026-07-13T12:30:00Z"),
    )
    assert "provenance" not in empty_meta


def test_source_loader_keeps_legacy_nested_provenance_frontmatter(tmp_path: Path):
    """Existing vault pages remain readable after the Obsidian-safe migration."""
    legacy = (
        "---\n"
        "title: Legacy\n"
        "provenance:\n"
        "  requested_url: https://example.com/requested\n"
        "  extractor_chain: [web, trafilatura]\n"
        "---\n"
        "Body\n"
    )
    path = tmp_path / "legacy.md"
    path.write_text(legacy, encoding="utf-8")

    source = load_source_file(path)

    assert source is not None
    assert source.provenance.requested_url == "https://example.com/requested"
    assert source.provenance.extractor_chain == ("web", "trafilatura")


def test_source_metadata_round_trips_through_source_frontmatter(tmp_path: Path):
    """Source aliases, tags, and classification survive rendering and reload."""
    source = SourceDoc(
        title="Article",
        content="Source body.",
        aliases=["Alternate title", "Short title"],
        tags=["research", "ai"],
        source_type="scientific-paper",
    )

    source_path = tmp_path / "article.md"
    source_path.write_text(render_source_page(source, "2026-07-13T12:30:00Z"), encoding="utf-8")
    metadata, _ = parse_frontmatter(source_path.read_text(encoding="utf-8"))
    reloaded = load_source_file(source_path)

    assert metadata["aliases"] == ["Alternate title", "Short title"]
    assert metadata["tags"] == ["research", "ai"]
    assert metadata["source_type"] == "scientific-paper"
    assert reloaded is not None
    assert reloaded.aliases == source.aliases
    assert reloaded.tags == source.tags
    assert reloaded.source_type == source.source_type
