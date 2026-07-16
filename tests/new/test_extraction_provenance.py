"""Tests for extraction-boundary provenance stamping."""

from __future__ import annotations

from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance


def test_stamp_source_populates_stable_retrieval_facts():
    from obsidian_llm_wiki.ingest.provenance import stamp_source

    source = SourceDoc(title="Paper", content="Important evidence", url="https://example.org/final")

    stamped = stamp_source(
        source,
        requested_url="https://example.org/landing",
        extractor="scientific_html",
        content_type="text/html",
        document_format="html",
        retrieved_at="2026-07-13T00:00:00Z",
    )

    assert stamped is not source
    assert stamped.provenance.requested_url == "https://example.org/landing"
    assert stamped.provenance.resolved_url == "https://example.org/final"
    assert stamped.provenance.extracted_url == "https://example.org/final"
    assert stamped.provenance.extractor_chain == ("scientific_html",)
    assert stamped.provenance.content_type == "text/html"
    assert stamped.provenance.document_format == "html"
    assert stamped.provenance.retrieved_at == "2026-07-13T00:00:00Z"
    assert len(stamped.provenance.content_sha256) == 64


def test_stamp_source_preserves_existing_facts_and_appends_chain():
    from obsidian_llm_wiki.ingest.provenance import stamp_source

    source = SourceDoc(
        title="Paper",
        content="Text",
        url="https://publisher.example/pdf",
        provenance=SourceProvenance(
            requested_url="https://publisher.example/landing",
            extractor_chain=("landing_discovery",),
            retrieved_at="2026-07-12T00:00:00Z",
            content_sha256="already-set",
        ),
    )

    stamped = stamp_source(source, requested_url="ignored", extractor="liteparse")

    assert stamped.provenance.requested_url == "https://publisher.example/landing"
    assert stamped.provenance.extractor_chain == ("landing_discovery", "liteparse")
    assert stamped.provenance.retrieved_at == "2026-07-12T00:00:00Z"
    assert stamped.provenance.content_sha256 == "already-set"


def test_registry_stamps_specialized_extraction(monkeypatch):
    from obsidian_llm_wiki.ingest import extractors

    def _specialized(_url: str) -> SourceDoc:
        return SourceDoc(title="Source", content="body " * 150, url="https://example.com/final")

    monkeypatch.setattr(extractors, "_EXTRACTORS", [(lambda _parsed, _raw: True, _specialized)])

    source = extractors.extract("https://example.com/requested")

    assert source.provenance.requested_url == "https://example.com/requested"
    assert source.provenance.resolved_url == "https://example.com/final"
    assert source.provenance.extractor_chain == ("_specialized",)
