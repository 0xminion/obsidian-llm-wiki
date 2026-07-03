"""Tests for pipeline.okf_renderer.

Each render function is exercised to verify:
  * frontmatter carries the correct ``type`` value,
  * body contains the title heading,
  * links are in the absolute OKF ``[text](/id.md)`` form,
  * citations / source / concept sections are formatted correctly.

Every test writes the rendered page to ``tmp_path`` and reads it back to
exercise the real file round-trip (frontmatter parse + body extraction).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import okf_markdown as om
from pipeline.okf_renderer import (
    render_concept_page,
    render_entry_page,
    render_moc_page,
    render_reference_page,
    render_source_page,
)

TS = "2025-01-01T00:00:00Z"


# ── helpers ────────────────────────────────────────────────────────────


def _write_and_read(tmp_path: Path, name: str, content: str) -> tuple[dict, str]:
    """Write ``content`` to ``tmp_path/name`` then parse it back."""
    target = tmp_path / name
    target.write_text(content, encoding="utf-8")
    raw = target.read_text(encoding="utf-8")
    return om.parse_frontmatter(raw)


# ── render_source_page ────────────────────────────────────────────────


def test_source_page_frontmatter_type(tmp_path: Path):
    out = render_source_page("My Source", "https://example.com", "raw body", timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "source.md", out)
    assert meta["type"] == "Source"


def test_source_page_body_contains_title(tmp_path: Path):
    out = render_source_page("My Source", "https://example.com", "raw body", timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "source.md", out)
    assert "# My Source" in body


def test_source_page_description_and_resource(tmp_path: Path):
    url = "https://example.com/page"
    out = render_source_page("My Source", url, "raw body", timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "source.md", out)
    assert meta["description"] == f"Original content from {url}"
    assert meta["resource"] == url
    assert meta["tags"] == ["source"]


def test_source_page_default_timestamp(tmp_path: Path):
    out = render_source_page("My Source", "https://example.com", "raw body")
    meta, _body = _write_and_read(tmp_path, "source.md", out)
    assert meta["timestamp"] is not None
    assert isinstance(meta["timestamp"], str)
    assert meta["timestamp"] != ""


# ── render_entry_page ─────────────────────────────────────────────────


def test_entry_page_frontmatter_type(tmp_path: Path):
    out = render_entry_page(
        "My Entry", "A summary", "sources/my-source", "body text", timestamp=TS
    )
    meta, _body = _write_and_read(tmp_path, "entry.md", out)
    assert meta["type"] == "Entry"


def test_entry_page_body_contains_title(tmp_path: Path):
    out = render_entry_page(
        "My Entry", "A summary", "sources/my-source", "body text", timestamp=TS
    )
    _meta, body = _write_and_read(tmp_path, "entry.md", out)
    assert "# My Entry" in body


def test_entry_page_description_truncated(tmp_path: Path):
    long_summary = "A" * 300
    out = render_entry_page(
        "My Entry", long_summary, "sources/my-source", "body text", timestamp=TS
    )
    meta, _body = _write_and_read(tmp_path, "entry.md", out)
    assert len(meta["description"]) == 200
    assert meta["description"] == long_summary[:200]


def test_entry_page_source_link_is_absolute(tmp_path: Path):
    out = render_entry_page(
        "My Entry", "A summary", "sources/my-source", "body text", timestamp=TS
    )
    _meta, body = _write_and_read(tmp_path, "entry.md", out)
    assert "## Source" in body
    # Absolute link: [Source](/sources/my-source.md)
    assert "[Source](/sources/my-source.md)" in body


def test_entry_page_tags(tmp_path: Path):
    out = render_entry_page(
        "My Entry", "A summary", "sources/my-source", "body text",
        tags=["foo", "bar"], timestamp=TS,
    )
    meta, _body = _write_and_read(tmp_path, "entry.md", out)
    assert meta["tags"] == ["foo", "bar"]


# ── render_concept_page ───────────────────────────────────────────────


def test_concept_page_frontmatter_type(tmp_path: Path):
    out = render_concept_page("My Concept", "summary", "body text", timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "concept.md", out)
    assert meta["type"] == "Concept"


def test_concept_page_body_contains_title(tmp_path: Path):
    out = render_concept_page("My Concept", "summary", "body text", timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "concept.md", out)
    assert "# My Concept" in body


def test_concept_page_sources_are_absolute_links(tmp_path: Path):
    out = render_concept_page(
        "My Concept", "summary", "body text",
        source_ids=["concepts/foo", "concepts/bar"], timestamp=TS,
    )
    _meta, body = _write_and_read(tmp_path, "concept.md", out)
    assert "## Sources" in body
    assert "[concepts/foo](/concepts/foo.md)" in body
    assert "[concepts/bar](/concepts/bar.md)" in body


def test_concept_page_citations_section_format(tmp_path: Path):
    out = render_concept_page(
        "My Concept", "summary", "body text",
        citations=["https://a.example.com", "https://b.example.com"], timestamp=TS,
    )
    _meta, body = _write_and_read(tmp_path, "concept.md", out)
    assert "# Citations" in body
    assert "1. https://a.example.com" in body
    assert "2. https://b.example.com" in body


def test_concept_page_no_sources_no_citations(tmp_path: Path):
    out = render_concept_page("My Concept", "summary", "body text", timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "concept.md", out)
    assert "## Sources" not in body
    assert "# Citations" not in body


def test_concept_page_description_truncated(tmp_path: Path):
    long_summary = "B" * 250
    out = render_concept_page("My Concept", long_summary, "body text", timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "concept.md", out)
    assert len(meta["description"]) == 200


# ── render_moc_page ────────────────────────────────────────────────────


def test_moc_page_frontmatter_type(tmp_path: Path):
    out = render_moc_page("My MOC", "summary", [], timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "moc.md", out)
    assert meta["type"] == "Map of Content"


def test_moc_page_body_contains_title(tmp_path: Path):
    out = render_moc_page("My MOC", "summary", [], timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "moc.md", out)
    assert "# My MOC" in body


def test_moc_page_concept_links_are_absolute(tmp_path: Path):
    links = [("concepts/foo", "Foo Concept"), ("concepts/bar", "Bar Concept")]
    out = render_moc_page("My MOC", "summary", links, timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "moc.md", out)
    assert "## Concepts" in body
    assert "[Foo Concept](/concepts/foo.md)" in body
    assert "[Bar Concept](/concepts/bar.md)" in body


def test_moc_page_empty_concept_links(tmp_path: Path):
    out = render_moc_page("My MOC", "summary", [], timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "moc.md", out)
    assert "## Concepts" in body


# ── render_reference_page ─────────────────────────────────────────────


def test_reference_page_frontmatter_type(tmp_path: Path):
    out = render_reference_page(
        "My Reference", "https://example.com/ref", "summary", "body text", timestamp=TS
    )
    meta, _body = _write_and_read(tmp_path, "reference.md", out)
    assert meta["type"] == "Reference"


def test_reference_page_body_contains_title(tmp_path: Path):
    out = render_reference_page(
        "My Reference", "https://example.com/ref", "summary", "body text", timestamp=TS
    )
    _meta, body = _write_and_read(tmp_path, "reference.md", out)
    assert "# My Reference" in body


def test_reference_page_resource_in_frontmatter(tmp_path: Path):
    url = "https://example.com/ref"
    out = render_reference_page("My Reference", url, "summary", "body text", timestamp=TS)
    meta, _body = _write_and_read(tmp_path, "reference.md", out)
    assert meta["resource"] == url


def test_reference_page_citations_section_with_url(tmp_path: Path):
    url = "https://example.com/ref"
    out = render_reference_page("My Reference", url, "summary", "body text", timestamp=TS)
    _meta, body = _write_and_read(tmp_path, "reference.md", out)
    assert "# Citations" in body
    assert f"1. {url}" in body


def test_reference_page_description_truncated(tmp_path: Path):
    long_summary = "C" * 300
    out = render_reference_page(
        "My Reference", "https://example.com", long_summary, "body text", timestamp=TS
    )
    meta, _body = _write_and_read(tmp_path, "reference.md", out)
    assert len(meta["description"]) == 200


# ── pytest entry ───────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
