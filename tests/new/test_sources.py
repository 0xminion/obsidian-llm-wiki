"""Tests for obsidian_llm_wiki.ingest.sources — source loading helpers."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.ingest.sources import load_source_file, load_sources_from_dir
from obsidian_llm_wiki.render.obsidian import build_frontmatter


def _write_source(dir_path: Path, name: str, title: str, body: str, url: str = "") -> Path:
    """Write a source markdown file with frontmatter."""
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / name
    fm: dict[str, object] = {"type": "Source", "title": title}
    if url:
        fm["url"] = url
    content = build_frontmatter(fm) + "\n" + body
    path.write_text(content)
    return path


def test_load_source_file_with_frontmatter(tmp_path: Path):
    """load_source_file extracts title, url, and body from frontmatter."""
    _write_source(
        tmp_path, "article.md", "My Article", "Body text here.", url="https://example.com",
    )
    doc = load_source_file(tmp_path / "article.md")
    assert doc is not None
    assert doc.title == "My Article"
    assert doc.url == "https://example.com"
    assert "Body text here." in doc.content
    assert doc.source_file == "article.md"


def test_load_source_file_no_frontmatter(tmp_path: Path):
    """load_source_file uses filename as title when no frontmatter."""
    path = tmp_path / "no-fm.md"
    path.write_text("# Heading\n\nBody text.")
    doc = load_source_file(path)
    assert doc is not None
    assert doc.title == "no-fm"  # falls back to filename
    assert "Body text." in doc.content


def test_load_source_file_empty(tmp_path: Path):
    """load_source_file returns None for empty files."""
    path = tmp_path / "empty.md"
    path.write_text("")
    assert load_source_file(path) is None


def test_load_source_file_missing(tmp_path: Path):
    """load_source_file returns None for missing files."""
    assert load_source_file(tmp_path / "nonexistent.md") is None


def test_load_sources_from_dir_multiple(tmp_path: Path):
    """load_sources_from_dir loads all .md files keyed by filename."""
    sources_dir = tmp_path / "sources"
    _write_source(sources_dir, "a.md", "Alpha", "Alpha body.")
    _write_source(sources_dir, "b.md", "Beta", "Beta body.")

    result = load_sources_from_dir(sources_dir)
    assert len(result) == 2
    assert "a.md" in result
    assert "b.md" in result
    assert result["a.md"].title == "Alpha"
    assert result["b.md"].title == "Beta"


def test_load_sources_from_dir_empty(tmp_path: Path):
    """load_sources_from_dir returns empty dict for empty directory."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    assert load_sources_from_dir(sources_dir) == {}


def test_load_sources_from_dir_missing_dir(tmp_path: Path):
    """load_sources_from_dir returns empty dict when directory doesn't exist."""
    assert load_sources_from_dir(tmp_path / "nonexistent") == {}


def test_load_sources_from_dir_skips_empty_files(tmp_path: Path):
    """load_sources_from_dir skips empty files."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    _write_source(sources_dir, "good.md", "Good", "Good body.")
    (sources_dir / "empty.md").write_text("")

    result = load_sources_from_dir(sources_dir)
    assert len(result) == 1
    assert "good.md" in result
    assert "empty.md" not in result
