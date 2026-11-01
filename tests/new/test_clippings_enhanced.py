"""Tests for clippings dedup, archiving, and metadata preservation."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "04-Wiki" / "sources").mkdir(parents=True)
    (vault / "02-Clippings").mkdir(parents=True)
    (vault / ".env").write_text("LLM_MODEL=test\n", encoding="utf-8")
    return vault


def _write_clipping(vault: Path, name: str, title: str, body: str, url: str = "") -> Path:
    clipping = vault / "02-Clippings" / name
    frontmatter = f"---\ntitle: {title}\n"
    if url:
        frontmatter += f"source_url: {url}\n"
    frontmatter += "---\n\n"
    clipping.write_text(frontmatter + body, encoding="utf-8")
    return clipping


def test_content_hash_dedup_skips_duplicate_source(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    _write_clipping(vault, "article-a.md", "Article A", body, url="https://example.com/a")

    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis", "--json"]
    )
    assert result.exit_code == 0

    sources_dir = vault / "04-Wiki" / "sources"
    assert len(list(sources_dir.glob("*.md"))) == 1

    # Re-run with the same content under a different clipping name.
    _write_clipping(vault, "article-b.md", "Article B", body, url="https://example.com/b")
    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis", "--json"]
    )
    assert result.exit_code == 0

    # The duplicate should NOT have created a second source file.
    md_files = [f for f in sources_dir.glob("*.md") if f.name != "failed_urls.md"]
    assert len(md_files) == 1


def test_processed_clippings_retained_and_reingest_is_idempotent(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    clipping = _write_clipping(
        vault, "clip1.md", "Clip One", body, url="https://example.com/clip1"
    )
    before = (clipping.read_bytes(), clipping.stat().st_ino, clipping.stat().st_mtime_ns)

    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis"]
    )
    assert result.exit_code == 0

    # Processing is recorded operationally; human clippings stay in place.
    processed_dir = vault / "02-Clippings" / "processed"
    assert not processed_dir.exists()
    assert (clipping.read_bytes(), clipping.stat().st_ino, clipping.stat().st_mtime_ns) == before

    # Re-running must deduplicate without moving or rewriting the clipping.
    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis", "--json"]
    )
    assert result.exit_code == 0
    assert (clipping.read_bytes(), clipping.stat().st_ino, clipping.stat().st_mtime_ns) == before
    assert len(list((vault / "04-Wiki" / "sources").glob("*.md"))) == 1


def test_duplicate_clipping_retained_without_duplicate_source(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    first = _write_clipping(vault, "first.md", "First", body, url="https://example.com/first")
    first_before = (first.read_bytes(), first.stat().st_ino, first.stat().st_mtime_ns)

    # First ingest processes without moving the human input.
    result = runner.invoke(app, ["ingest", str(vault), "--skip-synthesis"])
    assert result.exit_code == 0

    # A duplicate remains in place too; deduplication must not delete either input.
    second = _write_clipping(vault, "second.md", "Second", body, url="https://example.com/second")
    second_before = (second.read_bytes(), second.stat().st_ino, second.stat().st_mtime_ns)
    result = runner.invoke(app, ["ingest", str(vault), "--skip-synthesis"])
    assert result.exit_code == 0

    assert not (vault / "02-Clippings" / "processed").exists()
    assert (first.read_bytes(), first.stat().st_ino, first.stat().st_mtime_ns) == first_before
    assert (second.read_bytes(), second.stat().st_ino, second.stat().st_mtime_ns) == second_before

    # Only one source file should exist.
    sources = [
        f for f in (vault / "04-Wiki" / "sources").glob("*.md")
        if f.name != "failed_urls.md"
    ]
    assert len(sources) == 1


def test_clipping_metadata_preserved_in_source_frontmatter(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    _write_clipping(
        vault,
        "with-meta.md",
        "With Metadata",
        body,
        url="https://example.com/meta-test",
    )

    result = runner.invoke(app, ["ingest", str(vault), "--skip-synthesis"])
    assert result.exit_code == 0

    source_files = [
        f for f in (vault / "04-Wiki" / "sources").glob("*.md")
        if f.name != "failed_urls.md"
    ]
    assert len(source_files) == 1
    content = source_files[0].read_text(encoding="utf-8")
    # Source URL should be preserved in frontmatter.
    assert "https://example.com/meta-test" in content
    assert "type: Source" in content
