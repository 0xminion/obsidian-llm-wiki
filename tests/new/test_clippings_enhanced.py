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


def test_processed_clippings_archived_to_subdirectory(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    clipping = _write_clipping(
        vault, "clip1.md", "Clip One", body, url="https://example.com/clip1"
    )

    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis"]
    )
    assert result.exit_code == 0

    # Clipping should have been moved to 02-Clippings/processed/
    processed_dir = vault / "02-Clippings" / "processed"
    assert processed_dir.is_dir()
    assert (processed_dir / "clip1.md").exists()
    # Original location should no longer have the file.
    assert not clipping.exists()

    # Re-running should not find any new clippings.
    result = runner.invoke(
        app, ["ingest", str(vault), "--skip-synthesis", "--json"]
    )
    assert result.exit_code == 0


def test_duplicate_clipping_archived_even_though_skipped(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    body = "This is a substantial article body with enough content to pass the quality gate. " * 8
    _write_clipping(vault, "first.md", "First", body, url="https://example.com/first")

    # First ingest processes and archives.
    result = runner.invoke(app, ["ingest", str(vault), "--skip-synthesis"])
    assert result.exit_code == 0

    # Second clipping with same content — should be deduped AND archived.
    _write_clipping(vault, "second.md", "Second", body, url="https://example.com/second")
    result = runner.invoke(app, ["ingest", str(vault), "--skip-synthesis"])
    assert result.exit_code == 0

    # Both clippings should be in processed/ — even the duplicate.
    processed = vault / "02-Clippings" / "processed"
    assert (processed / "first.md").exists()
    assert (processed / "second.md").exists()

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
