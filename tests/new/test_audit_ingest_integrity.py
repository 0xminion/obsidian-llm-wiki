"""Offline regression tests for audit-confirmed ingest integrity failures."""

from __future__ import annotations

import importlib
import json
import re

from typer.testing import CliRunner

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.render.obsidian import render_source_page


def _invoke(monkeypatch, config, *args, extractor=None):
    from obsidian_llm_wiki.cli import app
    from obsidian_llm_wiki.ingest import extractors

    module = importlib.import_module("obsidian_llm_wiki.cli.ingest")
    monkeypatch.setattr(module, "resolve_vault", lambda _vault: (config.vault, config))
    if extractor is not None:
        monkeypatch.setattr(extractors, "extract", extractor)
    return CliRunner().invoke(
        app, ["ingest", str(config.vault), "--parallel", "0", "--json", *args]
    )


def _events(result):
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def test_edited_duplicate_target_does_not_discard_new_clipping(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    url = "https://example.test/article"
    old = ("Original evidence. " * 40).strip()
    new = ("Corrected evidence. " * 40).strip()
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        url,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Article", content=old, url=value),
    )
    assert result.exit_code == 0, result.output
    target = config.sources_dir / "article.md"
    target.write_text(render_source_page(SourceDoc(title="Article", content=new, url=url)))
    config.clippings_dir.mkdir()
    (config.clippings_dir / "old-evidence.md").write_text(f"---\ntitle: Evidence\n---\n{old}")
    result = _invoke(monkeypatch, config, "--skip-synthesis")
    assert result.exit_code == 0, result.output
    assert any(old in path.read_text() for path in config.sources_dir.glob("*.md"))
    assert new in target.read_text()


def test_refresh_failure_and_review_protection_keep_last_good_source(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    url = "https://example.test/protected"
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        url,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Protected", content="Evidence. " * 90, url=value),
    )
    assert result.exit_code == 0, result.output
    target = config.sources_dir / "protected.md"
    original = target.read_bytes()

    def unavailable(_url):
        raise RuntimeError("fixture unavailable")

    result = _invoke(
        monkeypatch, config, "--url", url, "--refresh", "--skip-synthesis", extractor=unavailable
    )
    assert any(event.get("status") == "failed" for event in _events(result))
    assert target.read_bytes() == original
    # The retry queue intentionally backs off this failed URL. Verify review
    # protection independently on a second source with no pending retry lease.
    reviewed_url = "https://example.test/reviewed"
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        reviewed_url,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(
            title="Reviewed", content="Human evidence. " * 90, url=value
        ),
    )
    assert result.exit_code == 0, result.output
    target = config.sources_dir / "reviewed.md"
    target.write_text(target.read_text().replace("type: Source", "reviewed: true\ntype: Source"))
    reviewed = target.read_bytes()
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        reviewed_url,
        "--refresh",
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Protected", content="Changed. " * 90, url=value),
    )
    assert any("reviewed" in event.get("error", "") for event in _events(result))
    assert target.read_bytes() == reviewed


def test_refresh_replaces_same_source_in_place_with_backup(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    url = "https://example.test/revised"
    old = "Version one evidence. " * 40
    new = "Version two corrected evidence. " * 40
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        url,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Original title", content=old, url=value),
    )
    assert result.exit_code == 0, result.output
    target = config.sources_dir / "original-title.md"
    original = target.read_bytes()
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        url,
        "--refresh",
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Revised title", content=new, url=value),
    )
    assert result.exit_code == 0, result.output
    assert new.strip() in target.read_text()
    assert list(config.sources_dir.glob("*.md")) == [target]
    assert any(p.read_bytes() == original for p in (config.llmwiki_dir / "backups").rglob("*.bak"))


def test_podcast_preview_does_not_write_transcript_cache(tmp_path, monkeypatch):
    from obsidian_llm_wiki.ingest.extractors import podcast

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(
        podcast,
        "_fetch_defuddle_md_metadata",
        lambda _url: {"title": "Episode", "body": "## Transcript\n" + "Evidence sentence. " * 50},
    )
    monkeypatch.setattr(podcast, "_resolve_episode_asset", lambda *_a, **_k: podcast.EpisodeAsset())
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        "https://example.test/episode",
        "--preview",
        extractor=podcast._extract_podcast,
    )
    assert result.exit_code == 0, result.output
    assert any(event.get("status") == "succeeded" for event in _events(result))
    assert list(tmp_path.iterdir()) == []


def test_saturated_archives_preserve_existing_files(tmp_path):
    from obsidian_llm_wiki.cli.ingest import _archive_clipping, _archive_inbox_url_files

    config = Config(vault_path=str(tmp_path))
    for parent, suffix in ((config.clippings_dir, ".md"), (tmp_path / "00-Inbox", ".url")):
        archive = parent / "processed"
        archive.mkdir(parents=True)
        original = archive / f"repeat{suffix}"
        original.write_text("original")
        for index in range(1, 100):
            (archive / f"repeat-{index}{suffix}").write_text("occupied")
        pending = parent / f"repeat{suffix}"
        pending.write_text("new")
        if suffix == ".md":
            assert _archive_clipping(pending, config) is None
        else:
            assert _archive_inbox_url_files([pending], config) == []
        assert original.read_text() == "original"
        assert pending.read_text() == "new"


def test_backfilled_heading_source_is_deduplicated_by_same_canonical_body(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    source = SourceDoc(
        title="Original heading", content="# Original heading\n\n" + "Evidence. " * 80
    )
    config.sources_dir.mkdir(parents=True)
    target = config.sources_dir / "original.md"
    target.write_text(render_source_page(source))
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        "https://example.test/alias",
        "--skip-synthesis",
        extractor=lambda _url: source,
    )
    assert result.exit_code == 0, result.output
    assert list(config.sources_dir.glob("*.md")) == [target]
    assert any(event.get("duplicate_of") == target.name for event in _events(result))


def test_resume_cannot_mutate_state_in_read_only_modes(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    for flag in ("--plan", "--preview"):
        result = _invoke(monkeypatch, config, "--resume", flag)
        assert result.exit_code == 2
        # Rich can split a flag across ANSI spans when CI forces color.
        visible_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--resume" in visible_output
        assert list(tmp_path.iterdir()) == []


def test_ingest_index_cannot_redirect_reads_outside_sources(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.registry import VaultRegistry

    config = Config(vault_path=str(tmp_path))
    config.sources_dir.mkdir(parents=True)
    outside = config.wiki_dir / "outside.md"
    outside.write_text("---\ntitle: Outside\n---\nShould not be an ingest duplicate.")
    url = "https://example.test/input"
    with VaultRegistry(config.llmwiki_dir) as registry:
        registry.upsert_ingest_hash(
            requested_url=url,
            canonical_url=url,
            content_hash="corrupted",
            output_file="../outside.md",
        )
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        url,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Safe", content="Real evidence. " * 40, url=value),
    )
    assert result.exit_code == 0, result.output
    assert not any(event.get("duplicate_of") for event in _events(result))
    assert (config.sources_dir / "safe.md").exists()


def test_numeric_clipping_metadata_does_not_abort_collection(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    config.clippings_dir.mkdir()
    (config.clippings_dir / "numeric.md").write_text(
        "---\ntitle: 123\nurl: [invalid]\n---\n" + "User evidence. " * 50
    )
    result = _invoke(monkeypatch, config, "--skip-synthesis")
    assert result.exit_code == 0, result.output
    assert (config.sources_dir / "numeric.md").is_file()


def test_bad_inbox_encoding_does_not_abort_valid_planned_urls(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "bad.url").write_bytes(b"\xff")
    (inbox / "good.url").write_text("https://example.test/valid")
    result = _invoke(monkeypatch, config, "--plan")
    assert result.exit_code == 0, result.output
    assert any(event.get("source") == "https://example.test/valid" for event in _events(result))
    assert not config.llmwiki_dir.exists()


def test_refresh_tracks_original_request_across_changed_redirects(tmp_path, monkeypatch):
    from obsidian_llm_wiki.ingest.provenance import stamp_source

    config = Config(vault_path=str(tmp_path))
    requested = "https://example.test/latest"

    def revision(number):
        return stamp_source(
            SourceDoc(
                title="Versioned article",
                content=f"Revision {number}. " * 80,
                url=f"https://example.test/v{number}",
            ),
            requested_url=requested,
            extractor="fixture",
        )

    result = _invoke(
        monkeypatch,
        config,
        "--url",
        requested,
        "--skip-synthesis",
        extractor=lambda _url: revision(1),
    )
    assert result.exit_code == 0, result.output
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        requested,
        "--refresh",
        "--skip-synthesis",
        extractor=lambda _url: revision(2),
    )
    assert result.exit_code == 0, result.output
    pages = list(config.sources_dir.glob("*.md"))
    assert len(pages) == 1
    assert "Revision 2." in pages[0].read_text()


def test_refreshed_shared_page_does_not_keep_stale_url_alias(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path))
    first = "https://example.test/first"
    alias = "https://example.test/alias"
    old = ("Shared old evidence. " * 50).strip()
    new = ("Updated first evidence. " * 50).strip()
    for url in (first, alias):
        result = _invoke(
            monkeypatch,
            config,
            "--url",
            url,
            "--skip-synthesis",
            extractor=lambda value: SourceDoc(title="Shared", content=old, url=value),
        )
        assert result.exit_code == 0, result.output
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        first,
        "--refresh",
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Shared", content=new, url=value),
    )
    assert result.exit_code == 0, result.output
    result = _invoke(
        monkeypatch,
        config,
        "--url",
        alias,
        "--skip-synthesis",
        extractor=lambda value: SourceDoc(title="Alias", content=old, url=value),
    )
    assert result.exit_code == 0, result.output
    assert any(old in path.read_text() for path in config.sources_dir.glob("*.md"))
