"""Tests for 00-Inbox URL scanner — _scan_inbox_urls in cli/ingest.py.

Covers: valid .url files, blank lines, non-http URLs, non-.url extensions,
unreadable files.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from obsidian_llm_wiki.cli.ingest import (
    _archive_inbox_url_files,
    _inbox_url_paths,
    _scan_inbox_urls,
)


def _make_config(vault_dir: Path) -> SimpleNamespace:
    """Build a minimal config-like object with a .vault attribute."""
    return SimpleNamespace(vault=vault_dir)


def test_scan_inbox_no_directory(tmp_path: Path):
    """When 00-Inbox doesn't exist, returns empty list."""
    config = _make_config(tmp_path)
    assert _scan_inbox_urls(config) == []


def test_scan_inbox_empty_directory(tmp_path: Path):
    """Empty 00-Inbox directory returns empty list."""
    (tmp_path / "00-Inbox").mkdir()
    config = _make_config(tmp_path)
    assert _scan_inbox_urls(config) == []


def test_scan_inbox_valid_url_files(tmp_path: Path):
    """Valid .url files with http/https URLs are returned."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "page1.url").write_text("https://example.com/article1\n", encoding="utf-8")
    (inbox / "page2.url").write_text("http://example.org/article2\n", encoding="utf-8")
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/article1", "http://example.org/article2"]


def test_scan_inbox_skips_blank_lines(tmp_path: Path):
    """Blank lines before the URL are skipped — first non-empty http line wins."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "blank.url").write_text(
        "\n\n   \nhttps://example.com/real\nhttps://example.com/second\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/real"]


def test_scan_inbox_skips_non_http_lines(tmp_path: Path):
    """Lines that don't start with http:// or https:// are skipped."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "non_http.url").write_text(
        "ftp://files.example.com/data\nmailto:test@test.com\nhttps://example.com/valid\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/valid"]


def test_scan_inbox_skips_non_url_extensions(tmp_path: Path):
    """Files without .url extension are ignored."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "notes.md").write_text("https://example.com/should-not-appear\n", encoding="utf-8")
    (inbox / "data.txt").write_text("https://example.com/also-not\n", encoding="utf-8")
    (inbox / "real.url").write_text("https://example.com/yes\n", encoding="utf-8")
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/yes"]


def test_scan_inbox_skips_unreadable_files(tmp_path: Path):
    """Unreadable .url files (OSError) are silently skipped."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "good.url").write_text("https://example.com/ok\n", encoding="utf-8")
    unreadable = inbox / "bad.url"
    unreadable.write_text("https://example.com/nope\n", encoding="utf-8")
    # Remove read permission.
    unreadable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    unreadable.chmod(0o000)

    config = _make_config(tmp_path)
    try:
        if os.geteuid() == 0:
            pytest.skip("Running as root — permissions are bypassed")
        urls = _scan_inbox_urls(config)
        # The unreadable file should be skipped, the good one returned.
        assert urls == ["https://example.com/ok"]
    finally:
        # Restore permissions for cleanup.
        unreadable.chmod(0o644)


def test_scan_inbox_sorted_order(tmp_path: Path):
    """URLs are returned in sorted file order."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "c.url").write_text("https://c.com\n", encoding="utf-8")
    (inbox / "a.url").write_text("https://a.com\n", encoding="utf-8")
    (inbox / "b.url").write_text("https://b.com\n", encoding="utf-8")
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://a.com", "https://b.com", "https://c.com"]


def test_scan_inbox_dedup_not_applied(tmp_path: Path):
    """Each .url file contributes one URL — duplicates across files are
    NOT deduplicated by _scan_inbox_urls (that's the caller's job)."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "a.url").write_text("https://example.com/dup\n", encoding="utf-8")
    (inbox / "b.url").write_text("https://example.com/dup\n", encoding="utf-8")
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/dup", "https://example.com/dup"]


def test_scan_inbox_skips_subdirectories(tmp_path: Path):
    """Directories inside 00-Inbox are skipped (is_file() check)."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "subdir.url").mkdir()  # a directory named *.url
    (inbox / "real.url").write_text("https://example.com/real\n", encoding="utf-8")
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://example.com/real"]


def test_scan_inbox_only_takes_first_url_per_file(tmp_path: Path):
    """Only the first valid URL per file is extracted (break after first match)."""
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "multi.url").write_text(
        "https://first.com\nhttps://second.com\nhttps://third.com\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path)
    urls = _scan_inbox_urls(config)
    assert urls == ["https://first.com"]


def test_inbox_url_paths_preserve_duplicate_queue_records(tmp_path: Path):
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    (inbox / "a.url").write_text("https://example.com/dup\n", encoding="utf-8")
    (inbox / "b.url").write_text("https://example.com/dup\n", encoding="utf-8")

    paths = _inbox_url_paths(_make_config(tmp_path))

    assert [path.name for path in paths["https://example.com/dup"]] == ["a.url", "b.url"]


def test_archive_inbox_url_files_moves_only_confirmed_records(tmp_path: Path):
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir()
    queued = inbox / "article.url"
    queued.write_text("https://example.com/article\n", encoding="utf-8")

    archived = _archive_inbox_url_files([queued], _make_config(tmp_path))

    assert not queued.exists()
    assert archived == [inbox / "processed" / "article.url"]
    assert archived[0].read_text(encoding="utf-8") == "https://example.com/article\n"
