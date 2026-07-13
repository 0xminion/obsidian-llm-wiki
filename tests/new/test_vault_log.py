"""Tests for render.log — vault chronological log."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.render.log import (
    append_log,
    format_log_entry,
    log_to_file,
    read_log_entries,
)

# ── format_log_entry ────────────────────────────────────────────────────


def test_format_log_entry_basic():
    entry = format_log_entry("build", "rendered vault", timestamp="2026-01-01T00:00:00Z")
    assert entry == "## [2026-01-01T00:00:00Z] BUILD: rendered vault"


def test_format_log_entry_uppercases_action():
    entry = format_log_entry("ingest", "fetched 3 sources", timestamp="2026-01-01T00:00:00Z")
    assert "INGEST" in entry


def test_format_log_entry_with_body_string():
    entry = format_log_entry("fix", "resolved contradiction",
                             timestamp="2026-01-01T00:00:00Z", body="some detail")
    assert entry.startswith("## [2026-01-01T00:00:00Z] FIX: resolved contradiction")
    assert "some detail" in entry


def test_format_log_entry_with_body_list():
    entry = format_log_entry("build", "done",
                             timestamp="2026-01-01T00:00:00Z",
                             body=["line 1", "line 2"])
    assert "line 1" in entry
    assert "line 2" in entry
    assert entry.startswith("## [2026-01-01T00:00:00Z] BUILD: done")


def test_format_log_entry_empty_body_returns_header_only():
    entry = format_log_entry("query", "search", timestamp="2026-01-01T00:00:00Z", body=[])
    assert entry == "## [2026-01-01T00:00:00Z] QUERY: search"


# ── log_to_file ─────────────────────────────────────────────────────────


def test_log_to_file_creates_new_file(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "first entry", timestamp="2026-01-01T00:00:00Z")
    content = log_path.read_text(encoding="utf-8")
    assert "# Vault Log" in content
    assert "## [2026-01-01T00:00:00Z] BUILD: first entry" in content


def test_log_to_file_appends_to_existing(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "first entry", timestamp="2026-01-01T00:00:00Z")
    log_to_file(log_path, "query", "second entry", timestamp="2026-01-01T01:00:00Z")
    content = log_path.read_text(encoding="utf-8")
    assert "BUILD: first entry" in content
    assert "QUERY: second entry" in content
    # Both should appear in chronological order.
    assert content.index("BUILD: first entry") < content.index("QUERY: second entry")


def test_log_to_file_preserves_header_on_append(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "first", timestamp="2026-01-01T00:00:00Z")
    log_to_file(log_path, "query", "second", timestamp="2026-01-01T01:00:00Z")
    content = log_path.read_text(encoding="utf-8")
    # Header should appear only once.
    assert content.count("# Vault Log") == 1


def test_log_to_file_with_body(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(
        log_path, "build", "done",
        timestamp="2026-01-01T00:00:00Z",
        body=["- sources: 3", "- concepts: 10"],
    )
    content = log_path.read_text(encoding="utf-8")
    assert "- sources: 3" in content
    assert "- concepts: 10" in content


# ── append_log ──────────────────────────────────────────────────────────


def test_append_log_resolves_bundle_dir(tmp_path: Path):
    bundle_dir = tmp_path / "04-Wiki"
    bundle_dir.mkdir()
    log_path = append_log(bundle_dir, "build", "done", timestamp="2026-01-01T00:00:00Z")
    assert log_path == bundle_dir / "log.md"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "BUILD: done" in content


def test_append_log_creates_bundle_dir_if_missing(tmp_path: Path):
    bundle_dir = tmp_path / "04-Wiki"
    log_path = append_log(bundle_dir, "ingest", "fetched", timestamp="2026-01-01T00:00:00Z")
    assert log_path.exists()


# ── read_log_entries ────────────────────────────────────────────────────


def test_read_log_entries_empty_when_no_file(tmp_path: Path):
    entries = read_log_entries(tmp_path / "nonexistent.md")
    assert entries == []


def test_read_log_entries_returns_all(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "first", timestamp="2026-01-01T00:00:00Z")
    log_to_file(log_path, "query", "second", timestamp="2026-01-01T01:00:00Z")
    entries = read_log_entries(log_path)
    assert len(entries) == 2
    assert "BUILD: first" in entries[0]
    assert "QUERY: second" in entries[1]


def test_read_log_entries_filters_by_action(tmp_path: Path):
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "first", timestamp="2026-01-01T00:00:00Z")
    log_to_file(log_path, "query", "second", timestamp="2026-01-01T01:00:00Z")
    log_to_file(log_path, "build", "third", timestamp="2026-01-01T02:00:00Z")
    build_entries = read_log_entries(log_path, action="build")
    assert len(build_entries) == 2
    assert all("BUILD" in e for e in build_entries)


def test_read_log_entries_grep_parseable(tmp_path: Path):
    """Entries must match `grep '^## ['` — start with '## ['."""
    log_path = tmp_path / "log.md"
    log_to_file(log_path, "build", "entry", timestamp="2026-01-01T00:00:00Z")
    entries = read_log_entries(log_path)
    for e in entries:
        assert e.startswith("## [")


# ── Multiple actions ────────────────────────────────────────────────────


def test_all_pipeline_actions_recorded(tmp_path: Path):
    """Simulate all pipeline actions appearing in the log."""
    log_path = tmp_path / "log.md"
    ts = "2026-01-01T00:00:00Z"
    for action, detail in [
        ("ingest", "fetched 3 sources"),
        ("build", "rendered vault"),
        ("query", "searched concepts"),
        ("fix", "resolved contradiction"),
    ]:
        log_to_file(log_path, action, detail, timestamp=ts)
    entries = read_log_entries(log_path)
    assert len(entries) == 4
    assert any("INGEST" in e for e in entries)
    assert any("BUILD" in e for e in entries)
    assert any("QUERY" in e for e in entries)
    assert any("FIX" in e for e in entries)
