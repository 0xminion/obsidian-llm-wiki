"""Tests for atomic, bounded page backups."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.core.backups import backup_file, list_backups


def test_backup_file_copies_current_content_under_supplied_root(tmp_path: Path) -> None:
    source = tmp_path / "concept.md"
    source.write_text("first version", encoding="utf-8")
    backups_root = tmp_path / ".llmwiki" / "backups"

    backup = backup_file(source, backups_root, max_backups=3)

    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == "first version"
    assert backup.parent.is_relative_to(backups_root)
    assert list_backups(source, backups_root) == [backup]


def test_backup_file_rotates_oldest_backups_without_touching_newest(tmp_path: Path) -> None:
    source = tmp_path / "concept.md"
    backups_root = tmp_path / "backups"

    for version in range(4):
        source.write_text(f"version {version}", encoding="utf-8")
        backup_file(source, backups_root, max_backups=2)

    backups = list_backups(source, backups_root)
    assert len(backups) == 2
    assert [path.read_text(encoding="utf-8") for path in backups] == ["version 2", "version 3"]
    assert not list(backups_root.rglob("*.tmp"))


def test_backup_file_bounds_utf8_backup_component_length(tmp_path: Path) -> None:
    source = tmp_path / f"{'你' * 70}.md"
    source.write_text("content", encoding="utf-8")

    backup = backup_file(source, tmp_path / "backups")

    assert len(backup.name.encode()) <= 255
    assert backup.read_text(encoding="utf-8") == "content"


def test_backup_file_rejects_non_positive_retention(tmp_path: Path) -> None:
    source = tmp_path / "concept.md"
    source.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="max_backups"):
        backup_file(source, tmp_path / "backups", max_backups=0)


def test_backup_file_requires_an_existing_regular_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backup_file(tmp_path / "missing.md", tmp_path / "backups")

    directory = tmp_path / "directory.md"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        backup_file(directory, tmp_path / "backups")
