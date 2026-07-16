"""Backups must support source filenames near filesystem component limits."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.backups import backup_file, list_backups


def test_backup_file_hashes_only_the_overlong_display_component(tmp_path: Path):
    source = tmp_path / (("中" * 79) + ".md")
    source.write_text("source content", encoding="utf-8")

    backup = backup_file(source, tmp_path / "backups")

    assert backup.exists()
    assert backup.name.endswith(".bak")
    assert "sha256-" in backup.name
    assert len(backup.name.encode("utf-8")) < 255
    assert list_backups(source, tmp_path / "backups") == [backup]
