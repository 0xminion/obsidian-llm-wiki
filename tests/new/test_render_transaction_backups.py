"""Render rollback treats historical backups as immutable append-only state."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.render.obsidian import _RenderTransaction


def test_rollback_keeps_existing_backups_and_removes_backups_created_mid_render(tmp_path: Path):
    bundle = tmp_path / "04-Wiki"
    page = bundle / "concepts" / "alpha.md"
    page.parent.mkdir(parents=True)
    page.write_text("before", encoding="utf-8")
    backups = bundle / ".llmwiki" / "backups" / "digest"
    backups.mkdir(parents=True)
    historical = backups / "old-alpha.md.bak"
    historical.write_text("historical", encoding="utf-8")

    transaction = _RenderTransaction(bundle)
    page.write_text("after", encoding="utf-8")
    new_backup = backups / "new-alpha.md.bak"
    new_backup.write_text("new", encoding="utf-8")
    transaction.rollback()

    assert page.read_text(encoding="utf-8") == "before"
    assert historical.read_text(encoding="utf-8") == "historical"
    assert not new_backup.exists()
