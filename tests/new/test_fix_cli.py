"""Human-safe CLI tests for deterministic maintenance repairs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_fix_dry_run_is_nonmutating_and_apply_repairs_only_safe_unreviewed_pages(
    tmp_path: Path, monkeypatch
) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    editable = wiki / "concepts" / "editable.md"
    reviewed = wiki / "concepts" / "reviewed.md"
    generated = wiki / "concepts" / "generated-empty.md"
    _write_page(
        editable,
        (
            "---\ntype: Concept\ntags:\n  - has space\nrelations:\n"
            "  - target: missing\n    type: related_to\n---\n# Editable\n"
        ),
    )
    _write_page(
        reviewed,
        (
            "---\ntype: Concept\nreviewed: true\ntags:\n  - keep space\nrelations:\n"
            "  - target: missing\n    type: related_to\n---\n# Curated\n"
        ),
    )
    _write_page(generated, "---\ngenerated: true\n---\n")
    original_editable = editable.read_text(encoding="utf-8")
    original_reviewed = reviewed.read_text(encoding="utf-8")
    monkeypatch.delenv("VAULT_PATH", raising=False)

    dry_run = runner.invoke(app, ["fix", str(vault), "--dry-run", "--json"])

    assert dry_run.exit_code == 0, dry_run.output
    assert editable.read_text(encoding="utf-8") == original_editable
    assert reviewed.read_text(encoding="utf-8") == original_reviewed
    assert generated.exists()
    plan = json.loads(dry_run.output)
    assert plan["mode"] == "dry-run"
    assert plan["summary"] == {"applicable": 3, "requires_review": 1, "skipped_reviewed": 3}

    applied = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])

    assert applied.exit_code == 0, applied.output
    assert "has-space" in editable.read_text(encoding="utf-8")
    assert "relations: []" in editable.read_text(encoding="utf-8")
    assert reviewed.read_text(encoding="utf-8") == original_reviewed
    assert not generated.exists()
    backups = sorted((wiki / ".llmwiki" / "backups").rglob("*.bak"))
    assert len(backups) == 2
    editable_backup = next(
        backup for backup in backups if backup.read_text(encoding="utf-8") == original_editable
    )
    assert json.loads(applied.output)["summary"] == {
        "applied": 3,
        "requires_review": 1,
        "skipped_reviewed": 3,
    }

    repeated = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output)["summary"]["applied"] == 0
    assert len(sorted((wiki / ".llmwiki" / "backups").rglob("*.bak"))) == 2

    restore = runner.invoke(app, ["fix", str(vault), "--restore", str(editable_backup), "--json"])
    assert restore.exit_code == 0, restore.output
    assert editable.read_text(encoding="utf-8") == original_editable


def test_fix_restore_refuses_paths_outside_vault_backup_root(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "vault"
    (vault / "04-Wiki").mkdir(parents=True)
    outside = tmp_path / "outside.bak"
    outside.write_text("not a trusted backup", encoding="utf-8")
    monkeypatch.delenv("VAULT_PATH", raising=False)

    result = runner.invoke(app, ["fix", str(vault), "--restore", str(outside)])

    assert result.exit_code == 1
    assert "outside the vault backup root" in result.output
