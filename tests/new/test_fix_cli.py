"""Human-safe CLI tests for deterministic maintenance repairs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def test_fix_help_describes_restore_as_refused_compatibility_option() -> None:
    result = CliRunner().invoke(app, ["fix", "--help"], terminal_width=200)

    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.replace("│", " ").split())
    assert "--restore" in help_text
    assert "Retained compatibility option; always refused under immutable publication" in help_text
    assert "Restore one trusted backup path" not in help_text


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
    assert plan["summary"] == {"applicable": 3, "requires_review": 2, "skipped_reviewed": 4}

    from unittest.mock import AsyncMock

    from obsidian_llm_wiki.config import Config
    from tests.new.grounding_doubles import legacy_fixture_llm, producer_call_count
    from tests.new.immutable_semantic_helpers import assert_preserved, file_state, published_root
    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    before = file_state(vault)
    refused = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])
    assert refused.exit_code == 1
    assert "Refusing maintenance without source inputs" in json.loads(refused.output)["error"]
    assert file_state(vault) == before
    _write_page(
        wiki / "sources" / "alpha.md",
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8,
    )
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    before = file_state(vault)
    applied = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])
    assert applied.exit_code == 0, applied.output
    payload = json.loads(applied.output)
    assert payload["mode"] == "publish" and payload["summary"]["pages_published"] > 0
    assert producer_call_count(llm) == 1
    assert_preserved(vault, before)
    root = published_root(Config(vault_path=str(vault)))
    assert root.name == payload["generation"]
    old = file_state(root)
    repeated = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output)["generation"] != root.name
    assert file_state(root) == old
    assert producer_call_count(llm) == 1


def test_fix_restore_refuses_paths_outside_vault_backup_root(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "vault"
    (vault / "04-Wiki").mkdir(parents=True)
    outside = tmp_path / "outside.bak"
    outside.write_text("not a trusted backup", encoding="utf-8")
    monkeypatch.delenv("VAULT_PATH", raising=False)

    result = runner.invoke(app, ["fix", str(vault), "--restore", str(outside)])

    assert result.exit_code == 1
    assert "Refusing --restore" in result.output
    assert outside.read_text(encoding="utf-8") == "not a trusted backup"
