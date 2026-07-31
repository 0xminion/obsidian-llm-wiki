"""Pipeline state must never be treated as live vault content."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_health_and_strict_validation_ignore_quarantined_pipeline_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    _write(
        wiki / "sources" / "live.md",
        "---\ntype: Source\ntitle: Live source\n---\n# Live source\n",
    )
    _write(
        wiki / ".llmwiki" / "quarantine" / "broken.md",
        "---\ntype: Concept\nrelations:\n  - missing|related_to|Missing\n---\n[[missing]]\n",
    )
    _write(wiki / "views" / "concepts-by-confidence.md", "```dataview\nTABLE\n```\n")
    monkeypatch.delenv("VAULT_PATH", raising=False)

    health = runner.invoke(app, ["health", str(vault), "--json"])
    validated = runner.invoke(app, ["validate", str(vault), "--strict"])

    assert health.exit_code == 0, health.output
    assert json.loads(health.output)["files_scanned"] == 1
    assert json.loads(health.output)["findings"] == []
    assert validated.exit_code == 0, validated.output
    assert ".llmwiki" not in validated.output
    assert "views" not in validated.output
