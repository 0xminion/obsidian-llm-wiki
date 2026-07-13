"""CLI tests for machine-readable vault-health findings."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.contradictions import (
    ContradictionRecord,
    ContradictionStatus,
    ContradictionStore,
)

runner = CliRunner()


def _write_page(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_health_json_is_stable_and_includes_typed_findings_and_contradiction_counts(
    tmp_path: Path, monkeypatch
) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    _write_page(
        wiki / "concepts" / "alpha.md",
        (
            "---\ntype: Concept\ntags:\n  - has space\nrelations:\n"
            "  - target: missing\n    type: related_to\n---\n# Alpha\n"
        ),
    )
    _write_page(wiki / "concepts" / "generated-empty.md", "---\ngenerated: true\n---\n")
    store = ContradictionStore(wiki / ".llmwiki" / "contradictions.json")
    store.add(
        ContradictionRecord(
            id="c-1",
            summary="Conflicting facts",
            status=ContradictionStatus.PENDING_FIX,
        )
    )
    monkeypatch.delenv("VAULT_PATH", raising=False)

    first = runner.invoke(app, ["health", str(vault), "--json"])
    second = runner.invoke(app, ["health", str(vault), "--json"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first.output == second.output
    payload = json.loads(first.output)
    assert payload["files_scanned"] == 2
    assert payload["summary"] == {
        "broken_relation": 1,
        "empty_generated_stub": 1,
        "orphan_concept": 1,
        "tag_normalization": 1,
    }
    assert payload["contradictions"] == [
        {
            "evidence": [],
            "id": "c-1",
            "sources": [],
            "status": "pending_fix",
            "summary": "Conflicting facts",
        }
    ]
    assert payload["contradiction_summary"] == {"pending_fix": 1}
    assert payload["findings"] == [
        {
            "details": {"target": "missing"},
            "kind": "broken_relation",
            "message": "Relation target does not exist: missing",
            "path": "concepts/alpha.md",
            "reviewed": False,
            "severity": "error",
        },
        {
            "details": {},
            "kind": "orphan_concept",
            "message": "Concept is not in any MoC",
            "path": "concepts/alpha.md",
            "reviewed": False,
            "severity": "warning",
        },
        {
            "details": {"tag": "has space", "normalized_tag": "has-space"},
            "kind": "tag_normalization",
            "message": "Tag needs Obsidian normalization: has space",
            "path": "concepts/alpha.md",
            "reviewed": False,
            "severity": "warning",
        },
        {
            "details": {},
            "kind": "empty_generated_stub",
            "message": "Generated page has an empty body",
            "path": "concepts/generated-empty.md",
            "reviewed": False,
            "severity": "warning",
        },
    ]


def test_health_and_fix_offer_parenthetical_aliases_without_applying_them(
    tmp_path: Path, monkeypatch
) -> None:
    """Alias suggestions are deterministic, visible, and always proposal-only."""
    vault = tmp_path / "vault"
    source = vault / "04-Wiki" / "sources" / "report.md"
    _write_page(
        source,
        "---\ntype: Source\ntitle: Quarterly Research Report (QRR)\n---\n# Report\n",
    )
    original = source.read_text(encoding="utf-8")
    monkeypatch.delenv("VAULT_PATH", raising=False)

    health = runner.invoke(app, ["health", str(vault), "--json"])
    plan = runner.invoke(app, ["fix", str(vault), "--dry-run", "--json"])
    applied = runner.invoke(app, ["fix", str(vault), "--apply", "--json"])

    assert health.exit_code == 0, health.output
    assert plan.exit_code == 0, plan.output
    assert applied.exit_code == 0, applied.output
    assert json.loads(health.output)["findings"] == [
        {
            "details": {"alias": "QRR", "title": "Quarterly Research Report (QRR)"},
            "kind": "alias_candidate",
            "message": "Alias candidate from title: QRR",
            "path": "sources/report.md",
            "reviewed": False,
            "severity": "warning",
        }
    ]
    assert json.loads(plan.output)["plan"] == [
        {
            "kind": "propose_alias",
            "path": "sources/report.md",
            "payload": {"alias": "QRR", "title": "Quarterly Research Report (QRR)"},
            "requires_review": True,
        }
    ]
    assert json.loads(applied.output)["summary"]["applied"] == 0
    assert source.read_text(encoding="utf-8") == original
