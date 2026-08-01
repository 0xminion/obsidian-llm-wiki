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
from obsidian_llm_wiki.core.registry import VaultRegistry

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
        "short_concept_stub": 1,
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
            "details": {},
            "kind": "short_concept_stub",
            "message": "Concept body is shorter than 500 characters: 7",
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


def test_health_reports_all_content_quality_classes_as_typed_findings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    _write_page(
        wiki / "concepts" / "alpha.md",
        "---\ntype: Concept\nconfidence: 0.2\n---\n# Alpha\n\n[[missing-target]]\n",
    )
    _write_page(
        wiki / "entries" / "entry.md",
        "---\ntype: Entry\n---\n# Entry\n\nNo provenance link.\n",
    )
    _write_page(
        wiki / "mocs" / "tiny.md",
        "---\ntype: Map of Content\n---\n# Tiny\n\n- [[alpha]]\n",
    )
    monkeypatch.delenv("VAULT_PATH", raising=False)

    result = runner.invoke(app, ["health", str(vault), "--json", "--full"])

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)["summary"]
    assert summary["broken_wikilink"] == 1
    assert summary["low_confidence"] == 1
    assert summary["short_concept_stub"] == 1
    assert summary["missing_source_link"] == 1
    assert summary["undersized_moc"] == 1


def test_incremental_health_returns_a_persistent_whole_vault_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A clean batch must not hide findings discovered in earlier batches."""
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    first_path = wiki / "concepts" / "first.md"
    second_path = wiki / "concepts" / "second.md"
    _write_page(
        first_path,
        "---\ntype: Concept\ntags:\n  - has space\n---\n# First\n",
    )
    _write_page(
        second_path,
        (
            "---\ntype: Concept\nrelations:\n"
            "  - target: missing\n    type: related_to\n---\n# Second\n"
        ),
    )
    llmwiki_dir = wiki / ".llmwiki"
    with VaultRegistry(llmwiki_dir) as registry:
        manifest = {
            "first": ("concepts/first.md", "hash-first"),
            "second": ("concepts/second.md", "hash-second"),
        }
        diff = registry.render_diff("concept-page", manifest)
        registry.commit_render_manifest("concept-page", manifest, diff)

    monkeypatch.delenv("VAULT_PATH", raising=False)
    first = runner.invoke(
        app, ["health", str(vault), "--json", "--batch-size", "1"]
    )
    second = runner.invoke(
        app, ["health", str(vault), "--json", "--batch-size", "1"]
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["snapshot_complete"] is False
    assert second_payload["snapshot_complete"] is True
    assert first_payload["summary"] == {
        "orphan_concept": 1,
        "short_concept_stub": 1,
        "tag_normalization": 1,
    }
    assert second_payload["summary"] == {
        "broken_relation": 1,
        "orphan_concept": 2,
        "short_concept_stub": 2,
        "tag_normalization": 1,
    }

    _write_page(first_path, "---\ntype: Concept\ntags: []\n---\n# First\n")
    refreshed = runner.invoke(
        app, ["health", str(vault), "--json", "--batch-size", "1"]
    )
    assert refreshed.exit_code == 0, refreshed.output
    assert json.loads(refreshed.output)["summary"] == {
        "broken_relation": 1,
        "orphan_concept": 2,
        "short_concept_stub": 2,
    }


def test_full_health_detects_same_size_source_edit_with_restored_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import os

    from obsidian_llm_wiki.core.models import SourceSynthesis
    from obsidian_llm_wiki.core.registry import ArtifactVersions
    from obsidian_llm_wiki.ingest.sources import load_source_file

    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    source_path = wiki / "sources" / "same.md"
    _write_page(source_path, "---\ntype: Source\n---\n# Same\n\nAAAA\n")
    loaded = load_source_file(source_path)
    assert loaded is not None
    with VaultRegistry(wiki / ".llmwiki") as registry:
        registry.upsert_synthesis(
            "same.md",
            loaded,
            SourceSynthesis(
                source_title="Same",
                source_summary="summary",
                source_file="same.md",
            ),
            ArtifactVersions("extract", "prompt", "schema", "model"),
        )

    original_stat = source_path.stat()
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace("AAAA", "BBBB"),
        encoding="utf-8",
    )
    os.utime(
        source_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    monkeypatch.delenv("VAULT_PATH", raising=False)

    result = runner.invoke(app, ["health", str(vault), "--json", "--full"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["source_hash_mismatch"] == 1


def test_full_health_flags_unmanaged_legacy_generated_pages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    managed = wiki / "entries" / "managed.md"
    stale = wiki / "entries" / "stale.md"
    _write_page(managed, "---\ntype: Entry\n---\n# Managed\n\n## Source\n")
    _write_page(stale, "---\ntype: Entry\n---\n# Stale\n\n## Source\n")
    with VaultRegistry(wiki / ".llmwiki") as registry:
        manifest = {"managed-source.md": ("entries/managed.md", "hash")}
        diff = registry.render_diff("entry-page", manifest)
        registry.commit_render_manifest("entry-page", manifest, diff)
    monkeypatch.delenv("VAULT_PATH", raising=False)

    result = runner.invoke(app, ["health", str(vault), "--json", "--full"])

    assert result.exit_code == 0, result.output
    findings = json.loads(result.output)["findings"]
    unmanaged = [
        finding for finding in findings if finding["kind"] == "unmanaged_generated_page"
    ]
    assert [finding["path"] for finding in unmanaged] == ["entries/stale.md"]
