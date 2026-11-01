"""Extra controls for the immutable migration fixture and semantic gate reach."""
from __future__ import annotations

import json
import sqlite3

import pytest

from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.synth.review import verify_review_receipt
from tests.new import test_identity_structural_cli as structural
from tests.new.migration_fixture_helpers import (
    closed_registry,
    edit_receipt_and_reseal,
    meta_for,
)
from tests.new.test_identity_migration_preview import _snapshot
from tests.new.test_identity_migration_schema7 import _preview, cli, committed

__all__ = ["cli", "committed"]


def test_closed_inspector_has_no_filesystem_sqlite_and_closes(committed, monkeypatch):
    fixture, *_ = committed
    before = _snapshot(fixture.wiki.parent)
    real_connect = sqlite3.connect
    databases = []

    def connect(database, *args, **kwargs):
        databases.append(database)
        assert database == ":memory:"
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    with closed_registry(meta_for(fixture)) as registry:
        connection = registry.connection
        assert connection.execute("SELECT count(*) FROM source").fetchone()[0] == 3
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM source")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    assert databases == [":memory:"]
    assert _snapshot(fixture.wiki.parent) == before


def test_reseal_noop_is_valid_and_preserves_semantic_proof(committed, monkeypatch):
    fixture, *_ = committed
    original = load_synthesis(meta_for(fixture), "reuse.md").review_receipt
    edit_receipt_and_reseal(committed, lambda _receipt: None)
    assert load_synthesis(meta_for(fixture), "reuse.md").review_receipt == original
    report = _preview(committed, monkeypatch)
    assert report["publication"]["verified"] is True
    assert report["summary"]["bindings"]["resolved"] == 3


@pytest.mark.parametrize("damage", ["output-digest", "catalog-version", "support-spans"])
def test_resealed_admission_proof_is_not_certified(committed, monkeypatch, damage):
    fixture, *_ = committed
    _vault, config = resolve_vault(str(fixture.wiki.parent))
    synthesis = load_synthesis(meta_for(fixture), "reuse.md")
    assert verify_review_receipt(synthesis, structural.KERNEL.strip(), "reuse.md", config) == ""

    def edit(receipt):
        if damage == "output-digest":
            receipt["output_sha256"] = "0" * 64
        elif damage == "catalog-version":
            receipt["catalog_version"] = "span-catalog-v999"
        else:
            assert receipt["units"]
            for unit in receipt["units"].values():
                unit["span_ids"] = ["invented-source#invented-span"]

    edit_receipt_and_reseal(committed, edit)
    synthesis = load_synthesis(meta_for(fixture), "reuse.md")
    reason = verify_review_receipt(synthesis, structural.KERNEL.strip(), "reuse.md", config)
    assert reason, "negative fixture must actually fail the ordinary admission verifier"
    # Keep the actual producer's invalid proof and all coherent consumer hashes.
    # A verified byte envelope cannot authorize a semantically invalid receipt.
    report = _preview(committed, monkeypatch, expected=3)
    reuse = next(b for b in report["bindings"] if b["source_file"] == "reuse.md")
    assert reuse["status"] == "unresolved", reason
    assert report["assessment_estimate"]["model_calls_made"] == 0


def test_unresealed_generated_source_corruption_refused(committed, monkeypatch):
    fixture, *_ = committed
    path = fixture.generation / "sources" / "owner.md"
    raw = path.read_text()
    assert "kernel" in raw
    path.write_text(raw.replace("kernel", "postal"))
    report = _preview(committed, monkeypatch, expected=2)
    assert report["status"] == "refused" and report["bindings"] == []
    assert "generation inventory/digest mismatch" in " ".join(report["refusal"]["reasons"])


def test_resealed_inner_plan_version_is_not_certified(committed, monkeypatch):
    from obsidian_llm_wiki.core.identity_plan import canonical, digest
    from tests.new.migration_fixture_helpers import reseal_outer

    fixture, _endpoint, marker, raw = committed
    raw["version"] = "identity-publication-plan-v999"
    plan_id = digest(raw)
    meta = meta_for(fixture)
    (meta / "publications" / (plan_id + ".json")).write_text(canonical(raw))
    marker["plan_id"] = plan_id
    (meta / "identity-publication.json").write_text(canonical(marker))
    pointer = fixture.wiki / "OLW-Publications" / "current.json"
    outer = json.loads(pointer.read_text())
    outer["plan_id"] = plan_id
    pointer.write_text(canonical(outer))
    reseal_outer(fixture)
    report = _preview(committed, monkeypatch, expected=3)
    assert report["publication"]["verified"] is False
    assert report["summary"]["bindings"]["resolved"] == 0
    assert any("unknown or corrupt archived publication plan" in b["detail"]
               for b in report["blockers"])
