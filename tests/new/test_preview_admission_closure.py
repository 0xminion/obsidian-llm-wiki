"""Semantic preview controls: resealed containers never replace admission proof."""
from __future__ import annotations

import copy

import pytest

from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.synth.review import verify_review_receipt
from tests.new import test_identity_structural_cli as structural
from tests.new.migration_fixture_helpers import (
    edit_payload_and_reseal,
    edit_receipt_and_reseal,
    meta_for,
    reseal_outer,
)
from tests.new.test_identity_migration_schema7 import _preview, cli, committed

__all__ = ["cli", "committed"]


def _reuse(report):
    return next(b for b in report["bindings"] if b["source_file"] == "reuse.md")


def _forbid_config(monkeypatch):
    from obsidian_llm_wiki import config

    def forbidden(*args, **kwargs):
        pytest.fail("preview constructed writable/full configuration")

    monkeypatch.setattr(config.Config, "__init__", forbidden)
    monkeypatch.setattr(config.LLMProviderConfig, "__init__", forbidden)


@pytest.mark.parametrize("damage", ["new-fact", "prose", "quote"])
def test_resealed_unreviewed_output_is_unresolved(committed, monkeypatch, damage):
    fixture, *_ = committed
    _vault, config = resolve_vault(str(fixture.wiki.parent))
    synthesis = load_synthesis(meta_for(fixture), "reuse.md")
    assert verify_review_receipt(synthesis, structural.KERNEL.strip(), "reuse.md", config) == ""

    def edit(payload):
        concept = payload["concepts"][0]
        if damage == "new-fact":
            claim = copy.deepcopy(concept["claims"][0])
            claim["text"] = "The scheduler was invented by a lunar colony."
            concept["claims"].append(claim)
        elif damage == "prose":
            payload["source_summary"] += "\n\nA lunar colony operates the scheduler."
        else:
            concept["claims"][0]["evidence"]["quote"] = "A fabricated quotation."

    edit_payload_and_reseal(committed, edit)
    synthesis = load_synthesis(meta_for(fixture), "reuse.md")
    reason = verify_review_receipt(synthesis, structural.KERNEL.strip(), "reuse.md", config)
    assert reason
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    assert report["publication"]["verified"] is True  # byte envelope only
    assert _reuse(report)["status"] == "unresolved"
    assert any(b["kind"] == "invalid_admission_receipt" and reason in b["detail"]
               for b in report["blockers"])


def test_preview_separates_historical_integrity_and_current_review_policy(committed, monkeypatch):
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch)
    evidence = _reuse(report)["evidence"]
    assert evidence["historical_admission_integrity"] is True
    assert evidence["current_review_policy_compatible"] is True
    assert report["publication"]["current_policy_verified"] is False
    assert "historical" in report["publication"]["verification_scope"]


@pytest.mark.parametrize("change", ["receipt-reviewer", "current-reviewer", "invalid-policy"])
def test_current_review_policy_is_not_inferred_from_receipt(committed, monkeypatch, change):
    if change == "receipt-reviewer":
        edit_receipt_and_reseal(committed, lambda r: r.update(reviewer_fingerprint="f" * 24))
    elif change == "current-reviewer":
        monkeypatch.setenv("LLM_REVIEW_MODEL", "different-reviewer")
    else:
        monkeypatch.setenv("GROUNDING_REVIEW", "not-a-mode")
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    evidence = _reuse(report)["evidence"]
    assert evidence["historical_admission_integrity"] is True
    assert evidence["current_review_policy_compatible"] is not True
    assert _reuse(report)["status"] == "unresolved"
    assert any(b["kind"] == "review_policy_unverified" for b in report["blockers"])


def test_truncated_sqlite_header_refuses_json_and_public_reader(committed, monkeypatch):
    fixture, *_ = committed
    (meta_for(fixture) / "registry.db").write_bytes(b"SQLite format 3\x00")
    reseal_outer(fixture)
    _forbid_config(monkeypatch)
    with pytest.raises(RuntimeError, match="registry"):
        current_publication(fixture.wiki)
    report = _preview(committed, monkeypatch, expected=2)
    assert report["status"] == "refused"
    assert report["bindings"] == []
