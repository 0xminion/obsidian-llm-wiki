"""Content-bearing verification and policy-unavailability acceptance controls."""
from __future__ import annotations

import re

import pytest

from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.synth.review import verify_review_receipt
from tests.new import test_identity_structural_cli as structural
from tests.new.migration_fixture_helpers import (
    closed_registry,
    edit_receipt_and_reseal,
    meta_for,
    reseal_inner_generation,
    reseal_outer,
)
from tests.new.test_identity_migration_schema7 import _preview, cli, committed
from tests.new.test_preview_admission_closure import _forbid_config, _reuse

__all__ = ["cli", "committed"]


def test_same_revision_unknown_span_requires_historical_content(committed, monkeypatch):
    fixture, *_ = committed
    _vault, config = resolve_vault(str(fixture.wiki.parent))

    def edit(receipt):
        outcome = next(u for u in receipt["units"].values() if u["kind"] != "claim")
        outcome["span_ids"] = [re.sub(r"\d+$", "999999", outcome["span_ids"][0])]

    edit_receipt_and_reseal(committed, edit)
    synthesis = load_synthesis(meta_for(fixture), "reuse.md")
    # A v3 context binding also requires exact bytes, so hash-only reuse now
    # refuses explicitly. Preview must still reach the deeper span-existence gate.
    assert "exact source content required" in verify_review_receipt(
        synthesis, "", "reuse.md", config,
        source_sha256=synthesis.review_receipt["source_sha256"],
    )
    reason = verify_review_receipt(synthesis, structural.KERNEL.strip(), "reuse.md", config)
    assert "unknown span" in reason
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    assert _reuse(report)["status"] == "unresolved"
    assert any(b["kind"] == "invalid_admission_receipt" and reason in b["detail"]
               for b in report["blockers"])


def test_wrong_historical_bytes_are_not_replaced_with_current_body(committed, monkeypatch):
    fixture, *_ = committed
    with closed_registry(meta_for(fixture), edit=True) as registry:
        registry.connection.execute(
            "UPDATE identity_source_context SET content=? WHERE filename='reuse.md'",
            ("Wrong revision despite the matching row key.",),
        )
    reseal_inner_generation(fixture)
    reseal_outer(fixture)
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    binding = _reuse(report)
    assert binding["evidence"]["source_revision_matches_receipt"] is True
    assert binding["evidence"]["historical_source_context"] is False
    assert binding["evidence"]["historical_admission_integrity"] is False
    assert binding["status"] == "unresolved"


def test_unreadable_review_config_reports_unresolved_not_invented_policy(committed, monkeypatch):
    from obsidian_llm_wiki import config

    def unavailable(*args, **kwargs):
        raise OSError("synthetic policy input unavailable")

    monkeypatch.setattr(config, "_config_environment", unavailable)
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    assert all(b["status"] == "unresolved" for b in report["bindings"])
    assert _reuse(report)["evidence"]["historical_admission_integrity"] is True
    assert any("current review policy unavailable" in b["detail"] for b in report["blockers"])


def test_review_off_never_bypasses_historical_admission_integrity(committed, monkeypatch):
    edit_receipt_and_reseal(committed, lambda r: r.update(output_sha256="0" * 64))
    monkeypatch.setenv("GROUNDING_REVIEW", "off")
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    assert _reuse(report)["evidence"]["historical_admission_integrity"] is False
    assert _reuse(report)["status"] == "unresolved"
    assert any(b["kind"] == "invalid_admission_receipt" for b in report["blockers"])


@pytest.mark.parametrize("settings", [
    {}, {"INGEST_MODEL": "alias-reviewer"}, {"LLM_INGEST_MODEL": "ingest-reviewer"},
    {"LLM_REVIEW_MODEL": "explicit-reviewer"}, {"LLM_REVIEW_MODEL": "  "},
])
def test_constructor_free_review_policy_matches_ordinary_config(tmp_path, monkeypatch, settings):
    from obsidian_llm_wiki.config import load_config, load_review_policy
    from obsidian_llm_wiki.synth.review import reviewer_fingerprint

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=file-model\nLLM_PROVIDER=openai\nLLM_HOST=https://fixture.invalid\n")
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    ordinary = load_config(str(env_file), VAULT_PATH=str(tmp_path))
    expected = reviewer_fingerprint(ordinary)
    _forbid_config(monkeypatch)
    pure = load_review_policy(str(env_file), VAULT_PATH=str(tmp_path))
    assert reviewer_fingerprint(pure) == expected
    assert pure.grounding_review == ordinary.grounding_review
    assert not hasattr(pure.llm, "api_key")
