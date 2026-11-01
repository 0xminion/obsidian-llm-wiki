"""Resealed-container acceptance controls for persisted context binding."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth import review
from tests.new.migration_fixture_helpers import edit_receipt_and_reseal
from tests.new.test_identity_migration_schema7 import _preview, cli, committed
from tests.new.test_preview_admission_closure import _forbid_config, _reuse

__all__ = ["cli", "committed"]


def test_resealed_context_binding_does_not_become_historical_admission(committed, monkeypatch):
    def damage(receipt):
        claim = next(o for o in receipt["units"].values() if o["kind"] == "claim")
        claim["context_binding"]["source_sha256"] = "0" * 64

    edit_receipt_and_reseal(committed, damage)
    _forbid_config(monkeypatch)
    report = _preview(committed, monkeypatch, expected=3)
    assert report["publication"]["verified"] is True
    binding = _reuse(report)
    assert binding["status"] == "unresolved"
    assert binding["evidence"]["historical_admission_integrity"] is False
    assert any(b["kind"] == "invalid_admission_receipt" and "context binding" in b["detail"]
               for b in report["blockers"])


@pytest.mark.parametrize("content,claim", [
    ("I save my handwritten notes as searchable text.",
     "The writer saves handwritten notes as searchable text."),
    ("我會將手寫紀錄轉成文字，方便搜尋。", "作者會將手寫紀錄轉成文字，方便搜尋。"),
    ("Editor Noor reports: Mira said, ‘I save my notes as text.’",
     "Editor Noor saves notes as text."),
    ("译者按：我核对了引文；原文作者没有参与。", "原文作者核对了引文。"),
])
def test_narrator_prompt_preserves_actor_guards_and_exact_citation(content, claim):
    catalog = build_span_catalog(content, "probe.md")
    ids = tuple(span.id for span in catalog.spans)
    evidence = catalog.cite(list(ids))
    unit = review.ReviewUnit(
        key=review._unit_key("claim", claim, ids, ["", "", [
            evidence.quote, evidence.start_offset, evidence.end_offset,
        ]]),
        kind="claim", path="claims[0]", text=claim,
        cited_span_ids=ids, cited_text=evidence.quote,
    )
    captured = []

    class RequestCapturedError(Exception):
        pass

    async def capture(prompt, messages, config, **kwargs):
        captured.append(prompt)
        assert messages[-1]["content"] == prompt
        assert kwargs["task"] == "review"
        raise RequestCapturedError

    with pytest.raises(RequestCapturedError):
        asyncio.run(review._review_units(
            [unit], catalog, catalog.windows(10000)[0],
            SimpleNamespace(review_batch_units=16), call=capture,
        ))
    prompt, = captured
    listed = json.loads(prompt.rsplit("--- UNITS ---\n", 1)[1])
    assert listed[0]["text"] == claim
    assert listed[0]["cited_text"] == content
    assert listed[0]["cited_span_ids"] == list(ids)
    assert listed[0]["context_binding"]["offered_context"] == []
    assert "Generic narrator references:" in prompt
    assert "does not establish who composed the document" in prompt
    assert "Missing quotation marks alone do not establish authorship" in prompt
    assert "Do not transfer actions from quoted, interviewed or fictional speakers" in prompt
    assert "Named identities, credentials and distinct roles" in prompt
    assert "If the voice or genre makes the attribution genuinely ambiguous" in prompt
    assert "Context must not supply uncited premises" in prompt


def test_narrator_policy_has_distinct_compatibility_fingerprint(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    current = review.reviewer_fingerprint(config)
    assert review.REVIEW_PROMPT_VERSION != "grounding-review-prompt-v4"
    monkeypatch.setattr(review, "REVIEW_PROMPT_VERSION", "grounding-review-prompt-v4")
    assert review.reviewer_fingerprint(config) != current
