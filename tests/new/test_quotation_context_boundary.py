"""Structural quotation-introduction controls, not semantic accuracy tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from obsidian_llm_wiki.synth.review import review_window
from tests.new.quotation_context_helpers import (
    INTRO,
    QUOTE,
    QuotationProvider,
    canonical_case,
    wire_claim,
)


@pytest.mark.parametrize("separator", ["\n", "\n\n", "\r\n\r\n"], ids=["single", "lf", "crlf"])
def test_canonical_request_offers_attached_introduction_without_widening_citation(
    separator, tmp_path
):
    synthesis, catalog, config = canonical_case(separator)
    evidence = synthesis.concepts[0].claims[0].evidence
    original = asdict(evidence)
    assert evidence.quote == QUOTE
    provider = QuotationProvider(declare=False)
    outcomes = asyncio.run(
        review_window(
            synthesis,
            catalog,
            catalog.windows(10000)[0],
            config,
            call=provider,
        )
    )
    claim = wire_claim(provider)
    (tmp_path / "request.json").write_text(
        json.dumps(provider.requests, ensure_ascii=False, indent=2)
    )
    assert asdict(evidence) == original
    assert claim["cited_text"] == QUOTE
    assert claim["cited_span_ids"] == list(evidence.span_ids)
    binding = claim["context_binding"]
    assert outcomes[binding["unit_key"]]["model_context_uses"] == []
    assert [c["text"] for c in binding["offered_context"]] == [INTRO + separator]


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n"])
def test_real_quote_only_window_never_recovers_unseen_introduction(separator):
    synthesis, catalog, config = canonical_case(separator)
    window = next(w for w in catalog.windows(1) if QUOTE in w.text and INTRO not in w.text)
    provider = QuotationProvider()
    asyncio.run(review_window(synthesis, catalog, window, config, call=provider))
    claim = wire_claim(provider)
    assert claim["cited_text"] == QUOTE
    assert claim["context_binding"]["offered_context"] == []


def test_unknown_policy_cannot_become_known_by_changing_current_constant(monkeypatch):
    from obsidian_llm_wiki.synth import review_context

    binding, inputs = offset_binding(INTRO + "\n\n" + QUOTE)
    monkeypatch.setattr(review_context, "CONTEXT_POLICY_VERSION", "review-context-v999")
    with pytest.raises(ValueError, match="unknown context policy"):
        review_context.build_context_binding(**inputs)
    with pytest.raises(ValueError, match="unknown context policy"):
        review_context.build_context_binding(**inputs, policy_version="review-context-v999")
    binding["policy"] = "review-context-v999"
    assert review_context.validate_context_binding(binding, **inputs) == "unknown context policy"
    old = review_context.build_context_binding(**inputs, policy_version="review-context-v1")
    assert review_context.validate_context_binding(old, **inputs) == ""


def offset_binding(content, *, start=None, visible=None, policy=None):
    from obsidian_llm_wiki.synth.review_context import build_context_binding

    start = content.index(QUOTE) if start is None else start
    inputs = {
        "content": content,
        "source_file": "offset.md",
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "catalog_version": "auditor-offset-v1",
        "unit_key": "auditor-claim",
        "cited_span_ids": ["q"],
        "citation_start": start,
        "citation_end": min(len(content), start + len(QUOTE)),
        "visible_intervals": visible or [[0, len(content)]],
    }
    return build_context_binding(**inputs, policy_version=policy), inputs


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\n \t\n"])
def test_only_immediate_complete_introduction_raw_offsets_and_history(separator):
    from obsidian_llm_wiki.synth.review_context import validate_context_binding

    introduction = "Ínez, explaining the **柳泵** [manual](local.md), said："
    content = (
        "Unrelated measurement: 44.\n\n" + introduction + separator + QUOTE + "\n\nUnrelated after."
    )
    binding, inputs = offset_binding(content)
    assert binding["policy"] == "review-context-v2"
    assert [r["text"] for r in binding["offered_context"]] == [introduction + separator]
    assert validate_context_binding(binding, **inputs) == ""
    record = binding["offered_context"][0]
    assert content[record["char_start"] : record["char_end"]] == record["text"]
    assert content.encode()[record["byte_start"] : record["byte_end"]] == record["text"].encode()
    old, _ = offset_binding(content, policy="review-context-v1")
    assert old["offered_context"] == []
    assert validate_context_binding(old, **inputs) == ""
    assert old["policy"] == "review-context-v1"


@pytest.mark.parametrize(
    "before",
    [
        "An unrelated ordinary paragraph.\n\n",
        INTRO + "\n\nIntervening paragraph.\n\n",
        INTRO + "\n\n\n",
        INTRO + "\n\n\n\n",
        "A wrapped\nintroduction:\n\n",
        "# Heading:\n\n",
        "- List item:\n\n",
        "1. List item:\n\n",
        "> Nested introduction:\n\n",
        "    Indented code:\n\n",
    ],
)
def test_unsupported_neighbor_shapes_do_not_cross_blank_lines(before):
    binding, _ = offset_binding(before + QUOTE)
    assert binding["offered_context"] == []


@pytest.mark.parametrize("cut", [1, 10, len(INTRO), len(INTRO) + 1])
def test_missing_introduction_and_unseen_separator_gaps_are_not_reconstructed(cut):
    content = INTRO + "\n\n" + QUOTE
    binding, _ = offset_binding(content, visible=[[cut, len(content)]])
    assert binding["offered_context"] == []
    binding, _ = offset_binding(content, visible=[[0, cut], [cut + 1, len(content)]])
    assert binding["offered_context"] == []


@pytest.mark.parametrize("length", [384, 385, 1000])
def test_introduction_is_whole_or_excluded_at_unchanged_character_and_byte_caps(length):
    # Four-byte code points exercise the UTF-8 cap without normalizing Unicode.
    prefix = "🪶" * (length - 3) + ":\n\n"
    binding, _ = offset_binding(prefix + QUOTE)
    records = binding["offered_context"]
    assert [r["text"] for r in records] == ([prefix] if length == 384 else [])
    assert all(len(r["text"]) <= 384 and len(r["text"].encode()) <= 1536 for r in records)


def test_quote_marker_is_required_no_forward_paragraph_search():
    content = INTRO + "\n\nOrdinary prose only."
    binding, _ = offset_binding(content, start=len(INTRO) + 2)
    assert binding["offered_context"] == []


@pytest.mark.parametrize("declare", [False, True])
def test_current_context_declaration_and_captured_v1_v2_history(declare):
    from obsidian_llm_wiki.core.models import source_synthesis_from_dict
    from obsidian_llm_wiki.synth.review import (
        build_review_receipt,
        verify_review_receipt,
        verify_review_receipt_integrity,
    )

    synthesis, catalog, config = canonical_case()
    provider = QuotationProvider(declare=declare)
    outcomes = asyncio.run(
        review_window(
            synthesis,
            catalog,
            catalog.windows(10000)[0],
            config,
            call=provider,
        )
    )
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    assert verify_review_receipt(synthesis, catalog.content, catalog.source_file, config) == ""
    outcome = next(o for o in synthesis.review_receipt["units"].values() if o["kind"] == "claim")
    assert bool(outcome["model_context_uses"]) is declare
    assert outcome["context_binding"] == wire_claim(provider)["context_binding"]
    current = deepcopy(synthesis.review_receipt)
    # Captured using the actual frozen v10 runtime, not a relabelled v3 binding.
    records = json.loads(Path(__file__).with_name("quotation_context_legacy.json").read_text())[
        "records"
    ]
    old_receipts = {}
    for policy, record in records.items():
        assert record["content"] == catalog.content
        assert record["source_file"] == catalog.source_file
        historical = source_synthesis_from_dict(record["synthesis"])
        old_receipt = deepcopy(historical.review_receipt)
        old_receipts[policy] = old_receipt
        assert (
            verify_review_receipt_integrity(historical, catalog.content, catalog.source_file) == ""
        )
        assert "not current" in verify_review_receipt(
            historical, catalog.content, catalog.source_file, config
        )
        assert historical.review_receipt == old_receipt
        assert current["reviewer_fingerprint"] != old_receipt["reviewer_fingerprint"]
    # A v2 header must not wrap a v1 claim binding (or conversely).
    for header, binding_policy in [
        ("review-context-v2", "review-context-v1"),
        ("review-context-v1", "review-context-v2"),
    ]:
        synthesis.review_receipt = deepcopy(old_receipts[binding_policy])
        synthesis.review_receipt["context_policy"] = header
        assert verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file)
    synthesis.review_receipt = deepcopy(current)
    synthesis.review_receipt["context_policy"] = "review-context-v999"
    assert "unknown" in verify_review_receipt_integrity(
        synthesis, catalog.content, catalog.source_file
    )


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n"])
def test_raw_introduction_request_receipt_cache_reload(separator, tmp_path):
    from obsidian_llm_wiki.core.cache import load_synthesis, save_synthesis
    from obsidian_llm_wiki.synth.review import build_review_receipt, verify_review_receipt

    synthesis, catalog, config = canonical_case(separator)
    provider = QuotationProvider()
    outcomes = asyncio.run(
        review_window(
            synthesis,
            catalog,
            catalog.windows(10000)[0],
            config,
            call=provider,
        )
    )
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    save_synthesis(synthesis, tmp_path, catalog.source_file)
    cached = load_synthesis(tmp_path, catalog.source_file)
    assert cached.review_receipt == synthesis.review_receipt
    binding = wire_claim(provider)["context_binding"]
    assert [c["text"] for c in binding["offered_context"]] == [INTRO + separator]
    assert verify_review_receipt(cached, catalog.content, catalog.source_file, config) == ""
    assert asdict(cached.concepts[0].claims[0].evidence) == asdict(
        synthesis.concepts[0].claims[0].evidence
    )


@pytest.mark.parametrize(
    "purpose", ["referent", "attribution", "local_qualifier", "independent_fact"]
)
def test_new_cross_boundary_context_use_has_only_allowed_purposes(purpose):
    from obsidian_llm_wiki.synth.review import SynthesisReviewError

    synthesis, catalog, config = canonical_case()
    provider = QuotationProvider()

    async def answer(system, messages, config, **kwargs):
        result = json.loads(await provider(system, messages, config, **kwargs))
        unit = wire_claim(provider)
        result["reviews"][unit["unit_id"]]["model_context_uses"][0]["purpose"] = purpose
        return json.dumps(result)

    call = review_window(synthesis, catalog, catalog.windows(10000)[0], config, call=answer)
    if purpose == "independent_fact":
        with pytest.raises(SynthesisReviewError, match="forbidden purpose"):
            asyncio.run(call)
    else:
        outcomes = asyncio.run(call)
        key = wire_claim(provider)["context_binding"]["unit_key"]
        assert outcomes[key]["model_context_uses"][0]["purpose"] == purpose


def test_offered_introduction_does_not_override_model_insufficient():
    from obsidian_llm_wiki.synth.review import SynthesisReviewError

    synthesis, catalog, config = canonical_case()
    provider = QuotationProvider(declare=False)

    async def answer(system, messages, config, **kwargs):
        result = json.loads(await provider(system, messages, config, **kwargs))
        unit = wire_claim(provider)
        assert unit["context_binding"]["offered_context"]
        result["reviews"][unit["unit_id"]]["verdict"] = "insufficient"
        return json.dumps(result)

    with pytest.raises(SynthesisReviewError, match="insufficient"):
        asyncio.run(
            review_window(synthesis, catalog, catalog.windows(10000)[0], config, call=answer)
        )
