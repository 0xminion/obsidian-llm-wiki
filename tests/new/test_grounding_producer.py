"""The single-pass producer selects catalog IDs; the machine builds evidence."""

from __future__ import annotations

import json

import pytest

from obsidian_llm_wiki.core.evidence import resolve_synthesis_evidence
from obsidian_llm_wiki.core.models import EvidenceVerification, SourceDoc
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth.grounding import (
    GroundingError,
    parse_grounded_synthesis,
    validate_grounded_anchors,
)
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from obsidian_llm_wiki.synth.response_schema import SYNTHESIS_RESPONSE_SCHEMA
from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

PDF = (
    "Results\n\nThe new solver reduced\nrouting   errors by 12%\tin trials.\n"
    "It did not reduce latency.\n\nThe new solver reduced\nrouting   errors by 12%\tin trials.\n"
)


def _response(ids, **claim_extra):
    """Producer response citing ``ids`` through interface v2 (one explicit range).

    A non-empty list becomes ``{start_id: ids[0], end_id: ids[-1]}``; an empty
    list omits the range; any other value is sent verbatim as the range.
    """
    claim = {"text": "The solver reduced routing errors by 12%.", "source_ref": "",
             **claim_extra}
    if isinstance(ids, list) and ids:
        claim["evidence_span_range"] = {"start_id": ids[0], "end_id": ids[-1]}
    elif not isinstance(ids, list):
        claim["evidence_span_range"] = ids
    return json.dumps({
        "source_title": "R", "source_summary": "A solver reduced routing errors.",
        "source_tags": [], "key_points": [], "open_questions": [], "language": "en",
        "concepts": [{
            "title": "Solver", "slug": "solver", "canonical_id": "",
            "summary": "A solver that reduced routing errors.", "tags": [], "aliases": [],
            "sections": [{"heading": "Effect", "points": ["Fewer routing errors."], "prose": ""}],
            "claims": [claim], "related": [], "confidence": 0.9,
            "provenance": "extracted", "is_new": True,
        }],
        "maps": [],
    })


def _catalog():
    return build_span_catalog(PDF, "paper.md")


def _solver_ids(catalog):
    return [s.id for s in catalog.spans if "solver" in s.text]


def test_prompt_offers_window_spans_by_id_with_raw_text_and_forbids_copying_quotes():
    catalog = _catalog()
    window = catalog.windows(10_000)[0]
    prompt = build_synthesis_prompt("R", PDF, span_window=window)
    for span in catalog.spans:
        assert f"⟦{span.id}⟧" in prompt
        assert span.text.rstrip() in prompt
    assert "evidence_span_range" in prompt and "evidence_span_ids" not in prompt
    assert '"quote"' not in prompt


def test_wire_schema_claims_are_closed_and_id_only():
    claim = SYNTHESIS_RESPONSE_SCHEMA["properties"]["concepts"]["items"]["properties"]["claims"]
    items = claim["items"]
    assert items["additionalProperties"] is False
    assert set(items["properties"]) == {"text", "evidence_span_ranges", "source_ref"}
    assert set(items["required"]) == set(items["properties"])
    ranges = items["properties"]["evidence_span_ranges"]
    assert ranges["minItems"] == 1 and ranges["maxItems"] == 3
    span_range = ranges["items"]
    assert span_range["type"] == "object" and span_range["additionalProperties"] is False


def test_machine_builds_exact_evidence_for_repeated_pdf_text_by_position():
    catalog = _catalog()
    window = catalog.windows(10_000)[0]
    first, second = _solver_ids(catalog)
    synthesis = parse_grounded_synthesis(_response([second]), catalog, window)
    evidence = synthesis.concepts[0].claims[0].evidence
    assert evidence.quote == "The new solver reduced\nrouting   errors by 12%\tin trials."
    assert evidence.start_offset == PDF.rindex("The new solver")
    assert evidence.span_ids == (second,)
    assert evidence.verification is EvidenceVerification.VERIFIED
    source = SourceDoc(title="R", content=PDF, source_file="paper.md")
    validate_grounded_anchors(synthesis, source, "paper.md")
    # Legacy quote resolution must not erase the machine anchor of a repeated span.
    validate_synthesis(synthesis, source, "paper.md")
    resolve_synthesis_evidence(synthesis, source, "paper.md")
    assert synthesis.concepts[0].claims[0].evidence.span_ids == (second,)
    assert synthesis.concepts[0].claims[0].evidence.start_offset == evidence.start_offset


@pytest.mark.parametrize("extra", [
    {"quote": "The new solver reduced"},
    {"evidence": {"quote": "x", "verification": "verified", "start_offset": 0, "end_offset": 1}},
])
def test_model_supplied_quotes_or_verified_flags_are_rejected(extra):
    catalog = _catalog()
    with pytest.raises(GroundingError, match="quote|evidence"):
        parse_grounded_synthesis(_response(_solver_ids(catalog)[:1], **extra), catalog, None)


def test_legacy_quote_only_claim_is_not_a_producer_fallback():
    catalog = _catalog()
    data = json.loads(_response([]))
    claim = data["concepts"][0]["claims"][0]
    claim.pop("evidence_span_range", None)
    claim["quote"] = "It did not reduce latency."
    with pytest.raises(GroundingError):
        parse_grounded_synthesis(json.dumps(data), catalog, None)


@pytest.mark.parametrize("ids_factory, message", [
    (lambda c: [], "no evidence_span_range"),
    (lambda c: ["deadbeef-1"], "different source|unknown"),
    (lambda c: [build_span_catalog(PDF, "other.md").spans[1].id], "different source"),
    (lambda c: [build_span_catalog(PDF + "x", "paper.md").spans[1].id], "different source"),
    (lambda c: [c.spans[0].id + "0"], "unknown"),
    (lambda c: "abc", "object"),
])
def test_unknown_stale_wrong_source_or_malformed_ids_fail(ids_factory, message):
    catalog = _catalog()
    with pytest.raises(GroundingError, match=message):
        parse_grounded_synthesis(_response(ids_factory(catalog)), catalog, None)


def test_ids_outside_the_offered_window_fail():
    content = "\n\n".join(f"Paragraph {n} reports fact {n}." for n in range(30))
    catalog = build_span_catalog(content, "w.md")
    windows = catalog.windows(200)
    outside = windows[0].span_ids[0]
    with pytest.raises(GroundingError, match="not offered"):
        parse_grounded_synthesis(_response([outside]), catalog, windows[1])


def test_grounding_error_is_a_retryable_validation_error():
    assert issubclass(GroundingError, SynthesisValidationError)
