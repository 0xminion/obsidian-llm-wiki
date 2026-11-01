"""Producer citation interface v2: one explicit contiguous ``evidence_span_range``.

A live preflight showed a producer citing ``[7, 9]`` (skipping span 8) when it
meant the passage 7..9. The free ID list made that adjacency mistake
representable. New producer responses therefore name ONE range by its
endpoints; the application materializes every intervening offered ID and then
applies the unchanged strict ``SpanCatalog.cite`` (bounds on the EXPANDED IDs
and characters, offered-window and revision checks). Legacy free lists are
rejected, never silently re-interpreted as ranges. Persisted evidence keeps its
expanded ``span_ids`` and stays readable.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import EvidenceSpan, SourceDoc, source_synthesis_from_dict
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.span_catalog import MAX_CITATION_IDS, build_span_catalog
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.synth.grounding import (
    CITATION_INTERFACE_VERSION,
    GroundingError,
    parse_grounded_synthesis,
)
from obsidian_llm_wiki.synth.prompts import DraftFeedback, build_synthesis_prompt
from obsidian_llm_wiki.synth.response_schema import SYNTHESIS_RESPONSE_SCHEMA
from tests.new.grounding_doubles import offered_spans
from tests.new.immutable_semantic_helpers import (
    published_meta,
    published_root,
)

# Shaped like the T08 preflight passage: a parenthetical "(e.g. " split puts the
# needed example list in the span AFTER the one that introduces it.
T08_LIKE = (
    "Software 1.0 was written by hand. "
    "With AI now we write programs that we could not write by hand before. "
    "We do it by specifying objectives (e.g. "
    "classification accuracy, reward functions) and searching with **gradient descent**.\n"
    "Tasks that are verifiable get automated fast. "
    "Tasks that are unverifiable lag behind. "
    "This drives the jagged frontier.\n"
)
CHINESE = "试点开始。新路由器将丢包率降低了百分之十二。该路由器没有降低延迟。研究团队认为仍需验证。"
RAWFORMAT = (
    "Header\n\nThe **router v2** cut drops\tby 12%\nin 2024. It did not\n  reduce latency.\n"
)


def _range(start: str, end: str) -> dict:
    return {"start_id": start, "end_id": end}


def _response(evidence, *, key="evidence_span_range", text="A supported claim."):
    claim = {"text": text, key: evidence, "source_ref": ""}
    return json.dumps(
        {
            "source_title": "S",
            "source_summary": "Summary.",
            "source_tags": [],
            "key_points": [],
            "open_questions": [],
            "language": "en",
            "concepts": [
                {
                    "title": "C",
                    "slug": "c",
                    "canonical_id": "",
                    "summary": "C summary.",
                    "tags": [],
                    "aliases": [],
                    "sections": [{"heading": "H", "points": ["p"], "prose": ""}],
                    "claims": [claim],
                    "related": [],
                    "confidence": 0.9,
                    "provenance": "extracted",
                    "is_new": True,
                }
            ],
            "maps": [],
        },
        ensure_ascii=False,
    )


def _ids_containing(catalog, *phrases):
    return [next(s.id for s in catalog.spans if phrase in s.text) for phrase in phrases]


# ── schema + prompts ─────────────────────────────────────────────────────────


def test_wire_schema_claim_cites_bounded_closed_fragment_ranges():
    claim = SYNTHESIS_RESPONSE_SCHEMA["properties"]["concepts"]["items"]["properties"]["claims"]
    items = claim["items"]
    assert items["additionalProperties"] is False
    assert (
        set(items["properties"])
        == set(items["required"])
        == {
            "text",
            "evidence_span_ranges",
            "source_ref",
        }
    )
    ranges = items["properties"]["evidence_span_ranges"]
    assert ranges["minItems"] == 1 and ranges["maxItems"] == 3
    span_range = ranges["items"]
    assert span_range["type"] == "object" and span_range["additionalProperties"] is False
    assert set(span_range["properties"]) == set(span_range["required"]) == {"start_id", "count"}
    assert span_range["properties"]["start_id"]["type"] == "string"
    assert span_range["properties"]["count"]["type"] == "integer"
    assert span_range["properties"]["count"]["enum"] == list(range(1, MAX_CITATION_IDS + 1))
    assert "evidence_span_ids" not in json.dumps(SYNTHESIS_RESPONSE_SCHEMA)


@pytest.mark.parametrize("mode", ["single", "refine", "feedback"])
def test_single_refine_and_feedback_prompts_share_the_range_interface(mode):
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    window = catalog.windows(10_000)[0]
    kwargs = {}
    if mode == "refine":
        kwargs["refine_draft"] = _response(_range(catalog.spans[0].id, catalog.spans[0].id))
    if mode == "feedback":
        kwargs["feedback"] = DraftFeedback.from_rejection(
            previous_draft="{}",
            error="GroundingError: x",
            diagnostics=(),
        )
    prompt = build_synthesis_prompt(
        "T08", T08_LIKE, span_window=window, source_file="t08.md", **kwargs
    )
    assert '"evidence_span_ranges"' in prompt and "start_id" in prompt and '"count"' in prompt
    assert "exactly count consecutive\noffered spans starting at start_id" in prompt
    assert "evidence_span_ids" not in prompt


def test_interface_is_versioned_in_the_synthesis_compatibility_epoch(monkeypatch):
    from obsidian_llm_wiki.core import pipeline
    from obsidian_llm_wiki.synth import grounding

    assert CITATION_INTERFACE_VERSION == "evidence-fragment-ranges-v1"
    assert "fragment-ranges" in pipeline._PROMPT_VERSION
    config = Config(vault_path="/tmp/x")
    before = pipeline._artifact_versions_for_config(config)
    monkeypatch.setattr(grounding, "CITATION_INTERFACE_VERSION", "evidence-span-range-v999")
    assert pipeline._artifact_versions_for_config(config) != before


# ── parser positives ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content, first, last, must_contain",
    [
        (
            T08_LIKE,
            "With AI now",
            "classification accuracy",
            "We do it by specifying objectives (e.g. ",
        ),
        (T08_LIKE, "verifiable get", "jagged frontier", "Tasks that are unverifiable lag behind."),
        (CHINESE, "新路由器", "没有降低延迟", "该路由器没有降低延迟。"),
        (RAWFORMAT, "**router v2**", "reduce latency", "in 2024. It did not\n  reduce latency."),
    ],
)
def test_range_materializes_every_intervening_span_exactly(content, first, last, must_contain):
    catalog = build_span_catalog(content, "s.md")
    window = catalog.windows(10_000)[0]
    start, end = _ids_containing(catalog, first, last)
    synthesis = parse_grounded_synthesis(_response(_range(start, end)), catalog, window)
    evidence = synthesis.concepts[0].claims[0].evidence
    start_index = catalog.span(start).index
    end_index = catalog.span(end).index
    assert evidence.span_ids == tuple(s.id for s in catalog.spans[start_index : end_index + 1])
    assert len(evidence.span_ids) == end_index - start_index + 1 >= 2
    assert must_contain in evidence.quote
    assert content[evidence.start_offset : evidence.end_offset] == evidence.quote


def test_single_span_range_is_start_equals_end():
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    (only,) = _ids_containing(catalog, "jagged frontier")
    synthesis = parse_grounded_synthesis(_response(_range(only, only)), catalog, None)
    assert synthesis.concepts[0].claims[0].evidence.span_ids == (only,)


# ── parser negatives ─────────────────────────────────────────────────────────


def _neg(catalog, window, evidence, message, key="evidence_span_range"):
    with pytest.raises(GroundingError, match=message):
        parse_grounded_synthesis(_response(evidence, key=key), catalog, window)


@pytest.mark.parametrize(
    "ids",
    [
        "gap",
        "contiguous",
        "single",
    ],
)
def test_legacy_free_lists_are_rejected_not_reinterpreted(ids):
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    s = [span.id for span in catalog.spans]
    legacy = {"gap": [s[1], s[3]], "contiguous": [s[1], s[2]], "single": [s[1]]}[ids]
    _neg(catalog, None, legacy, "evidence_span_ids", key="evidence_span_ids")


def test_both_interfaces_at_once_are_rejected():
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    data = json.loads(_response(_range(catalog.spans[0].id, catalog.spans[0].id)))
    data["concepts"][0]["claims"][0]["evidence_span_ids"] = [catalog.spans[0].id]
    with pytest.raises(GroundingError, match="evidence_span_ids"):
        parse_grounded_synthesis(json.dumps(data), catalog, None)


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda c, s: _range(s[0] + "0", s[1]), "unknown"),
        (lambda c, s: _range(s[0], c.tag + "-999"), "unknown"),
        (
            lambda c, s: _range(s[0], build_span_catalog(T08_LIKE + "x", "t08.md").spans[1].id),
            "different source",
        ),
        (
            lambda c, s: _range(build_span_catalog(T08_LIKE, "other.md").spans[0].id, s[1]),
            "different source",
        ),
        (lambda c, s: _range(s[3], s[1]), "reversed"),
        (lambda c, s: {"start_id": s[0]}, "exactly"),
        (lambda c, s: {"start_id": s[0], "end_id": s[1], "middle_id": s[0]}, "exactly"),
        (lambda c, s: {"start_id": s[0], "end_id": None}, "string"),
        (lambda c, s: [s[0], s[1]], "object"),
        (lambda c, s: None, "object"),
        (lambda c, s: _range("", ""), "unknown"),
    ],
)
def test_malformed_unknown_stale_or_reversed_ranges_fail(make, message):
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    s = [span.id for span in catalog.spans]
    _neg(catalog, None, make(catalog, s), message)


def test_range_bounds_apply_to_expanded_ids_and_characters():
    many = " ".join(f"Fact {n} holds." for n in range(MAX_CITATION_IDS + 2))
    catalog = build_span_catalog(many, "m.md")
    s = [span.id for span in catalog.spans]
    _neg(catalog, None, _range(s[0], s[MAX_CITATION_IDS]), f"at most {MAX_CITATION_IDS}")
    ok = parse_grounded_synthesis(
        _response(_range(s[0], s[MAX_CITATION_IDS - 1])),
        catalog,
        None,
    )
    assert len(ok.concepts[0].claims[0].evidence.span_ids) == MAX_CITATION_IDS
    long_text = " ".join(("word " * 230).strip() + "." for _ in range(5))
    long_catalog = build_span_catalog(long_text, "l.md")
    ls = [span.id for span in long_catalog.spans]
    _neg(long_catalog, None, _range(ls[0], ls[-1]), "exceeds")


@pytest.mark.parametrize("endpoint", ["start", "end"])
def test_out_of_window_endpoints_fail(endpoint):
    content = "\n\n".join(f"Paragraph {n} reports fact {n}." for n in range(30))
    catalog = build_span_catalog(content, "w.md")
    windows = catalog.windows(200)
    inside = windows[1].span_ids
    outside = windows[2].span_ids[0] if endpoint == "end" else windows[0].span_ids[-1]
    evidence = _range(inside[0], outside) if endpoint == "end" else _range(outside, inside[-1])
    _neg(catalog, windows[1], evidence, "not offered")


def test_persisted_legacy_evidence_span_ids_remain_readable():
    catalog = build_span_catalog(T08_LIKE, "t08.md")
    evidence = catalog.cite([catalog.spans[1].id, catalog.spans[2].id])
    payload = {
        "source_title": "S",
        "source_summary": "s",
        "concepts": [
            {
                "title": "C",
                "slug": "c",
                "summary": "c",
                "claims": [
                    {
                        "text": "t",
                        "evidence": {
                            "quote": evidence.quote,
                            "source_file": "t08.md",
                            "source_hash": evidence.source_hash,
                            "start_offset": evidence.start_offset,
                            "end_offset": evidence.end_offset,
                            "verification": "verified",
                            "span_ids": list(evidence.span_ids),
                            "catalog_version": evidence.catalog_version,
                        },
                    }
                ],
            }
        ],
    }
    loaded = source_synthesis_from_dict(payload).concepts[0].claims[0].evidence
    assert isinstance(loaded, EvidenceSpan) and loaded.span_ids == evidence.span_ids


# ── actual wire, public pipeline ─────────────────────────────────────────────


class _Endpoint:
    def __init__(self, evidence_for):
        self.evidence_for = evidence_for
        self.kinds: list[str] = []
        self.producer_bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        prompt = body["messages"][-1]["content"]
        if "reviews" in schema["properties"]:
            self.kinds.append("review")
            units = json.loads(prompt.split("--- UNITS ---", 1)[1])
            first = offered_spans(prompt.split("--- UNITS ---")[0])[0][0]
            content = json.dumps(
                {
                    "reviews": {
                        u["unit_id"]: {
                            "text_sha256": u["text_sha256"],
                            "verdict": "supported",
                            "support_span_ids": u.get("cited_span_ids") or [first],
                            "rationale": "scripted",
                            "model_context_uses": [],
                        }
                        for u in units
                    }
                }
            )
        else:
            self.kinds.append("producer")
            self.producer_bodies.append(body)
            spans = offered_spans(prompt)
            content = _response(
                self.evidence_for([sid for sid, _ in spans]),
                text="Objectives are specified and searched.",
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
        )


def _install(monkeypatch, endpoint):
    client_type = httpx.Client
    monkeypatch.setattr(
        llm.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(endpoint), trust_env=False, **kwargs
        ),
    )
    return endpoint


def _config(tmp_path):
    return Config(
        vault_path=str(tmp_path),
        retry_count=1,
        min_source_chars=1,
        embeddings_enabled=False,
        llm=LLMProviderConfig(
            provider="openai", host="https://fixture.invalid/v1", model="gen-model"
        ),
    )


def test_wire_range_round_trip_admits_the_whole_passage(tmp_path, monkeypatch):
    endpoint = _install(monkeypatch, _Endpoint(lambda ids: _range(ids[1], ids[3])))
    config = _config(tmp_path)
    result = asyncio.run(run_pipeline(tmp_path, {"t08.md": SourceDoc("T08", T08_LIKE)}, config))
    assert result.errors == [] and result.compiled == 1
    schema = endpoint.producer_bodies[0]["response_format"]["json_schema"]["schema"]
    assert "evidence_span_range" in json.dumps(schema)
    evidence = load_synthesis(published_meta(config), "t08.md").concepts[0].claims[0].evidence
    assert len(evidence.span_ids) == 3
    assert "(e.g. classification accuracy" in evidence.quote
    page = (published_root(config) / "concepts" / "c.md").read_text(encoding="utf-8")
    assert "(e.g. classification accuracy" in page


@pytest.mark.parametrize(
    "evidence_for",
    [
        lambda ids: _range(ids[3], ids[1]),  # reversed
        lambda ids: [ids[1], ids[3]],  # legacy gap list (the T08 failure)
    ],
)
def test_wire_bad_ranges_fail_closed_before_review(tmp_path, monkeypatch, evidence_for):
    if isinstance(evidence_for([f"x-{n}" for n in range(5)]), list):
        endpoint = _install(monkeypatch, _Endpoint(evidence_for))
        original = endpoint.__call__

        def legacy_call(request, _original=original):
            response = _original(request)
            body = json.loads(response.content)
            content = body["choices"][0]["message"]["content"]
            if '"evidence_span_range"' in content:
                body["choices"][0]["message"]["content"] = content.replace(
                    '"evidence_span_range"', '"evidence_span_ids"'
                )
            return httpx.Response(200, json=body)

        _install(monkeypatch, legacy_call)
    else:
        endpoint = _install(monkeypatch, _Endpoint(evidence_for))
    config = _config(tmp_path)
    result = asyncio.run(run_pipeline(tmp_path, {"t08.md": SourceDoc("T08", T08_LIKE)}, config))
    assert result.compiled == 0 and result.errors
    assert "review" not in endpoint.kinds
    assert load_synthesis(published_meta(config), "t08.md") is None
