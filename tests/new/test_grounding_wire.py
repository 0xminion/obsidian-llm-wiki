"""Grounded producer + review through the real OpenAI-compatible HTTP client.

Only httpx's transport is replaced; prompts, schemas, request serialization,
response parsing, admission, cache and render all run production code.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator, ValidationError

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import _call_structured_synthesis, run_pipeline
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.synth.grounding import GroundingError, parse_grounded_synthesis
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from tests.new.grounding_doubles import cite, grounded_response, identity_answer, review_answer
from tests.new.immutable_semantic_helpers import (
    published_meta,
)

MARKDOWN_PDF_EN = (
    "# Field trial\n\nThe **router v2** cut dropped packets by 12%\nin the 2024 pilot "
    "([report](https://example.org/r?id=7)).\nIt did\tnot reduce   latency.\n"
)
CHINESE = (
    "试点显示，新路由器将丢包率降低了百分之十二。该路由器没有降低延迟。研究团队认为仍需更多验证。"
)


def _summary_for(prompt: str) -> tuple[str, str, str]:
    if "试点" in prompt:
        return (
            "新路由器将丢包率降低了百分之十二，但没有降低延迟。",
            "新路由器将丢包率降低了百分之十二。",
            "降低了百分之十二",
        )
    return (
        "Router v2 cut dropped packets by 12% but did not reduce latency.",
        "Router v2 cut dropped packets by 12% in the 2024 pilot.",
        "cut dropped packets by 12%",
    )


class WireEndpoint:
    """In-process OpenAI-compatible endpoint scripted per request kind."""

    def __init__(self, *, producer=None, verdicts=None, review_status=200):
        self.requests: list[dict] = []
        self.producer = producer or self.default_producer
        self.verdicts = verdicts or {}
        self.review_status = review_status

    @staticmethod
    def default_producer(prompt: str) -> str:
        summary, claim, phrase = _summary_for(prompt)
        return grounded_response(
            summary=summary,
            concepts=[
                {
                    "title": "Router v2",
                    "slug": "router-v2",
                    "summary": claim,
                    "sections": [{"heading": "Effect", "points": [claim]}],
                    "claims": [{"text": claim, "evidence_span_ids": cite(prompt, phrase)}],
                }
            ],
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        self.requests.append(body)
        schema = body["response_format"]["json_schema"]["schema"]
        prompt = body["messages"][-1]["content"]
        if "reviews" in schema["properties"]:
            if self.review_status != 200:
                return httpx.Response(self.review_status, json={"error": "unavailable"})
            content = json.dumps(review_answer(prompt, schema, self.verdicts), ensure_ascii=False)
        elif "assessments" in schema["properties"]:
            content = json.dumps(identity_answer(prompt, schema), ensure_ascii=False)
        else:
            content = self.producer(prompt)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            },
        )


@pytest.fixture
def wire(monkeypatch):
    client_type = httpx.Client

    def install(endpoint: WireEndpoint):
        monkeypatch.setattr(
            llm.httpx,
            "Client",
            lambda **kwargs: client_type(
                transport=httpx.MockTransport(endpoint),
                trust_env=False,
                **kwargs,
            ),
        )
        return endpoint

    return install


def _config(tmp_path, **llm_overrides) -> Config:
    return Config(
        vault_path=str(tmp_path),
        llm=LLMProviderConfig(
            provider="openai",
            host="https://fixture.invalid/v1",
            model="gen-model",
            **llm_overrides,
        ),
        retry_count=1,
        min_source_chars=1,
        embeddings_enabled=False,
    )


def _build(tmp_path, config, sources):
    return asyncio.run(run_pipeline(tmp_path, sources, config))


@pytest.mark.parametrize(
    "filename, content",
    [
        ("trial-en.md", MARKDOWN_PDF_EN),
        ("trial-zh.md", CHINESE),
    ],
)
def test_wire_round_trip_admits_exact_raw_evidence_after_closed_review(
    tmp_path,
    wire,
    filename,
    content,
):
    endpoint = wire(WireEndpoint())
    config = _config(tmp_path, review_model="judge-model")
    result = _build(tmp_path, config, {filename: SourceDoc(title="Trial", content=content)})
    assert result.errors == [] and result.compiled == 1

    producer, review = endpoint.requests
    assert producer["model"] == "gen-model"
    assert "evidence_span_range" in json.dumps(producer["response_format"])
    assert review["model"] == "judge-model" and review["temperature"] == 0
    schema = review["response_format"]["json_schema"]["schema"]
    reviews = schema["properties"]["reviews"]
    assert schema["additionalProperties"] is False and reviews["additionalProperties"] is False
    assert set(reviews["required"]) == set(reviews["properties"]) and reviews["required"]
    for unit_schema in reviews["properties"].values():
        assert unit_schema["additionalProperties"] is False
        assert unit_schema["properties"]["verdict"]["enum"] == [
            "supported",
            "contradicted",
            "insufficient",
        ]

    evidence = load_synthesis(published_meta(config), filename).concepts[0].claims[0].evidence
    assert content[evidence.start_offset : evidence.end_offset] == evidence.quote
    if filename == "trial-en.md":
        # Raw Markdown emphasis and PDF hard wrap are preserved, not normalized.
        assert "**router v2** cut dropped packets by 12%\nin the 2024 pilot" in evidence.quote


def test_review_defaults_to_a_separate_call_of_the_generator_model(tmp_path, wire):
    endpoint = wire(WireEndpoint())
    config = _config(tmp_path, ingest_model="ingest-model")
    assert _build(tmp_path, config, {"t.md": SourceDoc("T", MARKDOWN_PDF_EN)}).compiled == 1
    producer, review = endpoint.requests
    assert producer["model"] == review["model"] == "ingest-model"
    assert producer["messages"] != review["messages"]


@pytest.mark.parametrize(
    "bad_ids",
    [
        ["00000000-1"],  # unknown / different catalog
        ["deadbeef-999"],  # not a span of this revision
    ],
)
def test_wire_unknown_ids_are_rejected_before_review(tmp_path, wire, bad_ids):
    def producer(prompt):
        data = json.loads(WireEndpoint.default_producer(prompt))
        data["concepts"][0]["claims"][0]["evidence_span_range"] = {
            "start_id": bad_ids[0],
            "end_id": bad_ids[-1],
        }
        return json.dumps(data)

    endpoint = wire(WireEndpoint(producer=producer))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"t.md": SourceDoc("T", MARKDOWN_PDF_EN)})
    assert result.compiled == 0 and result.errors
    assert all("reviews" not in json.dumps(r["response_format"]) for r in endpoint.requests)
    assert load_synthesis(published_meta(config), "t.md") is None


def test_wire_stale_revision_ids_are_rejected(tmp_path, wire):
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog

    stale = build_span_catalog(MARKDOWN_PDF_EN.replace("12%", "15%"), "t.md").spans[1].id

    def producer(prompt):
        data = json.loads(WireEndpoint.default_producer(prompt))
        data["concepts"][0]["claims"][0]["evidence_span_range"] = {
            "start_id": stale,
            "end_id": stale,
        }
        return json.dumps(data)

    wire(WireEndpoint(producer=producer))
    result = _build(tmp_path, _config(tmp_path), {"t.md": SourceDoc("T", MARKDOWN_PDF_EN)})
    assert result.compiled == 0
    assert any("different source or source revision" in error for error in result.errors)


@pytest.mark.parametrize(
    "claim, verdict",
    [
        ("Router v1 cut dropped packets by 12% in the 2024 pilot.", "contradicted"),  # actor
        ("Router v2 cut dropped packets by 21% in the 2024 pilot.", "contradicted"),  # number
        ("Router v2 also reduced latency.", "contradicted"),  # negation
    ],
)
def test_wire_review_verdicts_block_admission(tmp_path, wire, claim, verdict):
    def producer(prompt):
        data = json.loads(WireEndpoint.default_producer(prompt))
        data["concepts"][0]["claims"][0]["text"] = claim
        return json.dumps(data)

    endpoint = wire(WireEndpoint(producer=producer, verdicts={claim: verdict}))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"t.md": SourceDoc("T", MARKDOWN_PDF_EN)})
    assert result.compiled == 0
    assert any(verdict in error and claim in error for error in result.errors)
    assert len(endpoint.requests) == 4  # (producer + review) x bounded retry
    assert load_synthesis(published_meta(config), "t.md") is None


def test_wire_reviewer_http_failure_is_not_success(tmp_path, wire):
    wire(WireEndpoint(review_status=503))
    config = _config(tmp_path)
    result = _build(tmp_path, config, {"t.md": SourceDoc("T", MARKDOWN_PDF_EN)})
    assert result.compiled == 0 and result.errors
    assert load_synthesis(published_meta(config), "t.md") is None


def test_wire_unchanged_rebuild_sends_zero_requests(tmp_path, wire):
    sources = {"t.md": SourceDoc("T", MARKDOWN_PDF_EN), "z.md": SourceDoc("Z", CHINESE)}
    config = _config(tmp_path)
    wire(WireEndpoint())
    assert _build(tmp_path, config, sources).compiled == 2
    endpoint = wire(WireEndpoint())
    again = _build(tmp_path, config, sources)
    assert again.errors == [] and again.compiled == 0 and again.skipped == 2
    assert endpoint.requests == []


# Self-authored EN/CN controls; offline production HTTP wire, not inference.

EN = (
    "The laboratory evaluated **Sensor R**. It did\tnot improve accuracy\r\nin cold rooms. "
    + " ".join(f"Trial {n} recorded no change." for n in range(18))
)
CN = "实验室评估了**传感器R**。它在低温室内\t没有改善\r\n准确率。" + "".join(
    f"第{n}次试验没有发现变化。" for n in range(18)
)


def payload(selection):
    return {
        "source_title": "Sensor R",
        "source_summary": "Sensor R did not improve accuracy.",
        "source_tags": [],
        "key_points": [],
        "open_questions": [],
        "language": "en",
        "concepts": [
            {
                "title": "Sensor R",
                "slug": "sensor-r",
                "canonical_id": "",
                "summary": "Sensor R did not improve accuracy.",
                "tags": [],
                "aliases": [],
                "sections": [{"heading": "Trial", "points": ["No accuracy gain."], "prose": ""}],
                "claims": [
                    {
                        "text": "Sensor R did not improve accuracy in cold rooms.",
                        "evidence_span_ranges": [selection],
                        "source_ref": "",
                    }
                ],
                "related": [],
                "confidence": 0.9,
                "provenance": "extracted",
                "is_new": True,
            }
        ],
        "maps": [],
    }


def config(tmp_path, **overrides):
    return Config(
        vault_path=str(tmp_path),
        retry_count=1,
        min_source_chars=1,
        embeddings_enabled=False,
        llm=LLMProviderConfig(
            provider="openai", host="https://fixture.invalid/v1", model="fixture"
        ),
        **overrides,
    )


def install(monkeypatch, handler):
    client_type = httpx.Client
    monkeypatch.setattr(
        llm.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(handler), trust_env=False, **kwargs
        ),
    )


def completion(data):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": json.dumps(data, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ]
        },
    )


@pytest.mark.parametrize("content", [EN, CN], ids=["EN", "CN"])
@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_producer_real_wire_only_expresses_one_to_six_spans(
    tmp_path,
    monkeypatch,
    content,
    provider,
):
    catalog = build_span_catalog(content, "sensor.md")
    window = catalog.windows(100_000)[0]
    captured = []
    good = payload({"start_id": window.span_ids[0], "count": 6})

    def endpoint(request):
        assert request.url.path == ("/api/chat" if provider == "ollama" else "/v1/chat/completions")
        body = json.loads(request.content)
        captured.append(body)
        if provider == "ollama":
            return httpx.Response(
                200,
                json={
                    "message": {"content": json.dumps(good)},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        return completion(good)

    install(monkeypatch, endpoint)
    prompt = build_synthesis_prompt("Sensor R", content, span_window=window)
    cfg = config(tmp_path)
    cfg.llm.provider = provider
    if provider == "ollama":
        cfg.llm.host = "https://fixture.invalid"
    asyncio.run(_call_structured_synthesis(prompt, [{"role": "user", "content": prompt}], cfg))
    assert len(captured) == 1
    schema = (
        captured[0]["format"]
        if provider == "ollama"
        else captured[0]["response_format"]["json_schema"]["schema"]
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for count in (1, 6):
        validator.validate(payload({"start_id": window.span_ids[0], "count": count}))
    for count in (0, 7, 9, 16, -1, True, "6", 1.5, None):
        with pytest.raises(ValidationError):
            validator.validate(payload({"start_id": window.span_ids[0], "count": count}))
    with pytest.raises(ValidationError):
        validator.validate(payload({"start_id": window.span_ids[0], "end_id": window.span_ids[15]}))
    assert '"count"' in prompt
    assert "at most 6 spans" in prompt


@pytest.mark.parametrize("content", [EN, CN], ids=["EN", "CN"])
@pytest.mark.parametrize("count", [1, 2, 6])
def test_count_parser_preserves_exact_legacy_evidence(content, count):
    catalog = build_span_catalog(content, "sensor.md")
    window = catalog.windows(100_000)[0]
    start = catalog.spans[0].id
    new = parse_grounded_synthesis(
        json.dumps(payload({"start_id": start, "count": count})), catalog, window
    )
    legacy = parse_grounded_synthesis(
        json.dumps(payload({"start_id": start, "end_id": catalog.spans[count - 1].id})),
        catalog,
        window,
    )
    evidence = new.concepts[0].claims[0].evidence
    assert evidence == legacy.concepts[0].claims[0].evidence
    assert evidence.span_ids == tuple(s.id for s in catalog.spans[:count])
    raw = content[evidence.start_offset : evidence.end_offset]
    assert evidence.quote.encode("utf-8") == raw.encode("utf-8")
    if count >= 2:
        assert "\r\n" in evidence.quote and "\t" in evidence.quote and "**" in evidence.quote


@pytest.mark.parametrize("count", [0, 7, 9, 16, -1, True, "6", 1.0, 1.5, None])
def test_count_parser_rejects_invalid_count_without_coercion(count):
    catalog = build_span_catalog(EN, "sensor.md")
    with pytest.raises(GroundingError, match="count"):
        parse_grounded_synthesis(
            json.dumps(payload({"start_id": catalog.spans[0].id, "count": count})), catalog, None
        )


def test_count_parser_rejects_overflow_gap_char_bound_and_ambiguous_fields():
    catalog = build_span_catalog(EN, "sensor.md")
    window = catalog.windows(100_000)[0]

    def reject(selection, offered, message, cat=catalog):
        with pytest.raises(GroundingError, match=message):
            parse_grounded_synthesis(json.dumps(payload(selection)), cat, offered)

    reject({"start_id": catalog.spans[-1].id, "count": 2}, None, "beyond|exceed")
    gap = replace(
        window,
        span_ids=(window.span_ids[0], window.span_ids[2]),
        _id_set=frozenset((window.span_ids[0], window.span_ids[2])),
    )
    reject({"start_id": catalog.spans[0].id, "count": 3}, gap, "not offered")
    reject(
        {"start_id": catalog.spans[0].id, "count": 1, "end_id": catalog.spans[0].id},
        window,
        "exactly",
    )
    reject(
        {"start_id": build_span_catalog(EN, "foreign.md").spans[0].id, "count": 1},
        None,
        "different source",
    )
    long = build_span_catalog(" ".join("x" * 1000 + "." for _ in range(5)), "long.md")
    reject({"start_id": long.spans[0].id, "count": 5}, None, "exceeds", long)


def review_units(catalog, window):
    from obsidian_llm_wiki.synth.review import ReviewUnit

    evidence = catalog.cite([window.span_ids[1]], offered=window)
    return [
        ReviewUnit(
            "a" * 32,
            "claim",
            "claims[0]",
            "No accuracy improvement.",
            evidence.span_ids,
            evidence.quote,
        ),
        ReviewUnit(
            "b" * 32,
            "claim",
            "claims[1]",
            "Accuracy did not improve.",
            evidence.span_ids,
            evidence.quote,
        ),
        ReviewUnit("c" * 32, "source_summary", "summary", "The sensor was evaluated."),
    ]


def review_response(listed, first):
    return {
        "reviews": {
            u["unit_id"]: {
                "text_sha256": u["text_sha256"],
                "verdict": "supported",
                "support_span_ids": u.get("cited_span_ids") or [first],
                "rationale": "scripted fixture, not a semantic accuracy measurement",
                "model_context_uses": [],
            }
            for u in listed
        }
    }


@pytest.mark.parametrize("content", [EN, CN], ids=["EN", "CN"])
@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_review_real_wire_binds_context_to_each_unit(tmp_path, monkeypatch, content, provider):
    from obsidian_llm_wiki.synth.review import _review_units

    catalog = build_span_catalog(content, "sensor.md")
    window = catalog.windows(100_000)[0]
    units = review_units(catalog, window)
    captured = []

    def endpoint(request):
        body = json.loads(request.content)
        listed = json.loads(body["messages"][-1]["content"].split("--- UNITS ---", 1)[1])
        data = review_response(listed, window.span_ids[0])
        captured.append((body, listed, data))
        if provider == "ollama":
            return httpx.Response(
                200,
                json={
                    "message": {"content": json.dumps(data)},
                    "done": True,
                    "done_reason": "stop",
                },
            )
        return completion(data)

    install(monkeypatch, endpoint)
    cfg = config(tmp_path)
    cfg.llm.provider = provider
    if provider == "ollama":
        cfg.llm.host = "https://fixture.invalid"
    asyncio.run(_review_units(units, catalog, window, cfg))
    body, listed, data = captured[0]
    schema = (
        body["format"] if provider == "ollama" else body["response_format"]["json_schema"]["schema"]
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(data)
    for own, other in ((listed[0], listed[1]), (listed[1], listed[0])):
        for record in own["context_binding"]["offered_context"]:
            good = deepcopy(data)
            good["reviews"][own["unit_id"]]["model_context_uses"] = [
                {"context_id": record["context_id"], "purpose": "referent"}
            ]
            validator.validate(good)
        bad = deepcopy(data)
        bad["reviews"][own["unit_id"]]["model_context_uses"] = [
            {
                "context_id": other["context_binding"]["offered_context"][0]["context_id"],
                "purpose": "referent",
            }
        ]
        with pytest.raises(ValidationError):
            validator.validate(bad)
    bad = deepcopy(data)
    bad["reviews"][listed[2]["unit_id"]]["model_context_uses"] = [
        {
            "context_id": listed[0]["context_binding"]["offered_context"][0]["context_id"],
            "purpose": "referent",
        }
    ]
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_review_empty_binding_real_wire_forbids_any_context_use(tmp_path, monkeypatch):
    from obsidian_llm_wiki.synth.review import ReviewUnit, _review_units

    catalog = build_span_catalog("No gain.", "one.md")
    window = catalog.windows(1000)[0]
    evidence = catalog.cite(window.span_ids)
    unit = ReviewUnit("d" * 32, "claim", "claim", "No gain.", evidence.span_ids, evidence.quote)
    captured = []

    def endpoint(request):
        body = json.loads(request.content)
        listed = json.loads(body["messages"][-1]["content"].split("--- UNITS ---", 1)[1])
        assert listed[0]["context_binding"]["offered_context"] == []
        data = review_response(listed, window.span_ids[0])
        captured.append((body, data))
        return completion(data)

    install(monkeypatch, endpoint)
    asyncio.run(_review_units([unit], catalog, window, config(tmp_path)))
    body, data = captured[0]
    validator = Draft202012Validator(body["response_format"]["json_schema"]["schema"])
    validator.validate(data)
    data["reviews"]["u1"]["model_context_uses"] = [{"context_id": "", "purpose": "referent"}]
    with pytest.raises(ValidationError):
        validator.validate(data)


def test_wire_contract_versions_invalidate_only_current_compatibility(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core import pipeline
    from obsidian_llm_wiki.synth import grounding, review

    assert grounding.CITATION_INTERFACE_VERSION == "evidence-fragment-ranges-v1"
    assert pipeline._PROMPT_VERSION == "synthesis-v9-fragment-ranges-feedback-refine"
    assert review.REVIEW_PROMPT_VERSION == "grounding-review-prompt-v8-fragments-linked-context"
    cfg = config(tmp_path)
    current = pipeline._artifact_versions_for_config(cfg)
    fingerprint = review.reviewer_fingerprint(cfg)
    monkeypatch.setattr(review, "REVIEW_PROMPT_VERSION", "grounding-review-prompt-v4")
    assert review.reviewer_fingerprint(cfg) != fingerprint
    assert pipeline._artifact_versions_for_config(cfg) != current
    assert review.REVIEW_RECEIPT_VERSION == "grounded-review-v4"
    assert pipeline._SYNTHESIS_SCHEMA_VERSION == "synthesis-schema-v5-fragments"


@pytest.mark.parametrize("content", [EN, CN], ids=["EN", "CN"])
@pytest.mark.parametrize("count", [7, 9, 16])
def test_legacy_overlong_ranges_still_reject_without_truncation(content, count):
    catalog = build_span_catalog(content, "sensor.md")
    data = payload({"start_id": catalog.spans[0].id, "end_id": catalog.spans[count - 1].id})
    claim = data["concepts"][0]["claims"][0]
    claim["evidence_span_range"] = claim.pop("evidence_span_ranges")[0]
    with pytest.raises(GroundingError, match=f"covers {count} spans.*at most 6"):
        parse_grounded_synthesis(json.dumps(data), catalog, None)


@pytest.mark.parametrize("damage", ["foreign", "nonclaim", "duplicate", "purpose", "support"])
def test_reviewer_runtime_rejection_checks_remain_independent_of_schema(tmp_path, damage):
    from obsidian_llm_wiki.synth.review import (
        SynthesisReviewError,
        _validate_batch,
        context_binding_for_unit,
    )

    catalog = build_span_catalog(EN, "sensor.md")
    window = catalog.windows(100_000)[0]
    units = review_units(catalog, window)
    batch = [(f"u{i}", unit) for i, unit in enumerate(units)]
    bindings = {u.key: context_binding_for_unit(u, catalog, window) for u in units}
    listed = [
        {"unit_id": uid, "text_sha256": u.digest, "cited_span_ids": u.cited_span_ids}
        for uid, u in batch
    ]
    data = review_response(listed, window.span_ids[0])
    own = bindings[units[0].key]["offered_context"][0]["context_id"]
    foreign = bindings[units[1].key]["offered_context"][0]["context_id"]
    if damage == "foreign":
        data["reviews"]["u0"]["model_context_uses"] = [
            {"context_id": foreign, "purpose": "referent"}
        ]
    elif damage == "nonclaim":
        data["reviews"]["u2"]["model_context_uses"] = [{"context_id": own, "purpose": "referent"}]
    elif damage == "duplicate":
        data["reviews"]["u0"]["model_context_uses"] = [
            {"context_id": own, "purpose": "referent"}
        ] * 2
    elif damage == "purpose":
        data["reviews"]["u0"]["model_context_uses"] = [{"context_id": own, "purpose": "extra_fact"}]
    else:
        data["reviews"]["u0"]["support_span_ids"] = [window.span_ids[-1]]
    with pytest.raises(SynthesisReviewError):
        _validate_batch(json.dumps(data), batch, catalog, window, bindings)


@pytest.mark.parametrize("content", [EN, CN], ids=["EN", "CN"])
@pytest.mark.parametrize(
    "mode, chunk_size",
    [("single_pass", 48_000), ("two_pass", 48_000), ("single_pass", 160), ("two_pass", 160)],
)
def test_current_count_wire_pipeline_modes_and_noop(
    tmp_path,
    monkeypatch,
    content,
    mode,
    chunk_size,
):
    from tests.new.grounding_doubles import offered_spans

    captured = []
    producer_count = 0

    def endpoint(request):
        nonlocal producer_count
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        prompt = body["messages"][-1]["content"]
        if "reviews" in schema["properties"]:
            listed = json.loads(prompt.split("--- UNITS ---", 1)[1])
            first = offered_spans(prompt.split("--- UNITS ---", 1)[0])[0][0]
            data = review_response(listed, first)
        elif "assessments" in schema["properties"]:
            data = identity_answer(prompt, schema)
        else:
            ids = [sid for sid, _ in offered_spans(prompt)]
            data = payload({"start_id": ids[0], "count": min(2, len(ids))})
            producer_count += 1
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(data)
        captured.append(body)
        return completion(data)

    install(monkeypatch, endpoint)
    cfg = config(tmp_path, synthesis_mode=mode, chunk_size=chunk_size)
    sources = {"sensor.md": SourceDoc("Sensor R", content)}
    first = asyncio.run(run_pipeline(tmp_path, sources, cfg))
    assert first.errors == [] and first.compiled == 1
    assert producer_count >= (2 if mode == "two_pass" else 1)
    if chunk_size == 160:
        assert producer_count > (2 if mode == "two_pass" else 1)
    cached = load_synthesis(published_meta(cfg), "sensor.md")
    for claim in cached.concepts[0].claims:
        ev = claim.evidence
        assert 1 <= len(ev.span_ids) <= 6
        assert ev.quote == content[ev.start_offset : ev.end_offset]
    captured.clear()
    second = asyncio.run(run_pipeline(tmp_path, sources, cfg))
    assert second.errors == [] and second.compiled == 0 and second.skipped == 1
    assert captured == []
