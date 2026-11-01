"""Synthetic fragments: scripted review proves gates, not semantic entailment."""

import asyncio
import json
from copy import deepcopy

import pytest

from obsidian_llm_wiki.core.models import (
    SynthesisBundle,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.span_catalog import build_span_catalog, verify_evidence_anchor
from obsidian_llm_wiki.render.obsidian import render_concept_page
from obsidian_llm_wiki.synth.grounding import parse_grounded_synthesis
from obsidian_llm_wiki.synth.publication import plan_publication, verify_publication
from obsidian_llm_wiki.synth.review import (
    build_review_receipt,
    collect_review_units,
    review_window,
    verify_review_receipt,
)
from tests.new.test_grounding_review import ScriptedReviewer, _config

CONTENT = "青鸟可以游泳。\r\n未引用的间隔说明。\r\n青鸟也可以飞行。\r\n"


def payload(catalog):
    return {
        "source_title": "青鸟",
        "source_file": "bird.md",
        "source_summary": "青鸟可以游泳，也可以飞行。",
        "concepts": [
            {
                "title": "青鸟",
                "slug": "bird",
                "summary": "青鸟可以游泳，也可以飞行。",
                "sections": [{"heading": "能力", "points": ["青鸟可以游泳，也可以飞行。"]}],
                "claims": [
                    {
                        "text": "青鸟可以游泳，也可以飞行。",
                        "source_ref": "bird.md",
                        "evidence_span_ranges": [
                            {"start_id": catalog.spans[i].id, "count": 1} for i in (0, 2)
                        ],
                    }
                ],
            }
        ],
    }


def parsed(content=CONTENT):
    catalog = build_span_catalog(content, "bird.md")
    window = catalog.windows(10000)[0]
    synthesis = parse_grounded_synthesis(json.dumps(payload(catalog)), catalog, window)
    return catalog, window, synthesis


def approved():
    catalog, window, synthesis = parsed()
    config, reviewer = _config(), ScriptedReviewer()
    outcomes = asyncio.run(review_window(synthesis, catalog, window, config, call=reviewer))
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    return catalog, synthesis, config, reviewer


def test_two_separate_fragments_survive_parse_review_receipt_reload_publication_render():
    catalog, synthesis, config, reviewer = approved()
    evidence = synthesis.concepts[0].claims[0].evidence
    assert len(evidence.fragments) == 2 and not hasattr(evidence, "quote")
    assert [f.quote for f in evidence.fragments] == ["青鸟可以游泳。", "青鸟也可以飞行。"]
    for f in evidence.fragments:
        assert verify_evidence_anchor(f, CONTENT, "bird.md") == ""
        assert CONTENT.encode()[f.start_byte : f.end_byte] == f.quote.encode()
    units = [
        u
        for r in reviewer.requests
        for u in json.loads(r["messages"][-1]["content"].split("--- UNITS ---", 1)[1])
    ]
    claim = next(u for u in units if u["kind"] == "claim")
    assert [f["quote"] for f in claim["cited_fragments"]] == [f.quote for f in evidence.fragments]
    assert "cited_text" not in claim
    assert claim["cited_span_ids"] == [catalog.spans[i].id for i in (0, 2)]
    assert all(r["strict_schema"] is True for r in reviewer.requests)
    assert len(units) == len(collect_review_units(synthesis))
    wire = source_synthesis_to_dict(synthesis)
    restored = source_synthesis_from_dict(json.loads(json.dumps(wire)))
    assert source_synthesis_to_dict(restored) == wire
    assert verify_review_receipt(restored, CONTENT, "bird.md", config) == ""
    bundle = SynthesisBundle(sources=[restored], concepts=restored.concepts)
    plan = plan_publication(bundle)
    assert not plan.errors and not verify_publication(bundle, plan)
    rendered = render_concept_page(restored.concepts[0])
    for f in evidence.fragments:
        assert f.quote in rendered and f"@{f.start_offset}–{f.end_offset}" in rendered
    assert "未引用的间隔说明" not in rendered


def test_fragment_consumers_pipeline_cache_registry_preview_query_and_noop(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config, LLMProviderConfig
    from obsidian_llm_wiki.core.cache import load_synthesis
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from obsidian_llm_wiki.providers import llm
    from obsidian_llm_wiki.query.graph import build_graph_from_vault
    from tests.new.immutable_semantic_helpers import (
        file_state,
        published_meta,
        published_registry,
        published_root,
    )

    config = Config(vault_path=str(tmp_path), llm=LLMProviderConfig(model="offline-fragments"))
    config.sources_dir.mkdir(parents=True)
    human = config.sources_dir / "bird.md"
    human.write_bytes(CONTENT.encode())
    human_before = (human.read_bytes(), human.stat().st_mtime_ns)
    catalog = build_span_catalog(CONTENT, "bird.md")
    reviewer, calls = ScriptedReviewer(), []

    async def provider(system, messages, config, **kwargs):
        calls.append(kwargs)
        if kwargs.get("task") == "review":
            return await reviewer(system, messages, config, **kwargs)
        assert kwargs.get("task") != "identity"
        return json.dumps(payload(catalog))

    monkeypatch.setattr(llm, "acall_llm", provider)
    source = SourceDoc("青鸟", CONTENT, source_file="bird.md")
    result = asyncio.run(run_pipeline(tmp_path, {"bird.md": source}, config))
    assert result.errors == [] and result.compiled == 1
    cached = load_synthesis(published_meta(config), "bird.md")
    assert len(cached.concepts[0].claims[0].evidence.fragments) == 2
    assert verify_review_receipt(cached, CONTENT, "bird.md", config) == ""
    with published_registry(config) as registry:
        restored = registry.load_syntheses({"bird.md"})["bird.md"]
    assert source_synthesis_to_dict(restored) == source_synthesis_to_dict(cached)
    root = published_root(config)
    before = file_state(root)
    note = (root / "concepts/bird.md").read_text()
    assert "未引用的间隔说明" not in note and "青鸟也可以飞行。" in note
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert graph.pages and any(p.current_eligible for p in graph.pages.values())
    calls.clear()
    repeated = asyncio.run(run_pipeline(tmp_path, {"bird.md": source}, config))
    assert repeated.errors == [] and repeated.compiled == 0 and repeated.skipped == 1
    assert calls == [] and published_root(config) == root and file_state(root) == before
    assert (human.read_bytes(), human.stat().st_mtime_ns) == human_before


def test_producer_schema_uses_bounded_closed_fragment_ranges():
    from obsidian_llm_wiki.synth.response_schema import SYNTHESIS_RESPONSE_SCHEMA

    claim = SYNTHESIS_RESPONSE_SCHEMA["properties"]["concepts"]["items"]["properties"]["claims"][
        "items"
    ]
    assert claim["additionalProperties"] is False
    assert set(claim["required"]) == {"text", "source_ref", "evidence_span_ranges"}
    ranges = claim["properties"]["evidence_span_ranges"]
    assert ranges["minItems"] == 1 and ranges["maxItems"] == 3
    assert ranges["items"]["additionalProperties"] is False
    assert ranges["items"]["properties"]["count"]["enum"] == [1, 2, 3, 4, 5, 6]


def test_every_consumer_preserves_fragment_identity_and_reports_all_anchors():
    from obsidian_llm_wiki.core.evidence import resolve_synthesis_evidence
    from obsidian_llm_wiki.core.identity_migration import _claim_facts
    from obsidian_llm_wiki.core.models import SourceDoc, concept_note_to_dict
    from obsidian_llm_wiki.synth.dedupe import _published_claim_identity
    from obsidian_llm_wiki.synth.grounding import _claim_identity
    from obsidian_llm_wiki.synth.identity import _subject_span_ids
    from obsidian_llm_wiki.synth.publication import claim_key
    from obsidian_llm_wiki.synth.quality import _claim_key, _merge_claims

    catalog, _, synthesis = parsed()
    original = deepcopy(synthesis)
    resolve_synthesis_evidence(synthesis, SourceDoc("Bird", CONTENT), "bird.md")
    assert synthesis == original
    claim = synthesis.concepts[0].claims[0]
    other = deepcopy(claim)
    other.evidence.fragments[1].quote += "changed"
    for key in (claim_key, _claim_key, _claim_identity, _published_claim_identity):
        assert key(claim) != key(other), key.__name__
    assert len(_merge_claims([claim], [other])) == 2
    assert _subject_span_ids(synthesis, 0) == [catalog.spans[i].id for i in (0, 2)]
    facts = _claim_facts(concept_note_to_dict(synthesis.concepts[0]), catalog.revision)
    assert facts["catalog_span_citations"] == 2
    assert facts["cited_source_hash_matches_current"] is True


def test_completed_synthesis_validator_rejects_tampered_second_fragment():
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

    _, _, synthesis = parsed()
    synthesis.concepts[0].claims[0].evidence.fragments[1].quote += "伪造"
    with pytest.raises(SynthesisValidationError):
        validate_synthesis(synthesis, SourceDoc("青鸟", CONTENT), "bird.md")


@pytest.mark.parametrize(
    "bad",
    [
        "reorder",
        "overlap",
        "duplicate",
        "stale",
        "wrong-source",
        "hidden",
        "too-many",
        "too-many-ids",
        "bool",
        "both",
    ],
)
def test_parser_rejects_invalid_fragment_ranges(bad):
    from obsidian_llm_wiki.synth.grounding import GroundingError

    content = " ".join(f"Statement {i}." for i in range(10))
    catalog = build_span_catalog(content, "bird.md")
    window = catalog.windows(10000)[0]
    data = payload(catalog)
    claim = data["concepts"][0]["claims"][0]
    ranges = claim["evidence_span_ranges"]
    if bad == "reorder":
        ranges.reverse()
    elif bad == "overlap":
        ranges[0]["count"] = 3
    elif bad == "duplicate":
        ranges[1] = dict(ranges[0])
    elif bad in ("stale", "wrong-source"):
        other = build_span_catalog(
            content + "!" if bad == "stale" else content,
            "bird.md" if bad == "stale" else "other.md",
        )
        ranges[1]["start_id"] = other.spans[2].id
    elif bad == "hidden":
        window = catalog.windows(len(catalog.spans[0].text))[0]
    elif bad == "too-many":
        claim["evidence_span_ranges"] = [
            {"start_id": catalog.spans[i].id, "count": 1} for i in (0, 2, 4, 6)
        ]
    elif bad == "too-many-ids":
        ranges[:] = [
            {"start_id": catalog.spans[0].id, "count": 4},
            {"start_id": catalog.spans[6].id, "count": 3},
        ]
    elif bad == "bool":
        ranges[1]["count"] = True
    else:
        claim["evidence_span_range"] = dict(ranges[0])
    with pytest.raises(GroundingError):
        parse_grounded_synthesis(json.dumps(data), catalog, window)


def test_fragment_character_total_cannot_exceed_existing_quote_budget():
    from obsidian_llm_wiki.synth.grounding import GroundingError

    catalog = build_span_catalog(" ".join("x" * 1000 + "." for _ in range(8)), "bird.md")
    data = payload(catalog)
    data["concepts"][0]["claims"][0]["evidence_span_ranges"] = [
        {"start_id": catalog.spans[0].id, "count": 2},
        {"start_id": catalog.spans[3].id, "count": 2},
    ]
    with pytest.raises(GroundingError, match="4000"):
        parse_grounded_synthesis(json.dumps(data), catalog, catalog.windows(20000)[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("quote", "伪造"),
        ("source_hash", "0" * 64),
        ("source_file", "other.md"),
        ("start_byte", 0),
        ("start_offset", 0),
        ("catalog_version", "unknown"),
        ("span_ids", ["invalid"]),
    ],
)
def test_receipt_rejects_second_fragment_tamper_even_with_resealed_unit_digest(field, value):
    from obsidian_llm_wiki.synth.review import _output_digest, verify_review_receipt_integrity

    _, synthesis, config, _ = approved()
    before = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
    setattr(synthesis.concepts[0].claims[0].evidence.fragments[1], field, value)
    assert verify_review_receipt(synthesis, CONTENT, "bird.md", config)
    after = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
    units = synthesis.review_receipt["units"]
    units[after.key] = units.pop(before.key)
    synthesis.review_receipt["output_sha256"] = _output_digest(list(units))
    assert verify_review_receipt_integrity(synthesis, CONTENT, "bird.md")


def test_reviewer_cannot_use_gap_as_claim_support():
    from obsidian_llm_wiki.synth.review import SynthesisReviewError

    catalog, window, synthesis = parsed()

    def corrupt(answer):
        for item in answer["reviews"].values():
            item["support_span_ids"] = [catalog.spans[1].id]
        return json.dumps(answer)

    with pytest.raises(SynthesisReviewError, match="not one of its cited spans"):
        asyncio.run(
            review_window(
                synthesis, catalog, window, _config(), call=ScriptedReviewer(mutate=corrupt)
            )
        )
