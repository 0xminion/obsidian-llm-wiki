"""Self-authored bounded-context wire/persistence controls, not semantic evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy

import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.synth.review import (
    collect_review_units,
    verify_review_receipt,
    verify_review_receipt_integrity,
)
from tests.new.grounding_doubles import cite, grounded_response, review_answer
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
)

CASES = [
    (
        "en",
        "Mira described the Finch valve during the dry run. ",
        "She wrote: “It may open only under load.”",
        "Mira wrote that, during the dry run, the Finch valve may open only under load.",
    ),
    (
        "zh",
        "米拉在空载演练中描述了雀阀。",
        "她写道：“它可能只在负载下打开。”",
        "米拉写道，在空载演练中，雀阀可能只在负载下打开。",
    ),
]


class WireProvider:
    """Only model answers are scripted. Keep the real producer/parser/review pipeline."""

    def __init__(self, language, before, cited, claim):
        self.language, self.before, self.cited, self.claim = language, before, cited, claim
        self.requests = []

    async def __call__(self, system, messages, config, **kwargs):
        request = deepcopy({"system": system, "messages": messages, **kwargs})
        self.requests.append(request)
        prompt = messages[-1]["content"]
        if kwargs.get("task") == "review":
            answer = review_answer(prompt, kwargs["format"], {})
            units = json.loads(prompt.split("--- UNITS ---", 1)[1])
            for unit in units:
                fields = kwargs["format"]["properties"]["reviews"]["properties"][unit["unit_id"]]
                if "model_context_uses" in fields["properties"]:
                    contexts = unit.get("context_binding", {}).get("offered_context", [])
                    answer["reviews"][unit["unit_id"]]["model_context_uses"] = [
                        {"context_id": c["context_id"], "purpose": "referent"} for c in contexts
                    ]
            return json.dumps(answer, ensure_ascii=False)
        assert kwargs.get("task") != "identity"
        return grounded_response(
            summary=self.claim,
            language=self.language,
            concepts=[
                {
                    "title": "Finch valve" if self.language == "en" else "雀阀",
                    "slug": "finch-valve",
                    "summary": self.claim,
                    "sections": [{"heading": "Observation", "points": [self.claim]}],
                    "claims": [{"text": self.claim, "evidence_span_ids": cite(prompt, self.cited)}],
                }
            ],
        )


@pytest.mark.parametrize("language,before,cited,claim", CASES)
def test_real_pipeline_context_request_receipt_publication_reload_noop(
    monkeypatch,
    tmp_path,
    language,
    before,
    cited,
    claim,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config(vault_path=str(vault), llm=LLMProviderConfig(model="offline-wire"))
    content = before + cited
    source = SourceDoc(title="Observation", content=content, source_file="observation.md")
    sources = {"observation.md": source}
    config.sources_dir.mkdir(parents=True)
    human = config.sources_dir / "observation.md"
    human.write_bytes(content.encode("utf-8"))
    human_before = (human.read_bytes(), human.stat().st_mtime_ns)
    provider = WireProvider(language, before, cited, claim)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    first = asyncio.run(run_pipeline(vault, sources, config))
    assert first.errors == [] and first.compiled == 1
    reviews = [r for r in provider.requests if r["task"] == "review"]
    (tmp_path / "captured-requests.json").write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2)
    )
    offered_units = [
        u
        for r in reviews
        for u in json.loads(r["messages"][-1]["content"].split("--- UNITS ---", 1)[1])
    ]
    wire_claim = next(u for u in offered_units if u["kind"] == "claim")
    assert "context_binding" in wire_claim, "claim request needs explicit app-offered context"
    binding = wire_claim["context_binding"]
    assert binding["source_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert binding["source_file"] == "observation.md"
    assert [
        identifier for fragment in binding["cited_fragments"] for identifier in fragment["span_ids"]
    ] == wire_claim["cited_span_ids"]
    assert binding["offered_context"]
    for context in binding["offered_context"]:
        assert content[context["char_start"] : context["char_end"]] == context["text"]
        raw = content.encode()[context["byte_start"] : context["byte_end"]]
        assert raw == context["text"].encode()
        assert context["text_sha256"] == hashlib.sha256(raw).hexdigest()
        assert context["text"] not in (content, cited)
    cached = load_synthesis(published_meta(config), "observation.md")
    unit = next(u for u in collect_review_units(cached) if u.kind == "claim")
    outcome = cached.review_receipt["units"][unit.key]
    assert outcome["context_binding"] == binding
    assert outcome["model_context_uses"]
    assert binding["unit_key"] == unit.key
    assert verify_review_receipt(cached, content, "observation.md", config) == ""
    assert verify_review_receipt_integrity(cached, content, "observation.md") == ""
    evidence = cached.concepts[0].claims[0].evidence
    assert evidence.quote == cited and evidence.span_ids == tuple(wire_claim["cited_span_ids"])
    assert content[evidence.start_offset : evidence.end_offset] == cited
    assert cached.concepts[0].claims[0].text == claim
    with published_registry(config) as registry:
        loaded = registry.load_syntheses({"observation.md"})["observation.md"]
    assert loaded.review_receipt == cached.review_receipt
    root = published_root(config)
    note = (root / "concepts/finch-valve.md").read_text()
    assert cited in note and claim in note
    preserved = file_state(root)
    provider.requests.clear()
    repeated = asyncio.run(run_pipeline(vault, sources, config))
    assert repeated.errors == [] and repeated.compiled == 0 and repeated.skipped == 1
    assert provider.requests == []
    assert file_state(root) == preserved
    assert (human.read_bytes(), human.stat().st_mtime_ns) == human_before


def reviewed_case():
    from obsidian_llm_wiki.core.models import Claim, ConceptNote, SourceSynthesis
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog
    from obsidian_llm_wiki.synth.review import build_review_receipt, review_window

    language, before, cited, claim = CASES[1]
    content = before + cited + " 限定于这次演练。"
    catalog = build_span_catalog(content, "observation.md")
    evidence = catalog.cite([next(s.id for s in catalog.spans if cited in s.text)])
    synthesis = SourceSynthesis(
        source_title="Observation",
        source_file="observation.md",
        source_summary=claim,
        concepts=[
            ConceptNote(
                title="雀阀",
                slug="finch",
                summary=claim,
                claims=[
                    Claim(
                        text=claim,
                        concept_slug="finch",
                        evidence=evidence,
                    )
                ],
            ),
        ],
    )
    config = Config(llm=LLMProviderConfig(model="offline-wire"))
    provider = WireProvider(language, before, cited, claim)
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
    unit = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
    return synthesis, catalog, config, unit, outcomes


@pytest.mark.parametrize(
    "problem",
    [
        "foreign-source",
        "fabricated-source-hash",
        "fabricated-text-hash",
        "negative-offset",
        "byte-offset",
        "boolean-offset",
        "full-source",
        "oversized",
        "wrong-unit",
        "wrong-citation",
        "unknown-policy",
        "extra-field",
        "missing-context",
        "nonclaim-context",
        "missing-use",
        "distant-valid-hash",
        "wrong-context-id",
        "gap-bridge",
    ],
)
def test_receipt_and_builder_reject_tampered_context(problem):
    from obsidian_llm_wiki.synth.review import SynthesisReviewError, build_review_receipt

    synthesis, catalog, config, unit, outcomes = reviewed_case()
    outcome = synthesis.review_receipt["units"][unit.key]
    binding = outcome["context_binding"]
    record = binding["offered_context"][0]
    if problem == "foreign-source":
        binding["source_file"] = "foreign.md"
    elif problem == "fabricated-source-hash":
        binding["source_sha256"] = "0" * 64
    elif problem == "fabricated-text-hash":
        record["text_sha256"] = "0" * 64
    elif problem == "negative-offset":
        record["char_start"] = -1
    elif problem == "byte-offset":
        record["byte_end"] -= 1
    elif problem == "boolean-offset":
        record["char_start"] = False
    elif problem == "full-source":
        record["text"] = catalog.content
    elif problem == "oversized":
        record["text"] = "中" * 10000
    elif problem == "wrong-unit":
        binding["unit_key"] = "0" * 32
    elif problem == "wrong-citation":
        binding["cited_span_ids"] = [catalog.spans[0].id]
    elif problem == "unknown-policy":
        binding["policy"] = "review-context-future"
    elif problem == "extra-field":
        record["role"] = "independent-factual-support"
    elif problem == "missing-context":
        outcome.pop("context_binding")
    elif problem == "nonclaim-context":
        key = next(u.key for u in collect_review_units(synthesis) if u.kind != "claim")
        synthesis.review_receipt["units"][key]["context_binding"] = binding
    elif problem == "missing-use":
        outcome.pop("model_context_uses")
    elif problem == "distant-valid-hash":
        record.update(
            char_start=len(catalog.content) - 4,
            char_end=len(catalog.content),
            text=catalog.content[-4:],
        )
        record["text_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    elif problem == "wrong-context-id":
        record["context_id"] = "ctx-" + "0" * 64
    elif problem == "gap-bridge":
        binding["visible_intervals"] = [[0, 2], [3, len(catalog.content)]]
    assert verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file)
    assert verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)
    with pytest.raises(SynthesisReviewError):
        build_review_receipt(synthesis, catalog, synthesis.review_receipt["units"], config)


@pytest.mark.parametrize(
    "uses",
    [
        None,
        "ctx",
        [{"context_id": "ctx-foreign", "purpose": "referent"}],
        [{"context_id": "FIRST", "purpose": "new_fact"}],
        [{"context_id": "FIRST", "purpose": "referent", "fact": "invented"}],
        [{"context_id": "FIRST", "purpose": "referent"}] * 2,
        [{"context_id": ["FIRST"], "purpose": "referent"}],
    ],
)
def test_model_declared_uses_are_validated_before_admission(uses):
    from obsidian_llm_wiki.synth.review import SynthesisReviewError, review_window

    synthesis, catalog, config, unit, _ = reviewed_case()
    provider = WireProvider(*CASES[1])

    async def invalid(system, messages, config, **kwargs):
        answer = json.loads(await provider(system, messages, config, **kwargs))
        units = json.loads(messages[-1]["content"].split("--- UNITS ---", 1)[1])
        claim = next(u for u in units if u["kind"] == "claim")
        bad = deepcopy(uses)
        if isinstance(bad, list):
            for item in bad:
                if item["context_id"] == "FIRST":
                    item["context_id"] = claim["context_binding"]["offered_context"][0][
                        "context_id"
                    ]
        answer["reviews"][claim["unit_id"]]["model_context_uses"] = bad
        return json.dumps(answer)

    with pytest.raises(SynthesisReviewError, match="context"):
        asyncio.run(
            review_window(synthesis, catalog, catalog.windows(10000)[0], config, call=invalid)
        )


def test_only_context_policy_change_invalidates_reviewer_and_inner_versions(monkeypatch):
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config
    from obsidian_llm_wiki.synth import review_context
    from obsidian_llm_wiki.synth.review import reviewer_fingerprint

    synthesis, catalog, config, _, _ = reviewed_case()
    before = reviewer_fingerprint(config)
    versions = _artifact_versions_for_config(config)
    monkeypatch.setattr(
        review_context, "FRAGMENT_CONTEXT_POLICY_VERSION", "review-context-next-test"
    )
    assert reviewer_fingerprint(config) != before
    assert _artifact_versions_for_config(config) != versions
    assert verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)


def test_known_v2_receipt_is_readable_but_not_current_and_unknown_version_fails():
    synthesis, catalog, config, _, _ = reviewed_case()
    receipt = synthesis.review_receipt
    receipt["version"] = "grounded-review-v2"
    for outcome in receipt["units"].values():
        outcome.pop("context_binding")
        outcome.pop("model_context_uses")
    before = deepcopy(receipt)
    assert verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file) == ""
    assert verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)
    assert receipt == before
    receipt["version"] = "grounded-review-v999"
    assert verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file)


def offset_inputs(content=None):
    content = content or ("远" * 1500 + " BEFORE " + "It stopped." + " AFTER " + "近" * 1500)
    start = content.index("It stopped.")
    return {
        "content": content,
        "source_file": "offset-note.md",
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "catalog_version": "self-authored-offset-catalog-v1",
        "unit_key": "a" * 32,
        "cited_span_ids": ["explicit-offset-claim"],
        "citation_start": start,
        "citation_end": start + len("It stopped."),
        "visible_intervals": [[0, len(content)]],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "0" * 64),
        ("citation_start", -1),
        ("citation_end", 100000),
        ("citation_start", True),
        ("citation_start", "1"),
        ("catalog_version", ""),
        ("unit_key", ""),
        ("cited_span_ids", []),
        ("cited_span_ids", ["x", "x"]),
        ("visible_intervals", [[-1, 4000]]),
        ("visible_intervals", [[0, 4], [2, 4000]]),
        ("visible_intervals", [[0, 1]]),
        ("visible_intervals", None),
    ],
)
def test_pure_constructor_rejects_invalid_raw_inputs(field, value):
    from obsidian_llm_wiki.synth.review_context import build_context_binding

    inputs = offset_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match="context"):
        build_context_binding(**inputs)


def test_large_offset_catalog_context_is_bounded_and_raw_and_does_not_jump_gaps():
    from obsidian_llm_wiki.synth.review_context import (
        build_context_binding,
        validate_context_binding,
    )

    inputs = offset_inputs()
    result = build_context_binding(**inputs)
    assert validate_context_binding(result, **inputs) == ""
    records = result["offered_context"]
    assert [len(r["text"]) for r in records] == [384, 384]
    for record in records:
        assert (
            inputs["content"].encode()[record["byte_start"] : record["byte_end"]]
            == record["text"].encode()
        )
    # An explicitly visible island does not imply the bounding source interval was shown.
    start, end = inputs["citation_start"], inputs["citation_end"]
    inputs["visible_intervals"] = [
        [0, start - 5],
        [start - 2, end + 2],
        [end + 5, len(inputs["content"])],
    ]
    island = build_context_binding(**inputs)
    assert [len(r["text"]) for r in island["offered_context"]] == [2, 2]
    assert validate_context_binding(island, **inputs) == ""
    assert validate_context_binding(result, **inputs)


def test_raw_paragraph_boundaries_do_not_depend_on_catalog_segmentation():
    from obsidian_llm_wiki.synth.review_context import build_context_binding

    inputs = offset_inputs(
        "Unrelated facts.\r\n\r\nMira: It stopped. Only in this trial.\n\nDistant."
    )
    result = build_context_binding(**inputs)
    assert [r["text"] for r in result["offered_context"]] == ["Mira: ", " Only in this trial."]


@pytest.mark.parametrize(
    "problem", ["source-hash", "window-text", "span-text", "span-offset", "id-set", "anchor-offset"]
)
def test_forged_input_fails_before_reviewer_request(problem):
    from dataclasses import replace

    from obsidian_llm_wiki.synth.review import SynthesisReviewError, review_window

    synthesis, catalog, config, unit, _ = reviewed_case()
    window = catalog.windows(10000)[0]
    if problem == "source-hash":
        catalog = replace(catalog, revision="0" * 64)
    elif problem == "window-text":
        window = replace(window, text=window.text + " forged")
    elif problem == "span-text":
        window = replace(window, spans=(replace(window.spans[0], text="forged"), *window.spans[1:]))
    elif problem == "span-offset":
        window = replace(window, spans=(replace(window.spans[0], start=-1), *window.spans[1:]))
    elif problem == "id-set":
        window = replace(window, _id_set=frozenset())
    elif problem == "anchor-offset":
        synthesis.concepts[0].claims[0].evidence.start_offset += 1
    provider = WireProvider(*CASES[1])
    with pytest.raises(SynthesisReviewError):
        asyncio.run(review_window(synthesis, catalog, window, config, call=provider))
    assert provider.requests == []


def test_known_current_receipt_integrity_survives_new_policy_without_relabeling(monkeypatch):
    from obsidian_llm_wiki.synth import review_context

    synthesis, catalog, config, _, _ = reviewed_case()
    before = deepcopy(synthesis.review_receipt)
    monkeypatch.setattr(
        review_context, "FRAGMENT_CONTEXT_POLICY_VERSION", "review-context-next-test"
    )
    assert verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file) == ""
    assert synthesis.review_receipt == before
    assert verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)


@pytest.mark.parametrize("field", ["support_span_ids", "model_context_uses"])
def test_context_ids_cannot_become_factual_support_and_nonclaims_cannot_use_context(field):
    from obsidian_llm_wiki.synth.review import SynthesisReviewError, review_window

    synthesis, catalog, config, _, _ = reviewed_case()
    provider = WireProvider(*CASES[1])

    async def invalid(system, messages, config, **kwargs):
        answer = json.loads(await provider(system, messages, config, **kwargs))
        units = json.loads(messages[-1]["content"].split("--- UNITS ---", 1)[1])
        claim = next(u for u in units if u["kind"] == "claim")
        context_id = claim["context_binding"]["offered_context"][0]["context_id"]
        if field == "support_span_ids":
            answer["reviews"][claim["unit_id"]][field] = [context_id]
        else:
            nonclaim = next(u for u in units if u["kind"] != "claim")
            answer["reviews"][nonclaim["unit_id"]][field] = [
                {"context_id": context_id, "purpose": "referent"},
            ]
        return json.dumps(answer)

    with pytest.raises(SynthesisReviewError):
        asyncio.run(
            review_window(synthesis, catalog, catalog.windows(10000)[0], config, call=invalid)
        )


def test_exact_body_supplied_with_trusted_hash_still_validates_context():
    synthesis, catalog, config, _, _ = reviewed_case()
    assert (
        verify_review_receipt(
            synthesis,
            catalog.content,
            catalog.source_file,
            config,
            source_sha256=catalog.revision,
        )
        == ""
    )
    assert verify_review_receipt(
        synthesis,
        catalog.content + " forged",
        catalog.source_file,
        config,
        source_sha256=catalog.revision,
    )


def test_receipt_builder_reports_malformed_outcome_without_keyerror():
    from obsidian_llm_wiki.synth.review import SynthesisReviewError, build_review_receipt

    synthesis, catalog, config, unit, outcomes = reviewed_case()
    outcomes[unit.key].pop("verdict")
    with pytest.raises(SynthesisReviewError):
        build_review_receipt(synthesis, catalog, outcomes, config)
