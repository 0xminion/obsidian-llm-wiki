"""Self-authored contract fixtures, not semantic model evaluation.

Captures real review_window requests; only the provider response is scripted.
No benchmark cases, external reference judgments, credentials or live inference.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.models import Claim, ConceptNote, SourceSynthesis
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth.review import (
    SynthesisReviewError,
    build_review_receipt,
    collect_review_units,
    review_window,
    verify_review_receipt,
    verify_review_receipt_integrity,
)

SOURCE = (
    "During the bench demonstration, Leda evaluated the Kestrel module. "
    "It stopped the pump only while the cover was open."
)
CITED = "It stopped the pump only while the cover was open."
CLAIM = (
    "During the bench demonstration, the Kestrel module stopped the pump "
    "only while the cover was open."
)


def _case(text=CLAIM, *, source=SOURCE, cited=CITED, title="Kestrel module"):
    catalog = build_span_catalog(source, "bench.md")
    span = next(span for span in catalog.spans if span.text.strip() == cited)
    synthesis = SourceSynthesis(
        source_title="Bench notes", source_file="bench.md", source_summary=cited,
        concepts=[ConceptNote(title=title, slug="device", summary=cited, claims=[Claim(
            text=text, concept_slug="device", source_ref="bench.md",
            evidence=catalog.cite([span.id]),
        )])],
    )
    return synthesis, catalog


class CapturingReviewer:
    """Wire double only: it does NOT infer entailment from the prompt."""

    def __init__(self, mutate=None, claim_verdict="supported"):
        self.requests = []
        self.mutate = mutate
        self.claim_verdict = claim_verdict

    async def __call__(self, prompt, messages, config, **kwargs):
        self.requests.append(deepcopy({"prompt": prompt, "messages": messages, **kwargs}))
        units = json.loads(prompt.split("--- UNITS ---", 1)[1])
        # The source summary/title are scripted as supported; claim support stays cited-only.
        first_id = prompt.split("--- SOURCE SPANS ---", 1)[1].split("⟦", 1)[1].split("⟧", 1)[0]
        payload = {"reviews": {unit["unit_id"]: {
            "text_sha256": unit["text_sha256"],
            "verdict": self.claim_verdict if unit["kind"] == "claim" else "supported",
            "support_span_ids": unit.get("cited_span_ids", [first_id]),
            "rationale": "scripted wire response, not a semantic judgment",
            "model_context_uses": [],
        } for unit in units}}
        if self.mutate:
            return self.mutate(payload, units)
        return json.dumps(payload)


def _review(synthesis, catalog, reviewer, config=None):
    return asyncio.run(review_window(
        synthesis, catalog, catalog.windows(10_000)[0],
        config or Config(llm=LLMProviderConfig(model="local-wire-double")), call=reviewer,
    ))


def test_actual_prompt_permits_bounded_local_resolution(tmp_path):
    synthesis, catalog = _case()
    before = deepcopy(synthesis.concepts[0].claims[0].evidence)
    reviewer = CapturingReviewer()
    _review(synthesis, catalog, reviewer)
    request = reviewer.requests[0]
    (tmp_path / "review-request.json").write_text(json.dumps(request, indent=2))
    prompt = request["messages"][-1]["content"]
    units = json.loads(prompt.split("--- UNITS ---", 1)[1])
    claim = next(unit for unit in units if unit["kind"] == "claim")
    assert SOURCE.split("It stopped", 1)[0] in prompt  # adjacent antecedent is actually offered
    assert claim["cited_text"] == CITED  # context is not folded into the quote
    assert claim["cited_span_ids"] == list(before.span_ids)
    assert synthesis.concepts[0].claims[0].evidence == before
    policy = " ".join(prompt.split("--- SOURCE SPANS ---", 1)[0].split())
    assert "ignore all other source text" not in policy
    assert "same-argument context" in policy
    assert "ordinary faithful paraphrase" in policy
    assert "independent propositions must be supported by cited_text" in policy


def test_old_prompt_receipt_is_historical_not_current_cache(monkeypatch, tmp_path):
    from obsidian_llm_wiki.core.cache import load_synthesis, save_synthesis
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config
    from obsidian_llm_wiki.synth import review

    synthesis, catalog = _case()
    config = Config(llm=LLMProviderConfig(model="local-wire-double"))
    outcomes = _review(synthesis, catalog, CapturingReviewer(), config)
    with monkeypatch.context() as old:
        old.setattr(review, "REVIEW_PROMPT_VERSION", "grounding-review-prompt-v2")
        synthesis.review_receipt = review.build_review_receipt(synthesis, catalog, outcomes, config)
        old_versions = _artifact_versions_for_config(config)
        old_fingerprint = review.reviewer_fingerprint(config)
    save_synthesis(synthesis, tmp_path, "bench.md")
    cached = load_synthesis(tmp_path, "bench.md")
    assert cached is not None
    assert review.verify_review_receipt_integrity(cached, SOURCE, "bench.md") == ""
    assert "different reviewer configuration" in review.verify_review_receipt(
        cached, SOURCE, "bench.md", config,
    )
    assert review.reviewer_fingerprint(config) != old_fingerprint
    current_versions = _artifact_versions_for_config(config)
    assert current_versions.prompt != old_versions.prompt
    assert current_versions.schema == old_versions.schema
    current_receipt = review.build_review_receipt(cached, catalog, outcomes, config)
    assert current_receipt["units"] == cached.review_receipt["units"]
    assert current_receipt["output_sha256"] == cached.review_receipt["output_sha256"]
    assert current_receipt["version"] == cached.review_receipt["version"]
    cached.review_receipt = current_receipt
    assert review.verify_review_receipt(cached, SOURCE, "bench.md", config) == ""


def test_actual_policy_retains_context_boundaries_and_other_kind_guards():
    synthesis, catalog = _case()
    reviewer = CapturingReviewer()
    _review(synthesis, catalog, reviewer)
    policy = " ".join(reviewer.requests[0]["prompt"].split("--- SOURCE SPANS ---", 1)[0].split())
    for obligation in (
        "only explicit same-argument context in this unit's context_binding.offered_context",
        "pronouns or deictic references", "speaker/author attribution", "entity/model identity",
        "local framing qualifiers governing the cited passage",
        "Context must not supply uncited premises, roles, new quantities, causal links, "
        "list members, or unrelated entity facts",
        "Adjacency or topical similarity alone is not enough",
        "Missing or ambiguous referents/attribution remain insufficient",
        "Review the ENTIRE claim about its subject",
        "preserve actors, numbers, dates, negation, modality, attribution, causal direction, "
        "conditions and scope",
        "support_span_ids must be a subset of the unit's cited_span_ids",
        "source_ref must truthfully describe where the cited text is",
        "a statement that the source makes about a different entity "
        "is not supported for this subject",
        "section_heading — the heading the body unit is published under",
        "tag / source_tag / map_tag", "map_member — the source supports listing the concept",
    ):
        assert obligation in policy
    schema = reviewer.requests[0]["format"]
    assert schema["additionalProperties"] is False
    reviews = schema["properties"]["reviews"]
    assert reviews["additionalProperties"] is False
    assert set(reviews["required"]) == set(reviews["properties"])
    for item in reviews["properties"].values():
        assert item["additionalProperties"] is False
        assert set(item["required"]) == {
            "text_sha256", "verdict", "support_span_ids", "rationale", "model_context_uses",
        }
        assert item["properties"]["verdict"]["enum"] == [
            "supported", "contradicted", "insufficient",
        ]


# These invented pairs exercise exact request preservation and scripted verdict routing,
# not a model's ability to distinguish the pair. No examples are added to the policy.
PAIRS = [
    pytest.param(SOURCE, CITED, "Kestrel module", CLAIM,
                 CLAIM.replace("Kestrel module", "Leda"), "contradicted", id="actor-pronoun"),
    pytest.param(SOURCE, CITED, "Kestrel module", CLAIM,
                 CLAIM.replace("only while the cover was open", "even with the cover shut"),
                 "contradicted", id="condition"),
    pytest.param("NX denotes the Ibis controller. This device blocked the inlet.",
                 "This device blocked the inlet.", "Ibis controller",
                 "The Ibis controller blocked the inlet.",
                 "The Ibis flight controller blocked the inlet.", "insufficient",
                 id="deictic-identity-new-role"),
    pytest.param("Mara wrote this note. The valve may leak.", "The valve may leak.", "Valve",
                 "Mara wrote that the valve may leak.",
                 "Mara proved that the valve leaks.", "insufficient", id="speaker-modality"),
    pytest.param("The next result is from a dry run. The hatch remained shut.",
                 "The hatch remained shut.", "Hatch", "In the dry run, the hatch stayed shut.",
                 "The hatch always stays shut.", "insufficient", id="local-frame-paraphrase"),
    pytest.param("The Sable sensor was tested seven times. It logged a pulse.",
                 "It logged a pulse.", "Sable sensor", "The Sable sensor logged a pulse.",
                 "The Sable sensor logged seven pulses.", "insufficient", id="quantity-import"),
    pytest.param("Rin is the inspector. Rin closed the gate.", "Rin closed the gate.", "Rin",
                 "Rin closed the gate.", "Inspector Rin closed the gate.", "insufficient",
                 id="uncited-role"),
    pytest.param("A timer expired. The latch opened.", "The latch opened.", "Latch",
                 "The latch opened.", "The latch opened because a timer expired.", "insufficient",
                 id="causal-premise-import"),
    pytest.param("The cabinet also holds a prism. The kit contains a lens.",
                 "The kit contains a lens.", "Kit", "The kit includes a lens.",
                 "The kit includes a lens and a prism.", "insufficient", id="list-member-import"),
    pytest.param("The Lark is solar-powered. The Ibis blocked the inlet.",
                 "The Ibis blocked the inlet.", "Ibis", "The Ibis blocked the inlet.",
                 "The solar-powered Ibis blocked the inlet.", "insufficient",
                 id="unrelated-entity-fact"),
    pytest.param("Miro inspected the valve. It did not open.", "It did not open.", "Valve",
                 "The valve did not open.", "The valve opened.", "contradicted", id="negation"),
    pytest.param("Ori described the pulley. It pulled the sled.", "It pulled the sled.", "Pulley",
                 "The pulley pulled the sled.", "The sled pulled the pulley.", "contradicted",
                 id="direction"),
    pytest.param("Day four was reserved for repairs. On day three, the lamp was green.",
                 "On day three, the lamp was green.", "Lamp", "The lamp was green on day three.",
                 "The lamp was green on day four.", "contradicted", id="date"),
]


@pytest.mark.parametrize("source,cited,title,positive,negative,verdict", PAIRS)
def test_minimal_pairs_preserve_citations_and_route_scripted_rejections(
    source, cited, title, positive, negative, verdict,
):
    for text, scripted in ((positive, "supported"), (negative, verdict)):
        synthesis, catalog = _case(text, source=source, cited=cited, title=title)
        reviewer = CapturingReviewer(claim_verdict=scripted)
        evidence = deepcopy(synthesis.concepts[0].claims[0].evidence)
        if scripted == "supported":
            outcomes = _review(synthesis, catalog, reviewer)
            claim = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
            assert outcomes[claim.key]["span_ids"] == list(evidence.span_ids)
        else:
            with pytest.raises(SynthesisReviewError) as caught:
                _review(synthesis, catalog, reviewer)
            assert caught.value.diagnostics == [{
                "path": "concepts[0].claims[0]", "kind": "claim", "verdict": verdict,
                "text": text, "rationale": "scripted wire response, not a semantic judgment",
            }]
        units = json.loads(reviewer.requests[0]["prompt"].split("--- UNITS ---", 1)[1])
        claim = next(u for u in units if u["kind"] == "claim")
        assert (claim["text"], claim["subject"], claim["cited_text"]) == (text, title, cited)
        assert claim["cited_span_ids"] == list(evidence.span_ids)
        assert synthesis.concepts[0].claims[0].evidence == evidence


@pytest.mark.parametrize("problem,match", [
    ("missing-support", "no support span"),
    ("context-support", "not one of its cited spans"),
    ("foreign-support", "different source"),
    ("unknown-support", "unknown span"),
    ("duplicate-support", "duplicate support"),
    ("string-support", "must be strings"),
    ("missing-unit", "missing units"),
    ("unknown-unit", "unknown units"),
    ("duplicate-unit", "duplicate key"),
    ("wrong-digest", "digest"),
    ("extra-field", "exactly the fields"),
    ("missing-field", "exactly the fields"),
    ("invalid-verdict", "invalid verdict"),
    ("invalid-json", "not valid JSON"),
])
def test_actual_parser_rejects_malformed_or_uncited_support(problem, match):
    synthesis, catalog = _case()

    def mutate(payload, units):
        unit = next(u for u in units if u["kind"] == "claim")
        unit_id = unit["unit_id"]
        answer = payload["reviews"][unit_id]
        support = answer["support_span_ids"]
        if problem == "missing-support":
            answer["support_span_ids"] = []
        elif problem == "context-support":
            answer["support_span_ids"] = [catalog.spans[0].id]
        elif problem == "foreign-support":
            answer["support_span_ids"] = [build_span_catalog("Other.", "other.md").spans[0].id]
        elif problem == "unknown-support":
            answer["support_span_ids"] = [f"{catalog.tag}-999"]
        elif problem == "duplicate-support":
            answer["support_span_ids"] = support + support
        elif problem == "string-support":
            answer["support_span_ids"] = support[0]
        elif problem == "missing-unit":
            payload["reviews"].pop(unit_id)
        elif problem == "unknown-unit":
            payload["reviews"]["u999"] = answer
        elif problem == "duplicate-unit":
            return json.dumps(payload)[:-2] + f', "{unit_id}": ' + json.dumps(answer) + "}}"
        elif problem == "wrong-digest":
            answer["text_sha256"] = "0" * 16
        elif problem == "extra-field":
            answer["context_is_evidence"] = True
        elif problem == "missing-field":
            answer.pop("support_span_ids")
        elif problem == "invalid-verdict":
            answer["verdict"] = "partially_supported"
        elif problem == "invalid-json":
            return '{"reviews":'
        return json.dumps(payload)

    with pytest.raises(SynthesisReviewError, match=match):
        _review(synthesis, catalog, CapturingReviewer(mutate))


@pytest.mark.parametrize("problem,match", [
    ("context-support", "outside the claim's cited spans"),
    ("missing-support", "no support span"),
    ("output-digest", "output digest"),
    ("changed-subject", "unreviewed or unsupported"),
    ("changed-source-ref", "unreviewed or unsupported"),
    ("changed-text", "unreviewed or unsupported"),
    ("changed-quote", "unreviewed or unsupported"),
])
def test_receipt_integrity_guards_remain_active(problem, match):
    synthesis, catalog = _case()
    config = Config(llm=LLMProviderConfig(model="local-wire-double"))
    outcomes = _review(synthesis, catalog, CapturingReviewer(), config)
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    claim = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
    if problem == "context-support":
        synthesis.review_receipt["units"][claim.key]["span_ids"] = [catalog.spans[0].id]
    elif problem == "missing-support":
        synthesis.review_receipt["units"][claim.key]["span_ids"] = []
    elif problem == "output-digest":
        synthesis.review_receipt["output_sha256"] = "0" * 64
    elif problem == "changed-subject":
        synthesis.concepts[0].title = "Different module"
    elif problem == "changed-source-ref":
        synthesis.concepts[0].claims[0].source_ref = "another-note.md"
    elif problem == "changed-text":
        synthesis.concepts[0].claims[0].text = "The module never stopped the pump."
    elif problem == "changed-quote":
        synthesis.concepts[0].claims[0].evidence.quote = SOURCE
    assert match in verify_review_receipt_integrity(synthesis, SOURCE, "bench.md")
    assert match in verify_review_receipt(synthesis, SOURCE, "bench.md", config)


def test_missing_offered_antecedent_is_not_invented_or_added_to_citation():
    synthesis, catalog = _case()
    window = catalog.windows(1)[1]  # Only the pronoun sentence is offered.
    reviewer = CapturingReviewer(claim_verdict="insufficient")
    with pytest.raises(SynthesisReviewError, match="claim.*insufficient"):
        asyncio.run(review_window(synthesis, catalog, window, Config(), call=reviewer))
    rendered = reviewer.requests[0]["prompt"].split("--- SOURCE SPANS ---", 1)[1]
    source_spans = rendered.split("--- UNITS ---", 1)[0]
    assert "Leda evaluated" not in source_spans
    assert CITED in source_spans
    claim = next(u for u in json.loads(rendered.split("--- UNITS ---", 1)[1])
                 if u["kind"] == "claim")
    assert claim["cited_span_ids"] == [catalog.spans[1].id]
    assert claim["cited_text"] == CITED


@pytest.mark.parametrize("refresh_render", [False, True], ids=["default-noop", "explicit-replan"])
def test_prompt_upgrade_forces_pipeline_recompile_then_allows_noop(
    monkeypatch, tmp_path, refresh_render,
):
    from obsidian_llm_wiki.core.cache import load_synthesis
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from obsidian_llm_wiki.synth import review
    from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
    from tests.new.immutable_semantic_helpers import published_meta

    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config(vault_path=str(vault), llm=LLMProviderConfig(model="local-wire-double"))
    sources = {"bench.md": SourceDoc(title="Bench notes", content=SOURCE, source_file="bench.md")}

    def produce(prompt):
        return grounded_response(summary=CLAIM, title="Bench notes", concepts=[{
            "title": "Kestrel module", "slug": "kestrel-module", "summary": CLAIM,
            "sections": [{"heading": "Bench operation", "points": [CLAIM]}],
            "claims": [{"text": CLAIM, "evidence_span_ids": cite(prompt, CITED)}],
        }])

    def run(**options):
        provider = GroundedProvider(produce)
        monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
        result = asyncio.run(run_pipeline(vault, sources, config, **options))
        assert result.errors == []
        return result, provider

    with monkeypatch.context() as old:
        old.setattr(review, "REVIEW_PROMPT_VERSION", "grounding-review-prompt-v2")
        first, provider = run()
        assert first.compiled == 1 and len(provider.review_calls) == 1
        cached = load_synthesis(published_meta(config), "bench.md")
        old_fingerprint = cached.review_receipt["reviewer_fingerprint"]
    upgraded, provider = run(refresh_render=refresh_render)
    assert upgraded.compiled == 1 and len(provider.review_calls) == 1
    cached = load_synthesis(published_meta(config), "bench.md")
    assert cached.review_receipt["reviewer_fingerprint"] != old_fingerprint
    assert verify_review_receipt(cached, SOURCE, "bench.md", config) == ""
    assert cached.concepts[0].claims[0].evidence.quote == CITED
    repeated, provider = run()
    assert repeated.compiled == 0 and repeated.skipped == 1
    assert provider.calls == []
