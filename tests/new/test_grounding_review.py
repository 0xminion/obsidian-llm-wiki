"""Mandatory closed-schema support review of every generated unit.

The reviewer here is a scripted test double: these tests prove that the gate
enforces coverage, schema and verdicts, not that any model judges correctly.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth.review import (
    SynthesisReviewError,
    build_review_receipt,
    collect_review_units,
    review_window,
    verify_review_receipt,
)
from obsidian_llm_wiki.synth.validation import SynthesisValidationError

SOURCE = (
    "Acme acquired Beta in 2021. Acme did not acquire Gamma. "
    "Revenue grew 12% after the merger. 研究团队认为该结论仍需验证。"
)


def _config(**llm):
    return Config(llm=LLMProviderConfig(model="gen-model", **llm), retry_count=1)


def _synthesis(catalog, *, claim_text="Acme acquired Beta in 2021.", claims=True):
    spans = catalog.spans
    concept_claims = (
        [Claim(text=claim_text, concept_slug="acme", evidence=catalog.cite([spans[0].id]))]
        if claims else []
    )
    return SourceSynthesis(
        source_title="Deal", source_file="deal.md",
        source_summary="Acme acquired Beta in 2021; revenue then grew 12%.",
        key_points=["Revenue grew 12% after the merger."],
        open_questions=["Why did Acme not acquire Gamma?"],
        concepts=[
            ConceptNote(
                title="Acme", slug="acme", summary="Acme is the acquirer of Beta.",
                sections=[BodySection(heading="Deal", points=["Acme acquired Beta."],
                                      prose="Acme did not acquire Gamma.")],
                claims=concept_claims,
                related=[ConceptLink(slug="beta", relation="competes_with")],
            ),
            ConceptNote(title="Beta", slug="beta", summary="Beta was acquired in 2021.",
                        sections=[BodySection(heading="Fate", points=["Acquired by Acme."])]),
        ],
        maps=[MapOfContent(title="M&A", slug="m-and-a", summary="Acme's 2021 Beta deal.",
                           concept_slugs=["acme", "beta"])],
    )


class ScriptedReviewer:
    """Answers each review request; ``verdicts`` maps a text fragment to a verdict."""

    def __init__(self, verdicts=None, mutate=None):
        self.verdicts = verdicts or {}
        self.mutate = mutate
        self.requests = []

    async def __call__(self, system, messages, config, **kwargs):
        self.requests.append({"system": system, "messages": messages, **kwargs})
        schema = kwargs["format"]
        units = json.loads(messages[-1]["content"].split("--- UNITS ---", 1)[1])
        window_ids = re.findall(r"⟦([0-9a-f]{32}-\d+)⟧", messages[-1]["content"])
        reviews = {}
        for unit in units:
            verdict = "supported"
            for fragment, value in self.verdicts.items():
                if fragment in unit["text"]:
                    verdict = value
            support = unit.get("cited_span_ids") or window_ids[:1]
            reviews[unit["unit_id"]] = {
                "text_sha256": unit["text_sha256"], "verdict": verdict,
                "support_span_ids": support if verdict == "supported" else [],
                "rationale": "scripted",
                "model_context_uses": [],
            }
        assert set(schema["properties"]["reviews"]["required"]) == set(reviews)
        payload = {"reviews": reviews}
        if self.mutate is not None:
            return self.mutate(payload)
        return json.dumps(payload)


def _review(synthesis, catalog, reviewer, config=None, batch=None):
    config = config or _config()
    if batch is not None:
        config.review_batch_units = batch
    window = catalog.windows(10_000)[0]
    return asyncio.run(review_window(synthesis, catalog, window, config, call=reviewer))


def test_units_cover_every_generated_statement_kind():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    synthesis.source_tags = ["mergers"]
    synthesis.concepts[0].aliases = ["Acme Corp"]
    synthesis.concepts[0].tags = ["acquirer"]
    synthesis.maps[0].tags = ["deals"]
    kinds = {unit.kind for unit in collect_review_units(synthesis)}
    # Labels (titles, aliases, tags, headings, MoC titles, MoC tags and the
    # member labels a MoC publishes) carry assertions too.
    assert kinds == {
        "source_summary", "source_tag", "key_point", "open_question", "concept_title",
        "alias", "tag", "concept_summary", "section_heading", "body_point", "body_prose",
        "claim", "relation", "map_title", "map_summary", "map_tag", "map_member",
    }


def test_unit_keys_bind_exact_text_and_claim_citations():
    """review-unit-identity-v2: whitespace is content (re-indented code is a new unit).

    Supersedes the v1 contract that normalized whitespace (review finding L2).
    """
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    before = {u.kind: u.key for u in collect_review_units(synthesis)}
    synthesis.key_points = ["Revenue  grew 12%\nafter the merger."]
    synthesis.concepts[0].claims[0].evidence = catalog.cite([catalog.spans[1].id])
    after = {u.kind: u.key for u in collect_review_units(synthesis)}
    assert before["key_point"] != after["key_point"]
    assert before["claim"] != after["claim"]
    synthesis = _synthesis(catalog)
    synthesis.concepts[0].sections[0].prose = "```python\nif ok:\n    migrate()\ncleanup()\n```"
    prose_before = next(u.key for u in collect_review_units(synthesis) if u.kind == "body_prose")
    synthesis.concepts[0].sections[0].prose = "```python\nif ok:\n    migrate()\n    cleanup()\n```"
    prose_after = next(u.key for u in collect_review_units(synthesis) if u.kind == "body_prose")
    assert prose_before != prose_after


def test_review_request_is_strict_closed_and_lists_exact_units():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    reviewer = ScriptedReviewer()
    _review(synthesis, catalog, reviewer, config=_config(review_model="judge-model"))
    request = reviewer.requests[0]
    assert request["task"] == "review"
    schema = request["format"]
    reviews = schema["properties"]["reviews"]
    assert schema["additionalProperties"] is False and schema["required"] == ["reviews"]
    assert reviews["additionalProperties"] is False
    assert set(reviews["required"]) == set(reviews["properties"])
    units = collect_review_units(synthesis)
    listed = [
        unit_id for r in reviewer.requests
        for unit_id in r["format"]["properties"]["reviews"]["required"]
    ]
    assert len(listed) == len({u.key for u in units})  # every unit, across batches
    one = next(iter(reviews["properties"].values()))
    assert one["additionalProperties"] is False
    assert set(one["required"]) == {
        "text_sha256", "verdict", "support_span_ids", "rationale", "model_context_uses",
    }
    assert one["properties"]["verdict"]["enum"] == ["supported", "contradicted", "insufficient"]
    assert len(one["properties"]["text_sha256"]["enum"]) == 1
    prompt = request["messages"][-1]["content"]
    assert "Acme acquired Beta in 2021." in prompt  # claim's cited text is shown
    assert request["options"]["temperature"] == 0


def test_all_supported_yields_receipt_that_verifies():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    config = _config()
    outcomes = _review(synthesis, catalog, ScriptedReviewer(), config)
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    assert synthesis.review_receipt["status"] == "approved"
    assert verify_review_receipt(synthesis, SOURCE, "deal.md", config) == ""
    claim_unit = next(u for u in collect_review_units(synthesis) if u.kind == "claim")
    assert synthesis.review_receipt["units"][claim_unit.key]["span_ids"] == [catalog.spans[0].id]


@pytest.mark.parametrize("claim_text, verdict", [
    ("Gamma acquired Beta in 2021.", "contradicted"),     # wrong actor
    ("Acme acquired Beta in 2019.", "contradicted"),      # wrong number
    ("Acme acquired Gamma.", "contradicted"),             # lost negation
    ("Acme acquired Beta for $3B.", "insufficient"),      # unsupported detail
])
def test_non_supported_claim_verdicts_reject_the_synthesis(claim_text, verdict):
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog, claim_text=claim_text)
    reviewer = ScriptedReviewer({claim_text: verdict})
    with pytest.raises(SynthesisReviewError) as caught:
        _review(synthesis, catalog, reviewer)
    assert verdict in str(caught.value) and claim_text in str(caught.value)
    assert caught.value.diagnostics[0]["verdict"] == verdict
    assert issubclass(SynthesisReviewError, SynthesisValidationError)


def test_zero_claim_synthesis_with_unsupported_summary_is_rejected():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog, claims=False)
    synthesis.concepts = []
    synthesis.maps = []
    synthesis.source_summary = "Acme was founded by Beta's CEO."
    with pytest.raises(SynthesisReviewError, match="source_summary"):
        _review(synthesis, catalog, ScriptedReviewer({"founded": "insufficient"}))


def test_false_premise_question_and_typed_relation_are_reviewed():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    with pytest.raises(SynthesisReviewError, match="open_question"):
        _review(synthesis, catalog, ScriptedReviewer({"Why did Acme not": "contradicted"}))
    with pytest.raises(SynthesisReviewError, match="relation"):
        _review(synthesis, catalog, ScriptedReviewer({"competes_with": "insufficient"}))


def _drop_first(payload):
    payload["reviews"].pop(next(iter(payload["reviews"])))
    return json.dumps(payload)


def _add_unknown(payload):
    payload["reviews"]["u999"] = next(iter(payload["reviews"].values()))
    return json.dumps(payload)


def _duplicate_key(payload):
    first_key, first = next(iter(payload["reviews"].items()))
    body = json.dumps(payload)
    return body[:-2] + f', "{first_key}": ' + json.dumps(first) + "}}"


def _wrong_digest(payload):
    next(iter(payload["reviews"].values()))["text_sha256"] = "0" * 16
    return json.dumps(payload)


def _no_support_ids(payload):
    for review in payload["reviews"].values():
        review["support_span_ids"] = []
    return json.dumps(payload)


def _foreign_support_id(payload):
    for review in payload["reviews"].values():
        review["support_span_ids"] = ["deadbeef-1"]
    return json.dumps(payload)


def _extra_field(payload):
    next(iter(payload["reviews"].values()))["verified"] = True
    return json.dumps(payload)


def _uncertain_verdict(payload):
    next(iter(payload["reviews"].values()))["verdict"] = "partially_supported"
    return json.dumps(payload)


@pytest.mark.parametrize("mutate, message", [
    (_drop_first, "missing"),
    (_add_unknown, "unknown"),
    (_duplicate_key, "duplicate"),
    (_wrong_digest, "digest"),
    (_no_support_ids, "support"),
    (_foreign_support_id, "support"),
    (_extra_field, "field"),
    (_uncertain_verdict, "verdict"),
    (lambda payload: "not json", "JSON"),
    (lambda payload: json.dumps({"reviews": {}, "approved": True}), "field"),
])
def test_malformed_reviewer_output_fails_closed(mutate, message):
    catalog = build_span_catalog(SOURCE, "deal.md")
    with pytest.raises(SynthesisReviewError, match=message):
        _review(_synthesis(catalog), catalog, ScriptedReviewer(mutate=mutate))


def test_claim_support_must_come_from_its_own_cited_spans():
    catalog = build_span_catalog(SOURCE, "deal.md")
    other = catalog.spans[2].id

    def mutate(payload):
        for review in payload["reviews"].values():
            review["support_span_ids"] = [other]
        return json.dumps(payload)

    with pytest.raises(SynthesisReviewError, match="cited"):
        _review(_synthesis(catalog), catalog, ScriptedReviewer(mutate=mutate))


def test_reviewer_provider_failure_propagates_without_fallback():
    catalog = build_span_catalog(SOURCE, "deal.md")

    async def broken(*args, **kwargs):
        raise ConnectionError("reviewer down")

    with pytest.raises(ConnectionError, match="reviewer down"):
        _review(_synthesis(catalog), catalog, broken)


def test_units_are_batched_without_dropping_any():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    synthesis.key_points = [f"Point {n} about the merger." for n in range(30)]
    reviewer = ScriptedReviewer()
    outcomes = _review(synthesis, catalog, reviewer, batch=8)
    expected = {u.key for u in collect_review_units(synthesis)}
    assert set(outcomes) == expected
    sizes = [len(r["format"]["properties"]["reviews"]["required"]) for r in reviewer.requests]
    assert max(sizes) <= 8 and sum(sizes) == len(expected)


def test_receipt_rejects_stale_source_other_reviewer_new_units_and_unchecked():
    catalog = build_span_catalog(SOURCE, "deal.md")
    synthesis = _synthesis(catalog)
    config = _config()
    outcomes = _review(synthesis, catalog, ScriptedReviewer(), config)
    synthesis.review_receipt = build_review_receipt(synthesis, catalog, outcomes, config)
    assert "source" in verify_review_receipt(synthesis, SOURCE + " More.", "deal.md", config)
    assert "reviewer" in verify_review_receipt(
        synthesis, SOURCE, "deal.md", _config(review_model="other-judge"),
    )
    synthesis.key_points.append("An unreviewed statement.")
    assert "unreviewed" in verify_review_receipt(synthesis, SOURCE, "deal.md", config)
    synthesis.key_points.pop()
    synthesis.review_receipt = {**synthesis.review_receipt, "status": "unchecked"}
    assert "approved" in verify_review_receipt(synthesis, SOURCE, "deal.md", config)
    synthesis.review_receipt = None
    assert "receipt" in verify_review_receipt(synthesis, SOURCE, "deal.md", config)
