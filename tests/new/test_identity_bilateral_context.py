"""Bilateral scope validation tests. Proof from another side/pair never binds."""

import json

import pytest

from obsidian_llm_wiki.synth.identity import _validate
from obsidian_llm_wiki.synth.review import SynthesisReviewError


def request():
    return {
        "candidate_id": "entity-a",
        "candidate_sha256": "ca",
        "subject_sha256": "su",
        "subject_context": {"complete": True, "spans": [{"support_id": "subject:a"}]},
        "candidate_context": {
            "complete": True,
            "owners": [{"complete": True, "spans": [{"support_id": "candidate:a"}]}],
        },
    }


@pytest.mark.parametrize(
    "support", [["subject:a"], ["candidate:a"], ["subject:a", "candidate:other"]]
)
def test_same_requires_both_sides_of_this_pair(support):
    req = request()
    answer = {k: req[k] for k in ("candidate_id", "candidate_sha256", "subject_sha256")}
    answer.update(verdict="same", support_span_ids=support, rationale="fixture")
    with pytest.raises(SynthesisReviewError):
        _validate(
            json.dumps({"assessments": {"a1": answer}}),
            [("a1", req)],
            {"subject:a", "candidate:a", "candidate:other"},
        )


def test_same_accepts_complete_bilateral_support():
    req = request()
    answer = {k: req[k] for k in ("candidate_id", "candidate_sha256", "subject_sha256")}
    answer.update(
        verdict="same", support_span_ids=["subject:a", "candidate:a"], rationale="fixture"
    )
    result = _validate(
        json.dumps({"assessments": {"a1": answer}}), [("a1", req)], {"subject:a", "candidate:a"}
    )
    assert result["a1"]["verdict"] == "same"


def test_same_requires_every_candidate_owner():
    req = request()
    req["candidate_context"]["owners"].append(
        {"complete": True, "spans": [{"support_id": "candidate:b"}]}
    )
    answer = {k: req[k] for k in ("candidate_id", "candidate_sha256", "subject_sha256")}
    answer.update(
        verdict="same", support_span_ids=["subject:a", "candidate:a"], rationale="fixture"
    )
    with pytest.raises(SynthesisReviewError):
        _validate(
            json.dumps({"assessments": {"a1": answer}}),
            [("a1", req)],
            {"subject:a", "candidate:a", "candidate:b"},
        )


def test_large_source_context_exposes_incompleteness():
    from obsidian_llm_wiki.core.identity_context import MAX_CONTEXT_SPANS, source_context
    from obsidian_llm_wiki.core.models import ConceptNote, SourceSynthesis
    from obsidian_llm_wiki.core.span_catalog import source_revision

    content = "".join(f"Context statement {i}.\n\n" for i in range(MAX_CONTEXT_SPANS + 1))
    synthesis = SourceSynthesis(
        "Fixture",
        "Context",
        source_file="fixture.md",
        concepts=[ConceptNote("Context", "context", "A context")],
        review_receipt={"status": "approved", "source_sha256": source_revision(content)},
    )
    context = source_context(synthesis, 0, content)
    assert not context["complete"] and "incomplete" in context["reason"]
    assert len(context["spans"]) <= MAX_CONTEXT_SPANS


def test_same_rejects_incomplete_context_even_with_two_sided_support():
    req = request()
    req["candidate_context"]["complete"] = False
    answer = {k: req[k] for k in ("candidate_id", "candidate_sha256", "subject_sha256")}
    answer.update(
        verdict="same", support_span_ids=["subject:a", "candidate:a"], rationale="fixture"
    )
    with pytest.raises(SynthesisReviewError):
        _validate(
            json.dumps({"assessments": {"a1": answer}}), [("a1", req)], {"subject:a", "candidate:a"}
        )
