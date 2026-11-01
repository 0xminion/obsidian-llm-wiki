"""Explicit provider test doubles for the grounded producer and review gate.

These doubles answer the *real* request shapes (catalog prompts, closed review
schemas). An approving reviewer is a scripted test double: tests using it check
plumbing (cache, render, registry), never semantic support.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

SPAN_RE = re.compile(r"⟦([0-9a-f]{32}-\d+)⟧ ")


def offered_spans(prompt: str) -> list[tuple[str, str]]:
    """Return ``(span_id, rendered_text)`` pairs from a catalog prompt."""
    marks = list(SPAN_RE.finditer(prompt))
    spans = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(prompt)
        spans.append((mark.group(1), prompt[mark.end():end]))
    return spans


def cite(prompt: str, phrase: str) -> list[str]:
    """IDs of the single offered span containing ``phrase`` (fails loudly)."""
    found = [span_id for span_id, text in offered_spans(prompt) if phrase in text]
    if len(found) != 1:
        raise AssertionError(f"phrase {phrase!r} matched {len(found)} offered spans")
    return found


def span_range(ids: Any) -> dict[str, str] | None:
    """Express a fixture's ID selection in citation interface v2, never widening it.

    Only a non-empty list of strictly consecutive IDs of one catalog becomes
    ``{"start_id", "end_id"}``. Anything else (empty, gapped, duplicated,
    malformed) has no faithful range and returns ``None``: the caller then sends
    the legacy list, which the producer contract rejects.
    """
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        return None
    parsed = []
    for span_id in ids:
        tag, _, number = span_id.rpartition("-")
        if not tag or not number.isdigit():
            return None
        parsed.append((tag, int(number)))
    if any(tag != parsed[0][0] for tag, _ in parsed):
        return None
    if any(b[1] != a[1] + 1 for a, b in zip(parsed, parsed[1:], strict=False)):
        return None
    return {"start_id": ids[0], "end_id": ids[-1]}


def wire_claim(claim: dict[str, Any]) -> dict[str, Any]:
    """Producer-wire form of a fixture claim (``evidence_span_ids`` -> range)."""
    out = {"text": claim["text"]}
    if "evidence_span_range" in claim:
        out["evidence_span_range"] = claim["evidence_span_range"]
    else:
        selected = span_range(claim["evidence_span_ids"])
        if selected is None:
            out["evidence_span_ids"] = claim["evidence_span_ids"]
        else:
            out["evidence_span_range"] = selected
    out["source_ref"] = claim.get("source_ref", "")
    return out


def grounded_response(
    *,
    summary: str,
    concepts: list[dict[str, Any]] | None = None,
    key_points: list[str] | None = None,
    open_questions: list[str] | None = None,
    maps: list[dict[str, Any]] | None = None,
    title: str = "Source",
    language: str = "en",
) -> str:
    """A complete closed-schema producer response."""
    full_concepts = []
    for concept in concepts or []:
        full_concepts.append({
            "title": concept["title"], "slug": concept["slug"], "canonical_id": "",
            "summary": concept["summary"], "tags": concept.get("tags", []),
            "aliases": concept.get("aliases", []),
            "sections": [
                {"heading": s["heading"], "points": s.get("points", []),
                 "prose": s.get("prose", "")}
                for s in concept["sections"]
            ],
            "claims": [wire_claim(c) for c in concept.get("claims", [])],
            "related": [
                {"slug": r["slug"], "relation": r.get("relation", "related_to"),
                 "display": r.get("display", "")}
                for r in concept.get("related", [])
            ],
            "confidence": concept.get("confidence", 0.9),
            "provenance": "extracted", "is_new": True,
        })
    return json.dumps({
        "source_title": title, "source_summary": summary, "source_tags": [],
        "key_points": key_points or [], "open_questions": open_questions or [],
        "language": language, "concepts": full_concepts,
        "maps": [
            {"title": m["title"], "slug": m["slug"], "summary": m["summary"],
             "tags": [], "concept_slugs": m["concept_slugs"]}
            for m in maps or []
        ],
    }, ensure_ascii=False)


def review_answer(prompt: str, schema: dict[str, Any], verdicts: dict[str, str]) -> dict:
    """Answer a review request; ``verdicts`` maps text fragments to verdicts."""
    units = json.loads(prompt.split("--- UNITS ---", 1)[1])
    window_ids = [span_id for span_id, _text in offered_spans(prompt)]
    reviews = {}
    for unit in units:
        verdict = "supported"
        for fragment, value in verdicts.items():
            if fragment in unit["text"]:
                verdict = value
        support = unit.get("cited_span_ids") or window_ids[:1]
        reviews[unit["unit_id"]] = {
            "text_sha256": unit["text_sha256"], "verdict": verdict,
            "support_span_ids": support if verdict == "supported" else [],
            "rationale": "scripted test double",
            "model_context_uses": [],  # no claim of semantic context use by this double
        }
    assert set(schema["properties"]["reviews"]["required"]) == set(reviews)
    return {"reviews": reviews}


def identity_answer(
    prompt: str, schema: dict[str, Any], verdicts: Any = None,
) -> dict:
    """Answer an identity-assessment request (``task="identity"``).

    ``verdicts`` maps ``(subject_title, candidate_title)`` to a verdict, or is a
    callable ``(item) -> verdict``. The default is ``insufficient``: a scripted
    double never asserts that two subjects are the same entity unless a test
    says so explicitly.
    """
    listed = json.loads(prompt.split("--- IDENTITY ASSESSMENTS ---", 1)[1])
    answers = {}
    for item in listed:
        if callable(verdicts):
            verdict = verdicts(item)
        else:
            verdict = (verdicts or {}).get(
                (item["subject"]["title"], item["candidate"]["title"]), "insufficient",
            )
        answers[item["assessment_id"]] = {
            "candidate_id": item["candidate"]["candidate_id"],
            "candidate_sha256": item["candidate"]["candidate_sha256"],
            "subject_sha256": item["subject"]["subject_sha256"],
            "verdict": verdict,
            "support_span_ids": (
                [context["spans"][0]["support_id"] for context in
                 [item["subject_context"], *item["candidate_context"]["owners"]]]
                if verdict == "same" else []
            ),
            "rationale": "scripted identity double",
        }
    assert set(schema["properties"]["assessments"]["required"]) == set(answers)
    return {"assessments": answers}


def ground_legacy_response(response: str, prompt: str) -> str:
    """Convert a pre-grounding fixture's claims to the span-ID contract.

    A legacy claim's ``quote`` is mapped to the ID of the unique offered span
    that contains it. Legacy claims without a quote cannot be cited honestly,
    so this fails loudly and the fixture itself must be repaired.
    """
    if not isinstance(response, str):
        return response
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(data, dict):
        return response
    for concept in data.get("concepts", []) if isinstance(data.get("concepts"), list) else []:
        if not isinstance(concept, dict) or not isinstance(concept.get("claims"), list):
            continue
        for claim in concept["claims"]:
            if not isinstance(claim, dict) or "evidence_span_range" in claim:
                continue
            if "evidence_span_ids" in claim:
                legacy = claim.pop("evidence_span_ids")
                selected = span_range(legacy)
                if selected is None:
                    claim["evidence_span_ids"] = legacy
                else:
                    claim["evidence_span_range"] = selected
                continue
            quote = claim.pop("quote", None)
            if not isinstance(quote, str) or not quote:
                raise AssertionError(
                    f"legacy fixture claim {claim.get('text')!r} has no quote; "
                    "repair the fixture with source-supported evidence"
                )
            only = cite(prompt, quote)[0]
            claim["evidence_span_range"] = {"start_id": only, "end_id": only}
    return json.dumps(data, ensure_ascii=False)


def legacy_fixture_llm(
    base: Any, *, verdicts: dict[str, str] | None = None, identity_verdicts: Any = None,
):
    """Explicitly adapt an old mocked provider: approving review double + span IDs.

    ``base`` may be a response string, a sync ``prompt -> str`` callable, or an
    async provider ``(system, messages, config, **kwargs)``. Review requests are
    answered by the scripted approving double (plumbing tests only).
    """
    import inspect

    provider_calls: list[dict[str, Any]] = []

    async def call(system, messages, config, **kwargs):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        provider_calls.append({"task": kwargs.get("task"), "prompt": prompt})
        if kwargs.get("task") == "review":
            return json.dumps(review_answer(prompt, kwargs["format"], verdicts or {}))
        if kwargs.get("task") == "identity":
            return json.dumps(
                identity_answer(prompt, kwargs["format"], identity_verdicts), ensure_ascii=False,
            )
        if isinstance(base, str):
            response = base
        elif inspect.iscoroutinefunction(base) or (
            callable(base) and inspect.iscoroutinefunction(type(base).__call__)
        ):
            response = await base(system, messages, config, **kwargs)
        else:
            response = base(system, messages, config, **kwargs)
            if inspect.isawaitable(response):
                response = await response
        return ground_legacy_response(response, prompt)

    call.calls = provider_calls  # type: ignore[attr-defined]
    call.producer_calls = lambda: [  # type: ignore[attr-defined]
        c for c in provider_calls if c["task"] not in ("review", "identity")
    ]
    return call


def producer_call_count(mock) -> int:
    """Count producer (non-review) requests recorded by an AsyncMock provider."""
    return sum(
        1 for call in mock.call_args_list
        if call.kwargs.get("task") not in ("review", "identity")
    )


def producer_prompts(mock) -> list[str]:
    """Producer prompts recorded by an AsyncMock provider, in call order."""
    return [
        call.args[0] for call in mock.call_args_list
        if call.kwargs.get("task") not in ("review", "identity")
    ]


def reviewed(synthesis, source, filename: str, config):
    """Attach a scripted approval receipt to a test double's synthesis."""
    synthesis.review_receipt = approved_receipt(synthesis, source.content, filename, config)
    return synthesis


def approved_receipt(synthesis, content: str, source_file: str, config) -> dict:
    """Receipt for a test double that stands in for the reviewed retry boundary."""
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog
    from obsidian_llm_wiki.synth.review import (
        build_review_receipt,
        collect_review_units,
        context_binding_for_unit,
    )

    catalog = build_span_catalog(content, source_file)
    first = catalog.spans[0].id if catalog.spans else ""
    outcomes = {
        unit.key: {
            "kind": unit.kind, "verdict": "supported",
            "span_ids": list(unit.cited_span_ids) or [first],
            "context_binding": context_binding_for_unit(
                unit, catalog, catalog.windows(max(1, len(content)))[0],
            ),
            "model_context_uses": [],
        }
        for unit in collect_review_units(synthesis)
    }
    return build_review_receipt(synthesis, catalog, outcomes, config)


class GroundedProvider:
    """Dispatch producer vs review requests by the production ``task`` field."""

    def __init__(
        self,
        producer: Callable[[str], str],
        *,
        verdicts: dict[str, str] | None = None,
        review_hook: Callable[[dict], str] | None = None,
        identity_verdicts: Any = None,
    ) -> None:
        self.producer = producer
        self.verdicts = verdicts or {}
        self.review_hook = review_hook
        self.identity_verdicts = identity_verdicts
        self.calls: list[dict[str, Any]] = []

    @property
    def producer_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["task"] not in ("review", "identity")]

    @property
    def identity_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["task"] == "identity"]

    @property
    def review_calls(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["task"] == "review"]

    async def __call__(self, system, messages, config, **kwargs):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        call = {"task": kwargs.get("task"), "prompt": prompt, "kwargs": kwargs}
        self.calls.append(call)
        if kwargs.get("task") == "review":
            answer = review_answer(prompt, kwargs["format"], self.verdicts)
            if self.review_hook is not None:
                return self.review_hook(answer)
            return json.dumps(answer, ensure_ascii=False)
        if kwargs.get("task") == "identity":
            return json.dumps(
                identity_answer(prompt, kwargs["format"], self.identity_verdicts),
                ensure_ascii=False,
            )
        return self.producer(prompt)
