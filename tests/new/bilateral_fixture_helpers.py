"""Bilateral identity proof for registry-level fixtures, built through production code.

Under the approved contract, a cross-source ``same`` binding needs a record
that carries complete, revision-scoped source context from BOTH sides and
support tokens drawn from the subject's reviewed source AND every candidate
owner's reviewed source. A fixture cannot type that proof by hand: the
tokens embed catalog IDs of the exact source bytes the registry retained.

These helpers therefore run the real pipeline path for a direct API caller:

* ``reviewed`` attaches a real approved review receipt (``source_sha256`` of
  the exact ``SourceDoc`` content; every unit supported);
* ``assess`` registers the incoming source bytes the way the pipeline does
  (``registry._identity_source_contents``), asks
  ``VaultRegistry.identity_requests`` for the pending proposals and answers
  them with ``synth.identity.assess_identity`` driven by a *scripted*
  assessor whose verdict is keyed to explicit fixture facts present in the
  offered spans — never to names, slugs, summaries or word overlap.

Nothing here approves anything by itself. A fixture must declare the fact
that makes two sources the same referent; if that fact is not in the offered
source spans on both sides the verdict is ``insufficient``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.span_catalog import build_span_catalog, source_revision
from obsidian_llm_wiki.synth.identity import (
    assess_identity,
    assessor_fingerprint,
    store_assessments,
)
from tests.new.grounding_doubles import approved_receipt, identity_answer

__all__ = [
    "FIXTURE_CONFIG",
    "assess",
    "assess_same",
    "fact_verdict",
    "plan_pending",
    "reviewed",
    "upsert_reviewed",
]

# Local-only configuration: no provider is ever contacted; the assessor is the
# scripted ``identity_answer`` double below.
FIXTURE_CONFIG = Config(vault_path="", grounding_review="required", embeddings_enabled=False)


def _text(context: dict[str, Any]) -> str:
    return " ".join(span["text"] for span in context.get("spans", []))


def fact_verdict(
    *,
    same: Iterable[str] = (),
    different: Iterable[tuple[str, str]] = (),
    pairs: Iterable[tuple[str, str]] = (),
):
    """Build an assessor keyed to explicit source facts.

    ``same``: every listed fact must literally appear in the subject's offered
    spans AND in every candidate owner's offered spans, and both contexts must
    be complete. ``different``: ``(subject_marker, candidate_marker)`` pairs;
    when the subject spans contain the first marker and every owner's spans
    contain the second, the referents are declared different. Everything else
    is ``insufficient``. ``pairs`` (optional) restricts ``same`` further to
    the declared ``(subject_definition_marker, candidate_definition_marker)``
    pairs: when two concepts share the same two sources, the source fact alone
    cannot discriminate them, so the fixture must say which definition pairs
    it vouches for.
    """
    same_facts = tuple(same)
    different_pairs = tuple(different)
    definition_pairs = tuple(pairs)

    def verdict(item: dict[str, Any]) -> str:
        subject = item["subject_context"]
        candidate = item["candidate_context"]
        owners = candidate.get("owners", [])
        if not subject.get("complete") or not candidate.get("complete") or not owners:
            return "insufficient"
        subject_text = _text(subject)
        owner_texts = [_text(owner) for owner in owners]
        for mine, theirs in different_pairs:
            if mine in subject_text and all(theirs in text for text in owner_texts):
                return "different"
        if definition_pairs and not any(
            mine in item["subject"]["definition"] and theirs in item["candidate"]["definition"]
            for mine, theirs in definition_pairs
        ):
            return "insufficient"
        if same_facts and all(
            fact in subject_text and all(fact in text for text in owner_texts)
            for fact in same_facts
        ):
            return "same"
        return "insufficient"

    return verdict


def reviewed(synthesis: SourceSynthesis, source: SourceDoc) -> SourceSynthesis:
    """Construct registry fixture evidence, then attach its scripted approval.

    These registry tests deliberately use arbitrary distinguishable claim payloads;
    their scripted reviewer is not an entailment oracle. Anchor previously absent
    evidence to the exact complete declared fixture source, not an arbitrary first
    span. Never rewrite supplied evidence, claim text or source bytes. Runtime
    approval of missing/malformed evidence remains forbidden.
    """
    catalog = build_span_catalog(source.content, synthesis.source_file)
    for concept in synthesis.concepts:
        for claim in concept.claims:
            if claim.evidence is None:
                claim.evidence = catalog.cite([span.id for span in catalog.spans])
    synthesis.review_receipt = approved_receipt(
        synthesis, source.content, synthesis.source_file, FIXTURE_CONFIG
    )
    return synthesis


def upsert_reviewed(registry, filename: str, source: SourceDoc, synthesis, versions):
    """``reviewed`` + ``upsert_synthesis`` for a first owner (no proposal to assess)."""
    synthesis.source_file = filename
    reviewed(synthesis, source)
    registry.upsert_synthesis(filename, source, synthesis, versions)
    return synthesis


def _register_incoming(registry, filename: str, content: str) -> None:
    contents = getattr(registry, "_identity_source_contents", None)
    if contents is None:
        contents = registry._identity_source_contents = {}
    contents[filename] = content


def plan_pending(registry, synthesis: SourceSynthesis, source: SourceDoc) -> SourceSynthesis:
    """Assign IDs to a same-run (unpersisted) owner the way the pipeline does.

    The pipeline registers each planned synthesis under
    ``registry._planned_concepts`` so that later syntheses of the same run can
    propose its pending entity and be assessed against its source context.
    """
    reviewed(synthesis, source)
    _register_incoming(registry, synthesis.source_file, source.content)
    registry.assign_canonical_ids(synthesis)
    planned = getattr(registry, "_planned_concepts", None)
    if planned is None:
        planned = registry._planned_concepts = {}
    for index, concept in enumerate(synthesis.concepts):
        planned.setdefault(concept.canonical_id, []).append((synthesis, index))
    return synthesis


def assess(registry, synthesis: SourceSynthesis, source: SourceDoc, verdict) -> list[dict]:
    """Assess every pending proposal of ``synthesis`` with the real assessor path.

    ``source`` must be the exact source the synthesis was reviewed against;
    a disagreeing receipt/content pair is a fixture error, not a verdict.
    Returns the request list (``[]`` when nothing needed assessment).
    """
    if not isinstance(synthesis.review_receipt, dict):
        reviewed(synthesis, source)
    receipt = synthesis.review_receipt
    revision = source_revision(source.content)
    assert receipt.get("source_sha256") == revision, (
        f"fixture receipt and SourceDoc content disagree for {synthesis.source_file!r}"
    )
    _register_incoming(registry, synthesis.source_file, source.content)
    registry._identity_assessor = assessor_fingerprint(FIXTURE_CONFIG)
    requests = registry.identity_requests(synthesis)
    if not requests:
        return []

    async def call(_prompt, messages, _config, **kwargs):
        prompt = messages[-1]["content"]
        return json.dumps(
            identity_answer(prompt, kwargs["format"], verdict), ensure_ascii=False
        )

    records = asyncio.run(
        assess_identity(
            synthesis,
            requests,
            build_span_catalog(source.content, synthesis.source_file),
            FIXTURE_CONFIG,
            call=call,
        )
    )
    store_assessments(synthesis, requests, records, assessor_fingerprint(FIXTURE_CONFIG))
    return requests


def assess_same(registry, synthesis: SourceSynthesis, source: SourceDoc, *facts: str) -> list[dict]:
    """Shorthand: ``assess`` with a ``same`` verdict keyed to ``facts`` (at least one)."""
    assert facts, "declare the source fact that makes both contributions the same referent"
    return assess(registry, synthesis, source, fact_verdict(same=facts))
