"""Claim-level evidence parsing, verification, persistence, and merging."""

from __future__ import annotations

import hashlib
import json

import pytest

from obsidian_llm_wiki.core.models import (
    Claim,
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
    concept_note_to_dict,
    source_synthesis_from_dict,
)
from obsidian_llm_wiki.synth.dedupe import merge_concepts


def test_rendered_claim_evidence_has_numbered_marker_quote_and_clickable_source_link():
    from obsidian_llm_wiki.render.obsidian import render_concept_page

    concept = ConceptNote(
        title="Concept",
        slug="concept",
        summary="Summary",
        claims=[
            Claim(
                text="Supported claim",
                evidence={
                    "quote": "Exact supporting quotation.",
                    "source_file": "article.md",
                    "source_hash": "hash",
                    "start_offset": 0,
                    "end_offset": 27,
                    "verification": "verified",
                },
            )
        ],
    )

    page = render_concept_page(concept, timestamp="2026-07-14T00:00:00Z")

    assert "- Supported claim [1]" in page
    assert "## Evidence" in page
    assert "Exact supporting quotation." in page
    assert "[[sources/article|article]]" in page


def test_claim_quote_parses_into_unresolved_evidence_without_breaking_legacy_claims():
    synthesis = source_synthesis_from_dict(
        {
            "source_title": "Article",
            "concepts": [
                {
                    "title": "Concept",
                    "slug": "concept",
                    "summary": "Summary",
                    "claims": [
                        {"text": "New claim", "quote": "Exact source sentence."},
                        {"text": "Legacy claim", "source_ref": "section 1"},
                    ],
                }
            ],
        }
    )

    quoted, legacy = synthesis.concepts[0].claims
    assert quoted.evidence is not None
    assert quoted.evidence.quote == "Exact source sentence."
    assert quoted.evidence.start_offset is None
    assert quoted.evidence.end_offset is None
    assert quoted.evidence.verification == "unverified"
    assert legacy.evidence is None
    assert legacy.source_ref == "section 1"


def test_claim_evidence_json_round_trip_preserves_verified_span():
    claim = Claim(
        text="A claim",
        evidence={
            "quote": "A verified quote.",
            "source_file": "article.md",
            "source_hash": "abc123",
            "start_offset": 3,
            "end_offset": 20,
            "verification": "verified",
        },
    )
    concept = ConceptNote(title="Concept", slug="concept", summary="Summary", claims=[claim])

    restored = source_synthesis_from_dict(
        {"source_title": "Article", "concepts": [concept_note_to_dict(concept)]}
    )
    evidence = restored.concepts[0].claims[0].evidence
    assert evidence is not None
    assert evidence.quote == "A verified quote."
    assert evidence.source_file == "article.md"
    assert evidence.source_hash == "abc123"
    assert (evidence.start_offset, evidence.end_offset) == (3, 20)
    assert evidence.verification == "verified"


def test_malformed_verified_offsets_are_downgraded_instead_of_trusted():
    synthesis = source_synthesis_from_dict(
        {
            "source_title": "Article",
            "concepts": [
                {
                    "title": "Concept",
                    "slug": "concept",
                    "summary": "Summary",
                    "claims": [
                        {
                            "text": "Claim",
                            "evidence": {
                                "quote": "Quoted text.",
                                "verification": "verified",
                                "start_offset": "not an integer",
                                "end_offset": 12,
                            },
                        }
                    ],
                }
            ],
        }
    )

    evidence = synthesis.concepts[0].claims[0].evidence
    assert evidence is not None
    assert evidence.verification == "unverified"
    assert evidence.start_offset is None
    assert evidence.end_offset is None


def test_quote_resolver_returns_unique_utf8_character_offsets_and_source_hash():
    from obsidian_llm_wiki.core.evidence import resolve_quote

    content = "α Start exact quote. End"
    evidence = resolve_quote("exact quote.", content, "article.md")

    assert evidence.quote == "exact quote."
    assert evidence.source_file == "article.md"
    assert evidence.source_hash == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert (evidence.start_offset, evidence.end_offset) == (8, 20)
    assert evidence.verification == "verified"


@pytest.mark.parametrize(
    ("content", "expected_verification"),
    [
        ("Repeated quote. Repeated quote.", "ambiguous"),
        ("This source does not contain it.", "unmatched"),
    ],
)
def test_quote_resolver_never_invents_offsets_for_ambiguous_or_unmatched_quotes(
    content: str, expected_verification: str
):
    from obsidian_llm_wiki.core.evidence import resolve_quote

    evidence = resolve_quote("Repeated quote.", content, "article.md")

    assert evidence.verification == expected_verification
    assert evidence.start_offset is None
    assert evidence.end_offset is None


def test_resolve_synthesis_evidence_updates_only_quoted_claims():
    from obsidian_llm_wiki.core.evidence import resolve_synthesis_evidence

    synthesis = SourceSynthesis(
        source_title="Article",
        source_summary="Summary",
        concepts=[
            ConceptNote(
                title="Concept",
                slug="concept",
                summary="Summary",
                claims=[
                    Claim(text="Supported", evidence={"quote": "Supported sentence."}),
                    Claim(text="Legacy", source_ref="section 1"),
                ],
            )
        ],
    )

    resolve_synthesis_evidence(
        synthesis,
        SourceDoc(title="Article", content="Supported sentence.", source_file="article.md"),
        "article.md",
    )

    verified, legacy = synthesis.concepts[0].claims
    assert verified.evidence is not None
    assert verified.evidence.verification == "verified"
    assert verified.evidence.source_file == "article.md"
    assert legacy.evidence is None


def test_merge_concepts_preserves_distinct_evidence_for_same_claim_text():
    first = ConceptNote(
        title="Concept",
        slug="concept",
        summary="Summary",
        claims=[
            Claim(
                text="Same claim",
                evidence={
                    "quote": "First source quote.",
                    "source_file": "first.md",
                    "source_hash": "one",
                    "start_offset": 0,
                    "end_offset": 19,
                    "verification": "verified",
                },
            )
        ],
    )
    second = ConceptNote(
        title="Concept",
        slug="concept",
        summary="Summary",
        claims=[
            Claim(
                text="Same claim",
                evidence={
                    "quote": "Second source quote.",
                    "source_file": "second.md",
                    "source_hash": "two",
                    "start_offset": 0,
                    "end_offset": 20,
                    "verification": "verified",
                },
            )
        ],
    )

    merged = merge_concepts([first, second])

    assert [claim.evidence.source_file for claim in merged[0].claims if claim.evidence] == [
        "first.md",
        "second.md",
    ]


@pytest.mark.asyncio
async def test_pipeline_re_resolves_quote_with_changed_source_hash(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from obsidian_llm_wiki.providers import llm

    calls = 0

    async def fake_acall(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "source_title": "Article",
                "concepts": [
                    {
                        "title": "Concept",
                        "slug": "concept",
                        "summary": "Summary",
                        "claims": [{"text": "Claim", "quote": "Supported sentence."}],
                    }
                ],
            }
        )

    monkeypatch.setattr(llm, "acall_llm", fake_acall)
    config = Config(vault_path=str(tmp_path), min_source_chars=1, retry_count=1)
    filename = "article.md"

    first = await run_pipeline(
        tmp_path,
        {filename: SourceDoc(title="Article", content="Supported sentence.")},
        config,
        force=True,
    )
    first_evidence = first.concepts[0].claims[0].evidence
    assert first_evidence is not None

    second_content = "Prelude. Supported sentence."
    second = await run_pipeline(
        tmp_path,
        {filename: SourceDoc(title="Article", content=second_content)},
        config,
    )
    second_evidence = second.concepts[0].claims[0].evidence
    assert calls == 2
    assert second_evidence is not None
    assert second_evidence.source_hash == hashlib.sha256(second_content.encode("utf-8")).hexdigest()
    assert second_evidence.source_hash != first_evidence.source_hash
    assert (second_evidence.start_offset, second_evidence.end_offset) == (9, 28)
