"""Catalog anchors and review receipts survive serialization and cache reloads."""

from __future__ import annotations

import json

from obsidian_llm_wiki.core.cache import load_synthesis, save_synthesis
from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptNote,
    EvidenceVerification,
    SourceSynthesis,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.span_catalog import CATALOG_VERSION, build_span_catalog

CONTENT = "Alpha rose 5%. Beta fell. Alpha rose 5%."


def _synthesis() -> SourceSynthesis:
    catalog = build_span_catalog(CONTENT, "a.md")
    evidence = catalog.cite([catalog.spans[2].id])
    return SourceSynthesis(
        source_title="A", source_summary="Alpha rose.", source_file="a.md",
        concepts=[ConceptNote(
            title="Alpha", slug="alpha", summary="Alpha rose 5%.",
            sections=[BodySection(heading="H", prose="Alpha rose.")],
            claims=[Claim(text="Alpha rose 5%.", concept_slug="alpha", evidence=evidence)],
        )],
        review_receipt={"version": "grounded-review-v1", "status": "approved", "units": {}},
    )


def test_span_anchor_and_receipt_round_trip_through_dict():
    restored = source_synthesis_from_dict(
        json.loads(json.dumps(source_synthesis_to_dict(_synthesis())))
    )
    evidence = restored.concepts[0].claims[0].evidence
    assert evidence.span_ids == (build_span_catalog(CONTENT, "a.md").spans[2].id,)
    assert evidence.catalog_version == CATALOG_VERSION
    assert (evidence.start_offset, evidence.end_offset) == (26, 40)
    assert evidence.verification is EvidenceVerification.VERIFIED
    assert restored.review_receipt == {
        "version": "grounded-review-v1", "status": "approved", "units": {},
    }


def test_span_anchor_and_receipt_round_trip_through_portable_cache(tmp_path):
    save_synthesis(_synthesis(), tmp_path, "a.md")
    restored = load_synthesis(tmp_path, "a.md")
    assert restored is not None
    assert restored.concepts[0].claims[0].evidence.span_ids
    assert restored.review_receipt["status"] == "approved"


def test_legacy_quote_only_cache_remains_readable_without_receipt():
    legacy = {
        "source_title": "A", "source_summary": "Alpha rose.",
        "concepts": [{
            "title": "Alpha", "slug": "alpha", "summary": "Alpha rose.",
            "sections": [{"heading": "H", "points": ["Alpha rose."]}],
            "claims": [{"text": "Alpha rose.", "evidence": {
                "quote": "Alpha rose 5%.", "verification": "ambiguous",
            }}],
        }],
    }
    restored = source_synthesis_from_dict(legacy)
    evidence = restored.concepts[0].claims[0].evidence
    assert evidence.quote == "Alpha rose 5%."
    assert evidence.span_ids == ()
    assert restored.review_receipt is None


def test_malformed_receipt_is_not_coerced_into_a_receipt():
    data = source_synthesis_to_dict(_synthesis())
    data["review_receipt"] = "approved"
    assert source_synthesis_from_dict(data).review_receipt is None
