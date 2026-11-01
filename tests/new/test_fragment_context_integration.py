"""Parent-owned integration boundary; synthetic source, no reference answers."""

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import Claim, ConceptNote, SourceSynthesis
from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth import review


def case():
    text = (
        "The lab reports:\r\n\r\n> Trial A uses a sealed chamber.\r\n\r\n"
        "Unrelated discussion of scheduling.\r\n\r\nThe same report continues:\r\n\r\n"
        "> Trial B uses a cooled chamber."
    )
    catalog = build_span_catalog(text, "trials.md")
    a = next(s for s in catalog.spans if "Trial A uses" in s.text)
    b = next(s for s in catalog.spans if "Trial B uses" in s.text)
    evidence = catalog.cite_fragments([[a.id], [b.id]])
    synth = SourceSynthesis(
        source_title="Synthetic laboratory report",
        source_summary="The report describes two trials.",
        concepts=[
            ConceptNote(
                title="Trials",
                slug="trials",
                summary="The report describes chambers.",
                claims=[
                    Claim(
                        "Trial A uses a sealed chamber and Trial B uses a cooled chamber.",
                        evidence=evidence,
                    )
                ],
            )
        ],
    )
    unit = next(u for u in review.collect_review_units(synth) if u.kind == "claim")
    window = catalog.windows(len(text))[0]
    return catalog, unit, window


def test_current_multifragment_review_uses_pure_v4_context_not_transitional_envelope():
    catalog, unit, window = case()
    binding = review.context_binding_for_unit(unit, catalog, window)
    assert binding["policy"] == "review-context-v4"
    expected = review.review_context.build_fragment_context_binding(
        content=catalog.content,
        source_file=catalog.source_file,
        source_sha256=catalog.revision,
        catalog_version=catalog.version,
        unit_key=unit.key,
        cited_fragments=[
            {
                "span_ids": f["span_ids"],
                "start_offset": f["start_offset"],
                "end_offset": f["end_offset"],
            }
            for f in unit.cited_fragments
        ],
        visible_intervals=[[0, len(catalog.content)]],
    )
    assert binding == expected


def test_current_single_fragment_uses_active_v4_fragment_context():
    text = "The laboratory report describes a local test.\r\n\r\n> A sealed chamber was used."
    catalog = build_span_catalog(text, "single.md")
    selected = next(s for s in catalog.spans if "sealed chamber" in s.text)
    synth = SourceSynthesis(
        source_title="Synthetic laboratory report",
        source_summary="A chamber test.",
        concepts=[
            ConceptNote(
                title="Test",
                slug="test",
                summary="A chamber test.",
                claims=[
                    Claim(
                        "The report describes use of a sealed chamber.",
                        evidence=catalog.cite([selected.id]),
                    )
                ],
            )
        ],
    )
    unit = next(u for u in review.collect_review_units(synth) if u.kind == "claim")
    binding = review.context_binding_for_unit(unit, catalog, catalog.windows(len(text))[0])
    assert binding["policy"] == "review-context-v4"
    assert len(binding["cited_fragments"]) == 1


def test_fragment_context_policy_alone_invalidates_reviewer_fingerprint(monkeypatch):
    config = Config()
    before = review.reviewer_fingerprint(config)
    monkeypatch.setattr(
        review.review_context,
        "FRAGMENT_CONTEXT_POLICY_VERSION",
        "review-context-vNEXT",
        raising=False,
    )
    assert review.reviewer_fingerprint(config) != before


def test_summary_coverage_policy_alone_invalidates_producer_fingerprint(monkeypatch):
    from obsidian_llm_wiki.core import pipeline
    from obsidian_llm_wiki.synth import prompts

    config = Config()
    before = pipeline._artifact_versions_for_config(config)
    monkeypatch.setattr(
        prompts, "SUMMARY_COVERAGE_POLICY_VERSION", "summary-coverage-NEXT", raising=False
    )
    after = pipeline._artifact_versions_for_config(config)
    assert after.prompt != before.prompt
    assert after.schema == before.schema


def test_known_v3_integrity_survives_active_policy_selector_change(monkeypatch):
    from copy import deepcopy

    from tests.new.test_evidence_fragments import approved

    # Build a genuinely recorded v3 receipt, not a default-v4 receipt mislabeled
    # as historical. Current v4 and an unknown future selector both reject reuse.
    with monkeypatch.context() as historical:
        historical.setattr(
            review.review_context, "FRAGMENT_CONTEXT_POLICY_VERSION", "review-context-v3"
        )
        catalog, synthesis, config, _ = approved()
    assert review.review_context.FRAGMENT_CONTEXT_POLICY_VERSION == "review-context-v4"
    assert synthesis.review_receipt["context_policy"] == "review-context-v3"
    before = deepcopy(synthesis.review_receipt)
    assert review.verify_review_receipt_integrity(
        synthesis, catalog.content, catalog.source_file
    ) == ""
    assert review.verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)
    monkeypatch.setattr(
        review.review_context, "FRAGMENT_CONTEXT_POLICY_VERSION", "review-context-vNEXT"
    )
    assert (
        review.verify_review_receipt_integrity(synthesis, catalog.content, catalog.source_file)
        == ""
    )
    assert review.verify_review_receipt(synthesis, catalog.content, catalog.source_file, config)
    assert synthesis.review_receipt == before
