"""Acceptance is a conservative content/evidence gate, not semantic proof."""

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SourceSynthesis


@pytest.mark.parametrize(
    "summary",
    [
        "",
        " \n",
        "TODO",
        "This document provides useful information.",
        "This concept is important.",
        "本文讨论了相关概念。",
        "Lorem ipsum dolor sit amet.",
        "[Insert summary here]",
        "Summary goes here.",
        "This source discusses important topics.",
        "本文介绍了相关内容。",
        "待补充",
    ],
)
def test_completed_synthesis_rejects_blank_or_canned_summary(summary):
    from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

    source = SourceDoc(title="Trial", content="The trial found no benefit.")
    with pytest.raises(SynthesisValidationError, match="source_summary"):
        validate_synthesis(SourceSynthesis(source_title="Trial", source_summary=summary), source)


@pytest.mark.parametrize("summary", ["No benefit.", "未见收益。", "S"])
def test_concise_sources_and_zero_concepts_are_valid(summary):
    from obsidian_llm_wiki.synth.validation import validate_synthesis

    source = SourceDoc(title="Trial", content=summary)
    synthesis = SourceSynthesis(source_title="Trial", source_summary=summary)
    assert validate_synthesis(synthesis, source) is synthesis
    assert synthesis.concepts == []


def _complete_synthesis():
    from obsidian_llm_wiki.core.models import BodySection, ConceptNote

    return SourceSynthesis(
        source_title="Trial",
        source_summary="No benefit was observed.",
        concepts=[
            ConceptNote(
                title="Trial result",
                slug="trial-result",
                summary="No observed benefit.",
                sections=[BodySection(heading="Result", prose="No benefit in this trial.")],
            )
        ],
    )


@pytest.mark.parametrize(
    "location", ["concept_summary", "empty_body", "point", "prose", "key_point", "map"]
)
def test_placeholders_cannot_hide_in_completed_content(location):
    from obsidian_llm_wiki.core.models import MapOfContent
    from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

    synthesis = _complete_synthesis()
    concept = synthesis.concepts[0]
    if location == "concept_summary":
        concept.summary = "TODO"
    elif location == "empty_body":
        concept.sections = []
    elif location == "point":
        concept.sections[0].points = ["Insert content here"]
    elif location == "prose":
        concept.sections[0].prose = "待补充"
    elif location == "key_point":
        synthesis.key_points = ["Summary goes here"]
    else:
        synthesis.maps = [MapOfContent(title="Results", slug="results", summary="TODO")]
    with pytest.raises(SynthesisValidationError):
        validate_synthesis(synthesis, SourceDoc(title="Trial", content="No benefit in this trial."))


@pytest.mark.parametrize("verification", ["unmatched", "unverified", "verified"])
def test_unmatched_quote_never_becomes_confident_completed_claim(verification):
    from obsidian_llm_wiki.core.models import Claim, EvidenceSpan
    from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

    synthesis = _complete_synthesis()
    synthesis.concepts[0].claims = [
        Claim(
            text="The trial proved a benefit.",
            evidence=EvidenceSpan(
                quote="The trial proved a benefit.",
                verification=verification,
                start_offset=0,
                end_offset=27,
            ),
        )
    ]
    with pytest.raises(SynthesisValidationError, match="unmatched quote"):
        validate_synthesis(synthesis, SourceDoc(title="Trial", content="No benefit in this trial."))


def test_supplied_quote_is_checked_against_source_not_model_verification_flag():
    from obsidian_llm_wiki.core.models import Claim, EvidenceSpan
    from obsidian_llm_wiki.synth.validation import validate_synthesis

    synthesis = _complete_synthesis()
    synthesis.concepts[0].claims = [
        Claim(
            text="No benefit was observed.",
            evidence=EvidenceSpan(quote="No benefit in this trial."),
        )
    ]
    validate_synthesis(
        synthesis, SourceDoc(title="Trial", content="No benefit in this trial."), "trial.md"
    )
    evidence = synthesis.concepts[0].claims[0].evidence
    assert evidence.verification == "verified"
    assert evidence.source_file == "trial.md"
    assert evidence.start_offset == 0


def test_quote_occurrence_is_not_a_claim_of_semantic_entailment():
    from obsidian_llm_wiki.core.models import Claim, EvidenceSpan
    from obsidian_llm_wiki.synth.validation import validate_synthesis

    synthesis = _complete_synthesis()
    synthesis.concepts[0].claims = [
        Claim(
            text="A conclusion not checked by string matching.",
            evidence=EvidenceSpan(quote="No benefit in this trial."),
        )
    ]
    # Passing means only that the quote occurs, NOT that it entails this claim.
    assert (
        validate_synthesis(synthesis, SourceDoc(title="Trial", content="No benefit in this trial."))
        is synthesis
    )
