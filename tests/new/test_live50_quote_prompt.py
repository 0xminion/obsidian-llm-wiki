"""Prompt-only fidelity contracts; source and quote acceptance stay byte-faithful."""

import pytest

from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from obsidian_llm_wiki.synth.quality import build_expand_prompt, build_extract_prompt

RAW_SOURCE = (
    "The Pebbles [assembled](https://example.org/build) the watches.\n"
    "**A verifier** checks the result.[7](https://example.org/evidence)\n"
    "PDF: 原文保留  空格，\n换行和\t制表符。\n"
)


@pytest.fixture(params=["single", "extract", "expand"])
def prompt(request):
    if request.param == "single":
        return build_synthesis_prompt("Raw source", RAW_SOURCE)
    if request.param == "extract":
        return build_extract_prompt("Raw source", RAW_SOURCE)
    return build_expand_prompt("Result", "result", "Observed result", "Raw source", RAW_SOURCE, [])


def test_all_modes_instruct_raw_quote_format_preservation(prompt):
    assert RAW_SOURCE in prompt  # Fixtures must not be normalized on the prompt path.
    for instruction in (
        "raw source text, not its rendered appearance",
        "Markdown links (including URLs), emphasis markers, and citation markers",
        "capitalization, punctuation, and all whitespace",
        "PDF newlines, spaces, and tabs",
        r"JSON escapes such as \n and \t",
        "decoding the JSON string must reproduce the exact source span",
        "Do not remove markup, change case, or normalize whitespace",
    ):
        assert instruction in prompt


def test_all_modes_prefer_short_unique_contiguous_support(prompt):
    for instruction in (
        "Prefer a shorter, unique, contiguous exact span",
        "still supports the entire claim",
        "rather than reconstructing a readable sentence",
        "Never join separate passages, skip intervening text, or add ellipses",
        "If a span repeats, choose a uniquely identifying supporting span",
        "otherwise omit the claim",
        "Do not remove a supported claim merely to avoid copying its formatting",
    ):
        assert instruction in prompt


def test_all_modes_forbid_unsupported_actor_contribution_and_causal_inferences(prompt):
    for instruction in (
        "Do not infer an actor's identity, contribution, or causal role",
        "from ownership, association, chronology, or a shared label",
        "require explicit source support",
        "Apply this to summaries, key points, bodies, and claims alike",
        "Preserve uncertainty",
        "qualifications, negation, attribution, and disagreements",
        "Quote occurrence alone is not proof",
        "omit the claim rather than invent evidence",
        "Do not pad content",
    ):
        assert instruction in prompt


def test_prompt_compatibility_invalidates_old_synthesis_without_schema_change(monkeypatch):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline

    current = pipeline._artifact_versions_for_config(Config())
    assert current.prompt.startswith("synthesis-v4:")
    assert current.schema == "synthesis-schema-v3"
    with monkeypatch.context() as old_prompt:
        old_prompt.setattr(pipeline, "_PROMPT_VERSION", "synthesis-v3")
        previous = pipeline._artifact_versions_for_config(Config())
    assert current.prompt != previous.prompt
    assert current.schema == previous.schema
    assert current.extractor == previous.extractor
    assert current.synthesis_model == previous.synthesis_model
