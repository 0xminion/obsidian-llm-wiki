"""Tests for few-shot relationship examples in prompts."""

from __future__ import annotations

from obsidian_llm_wiki.synth.prompts import RELATIONSHIP_FEWSHOT, build_synthesis_prompt
from obsidian_llm_wiki.synth.quality import build_expand_prompt


def test_relationship_fewshot_constant_exists():
    """RELATIONSHIP_FEWSHOT constant should exist and be non-empty."""
    assert RELATIONSHIP_FEWSHOT
    assert isinstance(RELATIONSHIP_FEWSHOT, str)
    assert len(RELATIONSHIP_FEWSHOT) > 50


def test_relationship_fewshot_has_crypto_examples():
    """Few-shot should contain crypto/finance domain examples."""
    assert "Bitcoin" in RELATIONSHIP_FEWSHOT
    assert "enables" in RELATIONSHIP_FEWSHOT
    assert "AMM" in RELATIONSHIP_FEWSHOT
    assert "evolves_into" in RELATIONSHIP_FEWSHOT


def test_relationship_fewshot_has_prediction_market_examples():
    """Few-shot should contain prediction market examples."""
    assert "Prediction Markets" in RELATIONSHIP_FEWSHOT
    assert "competes_with" in RELATIONSHIP_FEWSHOT
    assert "Futarchy" in RELATIONSHIP_FEWSHOT
    assert "supersedes" in RELATIONSHIP_FEWSHOT


def test_relationship_fewshot_has_finance_examples():
    """Few-shot should contain finance examples."""
    assert "Clearinghouse" in RELATIONSHIP_FEWSHOT
    assert "part_of" in RELATIONSHIP_FEWSHOT
    assert "Kelly Criterion" in RELATIONSHIP_FEWSHOT
    assert "measures" in RELATIONSHIP_FEWSHOT


def test_build_synthesis_prompt_includes_fewshot():
    """The synthesis prompt should include the few-shot examples."""
    prompt = build_synthesis_prompt(
        "Test Source", "Some content about Bitcoin and prediction markets.",
    )
    assert "Bitcoin" in prompt
    assert "enables" in prompt
    assert "Prediction Markets" in prompt


def test_build_expand_prompt_includes_fewshot():
    """The expand prompt should include the few-shot examples."""
    prompt = build_expand_prompt(
        concept_title="Bitcoin",
        concept_slug="bitcoin",
        concept_rationale="Important cryptocurrency",
        source_title="Crypto Paper",
        source_content="Content about Bitcoin.",
        all_concepts=[{"slug": "mining", "title": "Mining"}],
    )
    assert "Bitcoin" in prompt
    assert "enables" in prompt
    assert "Prediction Markets" in prompt


def test_relationship_fewshot_has_all_six_examples():
    """All six specified examples should be present."""
    examples = [
        ("Bitcoin", "enables", "Proof of Work"),
        ("AMM", "evolves_into", "Concentrated Liquidity"),
        ("Prediction Markets", "competes_with", "Opinion Polls"),
        ("Clearinghouse", "part_of", "Exchange Infrastructure"),
        ("Kelly Criterion", "measures", "Optimal Position Size"),
        ("Futarchy", "supersedes", "Voting"),
    ]
    for subject, relation, obj in examples:
        assert subject in RELATIONSHIP_FEWSHOT, f"Missing: {subject}"
        assert relation in RELATIONSHIP_FEWSHOT, f"Missing: {relation}"
        assert obj in RELATIONSHIP_FEWSHOT, f"Missing: {obj}"
