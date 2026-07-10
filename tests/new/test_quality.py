"""Tests for obsidian_llm_wiki.synth.quality — two-pass synthesis.

Covers:
  - build_extract_prompt / build_expand_prompt (pure functions)
  - _concept_body_chars (pure function)
  - _parse_concept_json (JSON extraction)
  - quality_synthesize_source (mocked two-pass flow)
  - Quality gate: low body chars → confidence 0.3
  - Pipeline dispatch: two_pass vs single
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc
from obsidian_llm_wiki.synth.quality import (
    _concept_body_chars,
    _parse_concept_json,
    build_expand_prompt,
    build_extract_prompt,
    quality_synthesize_source,
)

# ── Pure function tests ──────────────────────────────────────────────────


def test_build_extract_prompt_contains_source():
    """The extract prompt includes the source title and content."""
    prompt = build_extract_prompt("My Article", "Some body text here.", language="en")
    assert "My Article" in prompt
    assert "Some body text here." in prompt
    assert "**en**" in prompt


def test_build_extract_prompt_includes_existing_concepts():
    """Existing concepts are listed for dedup context."""
    prompt = build_extract_prompt(
        "T", "C", existing_concepts=["alpha", "beta"]
    )
    assert "alpha" in prompt
    assert "beta" in prompt


def test_build_expand_prompt_contains_concept_and_source():
    """The expand prompt includes the concept title, rationale, and source."""
    prompt = build_expand_prompt(
        concept_title="Gradient Descent",
        concept_slug="gradient-descent",
        concept_rationale="Core optimization method",
        source_title="ML Textbook",
        source_content="Gradient descent minimizes loss...",
        all_concepts=[{"slug": "gradient-descent", "title": "GD"}, {"slug": "sgd", "title": "SGD"}],
    )
    assert "Gradient Descent" in prompt
    assert "gradient-descent" in prompt
    assert "Core optimization method" in prompt
    assert "ML Textbook" in prompt
    assert "Gradient descent minimizes loss" in prompt
    # Other concepts listed for cross-linking.
    assert "sgd" in prompt
    # The concept itself is excluded from the cross-link list.
    # (its slug appears in the "CONCEPT TO EXPAND" section, not in "OTHER CONCEPTS")


def test_concept_body_chars_counts_points_and_prose():
    """_concept_body_chars sums all point + prose lengths."""
    from obsidian_llm_wiki.core.models import BodySection, ConceptNote

    concept = ConceptNote(
        title="T",
        slug="t",
        summary="S",
        sections=[
            BodySection(heading="A", points=["point one", "point two"]),
            BodySection(heading="B", prose="Some prose here."),
        ],
    )
    assert _concept_body_chars(concept) == (
        len("point one") + len("point two") + len("Some prose here.")
    )


def test_concept_body_chars_empty():
    """Empty concept has 0 body chars."""
    from obsidian_llm_wiki.core.models import ConceptNote

    assert _concept_body_chars(ConceptNote(title="T", slug="t", summary="S")) == 0


def test_parse_concept_json_clean():
    """Clean JSON is parsed directly."""
    data = _parse_concept_json('{"title": "Test", "slug": "test"}')
    assert data is not None
    assert data["title"] == "Test"


def test_parse_concept_json_with_fences():
    """Code-fenced JSON is extracted."""
    data = _parse_concept_json('```json\n{"title": "Fenced"}\n```')
    assert data is not None
    assert data["title"] == "Fenced"


def test_parse_concept_json_with_prose():
    """JSON surrounded by prose is extracted."""
    data = _parse_concept_json('Here is the result:\n{"title": "Prose"}\nDone.')
    assert data is not None
    assert data["title"] == "Prose"


def test_parse_concept_json_empty():
    """Empty input returns None."""
    assert _parse_concept_json("") is None
    assert _parse_concept_json("   ") is None


def test_parse_concept_json_no_json():
    """Non-JSON input returns None."""
    assert _parse_concept_json("just plain text") is None


# ── Two-pass orchestration test (mocked) ──────────────────────────────────


# Pass 1 skeleton response: concepts with NO body (just title, slug, summary).
_SKELETON_RESPONSE = json.dumps({
    "source_title": "ML Textbook",
    "source_summary": "A comprehensive ML textbook chapter on optimization.",
    "source_tags": ["machine-learning", "optimization"],
    "concepts": [
        {"title": "Gradient Descent", "slug": "gradient-descent", "summary": "Core optimization method."},
        {"title": "Learning Rate", "slug": "learning-rate", "summary": "Controls step size."},
    ],
    "maps": [],
})

# Pass 2 expansion response for "gradient-descent" — deep content.
_EXPAND_GD = json.dumps({
    "title": "Gradient Descent",
    "slug": "gradient-descent",
    "summary": "Iterative optimization algorithm.",
    "sections": [
        {"heading": "Core concept", "points": [
            "Minimizes loss by following the negative gradient",
            "Learning rate controls step size",
            "Converges to local minima for non-convex functions",
        ]},
        {"heading": "Variants", "points": [
            "Batch GD uses full dataset",
            "Stochastic GD uses one sample",
            "Mini-batch GD uses small batches",
        ]},
    ],
    "claims": [
        {"text": "GD converges at O(1/sqrt(t)) rate", "source_ref": "section 4.2"},
    ],
    "related": [
        {"slug": "learning-rate", "relation": "depends_on"},
    ],
})

# Pass 2 expansion response for "learning-rate" — short content (triggers quality gate).
_EXPAND_LR = json.dumps({
    "title": "Learning Rate",
    "slug": "learning-rate",
    "summary": "Step size for gradient updates.",
    "sections": [
        {"heading": "Overview", "points": ["Controls how far to move"]},
    ],
    "related": [],
})


class _MockConfig:
    """Minimal config stub for two-pass tests."""
    output_language = ""
    compile_concurrency = 2
    concept_min_body_chars = 800


@pytest.mark.asyncio
async def test_quality_synthesize_source_two_pass():
    """Full two-pass flow: skeleton → expand → merge."""
    source = SourceDoc(
        title="ML Textbook",
        content="Gradient descent is an optimization algorithm. " * 20,
    )

    responses = [_SKELETON_RESPONSE, _EXPAND_GD, _EXPAND_LR]
    call_idx = 0

    async def _mock_acall(*args, **kwargs):
        nonlocal call_idx
        resp = responses[call_idx]
        call_idx += 1
        return resp

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall,
    ):
        result = await quality_synthesize_source(
            _MockConfig(),
            "ml-textbook.md",
            source,
            existing_concepts=[],
        )

    # Pass 1 + 2 expansions = 3 total LLM calls.
    assert call_idx == 3

    assert result is not None
    assert result.source_title == "ML Textbook"
    assert len(result.concepts) == 2

    # gradient-descent: expanded with deep content.
    gd = next(c for c in result.concepts if c.slug == "gradient-descent")
    assert len(gd.sections) == 2
    assert gd.sections[0].heading == "Core concept"
    assert len(gd.sections[0].points) == 3
    assert len(gd.claims) == 1
    assert gd.claims[0].source_ref == "section 4.2"
    assert len(gd.related) == 1
    assert gd.related[0].slug == "learning-rate"
    assert gd.related[0].relation == "depends_on"

    # learning-rate: short content → quality gate triggers.
    lr = next(c for c in result.concepts if c.slug == "learning-rate")
    body = _concept_body_chars(lr)
    assert body < 800  # below threshold
    assert lr.confidence == 0.3  # quality gate set confidence low


@pytest.mark.asyncio
async def test_quality_synthesize_source_empty_concepts():
    """Pass 1 returns no concepts → early return with skeleton."""
    source = SourceDoc(title="Empty", content="Some content here. " * 10)

    empty_response = json.dumps({
        "source_title": "Empty",
        "source_summary": "Nothing here.",
        "concepts": [],
        "maps": [],
    })

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value=empty_response,
    ):
        result = await quality_synthesize_source(
            _MockConfig(),
            "empty.md",
            source,
            existing_concepts=[],
        )

    assert result is not None
    assert len(result.concepts) == 0
    assert result.source_summary == "Nothing here."


@pytest.mark.asyncio
async def test_quality_synthesize_source_pass2_failure_sets_low_confidence():
    """Pass-2 failure (None or exception) sets confidence=0.3 on skeleton concept.

    Regression test for the trade-size-scale-effect bug: a skeleton concept
    whose Pass-2 expansion returned None was shipped at confidence=1.0
    with zero body content.
    """
    source = SourceDoc(
        title="Test Source",
        content="Some content here for the source. " * 10,
    )

    skeleton_response = json.dumps({
        "source_title": "Test Source",
        "source_summary": "A test source.",
        "concepts": [
            {"title": "Good Concept", "slug": "good-concept", "summary": "Will expand fine."},
            {"title": "Null Concept", "slug": "null-concept", "summary": "Pass 2 returns None."},
            {"title": "Error Concept", "slug": "error-concept", "summary": "Pass 2 raises."},
        ],
        "maps": [],
    })

    async def _mock_acall(prompt, *args, **kwargs):
        # Pass 1 only — return skeleton
        return skeleton_response

    async def _mock_expand(config, concept, source, all_concepts):
        # Pass 2 routed by slug — concurrency-safe under asyncio.gather
        if concept.slug == "good-concept":
            from obsidian_llm_wiki.core.models import BodySection
            return ConceptNote(
                title="Good Concept",
                slug="good-concept",
                summary="Expanded well.",
                sections=[BodySection(heading="Core", points=["x" * 900])],
                confidence=1.0,
            )
        if concept.slug == "null-concept":
            return None
        if concept.slug == "error-concept":
            raise RuntimeError("LLM timeout")
        return None

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall,
    ):
        with patch(
            "obsidian_llm_wiki.synth.quality._expand_one_concept",
            new_callable=AsyncMock,
            side_effect=_mock_expand,
        ):
            result = await quality_synthesize_source(
                _MockConfig(),
                "test.md",
                source,
                existing_concepts=[],
            )

    assert result is not None
    assert len(result.concepts) == 3

    good = next(c for c in result.concepts if c.slug == "good-concept")
    null_c = next(c for c in result.concepts if c.slug == "null-concept")
    error_c = next(c for c in result.concepts if c.slug == "error-concept")

    # Good expansion stays at confidence 1.0
    assert good.confidence == 1.0
    assert _concept_body_chars(good) >= 800

    # Failed expansions get confidence 0.3 (the fix)
    assert null_c.confidence == 0.3, f"Expected 0.3 for None expansion, got {null_c.confidence}"
    assert error_c.confidence == 0.3, f"Expected 0.3 for exception expansion, got {error_c.confidence}"

    # Failed expansions have no body (skeleton only)
    assert _concept_body_chars(null_c) == 0
    assert _concept_body_chars(error_c) == 0


@pytest.mark.asyncio
async def test_quality_synthesize_source_pass1_parse_failure():
    """Pass 1 returns unparseable JSON → returns None."""
    source = SourceDoc(title="Bad", content="Content. " * 10)

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value="not valid json {{{",
    ):
        result = await quality_synthesize_source(
            _MockConfig(),
            "bad.md",
            source,
            existing_concepts=[],
        )

    assert result is None


# ── Pipeline dispatch test ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_dispatches_to_two_pass():
    """When synthesis_mode='two_pass', pipeline calls quality_synthesize_source."""
    from obsidian_llm_wiki.config import Config, LLMProviderConfig
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.pipeline import _synthesize_source

    config = Config(
        llm=LLMProviderConfig(model="test"),
        synthesis_mode="two_pass",
        min_source_chars=10,
    )
    source = SourceDoc(title="Test", content="Enough content to pass the gate. " * 3)

    called = False

    async def _mock_quality(*args, **kwargs):
        nonlocal called
        called = True
        from obsidian_llm_wiki.core.models import SourceSynthesis
        return SourceSynthesis(source_title="Test", source_summary="S")

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        side_effect=_mock_quality,
    ):
        result = await _synthesize_source(config, "test.md", source, [])

    assert called is True
    assert result is not None
    assert result.source_title == "Test"
