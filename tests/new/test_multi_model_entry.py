"""Tests for multi-model entry synthesis with section merging.

Covers:
  - merge_entry_syntheses (pure function, no LLM calls)
  - multi_model_entry_synthesize_source (mocked two-model flow)
  - Pipeline dispatch: two_pass routes to multi_model_entry_synthesize_source
  - Fallback when expand_model is not configured
  - Graceful degradation when one model fails
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.synth.quality import (
    merge_entry_syntheses,
    multi_model_entry_synthesize_source,
)

# ── Pure-function tests for merge_entry_syntheses ─────────────────────


def _make_concept(
    slug: str,
    sections: list[tuple[str, list[str], str]] | None = None,
    *,
    title: str = "",
    summary: str = "S",
    tags: list[str] | None = None,
    claims: list[Claim] | None = None,
    related: list[ConceptLink] | None = None,
    confidence: float = 1.0,
) -> ConceptNote:
    body_sections = [
        BodySection(heading=h, points=list(p), prose=pr)
        for h, p, pr in (sections or [])
    ]
    return ConceptNote(
        title=title or slug.replace("-", " ").title(),
        slug=slug,
        summary=summary,
        tags=tags or [],
        sections=body_sections,
        claims=claims or [],
        related=related or [],
        confidence=confidence,
    )


def _make_synth(
    source_title: str = "Test",
    concepts: list[ConceptNote] | None = None,
    maps: list[MapOfContent] | None = None,
    *,
    summary: str = "Summary",
    key_points: list[str] | None = None,
    open_questions: list[str] | None = None,
    source_tags: list[str] | None = None,
) -> SourceSynthesis:
    return SourceSynthesis(
        source_title=source_title,
        source_summary=summary,
        source_tags=source_tags or [],
        key_points=key_points or [],
        open_questions=open_questions or [],
        concepts=concepts or [],
        maps=maps or [],
    )


def test_merge_takes_deeper_section():
    """When both syntheses have the same section heading, the deeper one wins."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["short point"], ""),
        ]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["much longer point one", "much longer point two"], ""),
        ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    concept = merged.concepts[0]
    assert len(concept.sections) == 1
    assert len(concept.sections[0].points) == 2  # secondary was deeper
    assert concept.sections[0].points[0] == "much longer point one"


def test_merge_primary_section_wins_when_deeper():
    """When primary section is deeper, it is kept."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["deep point one", "deep point two", "deep point three"], ""),
        ]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["short"], ""),
        ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert len(merged.concepts[0].sections[0].points) == 3


def test_merge_unique_sections_appended():
    """Sections only in secondary are appended to the merged concept."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["primary point"], ""),
        ]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Overview", ["secondary point"], ""),
            ("Limitations", ["unique section"], ""),
        ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    concept = merged.concepts[0]
    assert len(concept.sections) == 2
    headings = [s.heading for s in concept.sections]
    assert "Limitations" in headings


def test_merge_concept_only_in_secondary_appended():
    """Concepts found only by the secondary model are kept."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("beta", sections=[("S", ["b"], "")]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    slugs = {c.slug for c in merged.concepts}
    assert slugs == {"alpha", "beta"}


def test_merge_unions_tags():
    """Tags from both models are unioned."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")], tags=["ml", "opt"]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["b"], "")], tags=["opt", "math"]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert set(merged.concepts[0].tags) == {"ml", "opt", "math"}


def test_merge_unions_claims_dedup():
    """Claims are unioned, deduplicated by text."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")],
                      claims=[Claim(text="claim A", concept_slug="alpha", source_ref="p1")]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["b"], "")],
                      claims=[
                          Claim(text="claim A", concept_slug="alpha", source_ref="p1"),
                          Claim(text="claim B", concept_slug="alpha", source_ref="p2"),
                      ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    claim_texts = [c.text for c in merged.concepts[0].claims]
    assert claim_texts == ["claim A", "claim B"]


def test_merge_unions_related_dedup():
    """Related links are unioned, deduplicated by slug."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")],
                      related=[ConceptLink(slug="beta", relation="depends_on")]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["b"], "")],
                      related=[
                          ConceptLink(slug="beta", relation="depends_on"),
                          ConceptLink(slug="gamma", relation="complements"),
                      ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    rel_slugs = [r.slug for r in merged.concepts[0].related]
    assert rel_slugs == ["beta", "gamma"]


def test_merge_unions_entry_metadata():
    """source_tags, key_points, open_questions are unioned."""
    primary = _make_synth(
        source_tags=["a", "b"],
        key_points=["kp1"],
        open_questions=["q1"],
    )
    secondary = _make_synth(
        source_tags=["b", "c"],
        key_points=["kp1", "kp2"],
        open_questions=["q1", "q2"],
    )

    merged = merge_entry_syntheses(primary, secondary)

    assert set(merged.source_tags) == {"a", "b", "c"}
    assert merged.key_points == ["kp1", "kp2"]
    assert merged.open_questions == ["q1", "q2"]


def test_merge_mocs_unioned():
    """MoCs are unioned by slug, concept_slugs unioned."""
    primary = _make_synth(maps=[
        MapOfContent(title="M1", slug="m1", summary="S", concept_slugs=["alpha"]),
    ])
    secondary = _make_synth(maps=[
        MapOfContent(title="M1", slug="m1", summary="S", concept_slugs=["beta"]),
        MapOfContent(title="M2", slug="m2", summary="S", concept_slugs=["alpha"]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    moc_slugs = {m.slug for m in merged.maps}
    assert moc_slugs == {"m1", "m2"}
    m1 = next(m for m in merged.maps if m.slug == "m1")
    assert set(m1.concept_slugs) == {"alpha", "beta"}


def test_merge_provenance_set_to_merged():
    """Merged concepts have provenance='merged'."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["b"], "")]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert merged.concepts[0].provenance == "merged"


def test_merge_longer_summary_wins():
    """The longer summary from either model is used."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["a"], "")], summary="Short."),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("S", ["b"], "")],
                      summary="A much longer summary that explains more."),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert merged.concepts[0].summary == "A much longer summary that explains more."


def test_merge_case_insensitive_heading_match():
    """Section headings are matched case-insensitively."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("Overview", ["a"], "")]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[("overview", ["much longer point"], "")]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert len(merged.concepts[0].sections) == 1
    assert merged.concepts[0].sections[0].points == ["much longer point"]


def test_merge_empty_inputs():
    """Merging two empty syntheses produces an empty synthesis."""
    primary = _make_synth()
    secondary = _make_synth()

    merged = merge_entry_syntheses(primary, secondary)

    assert merged.concepts == []
    assert merged.maps == []


# ── Multi-model orchestration tests (mocked) ─────────────────────────


class _MockConfigWithLLM:
    """Config stub with LLMProviderConfig for multi-model tests."""

    def __init__(self, *, model="default-model", expand_model=None,
                 concept_min_body_chars=800, compile_concurrency=2,
                 chunk_size=30_000, output_language=""):
        from obsidian_llm_wiki.config import LLMProviderConfig
        self.llm = LLMProviderConfig(
            model=model,
            expand_model=expand_model,
        )
        self.concept_min_body_chars = concept_min_body_chars
        self.compile_concurrency = compile_concurrency
        self.chunk_size = chunk_size
        self.output_language = output_language
        self.retry_count = 1
        self.retry_base_ms = 1
        self.retry_multiplier = 1


_SKELETON_RESPONSE = json.dumps({
    "source_title": "ML Textbook",
    "source_summary": "A comprehensive ML textbook chapter.",
    "source_tags": ["ml"],
    "concepts": [
        {"title": "Gradient Descent", "slug": "gradient-descent", "summary": "Core method."},
    ],
    "maps": [],
})

_EXPAND_RESPONSE = json.dumps({
    "title": "Gradient Descent",
    "slug": "gradient-descent",
    "summary": "Iterative optimization algorithm.",
    "sections": [
        {"heading": "Core concept", "points": ["point one", "point two"]},
    ],
})


@pytest.mark.asyncio
async def test_multi_model_no_expand_model_falls_back():
    """When expand_model is not set, multi_model_entry_synthesize_source
    delegates to quality_synthesize_source (single two-pass run)."""
    config = _MockConfigWithLLM(model="default-model", expand_model=None)
    source = SourceDoc(title="Test", content="Content here. " * 20)

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        return_value=_make_synth(
            source_title="Test",
            concepts=[_make_concept("alpha", sections=[("S", ["a"], "")])],
        ),
    ) as mock_quality:
        result = await multi_model_entry_synthesize_source(
            config, "test.md", source, [],
        )

    assert result is not None
    assert mock_quality.call_count == 1
    assert len(result.concepts) == 1


@pytest.mark.asyncio
async def test_multi_model_runs_both_models_and_merges():
    """When expand_model is set, both the default and expand model are run,
    and their results are merged."""
    config = _MockConfigWithLLM(
        model="gemma4:31b-cloud",
        expand_model="glm-5.2:cloud",
    )
    source = SourceDoc(title="Test", content="Content here. " * 20)

    # Track which config.model is used in each call.
    call_models: list[str] = []
    call_expand_models: list[str | None] = []

    primary_synth = _make_synth(
        source_title="Test",
        concepts=[
            _make_concept("alpha", sections=[
                ("Overview", ["primary point"], ""),
            ]),
        ],
    )
    secondary_synth = _make_synth(
        source_title="Test",
        concepts=[
            _make_concept("alpha", sections=[
                ("Overview", ["secondary deeper point one", "secondary deeper point two"], ""),
                ("Limitations", ["unique to secondary"], ""),
            ]),
        ],
    )

    async def _mock_quality(config_arg, filename, src, existing, **kw):
        call_models.append(config_arg.llm.model)
        call_expand_models.append(config_arg.llm.expand_model)
        # First call = primary, second call = secondary
        if len(call_models) == 1:
            return primary_synth
        return secondary_synth

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        side_effect=_mock_quality,
    ) as mock_quality:
        result = await multi_model_entry_synthesize_source(
            config, "test.md", source, [],
        )

    assert mock_quality.call_count == 2
    # Primary run: model=default, expand_model=None
    assert call_models[0] == "gemma4:31b-cloud"
    assert call_expand_models[0] is None
    # Secondary run: model=expand_model, expand_model=expand_model
    assert call_models[1] == "glm-5.2:cloud"
    assert call_expand_models[1] == "glm-5.2:cloud"

    # Merged result has 2 sections (Overview + Limitations)
    assert result is not None
    assert len(result.concepts) == 1
    assert len(result.concepts[0].sections) == 2
    # The deeper Overview section from secondary won
    overview = next(s for s in result.concepts[0].sections if s.heading == "Overview")
    assert len(overview.points) == 2  # secondary was deeper


@pytest.mark.asyncio
async def test_multi_model_primary_failure_uses_secondary():
    """When the primary run fails (None), the secondary result is used."""
    config = _MockConfigWithLLM(
        model="gemma4:31b-cloud",
        expand_model="glm-5.2:cloud",
    )
    source = SourceDoc(title="Test", content="Content. " * 20)

    secondary_synth = _make_synth(
        source_title="Test",
        concepts=[_make_concept("alpha", sections=[("S", ["b"], "")])],
    )

    async def _mock_quality(config_arg, filename, src, existing, **kw):
        if config_arg.llm.expand_model is None:
            return None  # Primary fails
        return secondary_synth

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        side_effect=_mock_quality,
    ):
        result = await multi_model_entry_synthesize_source(
            config, "test.md", source, [],
        )

    assert result is not None
    assert len(result.concepts) == 1
    assert result.concepts[0].slug == "alpha"


@pytest.mark.asyncio
async def test_multi_model_secondary_failure_uses_primary():
    """When the secondary run fails (None), the primary result is used."""
    config = _MockConfigWithLLM(
        model="gemma4:31b-cloud",
        expand_model="glm-5.2:cloud",
    )
    source = SourceDoc(title="Test", content="Content. " * 20)

    primary_synth = _make_synth(
        source_title="Test",
        concepts=[_make_concept("alpha", sections=[("S", ["a"], "")])],
    )

    async def _mock_quality(config_arg, filename, src, existing, **kw):
        if config_arg.llm.expand_model is None:
            return primary_synth
        return None  # Secondary fails

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        side_effect=_mock_quality,
    ):
        result = await multi_model_entry_synthesize_source(
            config, "test.md", source, [],
        )

    assert result is not None
    assert len(result.concepts) == 1


@pytest.mark.asyncio
async def test_multi_model_both_failures_returns_none():
    """When both runs fail, None is returned."""
    config = _MockConfigWithLLM(
        model="gemma4:31b-cloud",
        expand_model="glm-5.2:cloud",
    )
    source = SourceDoc(title="Test", content="Content. " * 20)

    with patch(
        "obsidian_llm_wiki.synth.quality.quality_synthesize_source",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await multi_model_entry_synthesize_source(
            config, "test.md", source, [],
        )

    assert result is None


# ── Pipeline dispatch test ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_two_pass_routes_to_multi_model():
    """When synthesis_mode='two_pass', pipeline calls multi_model_entry_synthesize_source."""
    from obsidian_llm_wiki.config import Config, LLMProviderConfig
    from obsidian_llm_wiki.core.pipeline import _synthesize_source

    config = Config(
        llm=LLMProviderConfig(model="test"),
        synthesis_mode="two_pass",
        min_source_chars=10,
    )
    source = SourceDoc(title="Test", content="Enough content. " * 3)

    called = False

    async def _mock_multi(*args, **kwargs):
        nonlocal called
        called = True
        return SourceSynthesis(source_title="Test", source_summary="S")

    with patch(
        "obsidian_llm_wiki.synth.quality.multi_model_entry_synthesize_source",
        new_callable=AsyncMock,
        side_effect=_mock_multi,
    ):
        result = await _synthesize_source(config, "test.md", source, [])

    assert called is True
    assert result is not None
    assert result.source_title == "Test"


# ── Integration: merge with prose sections ────────────────────────────


def test_merge_prose_vs_points_deeper_wins():
    """When one model uses prose and the other uses points, the deeper one wins."""
    primary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Mechanism", [], "Short prose."),
        ]),
    ])
    secondary = _make_synth(concepts=[
        _make_concept("alpha", sections=[
            ("Mechanism", [], "Much longer prose that explains the mechanism in great detail."),
        ]),
    ])

    merged = merge_entry_syntheses(primary, secondary)

    assert len(merged.concepts[0].sections) == 1
    assert merged.concepts[0].sections[0].prose.startswith("Much longer")
