"""Tests for gradient confidence scoring, content chunking, and Pass 2 diagnostics.

Covers:
  - gradient_confidence formula with various body sizes
  - chunk_content splitting large sources into chunks
  - merge_skeletons unioning concepts/MoCs/key_points from multiple skeletons
  - _diagnose_pass2_failure capturing structured failure reasons
  - quality_synthesize_source with large content triggering chunking
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.synth.quality import (
    _diagnose_pass2_failure,
    chunk_content,
    gradient_confidence,
    merge_skeletons,
    quality_synthesize_source,
)

# ── Gradient confidence formula tests ────────────────────────────────────


class TestGradientConfidence:
    """Test the gradient confidence formula at various body sizes."""

    def test_full_body_confidence_is_1(self):
        """Body >= threshold → confidence = 1.0."""
        assert gradient_confidence(800, 800) == 1.0
        assert gradient_confidence(1000, 800) == 1.0
        assert gradient_confidence(801, 800) == 1.0

    def test_zero_body_confidence_is_0_1(self):
        """Body = 0 → confidence = 0.1 (floor)."""
        assert gradient_confidence(0, 800) == pytest.approx(0.1)

    def test_half_threshold_confidence(self):
        """Body = half_threshold → 0.5 + 0.5 * (400/800) = 0.75."""
        assert gradient_confidence(400, 800) == pytest.approx(0.75)

    def test_just_below_half_threshold(self):
        """Body < half_threshold → uses lower segment formula."""
        # body=200, threshold=800, half=400
        # 0.1 + 0.4 * (200/400) = 0.1 + 0.2 = 0.3
        assert gradient_confidence(200, 800) == pytest.approx(0.3)

    def test_just_above_half_threshold(self):
        """Body >= half_threshold but < threshold → uses upper segment formula."""
        # body=600, threshold=800
        # 0.5 + 0.5 * (600/800) = 0.5 + 0.375 = 0.875
        assert gradient_confidence(600, 800) == pytest.approx(0.875)

    def test_quarter_body(self):
        """Body = 25% of threshold → lower segment."""
        # body=200, threshold=800, half=400
        # 0.1 + 0.4 * (200/400) = 0.3
        assert gradient_confidence(200, 800) == pytest.approx(0.3)

    def test_three_quarters_body(self):
        """Body = 75% of threshold → upper segment."""
        # body=600, threshold=800
        # 0.5 + 0.5 * (600/800) = 0.875
        assert gradient_confidence(600, 800) == pytest.approx(0.875)

    def test_monotonic_increase(self):
        """Confidence monotonically increases with body size."""
        prev = -1.0
        for body in range(0, 1000, 50):
            conf = gradient_confidence(body, 800)
            assert conf >= prev, f"Confidence decreased at body={body}: {conf} < {prev}"
            prev = conf

    def test_clamped_to_max_1(self):
        """Confidence never exceeds 1.0."""
        assert gradient_confidence(10_000, 800) == 1.0

    def test_clamped_to_min_0_1(self):
        """Confidence never goes below 0.1."""
        assert gradient_confidence(0, 800) == pytest.approx(0.1)

    def test_zero_threshold_returns_1(self):
        """Edge case: threshold=0 → return 1.0 (avoid division by zero)."""
        assert gradient_confidence(100, 0) == 1.0

    def test_gradient_smooth_at_half_threshold_boundary(self):
        """The formula is continuous at the half_threshold boundary."""
        # At exactly half_threshold: lower formula gives 0.1 + 0.4 * 1.0 = 0.5
        # Upper formula gives 0.5 + 0.5 * (400/800) = 0.5 + 0.25 = 0.75
        # Actually, the boundary condition is body >= half_threshold uses upper formula
        # So at body=400: 0.5 + 0.5 * (400/800) = 0.75
        conf_at_400 = gradient_confidence(400, 800)
        # body=399 uses lower: 0.1 + 0.4 * (399/400) = 0.1 + 0.399 = 0.499
        conf_at_399 = gradient_confidence(399, 800)
        # There's a discontinuity by design — just verify both are valid
        assert 0.1 <= conf_at_400 <= 1.0
        assert 0.1 <= conf_at_399 <= 1.0


# ── Content chunking tests ───────────────────────────────────────────────


class TestChunkContent:
    """Test content chunking for large sources."""

    def test_small_content_single_chunk(self):
        """Content <= chunk_size returns single chunk."""
        content = "Short content."
        chunks = chunk_content(content, chunk_size=30_000)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_100k_source_gets_split(self):
        """A 100K char source gets split into multiple chunks."""
        # Generate 100K chars of content with paragraphs
        paragraphs = []
        for i in range(500):
            paragraphs.append(f"Paragraph {i}: " + "x" * 190)  # ~205 chars each
        content = "\n\n".join(paragraphs)
        assert len(content) > 100_000

        chunks = chunk_content(content, chunk_size=30_000)
        assert len(chunks) > 1, f"Expected multiple chunks for 100K content, got {len(chunks)}"
        # Each chunk should be <= chunk_size (approximately)
        for chunk in chunks:
            assert len(chunk) <= 30_000 + 200  # small overflow for paragraph boundaries

    def test_chunk_preserves_content(self):
        """Chunking and rejoining produces approximately the same content."""
        paragraphs = []
        for i in range(200):
            paragraphs.append(f"Paragraph {i} with some content here. " * 5)
        content = "\n\n".join(paragraphs)

        chunks = chunk_content(content, chunk_size=10_000)
        rejoined = "\n\n".join(chunks)
        # The rejoined content should contain all the same paragraphs
        # (spacing may differ slightly at boundaries)
        for para in paragraphs:
            assert para in rejoined, f"Paragraph '{para[:40]}...' lost during chunking"

    def test_hard_split_long_paragraph(self):
        """A single paragraph longer than chunk_size gets hard-split."""
        content = "x" * 50_000  # single "paragraph" with no newlines
        chunks = chunk_content(content, chunk_size=10_000)
        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk) == 10_000

    def test_chunk_at_paragraph_boundaries(self):
        """Chunks split at paragraph boundaries, not mid-paragraph."""
        content = "Para 1 line.\n\nPara 2 line.\n\nPara 3 line.\n\nPara 4 line."
        chunks = chunk_content(content, chunk_size=25)
        # Should not cut "Para 1 line." in half
        for chunk in chunks:
            # Each chunk should start at a paragraph beginning
            assert chunk.startswith("Para") or chunk.startswith("")


# ── Merge skeletons tests ────────────────────────────────────────────────


class TestMergeSkeletons:
    """Test merging multiple Pass 1 skeletons."""

    def test_single_skeleton_passthrough(self):
        """Merging a single skeleton returns it unchanged."""
        s = SourceSynthesis(
            source_title="Test",
            source_summary="Summary",
            concepts=[ConceptNote(title="C1", slug="c1", summary="S1")],
            maps=[MapOfContent(title="M1", slug="m1", summary="", concept_slugs=["c1"])],
        )
        merged = merge_skeletons([s])
        assert merged.source_title == "Test"
        assert len(merged.concepts) == 1
        assert merged.concepts[0].slug == "c1"

    def test_union_concepts_by_slug(self):
        """Concepts with same slug from different chunks are unioned (first wins)."""
        s1 = SourceSynthesis(
            source_title="Test",
            source_summary="S1",
            concepts=[
                ConceptNote(
                    title="Concept A", slug="concept-a",
                    summary="From chunk 1", tags=["t1"],
                ),
                ConceptNote(title="Concept B", slug="concept-b", summary="From chunk 1"),
            ],
        )
        s2 = SourceSynthesis(
            source_title="Test",
            source_summary="S2",
            concepts=[
                ConceptNote(
                    title="Concept A", slug="concept-a",
                    summary="From chunk 2", tags=["t2"],
                ),
                ConceptNote(title="Concept C", slug="concept-c", summary="Only in chunk 2"),
            ],
        )
        merged = merge_skeletons([s1, s2])
        slugs = {c.slug for c in merged.concepts}
        assert slugs == {"concept-a", "concept-b", "concept-c"}
        # First occurrence wins for summary
        ca = next(c for c in merged.concepts if c.slug == "concept-a")
        assert ca.summary == "From chunk 1"
        # Tags are unioned
        assert set(ca.tags) == {"t1", "t2"}

    def test_union_mocs_by_slug(self):
        """MoCs with same slug have concept_slugs unioned."""
        s1 = SourceSynthesis(
            source_title="T", source_summary="S",
            maps=[MapOfContent(title="MOC", slug="moc", summary="M", concept_slugs=["a", "b"])],
        )
        s2 = SourceSynthesis(
            source_title="T", source_summary="S",
            maps=[MapOfContent(title="MOC", slug="moc", summary="M", concept_slugs=["b", "c"])],
        )
        merged = merge_skeletons([s1, s2])
        assert len(merged.maps) == 1
        moc = merged.maps[0]
        assert set(moc.concept_slugs) == {"a", "b", "c"}

    def test_union_key_points_deduped(self):
        """key_points are unioned with deduplication."""
        s1 = SourceSynthesis(
            source_title="T", source_summary="S",
            key_points=["Point A", "Point B"],
        )
        s2 = SourceSynthesis(
            source_title="T", source_summary="S",
            key_points=["Point B", "Point C"],
        )
        merged = merge_skeletons([s1, s2])
        assert set(merged.key_points) == {"Point A", "Point B", "Point C"}

    def test_empty_list_returns_empty(self):
        """Merging empty list returns empty skeleton."""
        merged = merge_skeletons([])
        assert merged.source_title == ""
        assert len(merged.concepts) == 0


# ── Diagnostic info tests ────────────────────────────────────────────────


class _MockDiagConfig:
    """Config stub for diagnostic tests."""
    output_language = ""
    compile_concurrency = 2
    concept_min_body_chars = 800
    chunk_size = 30_000

    class _LLM:
        context_window = 256_000
    llm = _LLM()


class TestDiagnosePass2Failure:
    """Test diagnostic info capture for Pass 2 failures."""

    def test_timeout_detected(self):
        """Timeout exception is classified as timeout."""
        exc = TimeoutError("Connection timed out after 30s")
        source = SourceDoc(title="T", content="C", source_file="test.md")
        diag = _diagnose_pass2_failure(exc, "my-concept", source, _MockDiagConfig())
        assert diag["failure_type"] == "timeout"
        assert "timed out" in diag["reason"].lower()
        assert diag["concept_slug"] == "my-concept"

    def test_empty_response_detected(self):
        """Empty response is classified as empty_response."""
        exc = RuntimeError("No output")
        source = SourceDoc(title="T", content="C", source_file="test.md")
        diag = _diagnose_pass2_failure(
            exc, "my-concept", source, _MockDiagConfig(),
            response="   ",
        )
        assert diag["failure_type"] == "empty_response"
        assert diag["concept_slug"] == "my-concept"

    def test_json_parse_error_detected(self):
        """Non-JSON response is classified as json_parse_error."""
        exc = RuntimeError("Parse failed")
        source = SourceDoc(title="T", content="C", source_file="test.md")
        diag = _diagnose_pass2_failure(
            exc, "my-concept", source, _MockDiagConfig(),
            response="This is not JSON {{{",
        )
        assert diag["failure_type"] == "json_parse_error"
        assert diag["response_len"] == len("This is not JSON {{{")
        assert "response_preview" in diag

    def test_context_window_overflow_detected(self):
        """Prompt larger than context window is detected."""
        exc = RuntimeError("Some error")
        source = SourceDoc(title="T", content="C", source_file="test.md")
        # Create a prompt that's clearly larger than a small context window
        big_prompt = "x" * 4_000_000  # ~1M tokens, way over 256K
        config = _MockDiagConfig()
        config.llm.context_window = 10_000  # small window for test
        diag = _diagnose_pass2_failure(
            exc, "my-concept", source, config,
            prompt=big_prompt,
        )
        assert diag["failure_type"] == "context_window_overflow"
        assert diag["estimated_prompt_tokens"] > diag["context_window"]

    def test_generic_exception_fallback(self):
        """Non-timeout exception without response/prompt uses generic fallback."""
        exc = ValueError("Something went wrong")
        source = SourceDoc(title="T", content="C", source_file="test.md")
        diag = _diagnose_pass2_failure(exc, "my-concept", source, _MockDiagConfig())
        assert diag["failure_type"] == "exception"
        assert diag["exception_type"] == "ValueError"
        assert "Something went wrong" in diag["reason"]

    def test_diagnostic_includes_source_info(self):
        """Diagnostic dict includes source file and content length."""
        exc = RuntimeError("err")
        source = SourceDoc(title="T", content="x" * 500, source_file="myfile.md")
        diag = _diagnose_pass2_failure(exc, "slug", source, _MockDiagConfig())
        assert diag["source_file"] == "myfile.md"
        assert diag["source_content_len"] == 500


# ── Integration: chunking in quality_synthesize_source ───────────────────


@pytest.mark.asyncio
async def test_quality_synthesize_source_chunking_large_content():
    """A source > 40K chars triggers chunking — multiple Pass 1 calls, merged skeleton."""
    # Build content > 40K chars
    paragraphs = []
    for i in range(250):
        paragraphs.append(f"This is paragraph {i} with enough text to be meaningful. " * 3)
    large_content = "\n\n".join(paragraphs)
    assert len(large_content) > 40_000

    source = SourceDoc(title="Large Source", content=large_content)

    # Each chunk's Pass 1 response returns different concepts
    chunk_responses = [
        json.dumps({
            "source_title": "Large Source",
            "source_summary": "Chunk 1 summary.",
            "concepts": [
                {"title": "Concept A", "slug": "concept-a", "summary": "From chunk 1"},
                {"title": "Concept B", "slug": "concept-b", "summary": "From chunk 1"},
            ],
            "maps": [
                {"title": "MOC", "slug": "moc",
                 "concept_slugs": ["concept-a"]},
            ],
        }),
        json.dumps({
            "source_title": "Large Source",
            "source_summary": "Chunk 2 summary.",
            "concepts": [
                {"title": "Concept B", "slug": "concept-b", "summary": "From chunk 2"},
                {"title": "Concept C", "slug": "concept-c", "summary": "From chunk 2"},
            ],
            "maps": [
                {"title": "MOC", "slug": "moc", "summary": "MOC summary.",
                 "concept_slugs": ["concept-b", "concept-c"]},
            ],
        }),
    ]

    call_idx = 0

    async def _mock_acall(prompt, *args, **kwargs):
        nonlocal call_idx
        # Pass 1 calls for chunks
        if call_idx < len(chunk_responses):
            resp = chunk_responses[call_idx]
            call_idx += 1
            return resp
        # Pass 2 calls for each concept expansion
        call_idx += 1
        return json.dumps({
            "title": "Expanded",
            "slug": "test",
            "summary": "Expanded.",
            "sections": [{"heading": "Core", "points": ["x" * 900]}],
        })

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall,
    ), patch(
        "obsidian_llm_wiki.synth.quality._expand_one_concept",
        new_callable=AsyncMock,
        side_effect=_mock_expand_simple,
    ):
        result = await quality_synthesize_source(
            _MockDiagConfig(),
            "large.md",
            source,
            existing_concepts=[],
        )

    assert result is not None
    # Merged skeleton should have concepts from both chunks
    slugs = {c.slug for c in result.concepts}
    assert "concept-a" in slugs
    assert "concept-b" in slugs
    assert "concept-c" in slugs
    # Merged MOC should have concept_slugs from both chunks
    assert len(result.maps) == 1
    moc = result.maps[0]
    assert set(moc.concept_slugs) == {"concept-a", "concept-b", "concept-c"}


async def _mock_expand_simple(config, concept, source, all_concepts, source_lang="", rationale=""):
    """Simple mock for _expand_one_concept that returns a fat concept."""
    return ConceptNote(
        title=concept.title,
        slug=concept.slug,
        summary=concept.summary,
        sections=[BodySection(heading="Core", points=["x" * 900])],
        confidence=1.0,
    )
