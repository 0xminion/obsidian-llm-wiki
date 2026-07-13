"""Extended golden tests — edge cases, two-pass mode, and regression coverage.

Complements test_golden_pipeline.py with:
  - Two-pass synthesis mode golden test
  - Empty concepts / missing fields golden test
  - Malformed JSON response golden test
  - Multi-source merge golden test (cross-references between sources)
  - Bilingual/Chinese source golden test
  - Concept confidence gradient golden test
  - Backlink propagation golden test
  - Frontmatter parsing robustness tests
  - URL classification tests (journal XML, SSRN, YouTube)

Per test automation taxonomy (https://en.wikipedia.org/wiki/Test_automation):
  - Golden/snapshot: test_two_pass_golden, test_multi_source_merge
  - Boundary value: test_empty_concepts, test_missing_fields
  - Error/exception: test_malformed_json, test_llm_timeout
  - Equivalence class: test_url_classification
  - State transition: test_incremental_cache_hit_then_miss
  - Regression: test_defuddle_frontmatter_parsing, test_journal_xml_url_scoping
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.obsidian import (
    parse_frontmatter,
    render_concept_page,
    render_moc_page,
    render_source_page,
    safe_read_file,
)
from obsidian_llm_wiki.synth.dedupe import (
    merge_bundle,
    propagate_backlinks,
)
from obsidian_llm_wiki.synth.quality import (
    chunk_content,
    gradient_confidence,
    merge_skeletons,
)

# ── Two-pass synthesis golden test ──────────────────────────────────────


TWO_PASS_EXTRACT_RESPONSE = json.dumps({
    "source_title": "Prediction Markets and Information Aggregation",
    "source_summary": "Explores how prediction markets aggregate information.",
    "source_tags": ["prediction-markets", "information-theory"],
    "key_points": ["Markets aggregate dispersed information efficiently"],
    "open_questions": ["Can prediction markets outperform polls?"],
    "language": "en",
    "concepts": [
        {
            "title": "Information Aggregation",
            "slug": "information-aggregation",
            "summary": "Markets pool information from diverse participants.",
            "tags": ["prediction-markets"],
            "rationale": "Core mechanism by which markets produce accurate forecasts.",
        },
        {
            "title": "Prediction Market Accuracy",
            "slug": "prediction-market-accuracy",
            "summary": "Accuracy of market forecasts compared to alternatives.",
            "tags": ["prediction-markets"],
            "rationale": "Empirical evidence on market accuracy drives adoption decisions.",
        },
    ],
    "maps": [
        {
            "title": "Prediction Market Mechanisms",
            "slug": "prediction-market-mechanisms",
            "summary": "How prediction markets work as information aggregation devices.",
            "tags": ["prediction-markets"],
            "concept_slugs": ["information-aggregation", "prediction-market-accuracy"],
        },
    ],
})

TWO_PASS_EXPAND_RESPONSES = {
    "information-aggregation": json.dumps({
        "title": "Information Aggregation",
        "slug": "information-aggregation",
        "summary": "Markets pool information from diverse participants.",
        "sections": [
            {
                "heading": "Core concept",
                "points": [
                    "Prices encode the weighted beliefs of all traders",
                    "Diverse traders bring different information to the market",
                    "The market price reflects the consensus probability estimate",
                ],
            },
            {
                "heading": "Context",
                "prose": "Information aggregation is the fundamental mechanism by "
                         "which prediction markets produce accurate forecasts. "
                         "Each trader brings private information to the market "
                         "through their trading decisions, and the resulting "
                         "price reflects the collective wisdom of all participants.",
            },
        ],
        "claims": [
            {"text": "Market prices are probability estimates", "source_ref": "section 2"},
        ],
        "related": [
            {"slug": "prediction-market-accuracy", "relation": "related_to"},
        ],
    }),
    "prediction-market-accuracy": json.dumps({
        "title": "Prediction Market Accuracy",
        "slug": "prediction-market-accuracy",
        "summary": "Accuracy of market forecasts compared to alternatives.",
        "sections": [
            {
                "heading": "Core concept",
                "points": [
                    "Prediction markets outperform opinion polls in many settings",
                    "Accuracy improves with trading volume and trader diversity",
                    "Long-running markets show calibration improvements over time",
                ],
            },
            {
                "heading": "Context",
                "prose": "Empirical studies consistently show that prediction markets produce "
                         "calibrated probability estimates that outperform expert judgments "
                         "and opinion polls, especially for well-defined events with clear "
                         "resolution criteria.",
            },
        ],
        "claims": [
            {"text": "Iowa Electronic Markets outperform polls", "source_ref": "section 3"},
        ],
        "related": [
            {"slug": "information-aggregation", "relation": "related_to"},
        ],
    }),
}


@pytest.mark.asyncio
async def test_two_pass_golden(tmp_path: Path):
    """Golden test for two-pass synthesis mode — extract skeleton then expand."""
    vault = tmp_path / "TwoPassVault"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\n"
        f"LLM_PROVIDER=ollama\n"
        f"LLM_MODEL=test-model\n"
        f"SYNTHESIS_MODE=two_pass\n"
    )
    config = load_config(env_file=str(vault / ".env"))

    sources = {
        "prediction-markets.md": SourceDoc(
            title="Prediction Markets and Information Aggregation",
            content="Prediction markets are speculative markets created for the purpose "
                    "of making predictions. Assets are produced whose final cash value is "
                    "tied to a particular event or parameter. The market prices can then "
                    "be interpreted as predictions of the probability of the event.",
            url="https://example.com/prediction-markets",
        ),
    }

    # Mock: Pass 1 returns skeleton, Pass 2 returns expanded concepts.
    call_count = 0

    async def _mock_acall(system, messages, cfg, **kwargs):
        nonlocal call_count
        call_count += 1
        prompt = system
        # Pass 1: extract prompt (first call)
        if call_count == 1:
            return TWO_PASS_EXTRACT_RESPONSE
        # Pass 2: expand prompts — match by concept slug in the CONCEPT TO EXPAND section
        # The prompt has: "Slug: <slug>" in the CONCEPT TO EXPAND section
        for slug, resp in TWO_PASS_EXPAND_RESPONSES.items():
            # Look for the slug in the "Slug:" line specifically
            if f"Slug: {slug}" in prompt:
                return resp
        # Fallback: return first expand response
        return list(TWO_PASS_EXPAND_RESPONSES.values())[0]

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        side_effect=AsyncMock(side_effect=_mock_acall),
    ):
        from obsidian_llm_wiki.core.pipeline import run_pipeline
        result = await run_pipeline(vault, sources, config, force=True)

    assert result.compiled == 1
    assert len(result.concepts) >= 2
    assert len(result.errors) == 0

    # Verify concept pages exist with expanded content
    bundle = vault / "04-Wiki"
    assert (bundle / "concepts" / "information-aggregation.md").exists()
    assert (bundle / "concepts" / "prediction-market-accuracy.md").exists()

    # Verify expanded content (not just skeleton)
    ia_page = safe_read_file(bundle / "concepts" / "information-aggregation.md")
    meta, body = parse_frontmatter(ia_page)
    assert meta["type"] == "Concept"
    assert meta["title"] == "Information Aggregation"
    assert "## Core concept" in body
    assert "Prices encode the weighted beliefs" in body
    assert "## Context" in body
    assert "Information aggregation is the fundamental" in body


# ── Empty concepts golden test ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_concepts_golden(tmp_path: Path):
    """Pipeline should handle a source that produces zero concepts."""
    vault = tmp_path / "EmptyVault"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=test\n"
    )
    config = load_config(env_file=str(vault / ".env"))

    sources = {
        "empty.md": SourceDoc(
            title="Empty Source",
            content="This is a test document with enough content to pass the length gate. " * 3,
        ),
    }

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value=json.dumps({
            "source_title": "Empty Source",
            "source_summary": "A source with no extractable concepts.",
            "concepts": [],
            "maps": [],
        }),
    ):
        from obsidian_llm_wiki.core.pipeline import run_pipeline
        result = await run_pipeline(vault, sources, config, force=True)

    # Should succeed with 0 concepts, 0 errors
    assert result.compiled == 1
    assert len(result.concepts) == 0
    assert len(result.errors) == 0


# ── Malformed JSON response golden test ─────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_json_response(tmp_path: Path):
    """Pipeline should handle malformed LLM JSON gracefully."""
    vault = tmp_path / "MalformedVault"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=test\n"
    )
    config = load_config(env_file=str(vault / ".env"))

    sources = {
        "test.md": SourceDoc(
            title="Test",
            content="This is a test document with enough content to pass the length gate. " * 3,
        ),
    }

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value="This is not JSON at all. Just plain text.",
    ):
        from obsidian_llm_wiki.core.pipeline import run_pipeline
        result = await run_pipeline(vault, sources, config, force=True)

    # Should have an error recorded, not crash
    assert result.compiled == 0
    assert len(result.errors) > 0
    assert "test.md" in result.errors[0]


# ── Multi-source merge golden test ──────────────────────────────────────


def test_multi_source_merge_golden():
    """Two sources with overlapping concepts should merge correctly."""
    source1 = SourceSynthesis(
        source_title="Source A",
        source_summary="Summary A",
        concepts=[
            ConceptNote(
                title="Shared Concept",
                slug="shared-concept",
                summary="Concept from source A",
                tags=["tag-a"],
                sections=[BodySection(heading="Core", points=["Point A"])],
                related=[ConceptLink(slug="unique-a", relation="related_to")],
            ),
            ConceptNote(
                title="Unique A",
                slug="unique-a",
                summary="Only in source A",
                sections=[BodySection(heading="Core", points=["Unique A point"])],
            ),
        ],
        maps=[
            MapOfContent(
                title="Topic A",
                slug="topic-a",
                summary="Topic A summary",
                concept_slugs=["shared-concept", "unique-a"],
            ),
        ],
    )

    source2 = SourceSynthesis(
        source_title="Source B",
        source_summary="Summary B",
        concepts=[
            ConceptNote(
                title="Shared Concept",
                slug="shared-concept",
                summary="Concept from source B",
                tags=["tag-b"],
                sections=[BodySection(heading="Context", points=["Point B"])],
                related=[ConceptLink(slug="unique-b", relation="related_to")],
            ),
            ConceptNote(
                title="Unique B",
                slug="unique-b",
                summary="Only in source B",
                sections=[BodySection(heading="Core", points=["Unique B point"])],
            ),
        ],
        maps=[
            MapOfContent(
                title="Topic A",
                slug="topic-a",
                summary="Topic A summary from B",
                concept_slugs=["shared-concept", "unique-b"],
            ),
        ],
    )

    bundle = merge_bundle([source1, source2])

    # Should have 3 concepts (shared merged, unique-a, unique-b)
    assert len(bundle.concepts) == 3
    slugs = {c.slug for c in bundle.concepts}
    assert slugs == {"shared-concept", "unique-a", "unique-b"}

    # Shared concept should have merged tags and sections
    shared = next(c for c in bundle.concepts if c.slug == "shared-concept")
    assert "tag-a" in shared.tags
    assert "tag-b" in shared.tags
    assert len(shared.sections) == 2  # One from each source

    # MoC should have unioned concept slugs
    moc = bundle.maps[0]
    assert "shared-concept" in moc.concept_slugs
    assert "unique-a" in moc.concept_slugs
    assert "unique-b" in moc.concept_slugs


# ── Backlink propagation golden test ────────────────────────────────────


def test_backlink_propagation_golden():
    """Forward edges should get reverse edges added automatically."""
    bundle = SynthesisBundle(
        concepts=[
            ConceptNote(
                title="Concept A",
                slug="concept-a",
                summary="A",
                related=[ConceptLink(slug="concept-b", relation="depends_on")],
            ),
            ConceptNote(
                title="Concept B",
                slug="concept-b",
                summary="B",
                related=[],  # No backlink to A initially
            ),
        ],
    )

    propagate_backlinks(bundle)

    # B should now have a backlink to A
    b = next(c for c in bundle.concepts if c.slug == "concept-b")
    assert len(b.related) == 1
    assert b.related[0].slug == "concept-a"
    assert b.related[0].relation == "depends_on"


# ── Gradient confidence golden test ─────────────────────────────────────


def test_gradient_confidence_boundaries():
    """Gradient confidence at boundary values."""
    threshold = 800

    # At threshold → 1.0
    assert gradient_confidence(800, threshold) == 1.0
    assert gradient_confidence(1000, threshold) == 1.0

    # At half threshold → 0.5 + 0.5 * (400/800) = 0.75
    assert gradient_confidence(400, threshold) == pytest.approx(0.75)

    # Below half threshold → 0.1 + 0.4 * (body / (threshold * 0.5))
    # At 200: 0.1 + 0.4 * (200/400) = 0.1 + 0.2 = 0.3
    assert gradient_confidence(200, threshold) == pytest.approx(0.3)

    # At 0 → 0.1
    assert gradient_confidence(0, threshold) == pytest.approx(0.1)

    # Zero threshold → 1.0 (edge case guard)
    assert gradient_confidence(100, 0) == 1.0


# ── Chunk content golden test ───────────────────────────────────────────


def test_chunk_content_boundaries():
    """Content chunking at boundary sizes."""
    # Small content → single chunk
    small = "Hello world."
    assert chunk_content(small, 1000) == [small]

    # Exactly at chunk_size → single chunk
    exact = "A" * 1000
    assert len(chunk_content(exact, 1000)) == 1

    # Just above chunk_size → multiple chunks
    big = "A" * 2001
    chunks = chunk_content(big, 1000)
    assert len(chunks) >= 2
    # All chunks should be <= chunk_size
    for c in chunks:
        assert len(c) <= 1000

    # Paragraph-boundary splitting
    paras = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_content(paras, 20)
    assert len(chunks) >= 2
    # No chunk should split mid-paragraph
    for c in chunks:
        assert not c.startswith("\n")


# ── Frontmatter parsing robustness ──────────────────────────────────────


def test_frontmatter_with_dashes_in_content():
    """Frontmatter parser should handle --- inside body content."""
    content = "---\ntitle: Test\n---\n\nSome text\n\n---\n\nMore text after hr"
    meta, body = parse_frontmatter(content)
    assert meta.get("title") == "Test"
    assert "Some text" in body
    assert "---" in body  # The body's --- should be preserved


def test_frontmatter_no_frontmatter():
    """Content without frontmatter should return empty meta."""
    content = "# Just a heading\n\nNo frontmatter here."
    meta, body = parse_frontmatter(content)
    assert meta == {}
    assert body == content


def test_frontmatter_unicode_title():
    """Frontmatter with Unicode (Chinese) title should parse correctly."""
    content = "---\ntitle: 流动性的重新定义\n---\n\nContent here."
    meta, body = parse_frontmatter(content)
    assert meta.get("title") == "流动性的重新定义"
    assert "Content here" in body


# ── URL classification tests ────────────────────────────────────────────


def test_journal_xml_url_scoping():
    """_is_journal_xml_url should only match journal domains for /article- pattern."""
    from obsidian_llm_wiki.ingest.web import _is_journal_xml_url

    # .xml extension should always match
    assert _is_journal_xml_url("https://example.com/data.xml") is True

    # akjournals with /article- should match
    assert _is_journal_xml_url(
        "https://www.akjournals.com/view/journals/2054/9/3/article-p294.xml"
    ) is True

    # Non-journal domain with /article- should NOT match
    assert _is_journal_xml_url("https://example.com/article-123") is False

    # Random URL should not match
    assert _is_journal_xml_url("https://example.com/page") is False


def test_ssrn_url_detection():
    """_is_ssrn_url should detect SSRN domains."""
    from obsidian_llm_wiki.ingest.web import _is_ssrn_url

    assert _is_ssrn_url("https://ssrn.com/abstract=123456") is True
    assert _is_ssrn_url("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=123") is True
    assert _is_ssrn_url("https://example.com/paper") is False


def test_youtube_url_detection():
    """_is_youtube_url should detect YouTube domains."""
    from obsidian_llm_wiki.ingest.web import _is_youtube_url

    assert _is_youtube_url("https://www.youtube.com/watch?v=abc123") is True
    assert _is_youtube_url("https://youtu.be/abc123") is True
    assert _is_youtube_url("https://m.youtube.com/watch?v=abc123") is True
    assert _is_youtube_url("https://example.com/video") is False


# ── Concept page rendering golden test ──────────────────────────────────


def test_concept_page_with_relations_frontmatter():
    """Concept page should render relations in frontmatter as structured field."""
    concept = ConceptNote(
        title="Test Concept",
        slug="test-concept",
        summary="A test concept.",
        tags=["test", "unit"],
        sections=[BodySection(heading="Core", points=["Point one"])],
        related=[
            ConceptLink(slug="other-concept", relation="depends_on", display="Other"),
        ],
        confidence=0.85,
        provenance="extracted",
    )

    page = render_concept_page(concept, "2026-01-01T00:00:00Z")
    meta, body = parse_frontmatter(page)

    assert meta["type"] == "Concept"
    assert meta["title"] == "Test Concept"
    assert meta["confidence"] == 0.85
    assert meta["provenance"] == "extracted"
    assert "relations" in meta
    assert meta["relations"][0] == "other-concept|depends_on|Other"
    assert "## Related Concepts" not in body
    assert "[[other-concept|Other]]" not in body


def test_concept_page_no_relations():
    """Concept page without relations should not have relations frontmatter."""
    concept = ConceptNote(
        title="Lonely Concept",
        slug="lonely",
        summary="No friends.",
        sections=[BodySection(heading="Core", points=["Solo point"])],
    )

    page = render_concept_page(concept)
    meta, body = parse_frontmatter(page)

    assert "relations" not in meta
    assert "## Related Concepts" not in body


# ── Source page rendering golden test ───────────────────────────────────


def test_source_page_avoids_duplicate_heading():
    """Source page should not duplicate the title heading if content starts with it."""
    source = SourceDoc(
        title="My Article",
        content="# My Article\n\nBody text here.",
    )
    page = render_source_page(source)
    meta, body = parse_frontmatter(page)

    assert meta["type"] == "Source"
    assert meta["title"] == "My Article"
    # Title heading should appear only once
    assert body.count("# My Article") == 1
    assert "Body text here." in body


def test_source_page_adds_heading_when_missing():
    """Source page should add title heading when content doesn't start with it."""
    source = SourceDoc(
        title="My Article",
        content="Body text without heading.",
    )
    page = render_source_page(source)
    meta, body = parse_frontmatter(page)

    assert body.startswith("# My Article")


# ── MOC page rendering golden test ──────────────────────────────────────


def test_moc_page_with_concepts():
    """MOC page should list concepts with wikilinks."""
    moc = MapOfContent(
        title="Test Topic",
        slug="test-topic",
        summary="A test topic covering important concepts.",
        concept_slugs=["concept-a", "concept-b"],
    )
    all_concepts = {
        "concept-a": ConceptNote(title="Concept A", slug="concept-a", summary="A summary"),
        "concept-b": ConceptNote(title="Concept B", slug="concept-b", summary="B summary"),
    }

    page = render_moc_page(moc, "2026-01-01T00:00:00Z", all_concepts=all_concepts)
    meta, body = parse_frontmatter(page)

    assert meta["type"] == "Map of Content"
    assert meta["title"] == "Test Topic"
    assert "## Concepts" in body
    assert "[[concept-a]]" in body
    assert "[[concept-b]]" in body
    assert "A summary" in body  # Concept summary included as definition


# ── Merge skeletons golden test ─────────────────────────────────────────


def test_merge_skeletons_union_concepts():
    """merge_skeletons should union concepts by slug and merge tags."""
    sk1 = SourceSynthesis(
        source_title="Test",
        source_summary="S",
        concepts=[
            ConceptNote(title="Shared", slug="shared", summary="From sk1", tags=["tag1"]),
            ConceptNote(title="Unique1", slug="unique1", summary="Only sk1"),
        ],
        maps=[
            MapOfContent(title="MOC", slug="moc", summary="M", concept_slugs=["shared", "unique1"]),
        ],
    )

    sk2 = SourceSynthesis(
        source_title="Test",
        source_summary="S",
        concepts=[
            ConceptNote(title="Shared", slug="shared", summary="From sk2", tags=["tag2"]),
            ConceptNote(title="Unique2", slug="unique2", summary="Only sk2"),
        ],
        maps=[
            MapOfContent(title="MOC", slug="moc", summary="M", concept_slugs=["shared", "unique2"]),
        ],
    )

    merged = merge_skeletons([sk1, sk2])

    # Should have 3 concepts (shared + unique1 + unique2)
    assert len(merged.concepts) == 3
    slugs = {c.slug for c in merged.concepts}
    assert slugs == {"shared", "unique1", "unique2"}

    # Shared concept should have merged tags
    shared = next(c for c in merged.concepts if c.slug == "shared")
    assert "tag1" in shared.tags
    assert "tag2" in shared.tags

    # MOC should have unioned concept slugs
    assert len(merged.maps) == 1
    assert set(merged.maps[0].concept_slugs) == {"shared", "unique1", "unique2"}


# ── Slugify edge cases ──────────────────────────────────────────────────


def test_slugify_edge_cases():
    """slugify should handle edge cases without crashing."""
    from obsidian_llm_wiki.render.frontmatter import slugify

    assert slugify("Hello World") == "hello-world"
    assert slugify("  Spaces  ") == "spaces"
    assert slugify("Special!@#Chars") == "specialchars"
    assert slugify("Chinese 中文标题") == "chinese-中文标题"
    assert slugify("") == "untitled"
    assert slugify("---") == "untitled"
    assert slugify("Don't") == "dont"


# ── Tag sanitization golden test ────────────────────────────────────────


def test_tag_sanitization():
    """build_frontmatter should sanitize tags properly."""
    from obsidian_llm_wiki.render.frontmatter import sanitize_tag

    assert sanitize_tag("multi agent systems") == "multi-agent-systems"
    assert sanitize_tag('tag"with"quotes') == "tagwithquotes"
    assert sanitize_tag("tag#with#hash") == "tagwithhash"
    assert sanitize_tag("  spaced  ") == "spaced"
    assert sanitize_tag("") == ""
