"""Regression tests for rejecting unusable extracted sources."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.ingest import extractors
from obsidian_llm_wiki.render.bilingual import normalize_bilingual_titles_and_slugs


def test_quality_gate_rejects_broken_markdown_link_as_title() -> None:
    source = SourceDoc(
        title="](https://x.com/ericliujt)",
        content="Substantive body text. " * 40,
        url="https://x.com/ericliujt/status/1",
    )

    assert extractors._check_extraction_quality(source) == (False, "invalid source title")


def test_quality_gate_rejects_compact_url_fragment_title() -> None:
    source = SourceDoc(
        title="httpsxcomericliujt",
        content="Substantive body text. " * 40,
        url="https://x.com/ericliujt/status/1",
    )

    assert extractors._check_extraction_quality(source) == (False, "invalid source title")


def test_quality_gate_blocks_rejected_source_from_pipeline() -> None:
    source = SourceDoc(
        title="](https://x.com/ericliujt)",
        content="Substantive body text. " * 40,
        url="https://x.com/ericliujt/status/1",
    )

    with pytest.raises(RuntimeError, match="invalid source title"):
        extractors._require_usable_source(source)


def test_quality_gate_rejects_x_article_preview() -> None:
    source = SourceDoc(
        title="A plausible X article title",
        content=(
            "Article\n\nA short preview of a purported article.\n\n"
            "](https://x.com/i/article/123456789)"
        ),
        url="https://x.com/example/status/1",
    )

    assert extractors._check_extraction_quality(source) == (False, "X article preview stub")


def test_quality_gate_rejects_navigation_chrome() -> None:
    source = SourceDoc(
        title="A public policy report",
        content=(
            "skip to main content\nNavigation\nAdvanced Searches\n"
            "Browse\nBack to top\nLoading...\n" * 80
        ),
        url="https://www.congress.gov/crs-product/LSB11406",
    )

    assert extractors._check_extraction_quality(source) == (False, "navigation chrome")


def test_twitter_prefers_full_hosted_defuddle_over_cli_preview(monkeypatch) -> None:
    preview = SourceDoc(
        title="Article title",
        content="Article\n\nPreview only.\n\n](https://x.com/i/article/123)",
        url="https://x.com/example/status/1",
    )
    full = SourceDoc(
        title="Article title",
        content="# Article title\n\n" + "Evidence-backed paragraph. " * 100,
        url="https://x.com/example/status/1",
    )
    monkeypatch.setattr(extractors.twitter, "_extract_via_defuddle", lambda _url: preview)
    monkeypatch.setattr(extractors.twitter, "_extract_via_defuddle_md", lambda _url: full)

    source = extractors.twitter.extract_twitter("https://x.com/example/status/1")

    assert source is full


def test_bilingual_entry_normalization_is_idempotent_for_inline_latin() -> None:
    title = "泡沫持久度取决于PVP的浓度——PVP的核心是引出人性的贪婪和不甘心"
    synthesis = SourceSynthesis(
        source_title=title,
        source_summary="A source summary.",
        concepts=[
            ConceptNote(
                title="Bubble Persistence",
                slug="bubble-persistence",
                summary="A concept summary.",
            )
        ],
    )
    bundle = SynthesisBundle(sources=[synthesis])

    normalize_bilingual_titles_and_slugs(bundle)
    first = synthesis.source_title
    normalize_bilingual_titles_and_slugs(bundle)

    assert synthesis.source_title == first
    assert first.count(title) == 1
