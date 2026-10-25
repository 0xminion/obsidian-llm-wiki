"""Integration-style selection tests for public scientific full text."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from obsidian_llm_wiki.core.models import SourceDoc


def test_scientific_preflight_prefers_public_html_over_valid_landing_abstract() -> None:
    """A same-publisher cited full text wins before a generic abstract extractor."""
    from obsidian_llm_wiki.ingest import web

    landing_url = "https://journals.example.edu/article/10.1000/example"
    fulltext_url = "https://journals.example.edu/article/full-text"
    landing_html = '<meta name="citation_fulltext_html_url" content="/article/full-text">'
    abstract = SourceDoc(
        title="Study abstract",
        content="This is a valid generic extraction of an abstract, but not the paper.",
        url=landing_url,
    )
    full_text = SourceDoc(
        title="Complete study",
        content="Complete public full text. " * 40,
        url=fulltext_url,
    )

    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            side_effect=[landing_html, "<html><body>full article</body></html>"],
        ) as fetch_html,
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific.extract_scientific_html",
            return_value=full_text,
        ),
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._extract_pdf",
        ) as extract_pdf,
        patch.object(web, "_extract_defuddle_md", return_value=abstract) as generic_extract,
    ):
        result = web.extract_web(landing_url, timeout=11)

    assert result is full_text
    fetch_html.assert_has_calls([
        ((landing_url, 11),),
        ((fulltext_url, 11),),
    ])
    generic_extract.assert_not_called()
    extract_pdf.assert_not_called()
    assert result.provenance.requested_url == landing_url
    assert result.provenance.extracted_url == fulltext_url
    assert result.provenance.extractor_chain[-1] == "scientific_public_html"
    assert result.provenance.diagnostics[-1] == "scientific selection: official html candidate"


def test_thin_cited_html_falls_back_to_official_pdf_after_html_priority() -> None:
    """A cited HTML abstract is rejected even when metadata lists PDF first."""
    from obsidian_llm_wiki.ingest.extractors import scientific

    landing_url = "https://journals.example.edu/article/10.1000/example"
    html_url = "https://journals.example.edu/article/full-text"
    pdf_url = "https://journals.example.edu/downloads/article.pdf"
    landing_html = """
        <meta name="citation_pdf_url" content="/downloads/article.pdf">
        <meta name="citation_fulltext_html_url" content="/article/full-text">
    """
    thin_abstract = SourceDoc(
        title="Study abstract",
        content="Abstract: this is a short but valid abstract extraction.",
        url=html_url,
    )
    pdf_full_text = SourceDoc(
        title="Complete study PDF",
        content="Text extracted from the official PDF.",
        url=pdf_url,
    )

    with (
        patch.object(
            scientific,
            "_fetch_public_html",
            side_effect=[landing_html, "<html><body>abstract</body></html>"],
        ),
        patch.object(scientific, "extract_scientific_html", return_value=thin_abstract),
        patch.object(scientific, "_extract_pdf", return_value=pdf_full_text) as extract_pdf,
    ):
        result = scientific.extract_discovered_scientific_document(landing_url, timeout=12)

    assert result is pdf_full_text
    extract_pdf.assert_called_once_with(pdf_url)


def test_unavailable_public_candidates_fall_back_to_landing_extractor() -> None:
    """Candidate failures return control to the generic extractor with the landing URL."""
    from obsidian_llm_wiki.ingest import web

    landing_url = "https://journals.example.edu/article/10.1000/unavailable"
    fulltext_url = "https://journals.example.edu/article/full-text"
    landing_html = '<meta name="citation_fulltext_html_url" content="/article/full-text">'
    # Model a usable body on the landing page, not a stub: web candidates now
    # meet the same 500-character admission contract as public extract().
    landing_source = SourceDoc(
        "Landing page", "Public landing-page fallback body with study evidence. " * 20, landing_url,
    )

    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            side_effect=[landing_html, RuntimeError("public full text unavailable")],
        ) as fetch_html,
        patch.object(web, "_extract_defuddle_md", return_value=landing_source) as generic_extract,
    ):
        result = web.extract_web(landing_url, timeout=13)

    assert result is landing_source
    fetch_html.assert_has_calls([((landing_url, 13),), ((fulltext_url, 13),)])
    generic_extract.assert_called_once_with(landing_url, 13)


def test_offsite_citation_candidate_is_rejected() -> None:
    """A citation URL on another host cannot turn an official page into a mirror fetch."""
    from obsidian_llm_wiki.ingest.extractors.scientific import discover_scientific_documents

    candidates = discover_scientific_documents(
        '<meta name="citation_fulltext_html_url" content="https://mirror.example/paper">',
        "https://journals.example.edu/article/10.1000/example",
    )

    assert candidates == []


def test_ssrn_preflight_failure_preserves_semantic_scholar_fallthrough() -> None:
    """SSRN tries its metadata fallback but does not publish it as full text."""
    from obsidian_llm_wiki.ingest import web

    ssrn_url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522"
    abstract_source = SourceDoc(
        "Semantic Scholar Abstract",
        "Public metadata and abstract.",
        ssrn_url,
    )
    unavailable = RuntimeError("unavailable")

    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            side_effect=unavailable,
        ) as fetch_html,
        patch.object(web, "_extract_defuddle_md", side_effect=unavailable),
        patch.object(web, "_extract_liteparse_document", side_effect=unavailable),
        patch.object(web, "_extract_trafilatura", side_effect=unavailable),
        patch.object(web, "_extract_defuddle", side_effect=unavailable),
        patch.object(web, "_extract_wayback", side_effect=unavailable),
        patch(
            "obsidian_llm_wiki.ingest.alt_source.extract_via_semantic_scholar",
            return_value=abstract_source,
        ) as semantic_scholar,
        pytest.raises(RuntimeError, match="semantic_scholar: Extraction quality gate failed"),
    ):
        web.extract_web(ssrn_url, timeout=14)

    fetch_html.assert_called_once_with(ssrn_url, 14)
    semantic_scholar.assert_called_once_with(ssrn_url, 14)


def test_ordinary_blog_skips_scientific_preflight() -> None:
    """A generic blog succeeds without the scientific landing-page network request."""
    from obsidian_llm_wiki.ingest import web

    blog_url = "https://blog.example.com/posts/ordinary-article"
    # This is a successful article fixture, so it must clear the shared gate.
    expected = SourceDoc("Ordinary post", "Generic web article body and evidence. " * 20, blog_url)
    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific.extract_discovered_scientific_document"
        ) as discover_document,
        patch.object(web, "_extract_defuddle_md", return_value=expected),
    ):
        result = web.extract_web(blog_url, timeout=15)

    assert result is expected
    discover_document.assert_not_called()
