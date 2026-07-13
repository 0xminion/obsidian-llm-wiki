"""Tests for ingest/extractors/jats.py — _try_publisher_pdf fallback.

Covers:
  - URL transformation for akjournals (XML → PDF path)
  - Non-akjournals host returns None
  - Path mismatch (no /view/journals/) returns None
  - dispatch_document is called with the transformed PDF URL
  - dispatch_document failure returns None
"""

from __future__ import annotations

from unittest.mock import patch

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors.jats import _try_publisher_pdf

# ── URL transformation ───────────────────────────────────────────────────


def test_try_publisher_pdf_akjournals_url_transformation():
    """For an akjournals URL, _try_publisher_pdf should construct the PDF URL
    by replacing /view/ with /downloadpdf/view/ and .xml with .pdf, then call
    dispatch_document with that URL."""
    raw_url = "https://akjournals.com/view/journals/2054/9/3/article-p294.xml"
    expected_pdf_url = "https://akjournals.com/downloadpdf/view/journals/2054/9/3/article-p294.pdf"

    mock_source = SourceDoc(title="Test PDF", content="PDF content here", url=expected_pdf_url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
        return_value=mock_source,
    ) as mock_dispatch:
        result = _try_publisher_pdf(raw_url)

    assert result is not None
    assert result.content == "PDF content here"
    # dispatch_document should have been called with the transformed PDF URL.
    mock_dispatch.assert_called_once_with(expected_pdf_url)


def test_try_publisher_pdf_akjournals_with_www():
    """www.akjournals.com host also triggers the PDF fallback."""
    raw_url = "https://www.akjournals.com/view/journals/2054/9/3/article-p294.xml"
    expected_pdf_url = "https://www.akjournals.com/downloadpdf/view/journals/2054/9/3/article-p294.pdf"

    mock_source = SourceDoc(title="Test", content="Content", url=expected_pdf_url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
        return_value=mock_source,
    ) as mock_dispatch:
        result = _try_publisher_pdf(raw_url)

    assert result is not None
    mock_dispatch.assert_called_once_with(expected_pdf_url)


def test_try_publisher_pdf_non_akjournals_returns_none():
    """Non-akjournals host should return None without calling dispatch_document."""
    raw_url = "https://example.com/view/journals/2054/9/3/article-p294.xml"

    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
    ) as mock_dispatch:
        result = _try_publisher_pdf(raw_url)

    assert result is None
    mock_dispatch.assert_not_called()


def test_try_publisher_pdf_path_mismatch_returns_none():
    """akjournals URL without /view/journals/ in the path returns None."""
    raw_url = "https://akjournals.com/abstract/12345"

    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
    ) as mock_dispatch:
        result = _try_publisher_pdf(raw_url)

    assert result is None
    mock_dispatch.assert_not_called()


def test_try_publisher_pdf_dispatch_failure_returns_none():
    """When dispatch_document raises, _try_publisher_pdf returns None."""
    raw_url = "https://akjournals.com/view/journals/2054/9/3/article-p294.xml"

    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
        side_effect=RuntimeError("Download failed"),
    ):
        result = _try_publisher_pdf(raw_url)

    assert result is None


def test_try_publisher_pdf_xml_suffix_replaced():
    """The .xml suffix is correctly replaced with .pdf in the PDF URL."""
    raw_url = "https://akjournals.com/view/journals/123/4/5/article-p999.xml"
    expected_pdf_url = "https://akjournals.com/downloadpdf/view/journals/123/4/5/article-p999.pdf"

    mock_source = SourceDoc(title="T", content="C", url=expected_pdf_url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
        return_value=mock_source,
    ) as mock_dispatch:
        _try_publisher_pdf(raw_url)

    mock_dispatch.assert_called_once_with(expected_pdf_url)


def test_try_publisher_pdf_http_scheme_preserved():
    """The scheme (http vs https) from the original URL is preserved."""
    raw_url = "http://akjournals.com/view/journals/2054/9/3/article-p294.xml"
    expected_pdf_url = "http://akjournals.com/downloadpdf/view/journals/2054/9/3/article-p294.pdf"

    mock_source = SourceDoc(title="T", content="C", url=expected_pdf_url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document",
        return_value=mock_source,
    ) as mock_dispatch:
        result = _try_publisher_pdf(raw_url)

    assert result is not None
    mock_dispatch.assert_called_once_with(expected_pdf_url)
