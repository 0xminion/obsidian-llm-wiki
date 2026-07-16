"""Tests for the optional LiteParse document fallback."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.core.models import SourceDoc


def test_parse_document_runs_lit_cli_and_returns_text(tmp_path: Path):
    """LiteParse invokes the compatible quiet text CLI and builds a SourceDoc."""
    from obsidian_llm_wiki.ingest.liteparse import parse_document

    document = tmp_path / "paper.pdf"
    document.write_bytes(b"%PDF-pretend")
    command = [
        "/usr/bin/lit",
        "parse",
        str(document),
        "--format",
        "text",
        "--quiet",
    ]
    with (
        patch("obsidian_llm_wiki.ingest.liteparse.shutil.which", return_value="/usr/bin/lit"),
        patch(
            "obsidian_llm_wiki.ingest.liteparse._run_liteparse",
            return_value=(0, b"# Parsed Paper\n\nParsed document body.", b""),
        ) as run,
    ):
        source = parse_document(document, source_url="https://example.com/paper.pdf")

    assert source.title == "Parsed Paper"
    assert source.content == "# Parsed Paper\n\nParsed document body."
    assert source.url == "https://example.com/paper.pdf"
    assert run.call_args.args[0] == command
    assert run.call_args.args[1].parser_timeout_seconds == 120


def test_liteparse_pipe_capture_is_bounded():
    """Parser diagnostics cannot retain more than their configured byte cap."""
    from obsidian_llm_wiki.ingest.liteparse import _BoundedPipe

    pipe = _BoundedPipe(io.BytesIO(b"abcdefgh"), 3)
    pipe.start()
    pipe.join()

    assert pipe.value == b"abc"


def test_parse_document_reports_missing_cli_without_hiding_later_fallbacks(tmp_path: Path):
    """An uninstalled optional CLI produces a clear nonfatal exception."""
    from obsidian_llm_wiki.ingest.liteparse import LiteParseUnavailableError, parse_document

    with (
        patch("obsidian_llm_wiki.ingest.liteparse.shutil.which", return_value=None),
        pytest.raises(LiteParseUnavailableError, match="pip install liteparse"),
    ):
        parse_document(tmp_path / "paper.pdf")


def test_document_fallback_delegates_to_safe_discovery():
    """Web fallback uses the same dispatcher as discovered document candidates."""
    from obsidian_llm_wiki.ingest import liteparse

    expected = SourceDoc(
        title="Downloaded PDF", content="Parsed body", url="https://example.com/report.pdf"
    )
    with patch(
        "obsidian_llm_wiki.ingest.documents.extract_discovered_document", return_value=expected
    ) as discover:
        assert (
            liteparse.extract_document_fallback("https://example.com/report.pdf", timeout=9)
            == expected
        )

    discover.assert_called_once_with("https://example.com/report.pdf", 9)


def test_document_candidates_include_citation_html_and_pdf_link_metadata():
    """Citation fulltext HTML and PDF link metadata are both usable candidates."""
    from obsidian_llm_wiki.ingest.liteparse import _document_candidates

    candidates = _document_candidates(
        """
        <meta name="citation_fulltext_html_url" content="/fulltext/article">
        <link rel="alternate" type="application/pdf" href="/downloads/article">
        """,
        "https://journal.example/article",
    )

    assert candidates == [
        "https://journal.example/fulltext/article",
        "https://journal.example/downloads/article",
    ]


def test_document_candidates_reject_offsite_links():
    """Off-site PDF links must not be followed — prevents routing through mirrors."""
    from obsidian_llm_wiki.ingest.liteparse import _document_candidates

    candidates = _document_candidates(
        """
        <meta name="citation_pdf_url" content="/local/paper.pdf">
        <a href="https://evil-mirror.example/paper.pdf">Mirror</a>
        """,
        "https://journal.example/article",
    )

    assert candidates == ["https://journal.example/local/paper.pdf"]


def test_document_candidates_skip_empty_meta_content():
    """Empty content attributes do not produce the page URL itself as a candidate."""
    from obsidian_llm_wiki.ingest.liteparse import _document_candidates

    candidates = _document_candidates(
        '<meta name="citation_pdf_url" content="">',
        "https://journal.example/article",
    )

    assert candidates == []


def test_document_fallback_discovers_citation_pdf_from_html_page():
    """Landing-page fallback delegates candidate fetching to the safe dispatcher."""
    from obsidian_llm_wiki.ingest import liteparse

    page_url = "https://journal.example/article"
    expected = SourceDoc(title="Citation PDF", content="Body", url=page_url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.extract_discovered_document", return_value=expected
    ) as discover:
        assert liteparse.extract_document_fallback(page_url, timeout=9) == expected

    discover.assert_called_once_with(page_url, 9)


def test_extract_web_tries_document_fallback_before_trafilatura():
    """LiteParse document parsing is the first local fallback after hosted Defuddle."""
    from obsidian_llm_wiki.ingest import web

    expected = SourceDoc(
        title="Parsed PDF", content="LiteParse text", url="https://example.com/page"
    )
    with (
        patch.object(web, "_extract_defuddle_md", side_effect=RuntimeError("hosted failure")),
        patch.object(web, "_extract_liteparse_document", return_value=expected),
        patch.object(web, "_extract_trafilatura", side_effect=AssertionError("wrong order")),
    ):
        assert web.extract_web("https://example.com/page") == expected


def test_local_pdf_uses_liteparse_when_pymupdf_fails(tmp_path: Path):
    """A PyMuPDF open failure falls back to the optional LiteParse parser."""
    from obsidian_llm_wiki.ingest.extractors import pdf

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-broken")
    expected = SourceDoc(title="Recovered", content="LiteParse body", url=str(path))

    class FailingFitz:
        @staticmethod
        def open(*_args, **_kwargs):
            raise RuntimeError("pymupdf failure")

    with (
        patch.object(pdf, "fitz", FailingFitz),
        patch.object(pdf, "_extract_with_liteparse", return_value=expected) as fallback,
    ):
        assert pdf._extract_local_pdf(str(path)) == expected

    fallback.assert_called_once_with(path, str(path))


def test_local_pdf_uses_liteparse_when_pymupdf_returns_empty_text(tmp_path: Path):
    """A scanned PDF with no PyMuPDF text also reaches LiteParse."""
    from obsidian_llm_wiki.ingest.extractors import pdf

    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-scanned")
    expected = SourceDoc(title="Recovered", content="LiteParse OCR text", url=str(path))

    class EmptyDoc:
        metadata = {}

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return self

        def get_text(self, _mode):
            return ""

        def close(self):
            return None

    class EmptyTextFitz:
        @staticmethod
        def open(*_args, **_kwargs):
            return EmptyDoc()

    with (
        patch.object(pdf, "fitz", EmptyTextFitz),
        patch.object(pdf, "_extract_with_liteparse", return_value=expected) as fallback,
    ):
        assert pdf._extract_local_pdf(str(path)) == expected

    fallback.assert_called_once_with(path, str(path))


def test_remote_pdf_routes_through_shared_document_dispatcher():
    """Remote PDFs use the bounded shared dispatcher before any parser runs."""
    from obsidian_llm_wiki.ingest.extractors import pdf

    url = "https://example.com/corrupt.pdf"
    expected = SourceDoc(title="Recovered", content="LiteParse recovered text", url=url)
    with patch(
        "obsidian_llm_wiki.ingest.documents.dispatch_document", return_value=expected
    ) as dispatch:
        assert pdf._extract_remote_pdf(url) == expected

    dispatch.assert_called_once_with(url)
