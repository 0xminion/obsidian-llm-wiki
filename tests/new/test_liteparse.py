"""Tests for the optional LiteParse document fallback."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from obsidian_llm_wiki.core.models import SourceDoc


def test_parse_document_runs_lit_cli_and_returns_markdown(tmp_path: Path):
    """LiteParse invokes the documented quiet Markdown CLI and builds a SourceDoc."""
    from obsidian_llm_wiki.ingest.liteparse import parse_document

    document = tmp_path / "paper.pdf"
    document.write_bytes(b"%PDF-pretend")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="# Parsed Paper\n\nParsed document body.",
        stderr="",
    )

    with (
        patch("obsidian_llm_wiki.ingest.liteparse.shutil.which", return_value="/usr/bin/lit"),
        patch("obsidian_llm_wiki.ingest.liteparse.subprocess.run", return_value=completed) as run,
    ):
        source = parse_document(document, source_url="https://example.com/paper.pdf")

    assert source.title == "Parsed Paper"
    assert source.content == "# Parsed Paper\n\nParsed document body."
    assert source.url == "https://example.com/paper.pdf"
    run.assert_called_once_with(
        [
            "/usr/bin/lit",
            "parse",
            str(document),
            "--format",
            "markdown",
            "--image-mode",
            "off",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_parse_document_reports_missing_cli_without_hiding_later_fallbacks(tmp_path: Path):
    """An uninstalled optional CLI produces a clear nonfatal exception."""
    from obsidian_llm_wiki.ingest.liteparse import LiteParseUnavailableError, parse_document

    with (
        patch("obsidian_llm_wiki.ingest.liteparse.shutil.which", return_value=None),
        pytest.raises(LiteParseUnavailableError, match="pip install liteparse"),
    ):
        parse_document(tmp_path / "paper.pdf")


def test_document_fallback_downloads_direct_pdf_parses_and_removes_temp_file():
    """A direct document URL is downloaded to a temporary file for LiteParse."""
    from obsidian_llm_wiki.ingest import liteparse

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, _url):
            return httpx.Response(
                200,
                content=b"%PDF-direct-document",
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", "https://example.com/report.pdf"),
            )

    parsed_paths: list[Path] = []

    def fake_parse(path: str | Path, *, source_url: str | None = None) -> SourceDoc:
        parsed_path = Path(path)
        parsed_paths.append(parsed_path)
        assert parsed_path.read_bytes() == b"%PDF-direct-document"
        return SourceDoc(title="Downloaded PDF", content="Parsed body", url=source_url)

    with (
        patch("obsidian_llm_wiki.ingest.liteparse.httpx.Client", FakeClient),
        patch("obsidian_llm_wiki.ingest.liteparse.parse_document", side_effect=fake_parse),
    ):
        source = liteparse.extract_document_fallback("https://example.com/report.pdf", timeout=9)

    assert source == SourceDoc(
        title="Downloaded PDF",
        content="Parsed body",
        url="https://example.com/report.pdf",
    )
    assert len(parsed_paths) == 1
    assert not parsed_paths[0].exists()


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


def test_document_fallback_discovers_citation_pdf_from_html_page():
    """Citation PDF metadata is preferred over a generic landing-page HTML parser."""
    from obsidian_llm_wiki.ingest import liteparse

    page_url = "https://journal.example/article"
    pdf_url = "https://journal.example/files/article.pdf"
    html = '<meta name="citation_pdf_url" content="/files/article.pdf">'

    class FakeClient:
        def __init__(self, **_kwargs):
            self.urls: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url):
            self.urls.append(url)
            if url == page_url:
                return httpx.Response(200, text=html, request=httpx.Request("GET", url))
            assert url == pdf_url
            return httpx.Response(
                200,
                content=b"%PDF-citation",
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", url),
            )

    with (
        patch("obsidian_llm_wiki.ingest.liteparse.httpx.Client", FakeClient),
        patch(
            "obsidian_llm_wiki.ingest.liteparse.parse_document",
            return_value=SourceDoc(title="Citation PDF", content="Body", url=page_url),
        ) as parse,
    ):
        source = liteparse.extract_document_fallback(page_url, timeout=9)

    assert source.url == page_url
    assert parse.call_args.kwargs["source_url"] == page_url
    assert Path(parse.call_args.args[0]).suffix == ".pdf"


def test_extract_web_tries_document_fallback_before_trafilatura():
    """LiteParse document parsing is the first local fallback after hosted Defuddle."""
    from obsidian_llm_wiki.ingest import web

    expected = SourceDoc(title="Parsed PDF", content="LiteParse text", url="https://example.com/page")
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
