"""Tests for bounded, format-aware document dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.models import SourceDoc


class _StreamingClient:
    def __init__(self, response: httpx.Response, **_kwargs) -> None:
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def stream(self, _method: str, _url: str):
        response = self.response

        class _Stream:
            def __enter__(self):
                return response

            def __exit__(self, *_args) -> None:
                response.close()

        return _Stream()


def _response(
    url: str,
    content: bytes,
    content_type: str,
    *,
    content_length: str | None = None,
) -> httpx.Response:
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = content_length
    return httpx.Response(200, content=content, headers=headers, request=httpx.Request("GET", url))


def test_document_limits_load_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_DOCUMENT_BYTES", "123")
    monkeypatch.setenv("MAX_DOCUMENT_CANDIDATES", "4")
    monkeypatch.setenv("PARSER_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("MAX_PARSER_STDOUT_BYTES", "99")
    monkeypatch.setenv("MAX_PARSER_STDERR_BYTES", "7")

    config = load_config()

    assert (
        config.max_document_bytes,
        config.max_document_candidates,
        config.parser_timeout_seconds,
        config.max_parser_stdout_bytes,
        config.max_parser_stderr_bytes,
    ) == (123, 4, 8, 99, 7)


def test_download_rejects_document_with_oversized_content_length():
    from obsidian_llm_wiki.ingest.documents import DocumentTooLargeError, download_document

    url = "https://example.com/paper.pdf"
    response = _response(url, b"", "application/pdf", content_length="11")
    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **kwargs: _StreamingClient(response, **kwargs),
        ),
        pytest.raises(DocumentTooLargeError, match="Content-Length"),
    ):
        download_document(url, config=Config(max_document_bytes=10))


def test_download_rejects_stream_that_exceeds_byte_limit():
    from obsidian_llm_wiki.ingest.documents import DocumentTooLargeError, download_document

    url = "https://example.com/paper.pdf"
    response = _response(url, b"%PDF-" + b"x" * 20, "application/pdf")
    response.headers.pop("content-length", None)
    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **kwargs: _StreamingClient(response, **kwargs),
        ),
        pytest.raises(DocumentTooLargeError, match="stream exceeded"),
    ):
        download_document(url, config=Config(max_document_bytes=10))


@pytest.mark.parametrize(
    ("content_type", "content", "message"),
    [
        ("text/html", b"<html>download page</html>", "MIME"),
        ("application/pdf", b"<html>download page</html>", "signature"),
    ],
)
def test_download_rejects_invalid_direct_pdf_mime_or_signature(
    content_type: str, content: bytes, message: str
):
    from obsidian_llm_wiki.ingest.documents import InvalidDocumentError, download_document

    url = "https://example.com/paper.pdf"
    response = _response(url, content, content_type)
    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **kwargs: _StreamingClient(response, **kwargs),
        ),
        pytest.raises(InvalidDocumentError, match=message),
    ):
        download_document(url)


@pytest.mark.parametrize(
    ("suffix", "content_type", "content"),
    [
        (".epub", "application/epub+zip", b"PK\x03\x04epub"),
        (".ppt", "application/vnd.ms-powerpoint", b"\xd0\xcf\x11\xe0ppt"),
        (
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            b"PK\x03\x04pptx",
        ),
        (".xls", "application/vnd.ms-excel", b"\xd0\xcf\x11\xe0xls"),
        (
            ".xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            b"PK\x03\x04xlsx",
        ),
    ],
)
def test_dispatch_routes_remote_supported_non_pdf_documents_to_liteparse(
    suffix: str, content_type: str, content: bytes
):
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    url = f"https://example.com/report{suffix}"
    response = _response(url, content, content_type)
    seen_paths: list[Path] = []

    def parse(path: str | Path, *, source_url: str | None = None, **_kwargs) -> SourceDoc:
        local_path = Path(path)
        seen_paths.append(local_path)
        assert local_path.suffix == suffix
        assert local_path.read_bytes() == content
        return SourceDoc(title="Parsed", content="Body", url=source_url or str(local_path))

    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **kwargs: _StreamingClient(response, **kwargs),
        ),
        patch("obsidian_llm_wiki.ingest.documents.parse_document", side_effect=parse),
    ):
        result = dispatch_document(url)

    assert result.url == url
    assert len(seen_paths) == 1
    assert not seen_paths[0].exists()


def test_direct_binary_url_does_not_fall_back_to_web_html():
    from obsidian_llm_wiki.ingest.extractors import extract

    url = "https://example.com/report.pdf"
    response = _response(url, b"<html>download page</html>", "text/html")
    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **kwargs: _StreamingClient(response, **kwargs),
        ),
        patch(
            "obsidian_llm_wiki.ingest.extractors.extract_web",
            side_effect=AssertionError("must not parse HTML"),
        ),
        pytest.raises(RuntimeError, match="MIME"),
    ):
        extract(url)


def test_document_candidates_are_same_site_and_capped():
    from obsidian_llm_wiki.ingest.documents import document_candidates

    candidates = document_candidates(
        """
        <a href="/one.pdf">one</a><a href="/two.pdf">two</a>
        <a href="https://mirror.example/three.pdf">three</a>
        """,
        "https://journal.example/article",
        max_candidates=1,
    )

    assert candidates == ["https://journal.example/one.pdf"]


def test_discovered_document_keeps_landing_candidate_redirect_and_download_provenance(
    tmp_path: Path,
):
    """A landing-page PDF keeps every retrieval hop after document parsing."""
    from obsidian_llm_wiki.ingest import documents
    from obsidian_llm_wiki.ingest.documents import DownloadedDocument

    landing_url = "https://example.com/articles/42"
    candidate_url = "https://example.com/files/42.pdf"
    resolved_url = "https://journal.example/downloads/42-final.pdf"
    landing_response = httpx.Response(
        200,
        text='<a href="/files/42.pdf">PDF</a>',
        request=httpx.Request("GET", landing_url),
    )
    document_path = tmp_path / "42.pdf"
    document_path.write_bytes(b"%PDF-pretend")
    downloaded = DownloadedDocument(
        path=document_path,
        source_url=candidate_url,
        resolved_url=resolved_url,
        content_type="application/pdf",
        suffix=".pdf",
    )

    class _LandingClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str, **_kwargs: object) -> httpx.Response:
            assert url == landing_url
            return landing_response

    def parse(path: Path, *, source_url: str, config: Config) -> SourceDoc:
        assert path == document_path
        assert source_url == landing_url
        return SourceDoc(title="Paper", content="Extracted document text", url=source_url)

    config = Config()
    with (
        patch(
            "obsidian_llm_wiki.ingest.documents.httpx.Client",
            lambda **_kwargs: _LandingClient(),
        ),
        patch(
            "obsidian_llm_wiki.ingest.documents.download_document",
            return_value=downloaded,
        ) as download,
        patch("obsidian_llm_wiki.ingest.documents._parse_local_document", side_effect=parse),
    ):
        result = documents.extract_discovered_document(landing_url, timeout=9, config=config)

    download.assert_called_once_with(
        candidate_url,
        config=config,
        timeout=9,
        required_host="example.com",
    )
    assert result.provenance.requested_url == landing_url
    assert result.provenance.extracted_url == candidate_url
    assert result.provenance.resolved_url == resolved_url
    assert result.provenance.content_type == "application/pdf"
    assert result.provenance.document_format == "pdf"
    assert result.provenance.extractor_chain == ("document-discovery", "document-download")
    assert len(result.provenance.content_sha256) == 64
    persisted = tmp_path / "source.md"
    from obsidian_llm_wiki.ingest.sources import load_source_file
    from obsidian_llm_wiki.render.obsidian import render_source_page

    persisted.write_text(render_source_page(result, "2026-07-13T12:00:00Z"), encoding="utf-8")
    reloaded = load_source_file(persisted)
    assert reloaded is not None
    assert reloaded.provenance == result.provenance
    assert not document_path.exists()
