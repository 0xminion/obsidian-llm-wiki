"""Optional local LiteParse CLI integration for structured documents.

LiteParse is deliberately invoked through its ``lit`` CLI so this module has
no import-time dependency on the optional package. Install it with
``pip install liteparse`` to enable the fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs


class LiteParseUnavailableError(RuntimeError):
    """Raised when the optional LiteParse CLI is not installed."""


_DOCUMENT_SUFFIXES = frozenset((".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".epub"))
_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/epub+zip": ".epub",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "text/html": ".html",
}


def parse_document(path: str | Path, *, source_url: str | None = None) -> SourceDoc:
    """Parse a local document with LiteParse and return Markdown in a SourceDoc.

    Raises:
        LiteParseUnavailableError: When the optional ``lit`` executable is unavailable.
        RuntimeError: When LiteParse cannot parse the document.
    """
    document = Path(path)
    lit = shutil.which("lit")
    if not lit:
        raise LiteParseUnavailableError(
            "LiteParse CLI is unavailable; install it with `pip install liteparse`"
        )

    try:
        proc = subprocess.run(
            [
                lit,
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
    except FileNotFoundError as exc:
        raise LiteParseUnavailableError(
            "LiteParse CLI is unavailable; install it with `pip install liteparse`"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"LiteParse timed out parsing {document}") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip() or "no stderr"
        raise RuntimeError(f"LiteParse exited {proc.returncode}: {detail[:300]}")

    content = proc.stdout.strip()
    if not content:
        raise RuntimeError("LiteParse returned empty Markdown")

    return SourceDoc(
        title=_markdown_title(content) or document.stem,
        content=content,
        url=source_url or str(document),
    )


def extract_document_fallback(url: str, timeout: int) -> SourceDoc:
    """Download a direct or discovered document URL and parse it with LiteParse.

    The original page URL remains the SourceDoc URL when a citation link is
    discovered, preserving provenance for callers. Every temporary download is
    removed after parsing, including when LiteParse is unavailable.
    """
    with httpx.Client(
        **make_client_kwargs(timeout=timeout, follow_redirects=True), headers=BROWSER_HEADERS
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        if _is_document_response(url, response):
            return _parse_download(response.content, url, response.headers, source_url=url)

        candidates = _document_candidates(response.text, url)
        if not candidates:
            raise RuntimeError("No direct document or citation document link found")

        errors: list[str] = []
        for candidate in candidates:
            try:
                document_response = client.get(candidate)
                document_response.raise_for_status()
                return _parse_download(
                    document_response.content,
                    candidate,
                    document_response.headers,
                    source_url=url,
                )
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

    raise RuntimeError("LiteParse document fallback failed: " + "; ".join(errors))


def _parse_download(
    content: bytes, document_url: str, headers: httpx.Headers, *, source_url: str
) -> SourceDoc:
    """Write a document response to a temporary file and invoke LiteParse."""
    if not content:
        raise RuntimeError(f"Downloaded empty document from {document_url}")

    suffix = _document_suffix(document_url, headers)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(content)
            temp_path = handle.name
        return parse_document(temp_path, source_url=source_url)
    finally:
        if temp_path is not None:
            with suppress(OSError):
                os.unlink(temp_path)


def _is_document_response(url: str, response: httpx.Response) -> bool:
    """Whether an HTTP response is a direct downloadable document."""
    return Path(urlparse(url).path).suffix.lower() in _DOCUMENT_SUFFIXES or _content_type(
        response.headers
    ) in _CONTENT_TYPE_SUFFIXES


def _document_suffix(url: str, headers: httpx.Headers) -> str:
    """Choose a useful temporary-file suffix from the URL or content type."""
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix:
        return suffix
    content_type = _content_type(headers)
    if content_type in _CONTENT_TYPE_SUFFIXES:
        return _CONTENT_TYPE_SUFFIXES[content_type]
    if "word" in content_type:
        return ".docx"
    if "spreadsheet" in content_type or "excel" in content_type:
        return ".xlsx"
    if "presentation" in content_type or "powerpoint" in content_type:
        return ".pptx"
    return ".bin"


def _content_type(headers: httpx.Headers) -> str:
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _document_candidates(html: str, page_url: str) -> list[str]:
    """Find citation metadata and PDF links in an HTML landing page."""
    parser = _DocumentLinkParser()
    parser.feed(html)
    parser.close()
    return list(dict.fromkeys(urljoin(page_url, candidate) for candidate in parser.candidates))


class _DocumentLinkParser(HTMLParser):
    """Extract citation document URLs without adding an HTML-parser dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        href = attributes.get("href", "")
        if tag == "meta":
            name = attributes.get("name", attributes.get("property", "")).lower()
            if name in {"citation_pdf_url", "citation_fulltext_html_url"}:
                self.candidates.append(attributes.get("content", ""))
        elif tag in {"a", "link"} and href:
            relation = " ".join(
                (attributes.get("rel", ""), attributes.get("type", ""), attributes.get("title", ""))
            ).lower()
            if "pdf" in href.lower() or "pdf" in relation:
                self.candidates.append(href)


def _markdown_title(markdown: str) -> str:
    """Return the first level-one Markdown heading, if LiteParse produced one."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
