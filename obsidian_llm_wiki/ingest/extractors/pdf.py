"""PDF extractor — extracts text with PyMuPDF, then optional LiteParse.

PyMuPDF is preferred for direct text PDFs. When it is unavailable, cannot
open a document, or returns no text (for example a scanned PDF), the optional
LiteParse CLI is tried before extraction fails.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.pdf")

try:
    import fitz  # pymupdf  # type: ignore[import-untyped]

    _DEPS_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    _DEPS_AVAILABLE = False

_ARXIV_HOSTS = frozenset(("arxiv.org", "www.arxiv.org", "export.arxiv.org"))


def _is_pdf(parsed, raw: str) -> bool:
    """Match .pdf paths, file:// PDF URLs, and arxiv /pdf/ URLs."""
    path_str = raw if not raw.startswith("file://") else parsed.path
    if path_str.lower().endswith(".pdf"):
        return True
    host = (parsed.hostname or "").lower()
    return bool(host in _ARXIV_HOSTS and "/pdf/" in (parsed.path or ""))


@register_extractor(_is_pdf)
def extract_pdf(raw_url_or_path: str) -> SourceDoc:
    """Extract a local or remote PDF, falling back to LiteParse if needed."""
    if raw_url_or_path.startswith(("http://", "https://")):
        return _extract_remote_pdf(raw_url_or_path)
    return _extract_local_pdf(raw_url_or_path)


def _extract_local_pdf(raw_path: str) -> SourceDoc:
    """Extract a local PDF, invoking LiteParse after any PyMuPDF failure."""
    path = Path(raw_path)
    if not path.is_file():
        raise RuntimeError(f"PDF file not found: {raw_path}")
    if not _DEPS_AVAILABLE:
        return _fallback_to_liteparse(path, str(path), RuntimeError("PyMuPDF is unavailable"))

    try:
        doc = fitz.open(str(path))  # type: ignore[union-attr]
        try:
            return _extract_text_from_doc(doc, str(path))
        finally:
            doc.close()
    except Exception as exc:
        return _fallback_to_liteparse(path, str(path), exc)


def _extract_remote_pdf(url: str) -> SourceDoc:
    """Download a PDF once and use LiteParse if PyMuPDF cannot extract it."""
    with httpx.Client(**make_client_kwargs(follow_redirects=True, timeout=60)) as client:
        response = client.get(url)
        response.raise_for_status()
        pdf_bytes = response.content

    if not pdf_bytes or len(pdf_bytes) < 100:
        raise RuntimeError(f"Downloaded empty/short PDF from {url}")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(pdf_bytes)
            temp_path = handle.name
        path = Path(temp_path)

        if not _DEPS_AVAILABLE:
            return _fallback_to_liteparse(path, url, RuntimeError("PyMuPDF is unavailable"))

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[union-attr]
            try:
                return _extract_text_from_doc(doc, url)
            finally:
                doc.close()
        except Exception as exc:
            return _fallback_to_liteparse(path, url, exc)
    finally:
        if temp_path is not None:
            with suppress(OSError):
                os.unlink(temp_path)


def _fallback_to_liteparse(path: Path, source_url: str, pymupdf_error: Exception) -> SourceDoc:
    """Attempt the optional parser and retain both failure reasons if it cannot run."""
    logger.info("PyMuPDF extraction failed for %s; trying LiteParse: %s", source_url, pymupdf_error)
    try:
        return _extract_with_liteparse(path, source_url)
    except Exception as liteparse_error:
        raise RuntimeError(
            f"PyMuPDF extraction failed for {source_url}: {pymupdf_error}; "
            f"LiteParse fallback failed: {liteparse_error}"
        ) from liteparse_error


def _extract_with_liteparse(path: Path, source_url: str) -> SourceDoc:
    """Import the optional LiteParse integration only when the fallback is needed."""
    from obsidian_llm_wiki.ingest.liteparse import parse_document

    return parse_document(path, source_url=source_url)


def _extract_text_from_doc(doc, source_url: str) -> SourceDoc:
    """Common PyMuPDF extraction logic for local and remote PDFs."""
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    if not title:
        parsed = urlparse(source_url)
        title = Path(parsed.path).stem if parsed.path else Path(source_url).stem

    pages: list[str] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            pages.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

    content = "\n\n".join(pages)
    if not content.strip():
        raise RuntimeError(f"PDF has no extractable text: {source_url}")

    return SourceDoc(title=title, content=content, url=source_url)
