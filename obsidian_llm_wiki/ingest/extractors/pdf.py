"""PDF extractor — PyMuPDF-first with a bounded LiteParse fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

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
    """Extract a local or remote PDF through the shared document boundary."""
    if raw_url_or_path.startswith(("http://", "https://")):
        return _extract_remote_pdf(raw_url_or_path)
    return _extract_local_pdf(raw_url_or_path)


def _extract_local_pdf(
    raw_path: str,
    *,
    source_url: str | None = None,
    config: Config | None = None,
) -> SourceDoc:
    """Extract a local PDF, invoking LiteParse after any PyMuPDF failure."""
    path = Path(raw_path)
    display_url = source_url or str(path)
    if not path.is_file():
        raise RuntimeError(f"PDF file not found: {raw_path}")
    if not _DEPS_AVAILABLE:
        return _fallback_to_liteparse(
            path, display_url, RuntimeError("PyMuPDF is unavailable"), config=config
        )

    try:
        doc = fitz.open(str(path))  # type: ignore[union-attr]
        try:
            return _extract_text_from_doc(doc, display_url)
        finally:
            doc.close()
    except Exception as exc:
        return _fallback_to_liteparse(path, display_url, exc, config=config)


def _extract_remote_pdf(url: str) -> SourceDoc:
    """Route remote PDFs through the shared bounded document dispatcher."""
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    return dispatch_document(url)


def _fallback_to_liteparse(
    path: Path,
    source_url: str,
    pymupdf_error: Exception,
    *,
    config: Config | None = None,
) -> SourceDoc:
    """Attempt optional LiteParse and retain both failure reasons if it fails."""
    logger.info("PyMuPDF extraction failed for %s; trying LiteParse: %s", source_url, pymupdf_error)
    try:
        if config is None:
            return _extract_with_liteparse(path, source_url)
        return _extract_with_liteparse(path, source_url, config=config)
    except Exception as liteparse_error:
        raise RuntimeError(
            f"PyMuPDF extraction failed for {source_url}: {pymupdf_error}; "
            f"LiteParse fallback failed: {liteparse_error}"
        ) from liteparse_error


def _extract_with_liteparse(
    path: Path,
    source_url: str,
    *,
    config: Config | None = None,
) -> SourceDoc:
    """Import optional LiteParse only when the PDF text fallback is needed."""
    from obsidian_llm_wiki.ingest.liteparse import parse_document

    if config is None:
        return parse_document(path, source_url=source_url)
    return parse_document(path, source_url=source_url, config=config)


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
