"""PDF extractor — PyMuPDF-first with a bounded LiteParse fallback."""

from __future__ import annotations

import logging
import re
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
        # Convert GitHub blob URLs to raw URLs for direct download.
        # GitHub /blob/ URLs are HTML pages, not direct file links.
        raw_url_or_path = _github_blob_to_raw(raw_url_or_path)
        return _extract_remote_pdf(raw_url_or_path)
    return _extract_local_pdf(raw_url_or_path)


def _github_blob_to_raw(url: str) -> str:
    """Convert a GitHub blob URL to a raw.githubusercontent.com URL.

    ``https://github.com/owner/repo/blob/branch/path/file.pdf``
    → ``https://raw.githubusercontent.com/owner/repo/branch/path/file.pdf``

    Returns the original URL unchanged if it's not a GitHub blob URL.
    """
    if "/blob/" not in url:
        return url
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ("github.com", "www.github.com"):
        return url
    # Replace /blob/ with / and swap the host
    path = parsed.path.replace("/blob/", "/", 1)
    # URL-decode the path (GitHub blob URLs often have %26 etc.)
    from urllib.parse import unquote
    path = unquote(path)
    return f"https://raw.githubusercontent.com{path}"


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
    if not _is_plausible_pdf_title(title):
        # PDF metadata is often a paper number (e.g. NBER "35589") or empty.
        # Fall back to the URL stem, then try to extract a title from page 1.
        parsed = urlparse(source_url)
        title = Path(parsed.path).stem if parsed.path else Path(source_url).stem
        if not _is_plausible_pdf_title(title):
            title = _extract_title_from_first_page(doc) or title

    pages: list[str] = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            pages.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

    content = "\n\n".join(pages)
    if not content.strip():
        raise RuntimeError(f"PDF has no extractable text: {source_url}")

    return SourceDoc(title=title, content=content, url=source_url)


def _is_plausible_pdf_title(title: str) -> bool:
    """Return True when *title* looks like a real document title, not a paper number.

    PDF metadata frequently contains the NBER working paper number, a DOI suffix,
    or a bare filename — none of which are useful source titles.  Reject:
      * Pure numbers (NBER paper numbers like "35589")
      * Very short strings (< 5 chars)
      * Strings that are all non-word characters
    """
    cleaned = (title or "").strip()
    if len(cleaned) < 5:
        return False
    # Pure digits / paper numbers like "35589" or "2606.16852"
    return not re.fullmatch(r"[\d.]+", cleaned)


def _extract_title_from_first_page(doc) -> str | None:
    """Scan the first page text for a plausible article title.

    Looks at the first few non-empty lines and picks the first one that looks
    like a title (longer than 10 chars, not a page header/number).
    """
    if not doc or len(doc) == 0:
        return None
    try:
        text = doc[0].get_text("text")
    except Exception:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:10]:
        # Skip page numbers, headers, copyright notices
        if len(line) < 10:
            continue
        if re.fullmatch(r"[\d.]+", line):
            continue
        if line.lower().startswith(("copyright", "doi:", "http", "www.")):
            continue
        # Found a plausible title
        return line[:200]
    return None
