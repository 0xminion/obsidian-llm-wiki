"""PDF extractor — extracts text from PDF files using pymupdf (fitz).

Dependency (optional): ``pymupdf``.
Install with: ``pip install okf-pipeline[pdf]``

Handles both local file paths and remote URLs (httpx download → fitz from bytes).
For arxiv URLs (``/abs/`` or ``/pdf/`` without .pdf suffix), see the URL
normalization in ``extractors/__init__.py::extract()`` which rewrites
``/abs/`` → ``/pdf/`` before dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.pdf")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import fitz  # pymupdf  # type: ignore[import-untyped]

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# ── Matching ────────────────────────────────────────────────────────────

_ARXIV_HOSTS = frozenset(("arxiv.org", "www.arxiv.org", "export.arxiv.org"))


def _is_pdf(parsed, raw: str) -> bool:
    """Match .pdf file paths, file:// PDF URLs, and arxiv /pdf/ URLs.

    Matches:
    - Local/remote paths ending in ``.pdf`` (case-insensitive)
    - arxiv.org ``/pdf/XXXX.YYYYY`` URLs (no .pdf suffix in URL)
    """
    path_str = raw if not raw.startswith("file://") else parsed.path
    if path_str.lower().endswith(".pdf"):
        return True
    # arxiv PDF URLs: /pdf/XXXX.YYYYY (no .pdf suffix in URL path)
    host = (parsed.hostname or "").lower()
    return bool(host in _ARXIV_HOSTS and "/pdf/" in (parsed.path or ""))


# ── Registration (only if deps available) ────────────────────────────────

if _DEPS_AVAILABLE:

    @register_extractor(_is_pdf)
    def extract_pdf(raw_url_or_path: str) -> SourceDoc:
        """Extract text from a PDF file or remote URL.

        For local files, opens directly with pymupdf.
        For remote URLs (http/https), downloads via httpx then opens from bytes.

        Uses pymupdf (fitz) to iterate pages and concatenate text.
        Title is derived from PDF metadata or the first page's heading.
        """
        is_remote = raw_url_or_path.startswith(("http://", "https://"))

        if is_remote:
            return _extract_remote_pdf(raw_url_or_path)
        else:
            return _extract_local_pdf(raw_url_or_path)

    def _extract_local_pdf(raw_path: str) -> SourceDoc:
        """Extract text from a local PDF file."""
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"PDF file not found: {raw_path}")

        doc = fitz.open(str(path))  # type: ignore[union-attr]
        try:
            return _extract_text_from_doc(doc, str(path))
        finally:
            doc.close()  # type: ignore[union-attr]

    def _extract_remote_pdf(url: str) -> SourceDoc:
        """Download a remote PDF and extract text from bytes."""
        with httpx.Client(**make_client_kwargs(follow_redirects=True, timeout=60)) as client:
            resp = client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content

        if not pdf_bytes or len(pdf_bytes) < 100:
            raise RuntimeError(f"Downloaded empty/short PDF from {url}")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[union-attr]
        try:
            return _extract_text_from_doc(doc, url)
        finally:
            doc.close()  # type: ignore[union-attr]

    def _extract_text_from_doc(doc, source_url: str) -> SourceDoc:
        """Common extraction logic for both local and remote PDFs."""
        # ── Title from metadata ────────────────────────────────────
        title = ""
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        if not title:
            # Derive from URL/path stem
            parsed = urlparse(source_url)
            stem = Path(parsed.path).stem if parsed.path else Path(source_url).stem
            title = stem

        # ── Extract text from all pages ────────────────────────────
        pages: list[str] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

        content = "\n\n".join(pages)
        if not content.strip():
            raise RuntimeError(f"PDF has no extractable text: {source_url}")

        return SourceDoc(
            title=title,
            content=content,
            url=source_url,
        )
