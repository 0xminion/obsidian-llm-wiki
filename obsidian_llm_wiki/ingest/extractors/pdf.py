"""PDF extractor — extracts text from PDF files using pymupdf (fitz).

Dependency (optional): ``pymupdf``.
Install with: ``pip install okf-pipeline[pdf]``
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

logger = logging.getLogger("obswiki.ingest.extractors.pdf")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import fitz  # pymupdf  # type: ignore[import-untyped]

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ── Matching ────────────────────────────────────────────────────────────


def _is_pdf(parsed, raw: str) -> bool:
    """Match .pdf file paths and file:// PDF URLs."""
    path_str = raw if not raw.startswith("file://") else parsed.path
    return path_str.lower().endswith(".pdf")


# ── Registration (only if deps available) ────────────────────────────────

if _DEPS_AVAILABLE:

    @register_extractor(_is_pdf)
    def extract_pdf(raw_path: str) -> SourceDoc:
        """Extract text from a PDF file.

        Uses pymupdf (fitz) to iterate pages and concatenate text.
        Title is derived from PDF metadata or the first page's heading.
        """
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"PDF file not found: {raw_path}")

        doc = fitz.open(str(path))  # type: ignore[union-attr]
        try:
            # ── Title from metadata ────────────────────────────────────
            title = ""
            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip()
            if not title:
                title = path.stem

            # ── Extract text from all pages ────────────────────────────
            pages: list[str] = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pages.append(f"--- Page {page_num + 1} ---\n{text.strip()}")

            content = "\n\n".join(pages)
            if not content.strip():
                raise RuntimeError(f"PDF has no extractable text: {raw_path}")

            return SourceDoc(
                title=title,
                content=content,
                url=str(path),
            )
        finally:
            doc.close()  # type: ignore[union-attr]