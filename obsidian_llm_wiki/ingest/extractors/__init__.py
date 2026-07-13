"""Extractor registry — routes URLs and file paths to the right extractor.

Usage::

    from obsidian_llm_wiki.ingest.extractors import extract
    source = extract("https://youtube.com/watch?v=...")
    source = extract("https://arxiv.org/pdf/1706.03762.pdf")
    source = extract("~/Downloads/paper.pdf")

The registry matches on URL domain, URL scheme, and file extension.
Remote binary files (PDF, DOCX) are downloaded to a temp file first.
Unknown URLs fall back to ``extract_web`` (trafilatura).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.web import extract_web

logger = logging.getLogger("obswiki.ingest.extractors")

__all__ = ["ExtractorNotApplicableError", "extract", "register_extractor"]


class ExtractorNotApplicableError(RuntimeError):
    """Raised when an extractor matched a URL pattern but the content isn't its type.

    URL patterns are necessarily coarse — ``/feed`` and ``.xml`` identify a
    *possible* podcast feed, not a definite one. An extractor that can only tell
    from the fetched body raises this to disclaim the URL. Unlike a genuine
    extraction failure, it does not trip the fail-closed policy in ``extract``:
    dispatch continues to the remaining extractors and, if none claim the URL,
    to ``extract_web``.
    """


# ── Registry ────────────────────────────────────────────────────────────

# Each entry: (match_fn, extractor_fn)
# match_fn takes a parsed URL + raw input string, returns True if this
# extractor should handle it.
_EXTRACTORS: list[tuple[Callable[..., bool], Callable[..., SourceDoc]]] = []


def register_extractor(
    match_fn: Callable[..., bool],
) -> Callable[[Callable[..., SourceDoc]], Callable[..., SourceDoc]]:
    """Decorator to register an extractor.

    Usage::

        @register_extractor(lambda url, raw: url.hostname in ("youtube.com", "youtu.be"))
        def extract_youtube(raw_url: str) -> SourceDoc:
            ...
    """

    def decorator(
        fn: Callable[..., SourceDoc],
    ) -> Callable[..., SourceDoc]:
        _EXTRACTORS.append((match_fn, fn))
        return fn

    return decorator


# ── Dispatch ────────────────────────────────────────────────────────────


def extract(raw_url: str) -> SourceDoc:
    """Extract content from a URL or file path using the registered extractors.

    For remote PDF/DOCX files (URLs ending in .pdf/.docx), downloads the file
    to a temp location first, then routes to the appropriate extractor.

    Falls back to ``extract_web`` for unknown URLs.

    Args:
        raw_url: A URL (https://...) or a local file path (~/Downloads/paper.pdf).

    Returns:
        SourceDoc with title, content, and url.

    Raises:
        RuntimeError: If all extraction strategies fail.
    """
    # Local and direct remote binary documents are centrally dispatched before
    # generic extractors.  This prevents a failed download page from becoming
    # bogus HTML source content.
    if _looks_like_file_path(raw_url):
        from obsidian_llm_wiki.ingest.documents import dispatch_document, is_document_path

        if is_document_path(raw_url):
            return _stamp_extracted_source(dispatch_document(raw_url), raw_url, "document_dispatch")
        return _stamp_extracted_source(_extract_file(raw_url), raw_url, "local_file")

    from obsidian_llm_wiki.ingest.documents import dispatch_document, is_direct_document_url

    if is_direct_document_url(raw_url):
        return _stamp_extracted_source(dispatch_document(raw_url), raw_url, "document_dispatch")

    # Parse as URL.
    parsed = urlparse(raw_url)

    # Try registered extractors first (YouTube, PDF, JATS, etc.).
    # If a specialized extractor MATCHES the URL but fails, we fail closed —
    # do NOT fall through to extract_web, which would produce garbage from
    # cookie-walled pages (YouTube footer chrome, CF challenge HTML, etc.).
    # JATS does its own XML→HTML fallback internally; PDF downloads directly.
    matched_errors: list[tuple[str, Exception]] = []
    for match_fn, extractor_fn in _EXTRACTORS:
        if match_fn(parsed, raw_url):
            logger.debug("Routing '%s' to %s", raw_url, extractor_fn.__name__)
            try:
                return _stamp_extracted_source(
                    extractor_fn(raw_url), raw_url, extractor_fn.__name__
                )
            except ExtractorNotApplicableError as exc:
                # The extractor disclaimed the URL after inspecting the content.
                # This is not a failure, so it must not fail closed — keep the
                # ordinary fallback chain (other extractors, then extract_web).
                logger.debug(
                    "Extractor %s disclaimed %s: %s",
                    extractor_fn.__name__,
                    raw_url,
                    exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Extractor %s failed for %s: %s; trying fallback",
                    extractor_fn.__name__,
                    raw_url,
                    exc,
                )
                matched_errors.append((extractor_fn.__name__, exc))
                continue

    if matched_errors:
        raise RuntimeError(
            f"All specialized extractors failed for {raw_url}: "
            + "; ".join(f"{n}: {e}" for n, e in matched_errors)
        )

    # Fallback: web extraction.
    return _stamp_extracted_source(extract_web(raw_url), raw_url, "web_fallback")


def _stamp_extracted_source(source: SourceDoc, raw_url: str, extractor: str) -> SourceDoc:
    """Attach baseline immutable provenance at the public extractor boundary."""
    from obsidian_llm_wiki.ingest.provenance import stamp_source

    return stamp_source(source, requested_url=raw_url, extractor=extractor)


def _looks_like_file_path(raw_url: str) -> bool:
    """Check if the input looks like a local file path rather than a URL."""
    if raw_url.startswith(("http://", "https://", "ftp://", "file://")):
        return False

    # Expand and check if file exists.
    expanded = Path(os.path.expanduser(raw_url))
    return expanded.is_file()


def _extract_file(file_path: str) -> SourceDoc:
    """Route a local file to the appropriate extractor based on extension."""
    path = Path(os.path.expanduser(file_path))
    suffix = path.suffix.lower()

    # Try registered extractors for file extensions.
    for match_fn, extractor_fn in _EXTRACTORS:
        if match_fn(urlparse(""), str(path)):
            logger.debug("Routing file '%s' to %s", file_path, extractor_fn.__name__)
            return extractor_fn(str(path))

    # Plain text/markdown: read directly.
    if suffix in (".txt", ".md", ".markdown", ".rst"):
        content = path.read_text(encoding="utf-8")
        title = path.stem
        return SourceDoc(title=title, content=content, url=str(path))

    raise RuntimeError(f"No extractor available for file type: {suffix}")


# ── Import extractors to trigger registration ───────────────────────────
# These imports register their extractors via the @register_extractor
# decorator. Import errors are silently swallowed — the extractor just
# won't be available. Each module handles its own dependency checking.

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import youtube as _youtube  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import scientific as _scientific  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import pdf as _pdf  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import docx as _docx  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import jats as _jats  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import podcast as _podcast  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import twitter as _twitter  # noqa: F401
