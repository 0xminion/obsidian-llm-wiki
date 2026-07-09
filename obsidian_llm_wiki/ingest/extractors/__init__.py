"""Extractor registry — routes URLs and file paths to the right extractor.

Usage::

    from obsidian_llm_wiki.ingest.extractors import extract
    source = extract("https://youtube.com/watch?v=...")
    source = extract("~/Downloads/paper.pdf")

The registry matches on URL domain, URL scheme, and file extension.
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

__all__ = ["extract", "register_extractor"]

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

    Falls back to ``extract_web`` for unknown URLs.

    Args:
        raw_url: A URL (https://...) or a local file path (~/Downloads/paper.pdf).

    Returns:
        SourceDoc with title, content, and url.

    Raises:
        RuntimeError: If all extraction strategies fail.
    """
    # Check if it's a local file path.
    if _looks_like_file_path(raw_url):
        return _extract_file(raw_url)

    # Parse as URL.
    parsed = urlparse(raw_url)

    # Try registered extractors.
    for match_fn, extractor_fn in _EXTRACTORS:
        try:
            if match_fn(parsed, raw_url):
                logger.debug("Routing '%s' to %s", raw_url, extractor_fn.__name__)
                return extractor_fn(raw_url)
        except Exception:
            continue

    # Fallback: web extraction.
    return extract_web(raw_url)


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
        try:
            if match_fn(urlparse(""), str(path)):
                logger.debug("Routing file '%s' to %s", file_path, extractor_fn.__name__)
                return extractor_fn(str(path))
        except Exception:
            continue

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
    from obsidian_llm_wiki.ingest.extractors import pdf as _pdf  # noqa: F401

with suppress(ImportError):
    from obsidian_llm_wiki.ingest.extractors import docx as _docx  # noqa: F401
