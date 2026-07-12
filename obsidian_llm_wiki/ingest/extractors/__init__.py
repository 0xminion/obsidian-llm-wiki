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
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.web import extract_web

logger = logging.getLogger("obswiki.ingest.extractors")

__all__ = ["extract", "register_extractor"]

# ── Registry ────────────────────────────────────────────────────────────

# Each entry: (match_fn, extractor_fn)
# match_fn takes a parsed URL + raw input string, returns True if this
# extractor should handle it.
_EXTRACTORS: list[tuple[Callable[..., bool], Callable[..., SourceDoc]]] = []

# File extensions that require download-then-extract for remote URLs.
_REMOTE_FILE_EXTENSIONS = frozenset((".pdf", ".docx", ".doc", ".pptx", ".xlsx"))


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
    # Check if it's a local file path.
    if _looks_like_file_path(raw_url):
        return _extract_file(raw_url)

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
                return extractor_fn(raw_url)
            except Exception as exc:
                logger.warning(
                    "Extractor %s failed for %s: %s; trying fallback",
                    extractor_fn.__name__, raw_url, exc,
                )
                matched_errors.append((extractor_fn.__name__, exc))
                continue

    if matched_errors:
        raise RuntimeError(
            f"All specialized extractors failed for {raw_url}: "
            + "; ".join(f"{n}: {e}" for n, e in matched_errors)
        )

    # Remote binary file (PDF/DOCX in URL path) → download to temp, then extract.
    if _is_remote_file(raw_url):
        return _extract_remote_file(raw_url)

    # Fallback: web extraction.
    return extract_web(raw_url)


def _looks_like_file_path(raw_url: str) -> bool:
    """Check if the input looks like a local file path rather than a URL."""
    if raw_url.startswith(("http://", "https://", "ftp://", "file://")):
        return False

    # Expand and check if file exists.
    expanded = Path(os.path.expanduser(raw_url))
    return expanded.is_file()


def _is_remote_file(raw_url: str) -> bool:
    """Check if a URL points to a binary file requiring download-then-extract."""
    if not raw_url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(raw_url)
    path = parsed.path.lower()
    # Strip query string already done by urlparse, but check extension.
    return any(path.endswith(ext) for ext in _REMOTE_FILE_EXTENSIONS)


def _extract_remote_file(url: str) -> SourceDoc:
    """Download a remote binary file to a temp file, then extract it.

    Routes to the appropriate registered extractor based on file extension.
    Falls back to web extraction if no extractor matches.
    """
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()

    # Download to temp file.
    tmp_path: str | None = None
    try:
        from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            with httpx.Client(**make_client_kwargs(follow_redirects=True, timeout=60)) as client:
                resp = client.get(url)
                resp.raise_for_status()
                tmp.write(resp.content)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path) if tmp_path else 0
        logger.debug("Downloaded '%s' to '%s' (%d bytes)", url, tmp_path, file_size)

        # Route to registered extractors for this file type.
        for match_fn, extractor_fn in _EXTRACTORS:
            if match_fn(urlparse(""), tmp_path):
                logger.debug("Routing remote file '%s' to %s", url, extractor_fn.__name__)
                source = extractor_fn(tmp_path)
                # Preserve the original URL, not the temp file path.
                source.url = url
                return source

        # Plain text/markdown: read directly.
        if suffix in (".txt", ".md", ".markdown", ".rst"):
            content = Path(tmp_path).read_text(encoding="utf-8")
            title = Path(parsed.path).stem
            return SourceDoc(title=title, content=content, url=url)

        # No extractor matched — fall back to web (extracts the HTML download page).
        logger.warning("No extractor for remote file type '%s', trying web extraction", suffix)
        return extract_web(url)

    finally:
        if tmp_path is not None:
            with suppress(OSError):
                os.unlink(tmp_path)


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
