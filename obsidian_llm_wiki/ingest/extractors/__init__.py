"""Extractor registry — routes URLs and file paths to the right extractor.

Usage::

    from obsidian_llm_wiki.ingest.extractors import extract
    source = extract("https://youtube.com/watch?v=...")
    source = extract("https://arxiv.org/pdf/1706.03762.pdf")
    source = extract("~/Downloads/paper.pdf")

The registry matches on URL domain, URL scheme, and file extension.
Remote binary files (PDF, DOCX) are downloaded to a temp file first.
Unknown URLs fall back to ``extract_web`` (trafilatura).

When ``DEEP_SEARCH_FALLBACK=1`` is set and the primary extraction fails or
produces a stub (< 500 chars / quality gate fails), a last-resort *deep
search fallback* queries accessible scholarly sources — Semantic Scholar,
OpenAlex, arXiv, Crossref — for the same title and uses the best accessible
alternative.  This is off by default to keep CI deterministic and avoid
network calls during tests.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.web import extract_web

logger = logging.getLogger("obswiki.ingest.extractors")

__all__ = [
    "ExtractorNotApplicableError",
    "extract",
    "register_extractor",
    "_check_extraction_quality",
    "_require_usable_source",
    "_deep_search_fallback",
    "_deep_search_enabled",
]


# ── Deep search fallback (last-resort for stubs / failed extractions) ──


def _deep_search_enabled() -> bool:
    """Return True when the deep search fallback is enabled via env.

    Enabled by setting ``DEEP_SEARCH_FALLBACK=1`` (or any truthy value other
    than ``0``/empty/``false``).  Off by default so CI and offline tests stay
    deterministic — the fallback makes live network calls.
    """
    val = os.environ.get("DEEP_SEARCH_FALLBACK", "").strip().lower()
    return val not in ("", "0", "false", "no", "off")


def _title_from_url(raw_url: str) -> str:
    """Derive a best-effort search query from a URL when no title is known.

    Strips the scheme, leading ``www.``, common file extensions, and query
    strings, then collapses separators to spaces.  The result is a short,
    title-like string suitable for a scholarly search API.
    """
    parsed = urlparse(raw_url)
    path = parsed.path or raw_url
    # Drop scheme/host if urlparse didn't catch a bare string.
    if "://" in path:
        path = path.split("://", 1)[1]
    path = path.split("/", 1)[-1] if "/" in path else path
    # Strip common document extensions and trailing slashes.
    path = re.sub(r"\.(pdf|html?|xml|php|cfm|asp|jsp)(\?.*)?$", "", path, flags=re.IGNORECASE)
    path = path.rstrip("/")
    # Take the final path segment — usually the article slug.
    if "/" in path:
        path = path.rsplit("/", 1)[-1]
    # Replace separators with spaces.
    query = re.sub(r"[-_]+", " ", path).strip()
    if not query:
        query = parsed.hostname or raw_url
    return query


def _deep_search_semantic_scholar(title: str, timeout: int) -> SourceDoc | None:
    """Search Semantic Scholar by title and return abstract + metadata.

    Uses the public Graph API (no key required for low-volume reads).
    Returns ``None`` (not raises) when the search fails — the caller collects
    all alternatives and picks the best.
    """
    import urllib.parse

    import httpx

    from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

    search_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={urllib.parse.quote(title)}&limit=1"
        "&fields=title,abstract,year,authors,externalIds,url"
    )
    try:
        with httpx.Client(**make_client_kwargs(timeout=timeout, follow_redirects=True)) as client:
            resp = client.get(search_url, headers=BROWSER_HEADERS)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
        papers = data.get("data", []) or []
        if not papers:
            return None
        paper = papers[0]
        result_title = paper.get("title", "") or title
        abstract = paper.get("abstract", "") or ""
        year = paper.get("year", "") or ""
        authors = paper.get("authors", []) or []
        author_names = ", ".join(a.get("name", "") for a in authors if a.get("name"))
        ext = paper.get("externalIds", {}) or {}
        doi = ext.get("DOI", "")
        arxiv = ext.get("ArXiv", "")
        paper_url = paper.get("url", "") or ""
        content_parts = [f"Title: {result_title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if doi:
            content_parts.append(f"DOI: {doi}")
        if arxiv:
            content_parts.append(f"arXiv: {arxiv}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        content = "\n".join(content_parts).strip()
        if len(content) < 100:
            return None
        return SourceDoc(title=result_title, content=content, url=paper_url or "")
    except Exception as exc:
        logger.debug("Semantic Scholar deep search failed for '%s': %s", title, exc)
        return None


def _deep_search_openalex(title: str, timeout: int) -> SourceDoc | None:
    """Search OpenAlex by title and return abstract + metadata.

    OpenAlex is a fully-open scholarly catalog (no API key required).
    """
    import urllib.parse

    import httpx

    from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

    search_url = (
        "https://api.openalex.org/works"
        f"?search={urllib.parse.quote(title)}&per-page=1"
    )
    try:
        with httpx.Client(**make_client_kwargs(timeout=timeout, follow_redirects=True)) as client:
            resp = client.get(search_url, headers=BROWSER_HEADERS)
            resp.raise_for_status()
            data = resp.json()
        results = data.get("results", []) or []
        if not results:
            return None
        work = results[0]
        result_title = work.get("title", "") or title
        # OpenAlex abstracts are inverted-index; reconstruct if present.
        abstract_inverted = work.get("abstract_inverted_index")
        abstract = ""
        if isinstance(abstract_inverted, dict) and abstract_inverted:
            positions: list[tuple[int, str]] = []
            for word, idxs in abstract_inverted.items():
                for idx in idxs:
                    positions.append((idx, word))
            positions.sort()
            abstract = " ".join(w for _, w in positions)
        year = ""
        pub = work.get("publication_year")
        if pub:
            year = str(pub)
        authorships = work.get("authorships", []) or []
        author_names = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships[:10]
            if a.get("author", {}).get("display_name")
        )
        doi = work.get("doi", "") or ""
        ids = work.get("ids", {}) or {}
        openalex_url = ids.get("openalex", "") or work.get("id", "") or ""
        content_parts = [f"Title: {result_title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if doi:
            content_parts.append(f"DOI: {doi}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        content = "\n".join(content_parts).strip()
        if len(content) < 100:
            return None
        return SourceDoc(title=result_title, content=content, url=openalex_url)
    except Exception as exc:
        logger.debug("OpenAlex deep search failed for '%s': %s", title, exc)
        return None


def _deep_search_arxiv(title: str, timeout: int) -> SourceDoc | None:
    """Search arXiv by title and return abstract + metadata.

    arXiv OAI-PMH API is fully open.  Useful for preprints not yet indexed by
    Semantic Scholar / OpenAlex.
    """
    import urllib.parse

    import httpx

    from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

    search_url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=ti:{urllib.parse.quote(title)}"
        "&max_results=1"
    )
    try:
        with httpx.Client(**make_client_kwargs(timeout=timeout, follow_redirects=True)) as client:
            resp = client.get(search_url, headers=BROWSER_HEADERS)
            resp.raise_for_status()
            xml = resp.text
        # Lightweight XML parse — avoids a hard dep on defusedxml for this path.
        entry_match = re.search(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        if not entry_match:
            return None
        entry = entry_match.group(1)

        def _tag(tag: str, text: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
            return m.group(1).strip() if m else ""

        # arXiv uses Atom namespaces; strip prefixes.
        clean = re.sub(r"<(/?)[a-zA-Z0-9]+:", r"<\1", entry)
        result_title = _tag("title", clean) or title
        summary = _tag("summary", clean)
        published = _tag("published", clean)
        year = published[:4] if published else ""
        id_url = _tag("id", clean)
        authors_xml = re.findall(r"<name>(.*?)</name>", clean, re.DOTALL)
        author_names = ", ".join(a.strip() for a in authors_xml if a.strip())
        content_parts = [f"Title: {result_title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if summary:
            content_parts.extend(["", "Abstract:", summary])
        content = "\n".join(content_parts).strip()
        if len(content) < 100:
            return None
        return SourceDoc(title=result_title, content=content, url=id_url)
    except Exception as exc:
        logger.debug("arXiv deep search failed for '%s': %s", title, exc)
        return None


def _deep_search_crossref(title: str, timeout: int) -> SourceDoc | None:
    """Search Crossref by title and return abstract + metadata.

    Crossref REST API is open (mailto in User-Agent is polite but not required).
    """
    import urllib.parse

    import httpx

    from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
    from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

    search_url = (
        "https://api.crossref.org/works"
        f"?query.title={urllib.parse.quote(title)}&rows=1"
    )
    headers = dict(BROWSER_HEADERS)
    headers["User-Agent"] = (
        "obsidian-llm-wiki/3.0 (https://github.com/nousresearch; mailto:dev@nousresearch.com)"
    )
    try:
        with httpx.Client(**make_client_kwargs(timeout=timeout, follow_redirects=True)) as client:
            resp = client.get(search_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("message", {}).get("items", []) or []
        if not items:
            return None
        item = items[0]
        result_title = " ".join(item.get("title", []) or []) or title
        abstract = item.get("abstract", "") or ""
        # Crossref abstracts are JATS XML — strip tags.
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        year = ""
        date_parts = item.get("published", {}) or item.get("issued", {})
        if isinstance(date_parts, dict):
            dp = date_parts.get("date-parts", [[]])
            if dp and dp[0]:
                year = str(dp[0][0])
        authors = item.get("author", []) or []
        author_names = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors[:10]
            if a.get("family")
        )
        doi = item.get("DOI", "") or ""
        url = item.get("URL", "") or (f"https://doi.org/{doi}" if doi else "")
        content_parts = [f"Title: {result_title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if doi:
            content_parts.append(f"DOI: {doi}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        content = "\n".join(content_parts).strip()
        if len(content) < 100:
            return None
        return SourceDoc(title=result_title, content=content, url=url)
    except Exception as exc:
        logger.debug("Crossref deep search failed for '%s': %s", title, exc)
        return None


# Ordered list of deep-search providers.  Each returns a SourceDoc or None.
# Order matters: Semantic Scholar has the best abstract coverage for CS/AI;
# OpenAlex is the broadest open catalog; arXiv covers preprints; Crossref
# covers DOIs across all publishers.  The first non-None result with the
# longest content wins.
_DEEP_SEARCH_PROVIDERS: tuple[Callable[[str, int], SourceDoc | None], ...] = (
    _deep_search_semantic_scholar,
    _deep_search_openalex,
    _deep_search_arxiv,
    _deep_search_crossref,
)


def _deep_search_fallback(
    title: str,
    raw_url: str,
    *,
    timeout: int = 45,
) -> SourceDoc:
    """Last-resort deep search across accessible scholarly sources.

    Queries Semantic Scholar, OpenAlex, arXiv, and Crossref for *title* and
    returns the ``SourceDoc`` with the longest content (best alternative).
    The returned ``SourceDoc`` carries the original *raw_url* and a
    diagnostic noting which provider supplied the content.

    Args:
        title: Title or search query derived from the original source.
        raw_url: The original URL that failed or produced a stub.
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        ``SourceDoc`` with the best accessible alternative content.

    Raises:
        RuntimeError: When every provider fails to return usable content.
    """
    if not title or not title.strip():
        title = _title_from_url(raw_url)
    if not title:
        raise RuntimeError(f"deep search fallback: no title to search for {raw_url}")

    candidates: list[tuple[str, SourceDoc]] = []
    for provider in _DEEP_SEARCH_PROVIDERS:
        try:
            doc = provider(title, timeout)
        except Exception as exc:  # defensive — providers already swallow
            logger.debug("deep search provider %s raised: %s", provider.__name__, exc)
            continue
        if doc is not None and doc.content:
            candidates.append((provider.__name__, doc))
        if len(candidates) >= 2:
            # Two good candidates is enough to pick a winner without hitting
            # every provider — keeps latency bounded.
            break

    if not candidates:
        raise RuntimeError(
            f"deep search fallback: no accessible alternative found for '{title}' ({raw_url})"
        )

    # Pick the longest-content candidate as the best alternative.
    best_name, best_doc = max(candidates, key=lambda pair: len(pair[1].content))
    logger.info(
        "deep search fallback: using %s for '%s' (%d chars)",
        best_name, title, len(best_doc.content),
    )
    # Preserve the original requested URL and tag provenance with a diagnostic.
    from dataclasses import replace

    from obsidian_llm_wiki.core.models import SourceProvenance

    diag = f"deep_search_fallback: source={best_name}"
    new_prov = SourceProvenance(
        requested_url=raw_url,
        resolved_url=raw_url,
        extracted_url=raw_url,
        extractor_chain=("deep_search_fallback", best_name),
        diagnostics=(diag,),
    )
    return replace(best_doc, url=raw_url, provenance=new_prov)


# ── Extraction quality gate ────────────────────────────────────────────


def _check_extraction_quality(source: SourceDoc) -> tuple[bool, str]:
    """Verify the extraction produced full content, not a stub or abstract.

    Returns ``(passed, reason)``.  When *passed* is ``False``, *reason*
    describes the quality issue.  This is a diagnostic gate — it never
    fails extraction.  Callers should log a WARNING and add the reason to
    provenance diagnostics.

    Checks:
      1. Content < 500 chars → "too short, likely stub"
      2. Content contains the stub-fallback sentinel → "stub fallback"
      3. Content has "Abstract:" but no "## Full Text" or body sections → "abstract only"
      4. Content is mostly metadata (Title:/Channel:/Published:) → "metadata only"
    """
    content = source.content or ""
    title = (source.title or "").strip()

    # A title is source identity. Broken Markdown/link fragments and raw URLs
    # become meaningless filenames and infect every downstream artifact.
    if (
        len(title) < 3
        or title.startswith(("](", "[", "http://", "https://"))
        or "://" in title
        or title.endswith((" on X", " on Twitter"))
        or bool(re.match(r"https?(?:www)?(?:x|twitter|[a-z0-9-]+(?:com|org|net))", title, re.I))
    ):
        return (False, "invalid source title")

    # X status pages frequently contain an article card plus a 1-2 sentence
    # preview. That is not the linked article; fail it instead of synthesizing
    # a plausible-looking graph from the login shell.
    lowered = content.casefold()
    if (
        "https://x.com/i/article/" in lowered
        or "https://twitter.com/i/article/" in lowered
    ) and ("article\n" in lowered or "article\r\n" in lowered):
        return (False, "X article preview stub")

    # Generic extractors can return a large navigation shell. Length is not
    # quality: Congress.gov's shell is tens of thousands of characters.
    chrome_markers = (
        "skip to main content",
        "navigation",
        "advanced searches",
        "back to top",
        "loading...",
    )
    if sum(marker in lowered for marker in chrome_markers) >= 4:
        return (False, "navigation chrome")

    # 1. Too short to be real content.
    if len(content) < 500:
        return (False, "too short, likely stub")

    # 2. Stub-fallback sentinel from transcript resolver.
    stub_markers = (
        "Note: Full transcript unavailable",
        "Full transcript unavailable",
    )
    for marker in stub_markers:
        if marker in content:
            return (False, "stub fallback")

    # 3. Abstract-only — has an abstract but no full-text or body sections.
    if "Abstract:" in content:
        has_full_text = "## Full Text" in content or "## Full text" in content
        has_body_sections = bool(
            re.search(r"^#{1,4}\s+\S", content, re.MULTILINE)
        )
        # Allow content that has body sections beyond the abstract.
        if not has_full_text and not has_body_sections:
            return (False, "abstract only")

    # 4. Metadata-only — content is dominated by metadata fields.
    metadata_prefixes = ("Title:", "Channel:", "Published:", "URL:", "Duration:")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines:
        metadata_lines = sum(
            1 for line in lines
            if any(line.startswith(p) for p in metadata_prefixes)
        )
        # If >60% of non-blank lines are metadata, it's metadata-only.
        if metadata_lines / len(lines) > 0.6:
            return (False, "metadata only")

    return (True, "")


def _require_usable_source(source: SourceDoc) -> SourceDoc:
    """Return a source only when it clears the corpus-wide quality contract."""
    passed, reason = _check_extraction_quality(source)
    if not passed:
        raise RuntimeError(f"Extraction quality rejected source: {reason}")
    return source


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

    # Inbox URLs can originate outside the local trust boundary. Reject direct
    # local/private targets before a generic extractor or document downloader
    # can turn the CLI into an SSRF primitive.
    from obsidian_llm_wiki.ingest.url_safety import validate_remote_url

    validate_remote_url(raw_url)

    from obsidian_llm_wiki.ingest.documents import dispatch_document, is_direct_document_url

    if is_direct_document_url(raw_url):
        return _stamp_extracted_source(dispatch_document(raw_url), raw_url, "document_dispatch")

    # Route every ordinary remote URL through the connector contract. Existing
    # specialists remain first and retain their fail-closed behavior; the
    # generic web extractor is the dispatcher fallback rather than a side path.
    result = _dispatch_remote_connectors(raw_url)
    if result.succeeded:
        assert result.source is not None
        stamped = _stamp_extracted_source(result.source, raw_url, result.connector_name)
        if result.connector_name != "generic_web":
            return _require_usable_source(stamped)

        # Last-resort: if generic web extraction produced a stub, try deep
        # search for the same title across accessible scholarly sources and use
        # the best alternative. Gated to keep CI deterministic.
        if _deep_search_enabled():
            passed, _reason = _check_extraction_quality(stamped)
            if not passed:
                try:
                    ds = _deep_search_fallback(stamped.title or "", raw_url)
                    if len(ds.content or "") > len(stamped.content or ""):
                        logger.info(
                            "Deep search fallback replacing stub for %s "
                            "(stub=%d chars → deep_search=%d chars).",
                            raw_url, len(stamped.content or ""), len(ds.content or ""),
                        )
                        return _require_usable_source(
                            _stamp_extracted_source(ds, raw_url, "deep_search_fallback")
                        )
                except Exception as ds_exc:
                    logger.warning("Deep search fallback failed for stub %s: %s", raw_url, ds_exc)
        return _require_usable_source(stamped)

    assert result.failure is not None
    if result.connector_name == "specialist_dispatch" and _deep_search_enabled():
        try:
            ds = _deep_search_fallback("", raw_url)
            return _require_usable_source(
                _stamp_extracted_source(ds, raw_url, "deep_search_fallback")
            )
        except Exception as ds_exc:
            logger.warning("Deep search fallback also failed for %s: %s", raw_url, ds_exc)
    if result.connector_name == "specialist_dispatch":
        raise RuntimeError(
            f"All specialized extractors failed for {raw_url}: {result.failure.message}"
        )
    raise RuntimeError(result.failure.message)


def _dispatch_remote_connectors(raw_url: str):
    """Adapt the registered specialist functions to the connector contract."""
    from obsidian_llm_wiki.ingest.connectors import (
        CallableSourceConnector,
        GenericWebConnector,
        SourceConnectorDispatcher,
    )

    specialists = [
        CallableSourceConnector(
            extractor_fn.__name__,
            match_fn,
            extractor_fn,
            is_not_applicable=lambda exc: isinstance(exc, ExtractorNotApplicableError),
            validated_redirects=True,
        )
        for match_fn, extractor_fn in _EXTRACTORS
    ]
    return SourceConnectorDispatcher(
        specialists,
        GenericWebConnector(extractor=extract_web),
    ).dispatch(raw_url)


def _stamp_extracted_source(source: SourceDoc, raw_url: str, extractor: str) -> SourceDoc:
    """Attach baseline immutable provenance at the public extractor boundary.

    Also runs the extraction quality gate.  When the gate fails, a WARNING
    is logged and the reason is appended to provenance diagnostics.
    Extraction is never failed — the source is still returned.
    """
    from obsidian_llm_wiki.ingest.provenance import stamp_source

    # Run the quality gate (diagnostic only — never fails extraction).
    passed, reason = _check_extraction_quality(source)
    if not passed:
        logger.warning(
            "Extraction quality gate: %s for '%s' (extractor=%s, content_len=%d)",
            reason, raw_url, extractor, len(source.content or ""),
        )
        diagnostics = (f"extraction_quality: {reason}",)
    else:
        diagnostics = ()

    return stamp_source(
        source,
        requested_url=raw_url,
        extractor=extractor,
        diagnostics=diagnostics,
    )


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
