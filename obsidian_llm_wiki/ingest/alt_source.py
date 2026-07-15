"""Alt-source extraction — Invidious API, Semantic Scholar, journal direct pages.

These extractors supplement the primary trafilatura-based web extractor for URLs
that are blocked or poorly handled. They are tried in the multi-layer fallback
chain after trafilatura fails.

Architecture note: These are NOT registered extractors in the registry.
They are called directly by web.py's extract_web() fallback chain and by
the extractors/__init__.py dispatch when specialized extractors fail.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.url_safety import get_with_validated_redirects

logger = logging.getLogger("obswiki.ingest.alt_source")


# ── Invidious ────────────────────────────────────────────────────────────────


def extract_via_invidious(youtube_url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceDoc:
    """Extract YouTube video metadata and community-supplied description via Invidious.

    Invidious is a front-end proxy for YouTube that does not require JavaScript.
    This is a last-resort fallback for YouTube videos when yt-dlp and the
    transcript API both fail. It gives metadata + description, not transcript.

    Known Invidious instances (updated periodically):
      - invidious.snopyta.org
      - yewtu.be
      - invidious.privacyredirect.com
      - iv.nboeck.de

    Returns:
        SourceDoc with title from video metadata and content from description.
    Raises:
        RuntimeError: If all Invidious instances fail.
    """
    # Extract video ID
    video_id = _extract_youtube_video_id(youtube_url)
    if not video_id:
        raise RuntimeError(f"Could not extract video ID from {youtube_url}")

    # Try multiple Invidious instances
    instances = [
        "https://yewtu.be",
        "https://invidious.privacyredirect.com",
    ]

    errors: list[str] = []
    for instance in instances:
        url = f"{instance}/api/v1/videos/{video_id}"
        try:
            with httpx.Client(
                **make_client_kwargs(timeout=timeout, follow_redirects=True),
            ) as client:
                resp = client.get(url, headers=BROWSER_HEADERS)
                resp.raise_for_status()
            data = resp.json()

            title = data.get("title", "") or ""
            description = data.get("description", "") or ""
            view_count = data.get("viewCount", 0) or 0
            like_count = data.get("likeCount", 0) or 0
            published = data.get("published", "") or ""
            channel = data.get("author", "") or ""

            # Clean HTML from description
            description = re.sub(r"<[^>]+>", "", description)
            description = description.strip()

            content_parts = [
                f"Channel: {channel}",
                f"Published: {published}",
                f"Views: {view_count:,}",
                f"Likes: {like_count:,}",
                "",
                "Description:",
                description,
            ]
            content = "\n".join(content_parts).strip()

            if not content or len(content) < 100:
                errors.append(f"{instance}: empty/short description")
                continue

            return SourceDoc(title=title, content=content, url=youtube_url)

        except Exception as exc:
            errors.append(f"{instance}: {exc}")

    raise RuntimeError(
        f"All Invidious instances failed for {youtube_url}: " + "; ".join(errors)
    )


def _extract_youtube_video_id(url: str) -> str | None:
    """Extract the 11-char video ID from various YouTube URL formats."""
    parsed = urlparse(url)
    # youtube.com/watch?v=...
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com") and parsed.query:
        for param in parsed.query.split("&"):
            k, _, v = param.partition("=")
            if k == "v":
                return v[:11]
    # youtu.be/...
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/")[:11]
    # youtube.com/embed/... or /shorts/...
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        for segment in parsed.path.split("/"):
            if len(segment) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", segment):
                return segment
    return None


# ── Semantic Scholar ─────────────────────────────────────────────────────────


def extract_via_semantic_scholar(ssrn_url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceDoc:
    """Extract paper abstract and metadata via Semantic Scholar API.

    Semantic Scholar (api.semanticscholar.org) indexes most SSRN papers.
    This is a fallback for SSRN URLs blocked by Cloudflare. It gives abstract
    + metadata, not full text.

    Returns:
        SourceDoc with title, abstract as content.
    Raises:
        RuntimeError: If Semantic Scholar lookup fails.
    """
    # Try to extract SSRN paper ID from URL
    paper_id = _extract_ssrn_paper_id(ssrn_url)

    if paper_id:
        # Try SSRN ID directly
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/"
            f"SSRN:{paper_id}"
            f"?fields=title,abstract,year,authors"
        )
    else:
        raise RuntimeError(
            f"Could not extract SSRN paper ID from {ssrn_url} — "
            "Semantic Scholar requires a paper ID, not a URL"
        )

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=True),
        ) as client:
            resp = client.get(url, headers=BROWSER_HEADERS)
            if resp.status_code == 404:
                raise RuntimeError(f"Paper not found on Semantic Scholar: SSRN:{paper_id}")
            resp.raise_for_status()
        data = resp.json()

        title = data.get("title", "") or ssrn_url
        abstract = data.get("abstract", "") or ""
        year = data.get("year") or ""
        authors = data.get("authors", []) or []
        author_names = ", ".join(a.get("name", "") for a in authors if a.get("name"))

        content_parts = [f"Title: {title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        else:
            content_parts.extend(["", "Note: No abstract available via Semantic Scholar."])

        content = "\n".join(content_parts).strip()
        return SourceDoc(title=title, content=content, url=ssrn_url)

    except Exception as exc:
        raise RuntimeError(
            f"Semantic Scholar lookup failed for SSRN:{paper_id}: {exc}"
        ) from exc


def _extract_ssrn_paper_id(url: str) -> str | None:
    """Extract the numeric paper ID from a SSRN URL."""
    # papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522
    # or papers.ssrn.com/abstract_id=5910522
    parsed = urlparse(url)
    if "papers.ssrn.com" not in (parsed.hostname or ""):
        return None
    query = parsed.query or ""
    for param in query.split("&"):
        k, _, v = param.partition("=")
        if k in ("abstract_id", "id") and v.isdigit():
            return v
    # Try path: /sol3/papers.cfm/abstract_id=5910522
    m = re.search(r"abstract[_-]?id[=:]?(\d+)", url, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# ── Journal direct page ──────────────────────────────────────────────────────


def extract_via_journal_page(url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceDoc:
    """Try to fetch a journal article's HTML page directly (no XML suffix).

    akjournals.com/view/journals/2054/9/3/article-p294.xml
      → akjournals.com/view/journals/2054/9/3/article-p294

    Publishers sometimes serve the same article at a /article-p294 URL
    without the .xml suffix. This tries that rewrite before falling back to
    the Wayback Machine. If a captcha/verification wall is detected, tries
    Semantic Scholar search by article title extracted from the page.

    Returns:
        SourceDoc with title and extracted content.
    Raises:
        RuntimeError: If the direct page also fails.
    """
    # Rewrite .xml suffix
    if url.endswith(".xml"):
        direct_url = url[:-4]
    elif "/article-" in url:
        # Already no .xml — try as-is
        direct_url = url
    else:
        raise RuntimeError(f"URL does not appear to be an XML article page: {url}")

    html = ""
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=False),
        ) as client:
            resp = get_with_validated_redirects(client, direct_url, headers=BROWSER_HEADERS)
            if resp.status_code == 404:
                raise RuntimeError(f"Direct page returned 404: {direct_url}")
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        raise RuntimeError(f"Journal direct page failed for {direct_url}: {exc}") from exc

    # Check for Cloudflare or human verification walls
    lower_html = html.lower()
    if (
        "just a moment" in lower_html
        or "cf-challenge" in lower_html
        or "confirm you are human" in lower_html
        or "human verification" in lower_html
    ):
        # Try Semantic Scholar title search as a fallback for captchas
        title = _extract_title_from_html(html)
        if title:
            try:
                return _semantic_scholar_search(title, url, timeout)
            except Exception:
                pass
        raise RuntimeError("Cloudflare/human verification challenge on direct page")

    if not html.strip() or len(html.strip()) < 200:
        raise RuntimeError("Empty or near-empty response from direct page")

    # Extract title
    title = _extract_title_from_html(html) or direct_url

    # Strip tags for plain text
    content = _strip_journal_html(html)
    if not content or len(content.strip()) < 100:
        raise RuntimeError("Journal page extraction produced short/empty content")

    return SourceDoc(title=title, content=content, url=direct_url)


def _extract_title_from_html(html: str) -> str:
    """Extract the article title from meta tags or <title> element."""
    title_match = re.search(
        r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"',
        html, re.IGNORECASE,
    )
    title = title_match.group(1).strip() if title_match else ""
    if not title:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
    return title


def _semantic_scholar_search(
    title: str, original_url: str, timeout: int = DEFAULT_TIMEOUT
) -> SourceDoc:
    """Search Semantic Scholar by paper title and return abstract + metadata.

    This is a last-resort fallback for academic papers behind captcha walls.
    """
    search_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={title}&limit=1&fields=title,abstract,year,authors"
    )
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=True),
        ) as client:
            resp = client.get(search_url, headers=BROWSER_HEADERS)
            if resp.status_code == 404:
                raise RuntimeError(f"Paper not found on Semantic Scholar: {title}")
            resp.raise_for_status()
            data = resp.json()

        papers = data.get("data", []) or []
        if not papers:
            raise RuntimeError(f"No results from Semantic Scholar for: {title}")

        paper = papers[0]
        title_result = paper.get("title", "") or title
        abstract = paper.get("abstract", "") or ""
        year = paper.get("year", "") or ""
        authors = paper.get("authors", []) or []
        author_names = ", ".join(
            a.get("name", "") for a in authors if a.get("name")
        )

        content_parts = [f"Title: {title_result}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        else:
            content_parts.extend(["", "Note: No abstract available via Semantic Scholar."])

        content = "\n".join(content_parts).strip()
        return SourceDoc(title=title_result, content=content, url=original_url)

    except Exception as exc:
        raise RuntimeError(
            f"Semantic Scholar title search failed for '{title}': {exc}"
        ) from exc


def _doi_lookup(doi: str, original_url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceDoc:
    """Resolve a DOI via Semantic Scholar's DOI endpoint and return abstract + metadata."""
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        f"?fields=title,abstract,year,authors"
    )
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=True),
        ) as client:
            resp = client.get(url, headers=BROWSER_HEADERS)
            if resp.status_code == 404:
                raise RuntimeError(f"DOI not found on Semantic Scholar: {doi}")
            resp.raise_for_status()
            data = resp.json()

        title = data.get("title", "") or doi
        abstract = data.get("abstract", "") or ""
        year = data.get("year", "") or ""
        authors = data.get("authors", []) or []
        author_names = ", ".join(a.get("name", "") for a in authors if a.get("name"))

        content_parts = [f"Title: {title}"]
        if author_names:
            content_parts.append(f"Authors: {author_names}")
        if year:
            content_parts.append(f"Year: {year}")
        if abstract:
            content_parts.extend(["", "Abstract:", abstract])
        else:
            content_parts.extend(["", "Note: No abstract available."])

        content = "\n".join(content_parts).strip()
        return SourceDoc(title=title, content=content, url=original_url)
    except Exception as exc:
        raise RuntimeError(f"DOI lookup failed for {doi}: {exc}") from exc


def _strip_journal_html(html: str) -> str:
    """Strip journal-specific HTML to readable text."""
    # Remove script/style/nav/header/footer
    cleaned = re.sub(
        r"<(script|style|nav|header|footer|aside|form)[^>]*>.*?</\1>",
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove HTML comments
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    # Remove divs with common non-content classes
    cleaned = re.sub(
        r'<div[^>]*(?:class|id)="[^"]*(?:sidebar|menu|nav|footer|header|ad|cookie)[^"]*"[^>]*>.*?</div>',
        "", cleaned, flags=re.IGNORECASE | re.DOTALL,
    )
    # Convert block elements to newlines
    cleaned = re.sub(
        r"</?(?:div|p|br|h[1-6]|li|tr|section|article|main)[^>]*>",
        "\n", cleaned, flags=re.IGNORECASE,
    )
    # Remove remaining tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Decode entities
    for entity, char in {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
    }.items():
        cleaned = cleaned.replace(entity, char)
    # Collapse whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
