"""Web URL extraction — Stage 1 deterministic ingest.

Primary: httpx + trafilatura (pure Python, no subprocess).
Fallback: archive.org Wayback Machine via httpx.

Never truncates content.  Always returns full SourceDoc.
"""

from __future__ import annotations

import logging
import re

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT

logger = logging.getLogger("obswiki.ingest.web")

__all__ = ["extract_web"]

_TIMEOUT = DEFAULT_TIMEOUT


def extract_web(url: str, timeout: int = _TIMEOUT) -> SourceDoc:
    """Extract full article content from a web URL.

    Strategy:
      1. httpx fetch + trafilatura extract (primary)
      2. httpx fetch + regex HTML-to-text (fallback 1)
      3. archive.org Wayback Machine (fallback 2)

    Raises:
        RuntimeError: When all strategies fail.
    """
    errors: list[str] = []

    try:
        return _extract_trafilatura(url, timeout)
    except Exception as exc:
        errors.append(f"trafilatura: {exc}")

    try:
        return _extract_regex(url, timeout)
    except Exception as exc:
        errors.append(f"regex: {exc}")

    try:
        return _extract_wayback(url, timeout)
    except Exception as exc:
        errors.append(f"wayback: {exc}")

    raise RuntimeError(
        f"All extraction strategies failed for {url}:\n  " + "\n  ".join(errors)
    )


def _extract_trafilatura(url: str, timeout: int) -> SourceDoc:
    """Extract via httpx fetch + trafilatura."""
    import trafilatura

    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers=BROWSER_HEADERS) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    _check_cloudflare(resp, html)
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not extracted or len(extracted.strip()) < 50:
        raise RuntimeError("trafilatura returned empty/short content")

    metadata = trafilatura.extract_metadata(html)
    title = ""
    if metadata:
        title = (metadata.title or "").strip()
    if not title:
        title = _extract_title_from_html(html)

    return SourceDoc(title=title or url, content=extracted.strip(), url=url)


def _extract_regex(url: str, timeout: int) -> SourceDoc:
    """Extract via httpx + regex HTML-to-text."""
    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers=BROWSER_HEADERS) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text

    _check_cloudflare(resp, html)
    if not html.strip():
        raise RuntimeError("empty response")

    title = _extract_title_from_html(html)
    content = _strip_tags(html)
    if not content:
        raise RuntimeError("regex produced empty content")

    return SourceDoc(title=title or url, content=content, url=url)


def _extract_wayback(url: str, timeout: int) -> SourceDoc:
    """Extract via archive.org Wayback Machine."""
    wayback_url = f"https://web.archive.org/web/2/{url}"

    with httpx.Client(timeout=timeout + 20, follow_redirects=True,
                      headers=BROWSER_HEADERS) as client:
        resp = client.get(wayback_url)
        resp.raise_for_status()
        html = resp.text

    # Remove Wayback toolbar.
    html = re.sub(
        r"<!--\s*BEGIN WAYBACK TOOLBAR.*?END WAYBACK TOOLBAR.*?-->",
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    html = re.sub(
        r'<div[^>]*id="wm-ipp[^"]*"[^>]*>.*?</div>',
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )

    title = _extract_title_from_html(html)
    content = _strip_tags(html)
    if not content:
        raise RuntimeError("wayback produced empty content")

    return SourceDoc(title=title or url, content=content, url=url)


def _check_cloudflare(resp: httpx.Response, html: str) -> None:
    """Detect Cloudflare JS challenge pages and fail cleanly.

    CF challenges return 403 with specific headers/body markers. Feeding
    these into the LLM as source content would produce garbage notes.
    """
    # Header-based detection (most reliable)
    if resp.headers.get("cf-mitigated") == "challenge":
        raise RuntimeError(
            "Cloudflare JS challenge — cannot extract without a real browser. "
            "This site requires JavaScript execution or cookies."
        )
    # Body-based detection (fallback for 403 without cf-mitigated header)
    if resp.status_code == 403:
        title = _extract_title_from_html(html)
        if "just a moment" in title.lower() or "attention required" in title.lower():
            raise RuntimeError(
                "Cloudflare challenge page detected — cannot extract without a real browser."
            )


# ── HTML helpers ────────────────────────────────────────────────────────


def _extract_title_from_html(html: str) -> str:
    """Extract <title>, og:title, or first <h1> from HTML."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return _decode_entities(match.group(1).strip())

    match = re.search(
        r'<meta\s[^>]*property="og:title"\s[^>]*content="([^"]*)"',
        html, re.IGNORECASE,
    )
    if match:
        return _decode_entities(match.group(1))

    match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return _strip_tags(match.group(1)).strip()

    return ""


def _strip_tags(html: str) -> str:
    """Strip HTML tags and return plain text."""
    cleaned = re.sub(
        r"<(script|style|noscript|iframe)[^>]*>.*?</\1>",
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"</?(?:div|p|h[1-6]|li|tr|br|hr|section|article|header|footer|nav"
        r"|main|aside|blockquote|pre|table|ul|ol|dl|figure|figcaption)[^>]*>",
        "\n", cleaned, flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = _decode_entities(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _decode_entities(text: str) -> str:
    """Decode common HTML entities."""
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&apos;": "'",
        "&nbsp;": " ", "&#160;": " ",
        "&ndash;": "–", "&mdash;": "—",
        "&lsquo;": "'", "&rsquo;": "'",
        "&ldquo;": '"', "&rdquo;": '"',
        "&hellip;": "…", "&trade;": "™", "&reg;": "®",
        "&copy;": "©", "&deg;": "°",
        "&euro;": "€", "&pound;": "£", "&yen;": "¥",
    }
    for entity, char in entities.items():
        text = text.replace(entity, char)
    text = re.sub(
        r"&#x([0-9a-fA-F]+);",
        lambda m: _safe_chr(int(m.group(1), 16), m.group(0)),
        text,
    )
    text = re.sub(
        r"&#(\d+);",
        lambda m: _safe_chr(int(m.group(1)), m.group(0)),
        text,
    )
    return text


def _safe_chr(codepoint: int, original: str) -> str:
    """Safely convert a codepoint to a character."""
    try:
        return chr(codepoint)
    except (ValueError, OverflowError):
        return original
