"""Web URL extraction — Stage 1 deterministic ingest.

Multi-layer strategy:
  1. httpx fetch + trafilatura extract (primary)
  2. defuddle CLI (npm) — removes JS/clutter, often succeeds where trafilatura is blocked
  3. Invidious API (YouTube metadata fallback)
  4. archive.org Wayback Machine (last resort)

Each layer runs only if the previous one fails. The SourceDoc always carries
the full content — never truncated.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.web")

__all__ = ["extract_web"]

_TIMEOUT = DEFAULT_TIMEOUT


def extract_web(url: str, timeout: int = _TIMEOUT) -> SourceDoc:
    """Extract full article content from a web URL.

    Strategy (in order):
      1. trafilatura extract — primary HTML→text
      2. defuddle CLI — npm package, better JS removal, custom UA support
      3. Invidious API — YouTube-only metadata fallback
      4. archive.org Wayback Machine — last resort

    Raises:
        RuntimeError: When all strategies fail.
    """
    errors: list[str] = []

    # Layer 1: trafilatura
    try:
        return _extract_trafilatura(url, timeout)
    except Exception as exc:
        errors.append(f"trafilatura: {exc}")

    # Layer 2: defuddle CLI (no proxy — it has its own UA handling)
    try:
        return _extract_defuddle(url, timeout)
    except Exception as exc:
        errors.append(f"defuddle: {exc}")

    # Layer 3: Invidious (YouTube only)
    if _is_youtube_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_invidious
            return extract_via_invidious(url, timeout)
        except Exception as exc:
            errors.append(f"invidious: {exc}")

    # Layer 4: SSRN via Semantic Scholar (academic paper fallback)
    if _is_ssrn_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_semantic_scholar
            return extract_via_semantic_scholar(url, timeout)
        except Exception as exc:
            errors.append(f"semantic_scholar: {exc}")

    # Layer 5: akjournals / journal XML direct-page fallback
    if _is_journal_xml_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_journal_page
            return extract_via_journal_page(url, timeout)
        except Exception as exc:
            errors.append(f"journal_direct: {exc}")

    # Layer 6: Wayback Machine
    try:
        return _extract_wayback(url, timeout)
    except Exception as exc:
        errors.append(f"wayback: {exc}")

    raise RuntimeError(
        f"All extraction strategies failed for {url}:\n  " + "\n  ".join(errors)
    )


def _extract_defuddle(url: str, timeout: int) -> SourceDoc:
    """Extract via defuddle npm CLI (removes JS, ads, clutter).

    defuddle is a browser-quality extractor that handles JS-rendered pages.
    It is tried after trafilatura and before Wayback Machine.

    Requires: npm install -g defuddle
    The CLI accepts --user-agent to bypass simple 403 blocks.
    """
    # Find defuddle binary
    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    cmd: list[str]
    if not defuddle_path:
        raise RuntimeError("defuddle not found — npm install -g defuddle")
    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--md"]
    else:
        # npx defuddle parse ...
        cmd = [defuddle_path, "defuddle", "parse", url, "--md"]

    # Run with browser UA to reduce 403s
    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""  # Clear problematic CA certs

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=min(timeout, 60),
        env=env,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if stderr:
            raise RuntimeError(f"defuddle exited {proc.returncode}: {stderr[:200]}")
        raise RuntimeError(f"defuddle exited {proc.returncode} (no stderr)")

    output = proc.stdout
    if not output.strip():
        raise RuntimeError("defuddle returned empty output")

    # defuddle --md returns markdown. Parse title from first # heading.
    # If the first line is not a heading, try defuddle --json for metadata.
    lines = output.strip().split("\n", 3)
    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    # If no heading title, try defuddle --json for structured metadata
    if not title or _is_bad_title(title):
        title = _defuddle_metadata_title(url, timeout)

    # If still no title and first line isn't a heading, use the URL
    if not title or _is_bad_title(title):
        title = url

    content = output.strip()
    # Strip the title heading from content if present
    if title and content.startswith(f"# {title}"):
        content = content[len(title) + 2:].strip()
    # Also strip markdown image prefix from content if it's the first line
    if content.startswith("!["):
        # Remove the image line and any following blank line
        content = content.split("\n", 1)[-1].lstrip() if "\n" in content else content

    if not content or len(content) < 50:
        raise RuntimeError(f"defuddle produced short content ({len(content)} chars)")

    return SourceDoc(title=title, content=content, url=url)


def _defuddle_metadata_title(url: str, timeout: int) -> str:
    """Fetch page title via defuddle --json (includes metadata).

    defuddle --json returns a JSON object with title, author, etc.
    This is used as a fallback when --md output has no # heading.
    """
    import json

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return ""

    cmd: list[str]
    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--json"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", url, "--json"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=min(timeout, 60),
            env=env,
        )
        if proc.returncode != 0:
            return ""
        data = json.loads(proc.stdout)
        return (data.get("title") or "").strip()
    except Exception:
        return ""


def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video page."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return host in (
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"
        )
    except Exception:
        return False


def _is_ssrn_url(url: str) -> bool:
    """Check if URL is a SSRN paper page."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return "ssrn.com" in host
    except Exception:
        return False


def _is_journal_xml_url(url: str) -> bool:
    """Check if URL is a journal XML article URL."""
    return url.endswith(".xml") or "/article-" in url


def _is_bad_title(title: str) -> bool:
    """Check if a title is garbage (markdown image, URL, or too short).

    trafilatura sometimes picks up markdown image alt text or raw URLs
    as the title instead of the actual page title.
    """
    if not title or len(title.strip()) < 3:
        return True
    # Markdown image: ![alt](url)
    if title.startswith("![") and "](" in title:
        return True
    # Raw URL
    if title.startswith("http") and "://" in title:
        return True
    # HTML tag fragment
    return bool(title.startswith("<") and ">" in title)


def _extract_trafilatura(url: str, timeout: int) -> SourceDoc:
    """Extract via httpx fetch + trafilatura."""
    import trafilatura

    with httpx.Client(
        **make_client_kwargs(timeout=timeout, follow_redirects=True),
        headers=BROWSER_HEADERS,
    ) as client:
        resp = client.get(url)
        html = resp.text

    _check_cloudflare(resp, html)
    resp.raise_for_status()
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
    if not title or _is_bad_title(title):
        title = _extract_title_from_html(html)
    if not title or _is_bad_title(title):
        title = ""

    return SourceDoc(title=title or url, content=extracted.strip(), url=url)


def _extract_regex(url: str, timeout: int) -> SourceDoc:
    """Extract via httpx + regex HTML-to-text."""
    with httpx.Client(**make_client_kwargs(timeout=timeout, follow_redirects=True)) as client:
        resp = client.get(url)
        html = resp.text

    _check_cloudflare(resp, html)
    resp.raise_for_status()
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

    with httpx.Client(**make_client_kwargs(timeout=timeout + 20, follow_redirects=True)) as client:
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
