"""Web URL extraction — Stage 1 deterministic ingest.

Primary: defuddle CLI for content extraction (title + markdown).
Fallback 1: curl + regex-based HTML-to-text conversion.
Fallback 2: archive.org Wayback Machine snapshot.

Never truncates content. Always returns full IngestedSource.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from pipeline.models import IngestedSource

# ── Constants ──────────────────────────────────────────────────────────

_DEFUDDLE_BIN = "defuddle"
_CURL_BIN = "curl"
_TIMEOUT = 45


# ── Public API ─────────────────────────────────────────────────────────


def extract_web(url: str, timeout: int = _TIMEOUT) -> IngestedSource:
    """Extract full article content from a web URL.

    Strategy:
      1. defuddle --json <url>  (primary)
      2. curl <url> + strip HTML tags  (fallback 1)
      3. archive.org Wayback Machine via curl  (fallback 2)

    Args:
        url: The web URL to extract content from.
        timeout: Subprocess timeout in seconds (default 45).

    Returns:
        IngestedSource with title and full content.

    Raises:
        RuntimeError: When all extraction strategies fail.
    """
    errors: list[str] = []

    # ── Strategy 1: defuddle ─────────────────────────────────────
    try:
        return _extract_defuddle(url, timeout)
    except Exception as exc:
        errors.append(f"defuddle: {exc}")

    # ── Strategy 2: curl + regex ─────────────────────────────────
    try:
        return _extract_curl_regex(url, timeout)
    except Exception as exc:
        errors.append(f"curl+regex: {exc}")

    # ── Strategy 3: archive.org ──────────────────────────────────
    try:
        return _extract_wayback(url, timeout)
    except Exception as exc:
        errors.append(f"wayback: {exc}")

    raise RuntimeError(
        f"All extraction strategies failed for {url}:\n  "
        + "\n  ".join(errors)
    )


# ── Strategy implementations ───────────────────────────────────────────


def _extract_defuddle(url: str, timeout: int) -> IngestedSource:
    """Extract via defuddle CLI (--json)."""
    result = subprocess.run(
        [_DEFUDDLE_BIN, "parse", "-j", url],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"defuddle exited {result.returncode}: {result.stderr.strip()}"
        )

    data = json.loads(result.stdout)

    title = (data.get("title") or "").strip()
    # Prefer markdown rendering; fall back to raw HTML content
    content = (data.get("contentMarkdown") or data.get("content") or "").strip()

    if not content:
        raise RuntimeError("defuddle returned empty content")

    return IngestedSource(title=title, content=content)


def _extract_curl_regex(url: str, timeout: int) -> IngestedSource:
    """Extract via curl + regex HTML-to-text conversion."""
    # Fetch raw HTML
    result = subprocess.run(
        [
            _CURL_BIN, "-sSL", "--max-time", str(timeout),
            "-A", "Mozilla/5.0 (compatible; llmwiki-bot/1.0)",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 5,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(
            f"curl exited {result.returncode}: {result.stderr.strip()}"
        )

    html = result.stdout
    if not html.strip():
        raise RuntimeError("curl returned empty response")

    # Extract <title>
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = _decode_html_entities(title_match.group(1).strip())

    # If no title tag, try og:title or first h1
    if not title:
        og_match = re.search(
            r'<meta\s[^>]*property="og:title"\s[^>]*content="([^"]*)"',
            html, re.IGNORECASE,
        )
        if og_match:
            title = _decode_html_entities(og_match.group(1))
    if not title:
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if h1_match:
            title = _strip_tags(h1_match.group(1)).strip()

    # Strip tags to get body text
    content = _strip_tags(html)

    # Normalise whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    if not content:
        raise RuntimeError("curl+regex produced empty content")

    return IngestedSource(title=title, content=content)


def _extract_wayback(url: str, timeout: int) -> IngestedSource:
    """Extract via archive.org Wayback Machine snapshot."""
    wayback_url = f"https://web.archive.org/web/2/{url}"

    # Fetch the Wayback snapshot
    result = subprocess.run(
        [
            _CURL_BIN, "-sSL", "--max-time", str(timeout + 15),
            "-A", "Mozilla/5.0 (compatible; llmwiki-bot/1.0)",
            "-o", "-",
            wayback_url,
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 20,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(
            f"Wayback curl exited {result.returncode}: {result.stderr.strip()}"
        )

    html = result.stdout
    if not html.strip():
        raise RuntimeError("Wayback machine returned empty response")

    # Wayback wraps content — try to extract just the archived page body.
    # Remove Wayback toolbar/header scripts.
    # Wayback inserts banners like "<!-- BEGIN WAYBACK TOOLBAR INSERT -->"
    html = re.sub(
        r"<!--\s*BEGIN WAYBACK TOOLBAR.*?END WAYBACK TOOLBAR.*?-->",
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove Wayback-specific scripts
    html = re.sub(
        r'<script[^>]*archive\.org[^>]*>.*?</script>',
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove Wayback toolbar divs
    html = re.sub(
        r'<div[^>]*id="wm-ipp[^"]*"[^>]*>.*?</div>',
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )

    # Extract title
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = _decode_html_entities(title_match.group(1).strip())
    if not title:
        og_match = re.search(
            r'<meta\s[^>]*property="og:title"\s[^>]*content="([^"]*)"',
            html, re.IGNORECASE,
        )
        if og_match:
            title = _decode_html_entities(og_match.group(1))

    content = _strip_tags(html)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    if not content:
        raise RuntimeError("Wayback produced empty content")

    return IngestedSource(title=title, content=content)


# ── HTML helpers ───────────────────────────────────────────────────────


def _strip_tags(html: str) -> str:
    """Strip HTML tags and return plain text.

    Handles line-breaking for block-level elements, removes scripts/styles,
    and decodes common entities.
    """
    # Remove scripts, styles, and comments
    cleaned = re.sub(
        r"<(script|style|noscript|iframe)[^>]*>.*?</\1>",
        "", html, flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)

    # Replace block-level tags with newlines
    cleaned = re.sub(
        r"</?(?:div|p|h[1-6]|li|tr|br|hr|section|article|header|footer|nav|main|aside|blockquote|pre|table|ul|ol|dl|figure|figcaption)[^>]*>",
        "\n", cleaned, flags=re.IGNORECASE,
    )

    # Remove remaining tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # Decode HTML entities
    cleaned = _decode_html_entities(cleaned)

    # Collapse whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)

    return cleaned.strip()


def _safe_chr(codepoint: int, original: str) -> str:
    """Safely convert a codepoint to a character.

    Returns the original entity text if the codepoint is outside the
    valid Unicode range (> 0x10FFFF), avoiding a ValueError from chr().
    """
    try:
        return chr(codepoint)
    except (ValueError, OverflowError):
        return original


def _decode_html_entities(text: str) -> str:
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

    # Numeric entities
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


# ── CLI entry point (for testing) ──────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {__file__} <url>", file=sys.stderr)
        sys.exit(1)

    try:
        source = extract_web(sys.argv[1])
        print(f"Title: {source.title}")
        print(f"Content length: {len(source.content)} chars")
        print("---")
        print(source.content[:500])
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
