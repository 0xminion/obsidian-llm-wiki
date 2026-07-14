"""Web URL extraction — Stage 1 deterministic ingest.

Multi-layer strategy:
  1. hosted Defuddle markdown extraction
  2. LiteParse local structured-document fallback
  3. trafilatura article extraction
  4. defuddle CLI
  5. specialist and archive fallbacks

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

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.url_safety import stream_with_validated_redirects

logger = logging.getLogger("obswiki.ingest.web")

__all__ = ["extract_web"]

_TIMEOUT = DEFAULT_TIMEOUT
_MAX_EXTRACTION_ERRORS = 12
_MAX_EXTRACTION_ERROR_CHARS = 240
_MAX_HTML_STREAM_CHUNK_BYTES = 64 * 1024


def _record_error(errors: list[str], stage: str, exc: Exception) -> None:
    """Keep fallback diagnostics informative without allowing unbounded errors."""
    if len(errors) < _MAX_EXTRACTION_ERRORS:
        errors.append(f"{stage}: {str(exc)[:_MAX_EXTRACTION_ERROR_CHARS]}")


def _read_bounded_html(response: httpx.Response, max_bytes: int) -> str:
    """Read one HTML response incrementally without exceeding ``max_bytes``."""
    if max_bytes < 1:
        raise RuntimeError("MAX_HTML_BYTES must be at least 1")

    declared_size = response.headers.get("content-length")
    if declared_size is not None:
        try:
            if int(declared_size) > max_bytes:
                raise RuntimeError(
                    f"HTML response Content-Length {declared_size} exceeds {max_bytes} bytes"
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    chunk_size = min(_MAX_HTML_STREAM_CHUNK_BYTES, max_bytes + 1)
    for chunk in response.iter_bytes(chunk_size=chunk_size):
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError(f"HTML response exceeded {max_bytes} bytes")
        chunks.append(chunk)

    return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


def extract_web(url: str, timeout: int = _TIMEOUT) -> SourceDoc:
    """Extract full article content from a web URL.

    Strategy (in order):
      1. public scientific full-text preflight for known scholarly landing pages
      2. defuddle.md web service — best markdown quality for articles/blogs
      3. LiteParse local document parser — direct files and citation-linked papers
      4. trafilatura extract — fallback for non-JS pages
      5. defuddle CLI — local defuddle fallback
      6. specialist and archive fallbacks

    Raises:
        RuntimeError: When all strategies fail.
    """
    errors: list[str] = []
    scientific_preflight_attempted = False

    # A short abstract is often a perfectly valid generic extraction, so known
    # scholarly landing pages get their official public full-text links first.
    # This narrow URL gate avoids adding a network request for ordinary blogs.
    if not _is_youtube_url(url):
        from obsidian_llm_wiki.ingest.extractors.scientific import (
            extract_discovered_scientific_document,
            is_likely_scientific_landing_page,
        )

        if is_likely_scientific_landing_page(url):
            scientific_preflight_attempted = True
            try:
                return extract_discovered_scientific_document(url, timeout)
            except Exception as exc:
                _record_error(errors, "public_scientific_preflight", exc)

    # Layer 1: defuddle.md web service (best markdown quality for articles/blogs)
    try:
        return _extract_defuddle_md(url, timeout)
    except Exception as exc:
        _record_error(errors, "defuddle.md", exc)

    # Layer 2: LiteParse local document fallback. This is optional; a missing
    # CLI is recorded and the ordinary HTML fallbacks continue.
    try:
        return _extract_liteparse_document(url, timeout)
    except Exception as exc:
        _record_error(errors, "liteparse", exc)

    # Layer 3: trafilatura
    try:
        return _extract_trafilatura(url, timeout)
    except Exception as exc:
        _record_error(errors, "trafilatura", exc)

    # Layer 4: defuddle CLI (no proxy — it has its own UA handling)
    try:
        return _extract_defuddle(url, timeout)
    except Exception as exc:
        _record_error(errors, "defuddle", exc)

    # A non-scientific URL may still expose an official paper link, but avoid
    # repeating discovery when the preflight already tried the same landing.
    if not _is_youtube_url(url) and not scientific_preflight_attempted:
        try:
            return extract_discovered_scientific_document(url, timeout)
        except Exception as exc:
            _record_error(errors, "public_scientific_document", exc)

    # Layer 6: Invidious (YouTube only)
    if _is_youtube_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_invidious
            return extract_via_invidious(url, timeout)
        except Exception as exc:
            _record_error(errors, "invidious", exc)

    # Layer 7: SSRN via Semantic Scholar (academic paper fallback)
    if _is_ssrn_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_semantic_scholar
            return extract_via_semantic_scholar(url, timeout)
        except Exception as exc:
            _record_error(errors, "semantic_scholar", exc)

    # Layer 8: akjournals / journal XML direct-page fallback
    if _is_journal_xml_url(url):
        try:
            from obsidian_llm_wiki.ingest.alt_source import extract_via_journal_page
            return extract_via_journal_page(url, timeout)
        except Exception as exc:
            _record_error(errors, "journal_direct", exc)

    # Layer 9: Wayback Machine
    try:
        return _extract_wayback(url, timeout)
    except Exception as exc:
        _record_error(errors, "wayback", exc)

    raise RuntimeError(
        f"All extraction strategies failed for {url}:\n  " + "\n  ".join(errors)
    )


def _extract_liteparse_document(url: str, timeout: int) -> SourceDoc:
    """Try LiteParse for direct documents or citation-linked landing pages."""
    from obsidian_llm_wiki.ingest.liteparse import extract_document_fallback

    return extract_document_fallback(url, timeout)


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
        return host.endswith("ssrn.com")
    except Exception:
        return False


def _is_journal_xml_url(url: str) -> bool:
    """Check if URL is a journal XML article URL."""
    if url.endswith(".xml"):
        return True
    # akjournals.com uses /article-p294.xml pattern — only match on known journal domains
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        if "akjournals" in host and "/article-" in url:
            return True
    except Exception:
        pass
    return False


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


def _extract_defuddle_md(url: str, timeout: int) -> SourceDoc:
    """Extract via defuddle.md web service — best markdown for articles/blogs.

    defuddle.md renders JS-heavy pages and returns clean markdown with
    YAML frontmatter (title, author, published, word_count).
    """
    stripped = url.replace("https://", "").replace("http://", "")
    defuddle_url = f"https://defuddle.md/{stripped}"

    with (
        httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=False),
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "text/html",
            },
        ) as client,
        stream_with_validated_redirects(client, defuddle_url) as resp,
    ):
        if resp.status_code != 200:
            raise RuntimeError(f"defuddle.md returned HTTP {resp.status_code}")
        text = _read_bounded_html(resp, load_config().max_html_bytes).strip()

    if not text or len(text) < 100:
        raise RuntimeError("defuddle.md returned empty content")

    # Parse frontmatter
    title = ""
    content = text
    if text.startswith("---"):
        # Use partition to find the closing --- (handles --- in values better)
        _, sep, rest = text[3:].partition("\n---\n")
        if sep:
            # Split on lines to extract frontmatter block
            lines = text.split("\n")
            fm_lines = []
            found_close = False
            for _i, line in enumerate(lines[1:], 1):  # skip first ---
                if line.strip() == "---":
                    found_close = True
                    break
                fm_lines.append(line)
            if found_close:
                fm_text = "\n".join(fm_lines)
                content = "\n".join(lines[2 + len(fm_lines):]).lstrip()
                for line in fm_text.split("\n"):
                    if line.startswith("title:"):
                        title = line[6:].strip().strip('"').strip("'")

    if not title:
        for line in content.split("\n", 5):
            if line.startswith("# "):
                title = line[2:].strip()
                break
    if not title:
        title = url

    # Strip cover image markdown from content start
    if content.startswith("!["):
        content = content.split("\n", 1)[-1].lstrip() if "\n" in content else content

    if not content or len(content.strip()) < 50:
        raise RuntimeError("defuddle.md content too short")

    return SourceDoc(title=title, content=content.strip(), url=url)


def _extract_trafilatura(url: str, timeout: int) -> SourceDoc:
    """Extract via httpx fetch + trafilatura."""
    import trafilatura

    with (
        httpx.Client(
            **make_client_kwargs(timeout=timeout, follow_redirects=False),
            headers=BROWSER_HEADERS,
        ) as client,
        stream_with_validated_redirects(client, url) as resp,
    ):
        html = _read_bounded_html(resp, load_config().max_html_bytes)

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
