"""Twitter/X post and article extractor via VxTwitter API + defuddle.

Handles both regular tweets and X Articles (long-form posts).
Uses VxTwitter API (api.vxtwitter.com) for metadata + tweet text,
and defuddle CLI for full article body extraction when available.

For X Articles:
  - Title comes from article.title (the article headline)
  - Content comes from article.preview_text + defuddle extraction of the tweet page
  - Full article body requires JS rendering (not available without a headless browser)

For regular tweets:
  - Title comes from user_name + tweet text (first 80 chars)
  - Content is the full tweet text from VxTwitter API
"""

from __future__ import annotations

import logging
import re

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.twitter")

__all__ = ["extract_twitter"]


def _is_twitter_url(parsed, raw: str) -> bool:
    """Match Twitter/X URLs."""
    host = (parsed.hostname or "").lower()
    return host in ("x.com", "twitter.com", "www.x.com", "www.twitter.com")


def _extract_tweet_id(url: str) -> str | None:
    """Extract the numeric tweet ID from a Twitter/X URL."""
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else None


def _extract_handle(url: str) -> str | None:
    """Extract the @handle from a Twitter/X URL."""
    m = re.search(r"/([^/]+)/status/\d+", url)
    return m.group(1) if m else None


@register_extractor(_is_twitter_url)
def extract_twitter(raw_url: str) -> SourceDoc:
    """Extract content from a Twitter/X post or article.

    Strategy:
      1. VxTwitter API for tweet metadata (title, text, article info)
      2. Defuddle CLI for full article body extraction
      3. Fallback to trafilatura via extract_web

    Raises:
        RuntimeError: If all extraction strategies fail.
    """
    errors: list[str] = []
    tweet_id = _extract_tweet_id(raw_url)
    handle = _extract_handle(raw_url)

    # ── Primary: Defuddle.md web service ──────────────────────────────
    try:
        source = _extract_via_defuddle_md(raw_url)
        if source:
            logger.info(
                "Defuddle.md: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"defuddle_md: {exc}")

    # ── Fallback: VxTwitter API ────────────────────────────────────────
    try:
        source = _extract_via_vxtwitter(raw_url, tweet_id, handle)
        if source:
            logger.info(
                "VxTwitter: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"vxtwitter: {exc}")

    # ── Fallback: defuddle CLI ────────────────────────────────────────
    try:
        source = _extract_via_defuddle(raw_url)
        if source:
            logger.info("Defuddle CLI fallback: %d chars for %s", len(source.content), raw_url)
            return source
    except Exception as exc:
        errors.append(f"defuddle: {exc}")

    # ── Last resort: web extraction (trafilatura) ─────────────────────
    try:
        from obsidian_llm_wiki.ingest.web import extract_web
        return extract_web(raw_url)
    except Exception as exc:
        errors.append(f"web: {exc}")

    raise RuntimeError(
        f"Twitter extraction failed for {raw_url}: " + "; ".join(errors)
    )


def _extract_via_defuddle_md(url: str) -> SourceDoc | None:
    """Extract full content via defuddle.md web service.

    defuddle.md is a hosted version of defuddle that renders JS-heavy pages
    (including X Articles) and returns clean markdown with YAML frontmatter.
    URL format: https://defuddle.md/<original-url>
    """
    # Build defuddle.md URL
    # Strip https:// from the original URL
    stripped = url.replace("https://", "").replace("http://", "")
    defuddle_url = f"https://defuddle.md/{stripped}"

    with httpx.Client(
        **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
        headers={
            "User-Agent": BROWSER_HEADERS["User-Agent"],
            "Accept": "text/html",
        },
    ) as client:
        resp = client.get(defuddle_url)

    if resp.status_code != 200:
        logger.debug("defuddle.md returned %d for %s", resp.status_code, url)
        return None

    text = resp.text.strip()
    if not text or len(text) < 100:
        return None

    # Parse frontmatter (defuddle.md returns YAML frontmatter + markdown body)
    title = ""
    content = text

    if text.startswith("---"):
        fm_end = text.find("---", 3)
        if fm_end > 0:
            fm_text = text[3:fm_end].strip()
            content = text[fm_end + 3:].strip()

            # Parse YAML frontmatter manually (avoid yaml dependency)
            for line in fm_text.split("\n"):
                if line.startswith("title:"):
                    title = line[6:].strip().strip('"').strip("'")
                elif line.startswith("author:"):
                    pass  # Author available but not needed in SourceDoc
                elif line.startswith("word_count:"):
                    pass  # Available but not needed

    if not title:
        # Try first # heading
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
        return None

    return SourceDoc(title=title, content=content.strip(), url=url)


def _extract_via_vxtwitter(
    url: str, tweet_id: str | None, handle: str | None,
) -> SourceDoc | None:
    """Fetch tweet data via VxTwitter API (api.vxtwitter.com).

    Returns None if the API is unavailable. Raises RuntimeError on auth errors.
    """
    if not tweet_id or not handle:
        return None

    api_url = f"https://api.vxtwitter.com/{handle}/status/{tweet_id}"

    with httpx.Client(
        **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
        headers={
            "User-Agent": BROWSER_HEADERS["User-Agent"],
            "Accept": "application/json",
        },
    ) as client:
        resp = client.get(api_url)

    if resp.status_code != 200:
        logger.debug("VxTwitter returned %d for %s", resp.status_code, url)
        return None

    data = resp.json()

    # Extract tweet metadata
    author_name = data.get("user_name", "") or ""
    author_handle = data.get("user_screen_name", "") or handle or ""
    tweet_text = data.get("text", "") or ""
    article = data.get("article", {}) or {}
    article_title = article.get("title", "") or ""
    article_preview = article.get("preview_text", "") or ""
    likes = data.get("likes", 0) or 0
    retweets = data.get("retweets", 0) or 0
    date = data.get("date", "") or ""
    data.get("lang", "") or ""

    # Determine if this is an X Article
    is_article = bool(article) and bool(article_title)

    if is_article:
        # X Article: use article title as the source title
        title = article_title

        # Try to get full article body via defuddle
        body = _extract_article_body_via_defuddle(url)

        if not body or len(body) < 100:
            # Fall back to preview text + tweet text
            body = article_preview
            if tweet_text and tweet_text != f"http://x.com/i/article/{tweet_id}":
                body = f"{body}\n\n{tweet_text}"

        if not body or len(body.strip()) < 50:
            return None

        # Add metadata
        content_parts = [
            f"Author: @{author_handle} ({author_name})",
            f"Date: {date}",
            f"Engagement: {likes:,} likes, {retweets:,} retweets",
            "",
            body.strip(),
        ]
        content = "\n".join(content_parts)
    else:
        # Regular tweet: use tweet text as both title and content
        if not tweet_text or "http://x.com/i/article/" in tweet_text:
            # Tweet text is just a link to an article — use preview
            if article_preview:
                tweet_text = article_preview
            else:
                return None

        # Title: first ~80 chars of tweet text, or author name
        title = tweet_text[:80].replace("\n", " ").strip()
        if not title:
            title = f"@{author_handle} ({author_name})"

        content_parts = [
            f"Author: @{author_handle} ({author_name})",
            f"Date: {date}",
            f"Engagement: {likes:,} likes, {retweets:,} retweets",
            "",
            tweet_text.strip(),
        ]
        content = "\n".join(content_parts)

    if not content or len(content.strip()) < 50:
        return None

    return SourceDoc(title=title, content=content.strip(), url=url)


def _extract_article_body_via_defuddle(tweet_url: str) -> str:
    """Try to extract full article body using defuddle CLI on the tweet URL.

    Defuddle can extract article preview text from the tweet page,
    even though it can't render the full JS article body.
    """
    import os
    import shutil
    import subprocess

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return ""

    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", tweet_url, "--md"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", tweet_url, "--md"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, env=env,
        )
        if proc.returncode != 0:
            return ""

        output = proc.stdout.strip()
        if not output or len(output) < 50:
            return ""

        # Extract the article body from defuddle output
        # Defuddle returns markdown starting with image + title + preview text
        lines = output.split("\n")
        body_lines: list[str] = []
        in_body = False
        for line in lines:
            # Skip image markdown
            if line.startswith("!["):
                continue
            # Skip empty lines at the start
            if not in_body and not line.strip():
                continue
            in_body = True
            body_lines.append(line)

        body = "\n".join(body_lines).strip()
        # Remove the article URL link if present
        body = re.sub(r"\[https?://x\.com/i/article/\d+\]\(.*?\)", "", body).strip()

        return body if len(body) > 50 else ""

    except Exception:
        return ""


def _extract_via_defuddle(url: str) -> SourceDoc | None:
    """Fallback: extract via defuddle CLI directly."""
    import os
    import shutil
    import subprocess

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return None

    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--md"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", url, "--md"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, env=env,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None

        output = proc.stdout.strip()
        if len(output) < 50:
            return None

        # Extract title from first # heading
        lines = output.split("\n", 3)
        title = ""
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()

        # If no heading, try defuddle --json for metadata
        if not title:
            title = _defuddle_metadata_title(url)

        if not title:
            title = url

        # Strip image markdown from content
        content = output
        if content.startswith("!["):
            content = content.split("\n", 1)[-1].lstrip() if "\n" in content else content

        if len(content) < 50:
            return None

        return SourceDoc(title=title, content=content, url=url)

    except Exception:
        return None


def _defuddle_metadata_title(url: str) -> str:
    """Fetch page title via defuddle --json."""
    import json
    import os
    import shutil
    import subprocess

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return ""

    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--json"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", url, "--json"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, env=env,
        )
        if proc.returncode != 0:
            return ""
        data = json.loads(proc.stdout)
        return (data.get("title") or "").strip()
    except Exception:
        return ""
