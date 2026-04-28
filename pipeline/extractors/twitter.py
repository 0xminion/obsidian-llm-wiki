"""Twitter/X content extraction via FxTwitter API.

Unlike the general web extractor, this uses the FxTwitter REST API
(api.fxtwitter.com) which bypasses X/Twitter's Cloudflare JS challenge.

Handles both plain tweets and long-form article tweets.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from pipeline.extractors._shared import _validate_url
from pipeline.models import ExtractedSource, SourceType

log = logging.getLogger(__name__)

# ── URL helpers ───────────────────────────────────────────────────────────────

_RE_X_HOST = re.compile(r"^(https?://)(?:x\.com|twitter\.com)(/.*)", re.IGNORECASE)


def _to_fxtwitter(url: str) -> str:
    """Convert an x.com / twitter.com URL into a FxTwitter API URL."""
    m = _RE_X_HOST.match(url)
    if not m:
        return url
    return f"https://api.fxtwitter.com{m.group(2)}".split("?")[0]


def _tweet_text(tweet: dict[str, Any]) -> str:
    """Best-effort tweet text extraction."""
    text = tweet.get("text", "")
    raw = tweet.get("raw_text")
    if isinstance(raw, dict):
        text = raw.get("text", text)
    return text.strip()


def _article_content(tweet: dict[str, Any]) -> tuple[str, str]:
    """Return (title, body) from an article tweet, or ('', '')."""
    article = tweet.get("article") or {}
    if not isinstance(article, dict):
        return "", ""
    title = article.get("title", "").strip()
    body = article.get("preview_text", "") or article.get("text", "") or article.get("content", "")
    return title, body.strip()


# ── Public entry point ────────────────────────────────────────────────────────

def extract_twitter(url: str, **_: Any) -> ExtractedSource:
    """Extract a tweet / X post via FxTwitter.

    Falls back to the article title/body when the tweet itself is just a
    t.co preview card link.
    """
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ExtractedSource(url=url, title=url, content="", type=SourceType.TWITTER)

    api_url = _to_fxtwitter(url)
    log.debug("FxTwitter API: %s", api_url)

    try:
        resp = requests.get(
            api_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning("FxTwitter error for %s: %s", url[:80], e)
        return ExtractedSource(url=url, title=url, content="", type=SourceType.TWITTER)
    except json.JSONDecodeError as e:
        log.warning("FxTwitter JSON parse error for %s: %s", url[:80], e)
        return ExtractedSource(url=url, title=url, content="", type=SourceType.TWITTER)

    tweet = data.get("tweet") or {}
    if not isinstance(tweet, dict):
        log.warning("FxTwitter unexpected response shape for %s", url[:80])
        return ExtractedSource(url=url, title=url, content="", type=SourceType.TWITTER)

    # ── Build content ─────────────────────────────────────────────────────────────
    parts: list[str] = []

    author = tweet.get("author", {}) or {}
    author_name = author.get("name", "").strip()
    screen_name = author.get("screen_name", "").strip()
    created = tweet.get("created_at", tweet.get("date", "")).strip()
    views = tweet.get("views", 0)

    header = ""
    if author_name:
        header = f"Tweet by {author_name} (@{screen_name})" if screen_name else f"Tweet by {author_name}"
    if created:
        header += f" — {created}"
    if header:
        parts.append(header)
    if views:
        parts.append(f"Views: {views:,}")

    tweet_text = _tweet_text(tweet)
    art_title, art_body = _article_content(tweet)

    # If the "tweet text" is just a t.co link and there is an article,
    # the article *is* the real content.
    if tweet_text and not tweet_text.startswith("https://t.co/"):
        parts.append(f"\n{tweet_text}")

    if art_title or art_body:
        parts.append("\n--- Article ---")
        if art_title:
            parts.append(f"\n{art_title}")
        if art_body:
            parts.append(f"\n{art_body}")

    # Quoted tweet
    quote = tweet.get("quote") or {}
    if isinstance(quote, dict) and quote.get("text"):
        parts.append(f"\n--- Quoting @{quote.get('author', {}).get('screen_name', '')} ---")
        parts.append(quote["text"])

    # Reply context
    reply = tweet.get("replying_to_status") or {}
    if isinstance(reply, dict) and reply.get("text"):
        parts.append(f"\n--- In reply to @{reply.get('author', {}).get('screen_name', '')} ---")
        parts.append(reply["text"])

    # Media
    media = tweet.get("media", {}) or {}
    if isinstance(media, dict):
        for m in media.get("all", []):
            if isinstance(m, dict) and m.get("url"):
                parts.append(f"\n[Media: {m['url']}]")

    content = "\n".join(parts).strip()

    # ── Title ───────────────────────────────────────────────────────────────────
    if art_title:
        title = art_title
    elif tweet_text and len(tweet_text) > 5 and not tweet_text.startswith("https://"):
        title = tweet_text[:120]
    elif author_name:
        title = f"Tweet by {author_name}"
    else:
        title = "X post"

    return ExtractedSource(
        url=url,
        title=title,
        content=content,
        type=SourceType.TWITTER,
    )
