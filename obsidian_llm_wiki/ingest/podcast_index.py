"""Podcast Index feed discovery for canonical RSS resolution.

Podcast Index is used only to discover publisher-owned RSS feeds. It does not
supply transcript text itself: callers must resolve RSS ``podcast:transcript``
tags or use the normal generated-transcript fallback chain afterwards.

The API is free to use with a developer key and secret. With no configured
credentials every public function returns an empty result without networking.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass

import httpx

from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.podcast_index")

_API_URL = "https://api.podcastindex.org/api/1.0/search/byterm"

__all__ = [
    "PodcastIndexFeed",
    "discover_feed_urls",
    "get_podcast_index_credentials",
]


@dataclass(frozen=True)
class PodcastIndexFeed:
    """A discovered publisher RSS feed from Podcast Index."""

    feed_url: str
    title: str = ""
    author: str = ""
    feed_id: int | None = None


def get_podcast_index_credentials() -> tuple[str, str]:
    """Return API credentials without ever logging them."""
    return (
        os.environ.get("PODCAST_INDEX_API_KEY", "").strip(),
        os.environ.get("PODCAST_INDEX_API_SECRET", "").strip(),
    )


def _auth_headers(api_key: str, api_secret: str, now: int | None = None) -> dict[str, str]:
    """Build Podcast Index's documented SHA-1 request authentication headers."""
    auth_date = str(now if now is not None else int(time.time()))
    authorization = hashlib.sha1(
        f"{api_key}{api_secret}{auth_date}".encode(),
    ).hexdigest()
    return {
        "User-Agent": BROWSER_HEADERS["User-Agent"],
        "X-Auth-Date": auth_date,
        "X-Auth-Key": api_key,
        "Authorization": authorization,
    }


def discover_feed_urls(
    query: str,
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    max_results: int = 10,
) -> list[PodcastIndexFeed]:
    """Discover candidate canonical RSS feeds for a show/episode query.

    Results are intentionally candidates, not truth. Callers must still match
    an episode title against the fetched RSS feed before treating a result as
    the source for audio or transcript acquisition.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []
    configured_key, configured_secret = get_podcast_index_credentials()
    key = (api_key or configured_key).strip()
    secret = (api_secret or configured_secret).strip()
    if not key or not secret:
        return []

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
        ) as client:
            response = client.get(
                _API_URL,
                params={"q": cleaned_query, "max": max(1, min(max_results, 50))},
                headers=_auth_headers(key, secret),
            )
        if response.status_code in (401, 403):
            logger.warning("Podcast Index rejected the configured API credentials.")
            return []
        response.raise_for_status()
        feeds = response.json().get("feeds", [])
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Podcast Index discovery failed for %r: %s", cleaned_query, exc)
        return []

    discovered: list[PodcastIndexFeed] = []
    seen_urls: set[str] = set()
    for feed in feeds:
        if not isinstance(feed, dict):
            continue
        feed_url = str(feed.get("url") or feed.get("originalUrl") or "").strip()
        if not feed_url or feed_url in seen_urls:
            continue
        seen_urls.add(feed_url)
        raw_id = feed.get("id")
        discovered.append(
            PodcastIndexFeed(
                feed_url=feed_url,
                title=str(feed.get("title", "")),
                author=str(feed.get("author", "")),
                feed_id=raw_id if isinstance(raw_id, int) else None,
            ),
        )
    return discovered
