"""Spotify + generic podcast extractor — HTML scrape episode descriptions + metadata.

Spotify podcast episodes expose a JSON-LD page via their web player URL.
Generic podcasts use an RSS feed.  This extractor tries Spotify first, then RSS.

Dependencies (optional): ``httpx``.
Install with: ``pip install okf-pipeline[podcast]``
"""

from __future__ import annotations

import logging
import re

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

logger = logging.getLogger("obswiki.ingest.extractors.podcast")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import httpx

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# ── Domain matching ─────────────────────────────────────────────────────

_SPOTIFY_HOSTS = frozenset(("open.spotify.com", "Spotify.com"))
_PODCAST_HOSTS = frozenset((
    "anchor.fm", "overcast.fm", "podbean.com", "buzzsprout.com",
    "transistor.fm", "captivate.fm", "ausha.co",
))


def _is_spotify(parsed, raw: str) -> bool:
    return bool(parsed.hostname and parsed.hostname.lower() in _SPOTIFY_HOSTS)


def _is_generic_podcast(parsed, raw: str) -> bool:
    return bool(parsed.hostname and any(h in parsed.hostname.lower() for h in _PODCAST_HOSTS))


def _is_rss_feed(url: str) -> bool:
    return url.endswith(".xml") or "/feed" in url or "/rss" in url


# ── Registration ───────────────────────────────────────────────────────

if _DEPS_AVAILABLE:

    @register_extractor(_is_spotify)
    def extract_spotify(raw_url: str) -> SourceDoc:  # type: ignore[valid-type]
        """Extract a Spotify podcast episode via HTML + JSON-LD metadata.

        Falls back to description-only if full metadata is unavailable.
        """
        try:
            response = httpx.get(raw_url, headers=_HEADERS, timeout=30, follow_redirects=True)  # type: ignore[union-attr]
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Spotify URL '{raw_url}': {exc}") from exc

        html = response.text

        # Extract JSON-LD structured data.
        json_ld = _extract_json_ld(html)
        if json_ld:
            title = json_ld.get("name") or json_ld.get("headline", raw_url)
            description = json_ld.get("description", "")
            author_raw = json_ld.get("author", "")
            author = (
                author_raw.get("name", "") if isinstance(author_raw, dict)
                else str(author_raw)
            )
            date = json_ld.get("datePublished", "")
        else:
            title = _extract_og_title(html) or raw_url
            description = _extract_meta_description(html) or ""
            author = _extract_og_site_name(html) or ""
            date = ""

        # Spotify episodes embed audio duration in the page.
        duration = _extract_duration(html)

        parts: list[str] = []
        if author:
            parts.append(f"Host: {author}")
        if date:
            parts.append(f"Published: {date}")
        if duration:
            parts.append(f"Duration: {duration}")
        if description:
            parts.append(f"Description:\n{description}")

        content = "\n\n".join(parts)
        if not content.strip():
            raise RuntimeError(f"Could not extract content from Spotify URL: {raw_url}")

        return SourceDoc(title=title, content=content, url=raw_url)

    @register_extractor(_is_generic_podcast)
    def extract_podcast_rss(raw_url: str) -> SourceDoc:  # type: ignore[valid-type]
        """Extract a podcast episode via RSS feed.

        Fetches the RSS XML and extracts title, description, publication date,
        and enclosure (audio file) URL.
        """
        try:
            response = httpx.get(raw_url, headers=_HEADERS, timeout=30, follow_redirects=True)  # type: ignore[union-attr]
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch podcast RSS '{raw_url}': {exc}") from exc

        xml_text = response.text

        title = _extract_rss_title(xml_text) or raw_url
        description = _extract_rss_description(xml_text) or ""
        pub_date = _extract_rss_date(xml_text) or ""
        audio_url = _extract_rss_enclosure(xml_text) or ""

        parts: list[str] = []
        if pub_date:
            parts.append(f"Published: {pub_date}")
        if audio_url:
            parts.append(f"Audio: {audio_url}")
        if description:
            parts.append(f"Description:\n{description}")

        content = "\n\n".join(parts)
        if not content.strip():
            raise RuntimeError(f"Could not extract content from podcast RSS: {raw_url}")

        return SourceDoc(title=title, content=content, url=raw_url)


# ── Helpers ─────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    ),
}

_OG_TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]+)"')
_META_DESC_RE = re.compile(r'<meta name="description" content="([^"]+)"')
_JSON_LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)
_DURATION_RE = re.compile(r'"duration":"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)"')


def _extract_og_title(html: str) -> str:
    m = _OG_TITLE_RE.search(html)
    return m.group(1) if m else ""


def _extract_meta_description(html: str) -> str:
    m = _META_DESC_RE.search(html)
    return m.group(1) if m else ""


def _extract_og_site_name(html: str) -> str:
    m = re.search(r'<meta property="og:site_name" content="([^"]+)"', html)
    return m.group(1) if m else ""


def _extract_duration(html: str) -> str:
    m = _DURATION_RE.search(html)
    if not m:
        return ""
    h, m_i, s = (int(x) if x else 0 for x in m.groups())
    if h:
        return f"{h}h {m_i}m"
    if m_i:
        return f"{m_i}m {s}s"
    return f"{s}s"


def _extract_json_ld(html: str) -> dict | None:
    """Extract and parse JSON-LD structured data from HTML."""
    m = _JSON_LD_RE.search(html)
    if not m:
        return None
    try:
        import json
        return json.loads(m.group(1))
    except Exception:
        return None


# RSS extraction helpers (regex-based, no feedparser dep).

_RSS_TITLE_RE = re.compile(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", re.IGNORECASE)
_RSS_DESC_RE = re.compile(
    r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>",
    re.IGNORECASE | re.DOTALL,
)
_RSS_DATE_RE = re.compile(r"<pubDate>(.*?)</pubDate>", re.IGNORECASE)
_RSS_ENC_RE = re.compile(r"<enclosure[^>]+url=\"([^\"]+)\"", re.IGNORECASE)


def _extract_rss_title(xml: str) -> str:
    # Skip the channel-level <title> (first one), get item-level.
    titles = _RSS_TITLE_RE.findall(xml)
    return titles[1].strip() if len(titles) > 1 else ""


def _extract_rss_description(xml: str) -> str:
    descs = _RSS_DESC_RE.findall(xml)
    desc = descs[1].strip() if len(descs) > 1 else ""
    # Strip HTML tags.
    desc = re.sub(r"<[^>]+>", "", desc)
    return desc.strip()


def _extract_rss_date(xml: str) -> str:
    dates = _RSS_DATE_RE.findall(xml)
    return dates[0].strip() if dates else ""


def _extract_rss_enclosure(xml: str) -> str:
    m = _RSS_ENC_RE.search(xml)
    return m.group(1).strip() if m else ""
