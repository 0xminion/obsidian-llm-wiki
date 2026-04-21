"""Podcast episode extraction.

Supports: Apple Podcasts, Spotify, Overcast, Pocket Casts, Castbox, Podbean,
Podchaser, Podcast Addict, direct RSS feeds, PodcastIndex.org, and any
provider with an RSS link in the page.

Chain: provider lookup → RSS parse → transcription.
Fails loudly if transcript unavailable — never metadata-only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote, urlparse, parse_qs

from pipeline.config import Config
from pipeline.models import ExtractedSource, SourceType
from pipeline.extractors._shared import (
    _curl_get,
    _run,
    transcribe_with_whisper,
    transcribe_assemblyai,
    ExtractionError,
)

log = logging.getLogger(__name__)


# ─── Provider Patterns ──────────────────────────────────────────────────────

# Direct RSS feed indicators
_RSS_EXTENSIONS = (".xml", ".rss", "/feed", "/rss", "/podcast.xml")
_RSS_HOSTS = (
    "feeds.", "feed.", "buzzsprout.com", "libsyn.com", "megaphone.fm",
    "anchor.fm", "podbean.com", "transistor.fm", "simplecast.com",
    "captivate.fm", "fireside.fm", "rss.com", "podomatic.com",
    "spreaker.com", "audioboom.com", "omnycontent.com", "chtbl.com",
    "art19.com", "acast.com", "redcircle.com", "podigee.com",
)

# Platform-specific resolvers
_KNOWN_PROVIDERS = {
    "podcasts.apple.com": "apple",
    "open.spotify.com": "spotify",
    "spotify.com": "spotify",
    "overcast.fm": "overcast",
    "pocketcasts.com": "pocketcasts",
    "castbox.fm": "castbox",
    "podchaser.com": "podchaser",
    "podcastaddict.com": "podcastaddict",
    "pod.link": "podlink",
    "podcastindex.org": "podcastindex",
    "google.com/podcasts": "google",
    "youtube.com": "youtube_music",
    "music.youtube.com": "youtube_music",
}


def extract_podcast(url: str, cfg: Config) -> ExtractedSource:
    """Extract podcast episode via provider-specific lookup → RSS → transcription.

    Supports any podcast platform that either:
    1. Has a direct RSS feed URL
    2. Can be resolved to RSS via platform-specific logic
    3. Has an RSS link embedded in the page

    Chain: provider lookup → RSS parse → transcription.
    Falls back to RSS description if transcription unavailable.
    FAILS LOUDLY if transcript unavailable — never metadata-only.
    """
    timeout = cfg.extract_timeout

    # Step 1: Find podcast RSS feed URL and episode info
    feed_url, podcast_name, episode_id, episode_slug, description = (
        _resolve_podcast_feed(url, timeout)
    )

    # Step 2: Parse RSS feed for episode
    audio_url = ""
    rss_description = ""
    episode_title = ""

    if feed_url:
        try:
            audio_url, rss_description, episode_title = _parse_rss_episode(
                feed_url, episode_id, episode_slug, timeout
            )
        except Exception as e:
            log.debug("RSS parse failed: %s", e)

    if not description:
        description = rss_description

    # Step 3: Transcribe audio — FAIL LOUDLY if no audio
    if not audio_url:
        log.error("No audio URL found for podcast: %s", url)
        raise ExtractionError(
            f"Podcast extraction failed: no audio URL found. "
            f"Feed: {feed_url or 'not found'}, Episode: {episode_title or 'not found'}. "
            f"Cannot extract transcript without audio."
        )

    transcript = _transcribe_podcast_audio(audio_url, cfg)
    if not transcript or len(transcript) < 100:
        log.error("Transcription failed for podcast audio: %s", audio_url)
        raise ExtractionError(
            f"Podcast transcription failed for {url}. "
            f"Audio URL: {audio_url}. "
            f"Transcript length: {len(transcript) if transcript else 0} chars. "
            f"Check AssemblyAI/whisper configuration."
        )

    content = (
        f"Podcast: {podcast_name}\n"
        f"Episode: {episode_title}\n"
        f"URL: {url}\n\n"
        f"## Transcript\n\n{transcript}"
    )

    return ExtractedSource(
        url=url,
        title=episode_title or podcast_name or url,
        content=content,
        type=SourceType.PODCAST,
        author=podcast_name,
    )


def _detect_provider(url: str) -> str:
    """Detect podcast provider from URL. Returns provider key or 'unknown'."""
    url_lower = url.lower()
    for pattern, provider in _KNOWN_PROVIDERS.items():
        if pattern in url_lower:
            return provider

    # Check for direct RSS
    if url.startswith("http") and any(ext in url_lower for ext in _RSS_EXTENSIONS):
        return "direct_rss"
    if url.startswith("http") and any(host in url_lower for host in _RSS_HOSTS):
        return "direct_rss"

    return "unknown"


def _resolve_podcast_feed(
    url: str, timeout: int
) -> tuple[str, str, str, str, str]:
    """Resolve podcast feed URL from any provider.

    Returns (feed_url, podcast_name, episode_id, episode_slug, description).
    """
    # Extract IDs and slug from URL
    id_match = re.search(r"id(\d+)", url)
    ep_match = re.search(r"[?&]i=(\d+)", url)
    podcast_id = id_match.group(1) if id_match else ""
    episode_id = ep_match.group(1) if ep_match else ""
    episode_slug = ""
    slug_match = re.search(r"/podcast/([^/]+)/id\d+", url)
    if slug_match:
        episode_slug = slug_match.group(1).replace("-", " ").strip()

    feed_url = ""
    podcast_name = ""
    description = ""
    provider = _detect_provider(url)

    log.debug("Podcast provider detected: %s for %s", provider, url[:80])

    # ─── Apple Podcasts: iTunes API ───────────────────────────────────────
    if provider == "apple":
        feed_url, podcast_name, description = _resolve_apple_podcast(
            podcast_id, episode_slug, timeout
        )

    # ─── Spotify ──────────────────────────────────────────────────────────
    elif provider == "spotify":
        feed_url, podcast_name = _resolve_spotify_podcast(url, timeout)

    # ─── Overcast ─────────────────────────────────────────────────────────
    elif provider == "overcast":
        feed_url, podcast_name = _resolve_overcast_podcast(url, timeout)

    # ─── Pocket Casts ─────────────────────────────────────────────────────
    elif provider == "pocketcasts":
        feed_url, podcast_name = _resolve_pocketcasts_podcast(url, timeout)

    # ─── Castbox ──────────────────────────────────────────────────────────
    elif provider == "castbox":
        feed_url, podcast_name = _resolve_castbox_podcast(url, timeout)

    # ─── Podchaser ────────────────────────────────────────────────────────
    elif provider == "podchaser":
        feed_url, podcast_name = _resolve_podchaser_podcast(url, timeout)

    # ─── PodcastIndex.org ─────────────────────────────────────────────────
    elif provider == "podcastindex":
        feed_url, podcast_name = _resolve_podcastindex(url, timeout)

    # ─── Direct RSS feed URL ──────────────────────────────────────────────
    elif provider == "direct_rss":
        feed_url = url
        podcast_name = _guess_name_from_feed_url(url)

    # ─── Generic: fetch page, look for RSS link ───────────────────────────
    else:
        feed_url, podcast_name = _resolve_generic_podcast(url, timeout)

    # Final fallback: try PodcastIndex search if still no feed
    if not feed_url and podcast_name:
        log.info("No feed from provider, trying PodcastIndex search for: %s", podcast_name)
        feed_url, _ = _search_podcastindex(podcast_name, timeout)

    return feed_url, podcast_name, episode_id, episode_slug, description


# ─── Provider-Specific Resolvers ────────────────────────────────────────────

def _resolve_apple_podcast(
    podcast_id: str, episode_slug: str, timeout: int
) -> tuple[str, str, str]:
    """Resolve Apple Podcasts feed via iTunes API."""
    feed_url = ""
    podcast_name = ""
    description = ""

    if not podcast_id:
        return "", "", ""

    # Strategy 1: iTunes lookup (entity=podcast)
    lookup_json = _curl_get(
        f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast",
        timeout=timeout,
    )
    if lookup_json:
        try:
            lookup = json.loads(lookup_json)
            if lookup.get("resultCount", 0) > 0 and lookup.get("results"):
                feed_url = lookup["results"][0].get("feedUrl", "")
                podcast_name = lookup["results"][0].get("collectionName", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Strategy 2: iTunes lookup (entity=podcastEpisode)
    if not feed_url:
        lookup_ep_json = _curl_get(
            f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcastEpisode&limit=50",
            timeout=timeout,
        )
        if lookup_ep_json:
            try:
                lookup_ep = json.loads(lookup_ep_json)
                if lookup_ep.get("resultCount", 0) > 0:
                    for r in lookup_ep.get("results", []):
                        if r.get("feedUrl"):
                            feed_url = r["feedUrl"]
                            podcast_name = r.get("collectionName", podcast_name)
                            break
                    if episode_slug and feed_url:
                        for r in lookup_ep.get("results", []):
                            ep_title = r.get("trackName", "")
                            if ep_title and _episode_title_match(episode_slug, ep_title):
                                if not description:
                                    description = r.get("description", "")
                                break
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # Strategy 3: iTunes search by name
    if not feed_url and episode_slug:
        search_json = _curl_get(
            f"https://itunes.apple.com/search?term={quote(episode_slug)}&media=podcast&limit=5",
            timeout=timeout,
        )
        if search_json:
            try:
                search = json.loads(search_json)
                if search.get("results"):
                    feed_url = search["results"][0].get("feedUrl", "")
                    podcast_name = search["results"][0].get("collectionName", podcast_name)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    return feed_url, podcast_name, description


def _resolve_spotify_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve Spotify podcast to RSS feed via show/episode page scraping."""
    content = _curl_get(url, timeout=timeout)
    podcast_name = ""

    # Try to extract show name from page title/meta
    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        title_text = title_match.group(1)
        podcast_name = re.sub(r"\s*[|\-]\s*(?:Podcast\s+)?on\s+Spotify.*", "", title_text)
        podcast_name = re.sub(r"\s*[|\-]\s*Spotify.*", "", podcast_name).strip()

    # Try PodcastIndex first (non-iTunes)
    if podcast_name:
        feed_url, _ = _search_podcastindex(podcast_name, timeout)
        if feed_url:
            return feed_url, podcast_name

    # Fallback: iTunes search
    if podcast_name:
        search_json = _curl_get(
            f"https://itunes.apple.com/search?term={quote(podcast_name)}&media=podcast&limit=1",
            timeout=timeout,
        )
        if search_json:
            try:
                search = json.loads(search_json)
                if search.get("results"):
                    feed_url = search["results"][0].get("feedUrl", "")
                    if feed_url:
                        return feed_url, podcast_name
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    return "", podcast_name


def _resolve_overcast_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve Overcast.fm podcast/episode to RSS feed.

    Overcast pages contain <link rel="alternate" type="application/rss+xml" href="...">
    """
    content = _curl_get(url, timeout=timeout)
    podcast_name = ""

    # Extract show name from title
    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        podcast_name = re.sub(r"\s*[–—\-|]\s*Overcast.*", "", title_match.group(1)).strip()

    # Look for RSS link
    rss_match = re.search(
        r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if rss_match:
        return rss_match.group(1), podcast_name

    # Overcast also stores feed URL in meta
    feed_match = re.search(r'data-feedurl="([^"]+)"', content)
    if feed_match:
        return feed_match.group(1), podcast_name

    return "", podcast_name


def _resolve_pocketcasts_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve Pocket Casts podcast to RSS feed.

    Pocket Casts URLs: pocketcasts.com/podcasts/SHOW_ID or /episodes/EP_ID
    """
    content = _curl_get(url, timeout=timeout)
    podcast_name = ""

    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        podcast_name = re.sub(r"\s*\|\s*Pocket Casts.*", "", title_match.group(1)).strip()

    # Look for RSS link
    rss_match = re.search(
        r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if rss_match:
        return rss_match.group(1), podcast_name

    # Look for feed URL in page data
    feed_match = re.search(r'"feedUrl"\s*:\s*"([^"]+)"', content)
    if feed_match:
        return feed_match.group(1), podcast_name

    return "", podcast_name


def _resolve_castbox_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve Castbox.fm podcast to RSS feed.

    Castbox URLs: castbox.fm/channel/... or /episode/...
    """
    content = _curl_get(url, timeout=timeout)
    podcast_name = ""

    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        podcast_name = re.sub(r"\s*[-|]\s*Castbox.*", "", title_match.group(1)).strip()

    # Castbox embeds feed data in page
    feed_match = re.search(r'"feedUrl"\s*:\s*"([^"]+)"', content)
    if feed_match:
        return feed_match.group(1), podcast_name

    rss_match = re.search(
        r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if rss_match:
        return rss_match.group(1), podcast_name

    return "", podcast_name


def _resolve_podchaser_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve Podchaser podcast to RSS feed.

    Podchaser URLs: podchaser.com/podcasts/SLUG-ID or /episodes/SLUG-ID
    """
    content = _curl_get(url, timeout=timeout)
    podcast_name = ""

    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        podcast_name = re.sub(r"\s*[-|]\s*Podchaser.*", "", title_match.group(1)).strip()

    # Look for RSS link
    rss_match = re.search(
        r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if rss_match:
        return rss_match.group(1), podcast_name

    # Podchaser embeds feed URL in JSON-LD or page data
    feed_match = re.search(r'"url"\s*:\s*"(https?://[^"]*\.(xml|rss)[^"]*)"', content)
    if feed_match:
        return feed_match.group(1), podcast_name

    return "", podcast_name


def _resolve_podcastindex(url: str, timeout: int) -> tuple[str, str]:
    """Resolve PodcastIndex.org URL to RSS feed.

    PodcastIndex URLs: podcastindex.org/podcast/ID or /episode/ID
    """
    # Extract ID from URL
    id_match = re.search(r"/(?:podcast|episode)/(\d+)", url)
    if not id_match:
        return "", ""

    podcast_id = id_match.group(1)

    # Use PodcastIndex API (public, no auth needed for lookups)
    api_url = f"https://api.podcastindex.org/api/1.0/podcasts/byfeedid?id={podcast_id}"
    response = _curl_get(api_url, timeout=timeout)
    if response:
        try:
            data = json.loads(response)
            feed = data.get("feed", {})
            if feed:
                return feed.get("url", ""), feed.get("title", "")
        except (json.JSONDecodeError, KeyError):
            pass

    return "", ""


# ─── PodcastIndex Search (non-iTunes fallback) ──────────────────────────────

def _search_podcastindex(query: str, timeout: int) -> tuple[str, str]:
    """Search PodcastIndex.org for a podcast by name. Returns (feed_url, title).

    This is the primary non-iTunes search backend.
    PodcastIndex is open, community-maintained, and includes podcasts
    not listed in Apple's directory.
    """
    if not query or len(query.strip()) < 3:
        return "", ""

    api_url = f"https://api.podcastindex.org/api/1.0/search/byterm?q={quote(query)}&max=3"
    response = _curl_get(api_url, timeout=timeout)
    if not response:
        return "", ""

    try:
        data = json.loads(response)
        feeds = data.get("feeds", [])
        if feeds:
            best = feeds[0]
            return best.get("url", ""), best.get("title", "")
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    return "", ""


# ─── Generic Resolver ───────────────────────────────────────────────────────

def _resolve_generic_podcast(url: str, timeout: int) -> tuple[str, str]:
    """Resolve generic podcast URL by fetching page and looking for RSS feed link.

    Tries multiple strategies:
    1. <link type="application/rss+xml"> tag
    2. RSS URL patterns in page text
    3. JSON-LD structured data with feed URL
    4. og:audio or podcast metadata in meta tags
    """
    content = _curl_get(url, timeout=timeout)
    if not content:
        return "", ""

    podcast_name = ""

    # Extract name from title
    title_match = re.search(r"<title>([^<]+)</title>", content)
    if title_match:
        podcast_name = title_match.group(1).strip()
        # Clean common suffixes
        for suffix in [" | Podcast", " - Podcast", " Podcast", " on Apple Podcasts",
                       " on Spotify", " | Listen on", " - Listen Free"]:
            podcast_name = podcast_name.replace(suffix, "").strip()

    # Strategy 1: <link type="application/rss+xml" href="...">
    rss_match = re.search(
        r'<link[^>]+type=["\']application/rss\+xml["\'][^>]+href=["\']([^"\']+)["\']',
        content, re.IGNORECASE,
    )
    if rss_match:
        feed_url = rss_match.group(1)
        # Make relative URLs absolute
        if feed_url.startswith("/"):
            parsed = urlparse(url)
            feed_url = f"{parsed.scheme}://{parsed.netloc}{feed_url}"
        return feed_url, podcast_name

    # Strategy 2: RSS URL pattern in page
    rss_url_match = re.search(
        r'https?://[^\s"<>]+/(?:feed|rss|podcast\.xml)[^\s"<>]*',
        content, re.IGNORECASE,
    )
    if rss_url_match:
        return rss_url_match.group(0), podcast_name

    # Strategy 3: JSON-LD structured data
    jsonld_match = re.search(r'"url"\s*:\s*"(https?://[^"]*\.(xml|rss)[^"]*)"', content)
    if jsonld_match:
        return jsonld_match.group(1), podcast_name

    # Strategy 4: feedUrl in JavaScript data
    feed_match = re.search(r'"feedUrl"\s*:\s*"(https?://[^"]+)"', content)
    if feed_match:
        return feed_match.group(1), podcast_name

    # Strategy 5: data-feed-url attribute
    data_feed_match = re.search(r'data-feed-?url=["\']([^"\']+)["\']', content, re.IGNORECASE)
    if data_feed_match:
        return data_feed_match.group(1), podcast_name

    return "", podcast_name


# ─── RSS Parsing ─────────────────────────────────────────────────────────────

def _guess_name_from_feed_url(url: str) -> str:
    """Guess podcast name from feed URL path."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if "/" in path:
        path = path.split("/")[-1]
    return path.replace("-", " ").replace("_", " ").replace(".xml", "").replace(".rss", "").title()


def _episode_title_match(slug: str, title: str) -> bool:
    """Check if an episode title matches the URL slug.

    Uses keyword overlap: at least 60% of slug words must appear in the title.
    """
    slug_words = set(re.sub(r"[^a-z0-9 ]", "", slug.lower()).split())
    title_words = set(re.sub(r"[^a-z0-9 ]", "", title.lower()).split())
    slug_words = {w for w in slug_words if len(w) > 2}
    title_words = {w for w in title_words if len(w) > 2}
    if not slug_words:
        return False
    overlap = slug_words & title_words
    return len(overlap) / len(slug_words) >= 0.6


def _parse_rss_episode(feed_url: str, episode_id: str, episode_slug: str = "",
                       timeout: int = 30) -> tuple[str, str, str]:
    """Parse RSS feed to find episode audio URL, description, and title.

    Tries matching in order:
      1. By episode ID in GUID/link
      2. By episode title slug (keyword overlap)
      3. Fallback to latest episode

    Returns (audio_url, description, episode_title).
    """
    rss_xml = _curl_get(feed_url, timeout=timeout)
    if not rss_xml:
        return "", "", ""

    try:
        root = ET.fromstring(rss_xml)
    except ET.ParseError:
        return "", "", ""

    items = list(root.iter("item"))

    target_item = None

    # Try to match by episode ID
    if episode_id:
        for item in items:
            guid = item.find("guid")
            link = item.find("link")
            guid_text = guid.text if guid is not None else ""
            link_text = link.text if link is not None else ""
            if (guid_text and episode_id in guid_text) or \
               (link_text and episode_id in link_text):
                target_item = item
                break

    # Try to match by episode title slug (keyword overlap)
    if target_item is None and episode_slug:
        best_score = 0.0
        best_item = None
        for item in items:
            title_elem = item.find("title")
            if title_elem is None or not title_elem.text:
                continue
            item_title = title_elem.text
            slug_words = set(re.sub(r"[^a-z0-9 ]", "", episode_slug.lower()).split())
            title_words = set(re.sub(r"[^a-z0-9 ]", "", item_title.lower()).split())
            slug_words = {w for w in slug_words if len(w) > 2}
            title_words = {w for w in title_words if len(w) > 2}
            if not slug_words:
                continue
            overlap = slug_words & title_words
            score = len(overlap) / len(slug_words)
            if score > best_score:
                best_score = score
                best_item = item
        if best_score >= 0.5 and best_item is not None:
            target_item = best_item
            log.info("RSS: matched episode by title (score=%.2f): %s",
                     best_score,
                     (best_item.find("title").text if best_item.find("title") is not None else "?"))

    # Fallback to latest episode
    if target_item is None and items:
        target_item = items[0]
        log.warning("RSS: no episode match found, falling back to latest episode: %s",
                    (target_item.find("title").text if target_item.find("title") is not None else "?"))

    if target_item is None:
        return "", "", ""

    enclosure = target_item.find("enclosure")
    audio_url = enclosure.get("url", "") if enclosure is not None else ""

    desc_elem = target_item.find("description")
    description = (desc_elem.text or "")[:5000] if desc_elem is not None else ""

    title_elem = target_item.find("title")
    episode_title = (title_elem.text or "") if title_elem is not None else ""

    return audio_url, description, episode_title


# ─── Transcription ───────────────────────────────────────────────────────────

def _transcribe_podcast_audio(audio_url: str, cfg: Config) -> str:
    """Download podcast audio and transcribe with AssemblyAI.

    Falls back to local whisper if AssemblyAI fails.
    """
    timeout = cfg.extract_timeout

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_audio = f.name

    try:
        # Download audio
        dl = _run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", tmp_audio, audio_url],
            timeout=120,
        )
        if dl.returncode != 0 or not os.path.exists(tmp_audio):
            return ""

        # Try AssemblyAI first
        if cfg.assemblyai_api_key:
            transcript = transcribe_assemblyai(tmp_audio, cfg.assemblyai_api_key, timeout)
            if transcript:
                return transcript

        # Fallback to local whisper
        return transcribe_with_whisper(tmp_audio, cfg.whisper_language)

    finally:
        if os.path.exists(tmp_audio):
            os.unlink(tmp_audio)
