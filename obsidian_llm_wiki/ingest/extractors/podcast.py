"""Podcast extractor — Spotify, Apple Podcasts, and generic RSS.

Extracts podcast episode transcripts and metadata via:
1. defuddle.md for episode metadata (title, description, date)
2. iTunes Lookup API → RSS feed → episode audio URL
3. Supadata API for audio transcription (async job polling)
4. Fallback to defuddle.md metadata only if no transcript available

Supported platforms:
  - open.spotify.com/episode/*
  - podcasts.apple.com/*/podcast/*/id*?i=*
  - Generic RSS feeds (*.xml, /feed, /rss)
  - anchor.fm, overcast.fm, podbean.com, buzzsprout.com, etc.

Dependencies: httpx (required), defuddle.md (web service), Supadata API (optional)
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import parse_qs, urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.podcast")

__all__ = ["extract_spotify", "extract_apple_podcast", "extract_generic_podcast"]

# ── Domain matching ─────────────────────────────────────────────────────

_SPOTIFY_HOSTS = frozenset(("open.spotify.com", "spotify.com"))
_APPLE_HOSTS = frozenset(("podcasts.apple.com", "podcasters.apple.com"))
_PODCAST_HOSTS = frozenset((
    "anchor.fm", "overcast.fm", "podbean.com", "buzzsprout.com",
    "transistor.fm", "captivate.fm", "ausha.co",
))


def _is_spotify(parsed, raw: str) -> bool:
    return bool(parsed.hostname and parsed.hostname.lower() in _SPOTIFY_HOSTS)


def _is_apple_podcast(parsed, raw: str) -> bool:
    return bool(parsed.hostname and parsed.hostname.lower() in _APPLE_HOSTS)


def _is_generic_podcast(parsed, raw: str) -> bool:
    return bool(
        parsed.hostname and
        any(h in parsed.hostname.lower() for h in _PODCAST_HOSTS)
    )


def _is_rss_feed(url: str) -> bool:
    return url.endswith(".xml") or "/feed" in url or "/rss" in url


# ── Registration ───────────────────────────────────────────────────────

@register_extractor(_is_spotify)
def extract_spotify(raw_url: str) -> SourceDoc:
    """Extract a Spotify podcast episode with transcript if available."""
    return _extract_podcast(raw_url, platform="spotify")


@register_extractor(_is_apple_podcast)
def extract_apple_podcast(raw_url: str) -> SourceDoc:
    """Extract an Apple Podcasts episode with transcript if available."""
    return _extract_podcast(raw_url, platform="apple")


@register_extractor(_is_generic_podcast)
def extract_generic_podcast(raw_url: str) -> SourceDoc:
    """Extract a generic podcast episode via RSS."""
    return _extract_podcast(raw_url, platform="generic")


# ── Core extraction ─────────────────────────────────────────────────────

def _extract_podcast(raw_url: str, platform: str = "generic") -> SourceDoc:
    """Extract podcast episode with transcript.

    Strategy:
      1. defuddle.md for metadata (title, description, date, host)
      2. Find audio URL via iTunes Lookup API + RSS feed
      3. Supadata API for transcript from audio URL
      4. Fallback to metadata only
    """
    errors: list[str] = []

    # Step 1: Get metadata via defuddle.md
    metadata = _fetch_defuddle_md_metadata(raw_url)
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    author = metadata.get("author", "")
    date = metadata.get("published", "")

    # Step 2: Find audio URL
    audio_url = ""
    if platform == "apple":
        audio_url = _find_apple_audio_url(raw_url)
    elif platform == "spotify":
        audio_url = _find_spotify_audio_url(raw_url)

    # Step 3: Try Supadata transcript from audio URL
    transcript = ""
    if audio_url:
        supadata_key = _get_supadata_key()
        if supadata_key:
            transcript = _supadata_transcribe_audio(audio_url, supadata_key)
            if not transcript:
                errors.append("supadata: no transcript returned")
        else:
            logger.debug("No SUPADATA_API_KEY — skipping transcript")

    # Step 4: Build content
    parts: list[str] = []
    if author:
        parts.append(f"Host: {author}")
    if date:
        parts.append(f"Published: {date}")
    if audio_url:
        parts.append(f"Audio: {audio_url}")
    if transcript:
        parts.append("")
        parts.append("## Transcript")
        parts.append("")
        parts.append(transcript)
    elif description:
        parts.append("")
        parts.append("## Episode Description")
        parts.append("")
        parts.append(description)
    elif description:
        parts.append(f"Description:\n{description}")

    content = "\n".join(parts)
    if not content.strip():
        raise RuntimeError(
            f"Could not extract podcast content from {raw_url}: "
            + "; ".join(errors)
        )

    if not title:
        title = raw_url

    return SourceDoc(title=title, content=content.strip(), url=raw_url)


# ── defuddle.md metadata ────────────────────────────────────────────────

def _fetch_defuddle_md_metadata(url: str) -> dict[str, str]:
    """Fetch episode metadata via defuddle.md web service."""
    stripped = url.replace("https://", "").replace("http://", "")
    defuddle_url = f"https://defuddle.md/{stripped}"

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={
                "User-Agent": BROWSER_HEADERS["User-Agent"],
                "Accept": "text/html",
            },
        ) as client:
            resp = client.get(defuddle_url)
        if resp.status_code != 200:
            return {}

        text = resp.text.strip()
        if not text or len(text) < 50:
            return {}

        metadata: dict[str, str] = {}
        if text.startswith("---"):
            fm_end = text.find("---", 3)
            if fm_end > 0:
                fm_text = text[3:fm_end].strip()
                for line in fm_text.split("\n"):
                    if line.startswith("title:"):
                        metadata["title"] = line[6:].strip().strip('"').strip("'")
                    elif line.startswith("author:"):
                        metadata["author"] = line[7:].strip().strip('"').strip("'")
                    elif line.startswith("published:"):
                        metadata["published"] = line[11:].strip().strip('"').strip("'")
                    elif line.startswith("description:"):
                        metadata["description"] = line[12:].strip().strip('"').strip("'")
        return metadata
    except Exception as exc:
        logger.debug("defuddle.md failed for %s: %s", url, exc)
        return {}


# ── Apple Podcasts audio URL via iTunes Lookup + RSS ────────────────────

def _find_apple_audio_url(url: str) -> str:
    """Find the audio enclosure URL for an Apple Podcasts episode.

    Uses iTunes Lookup API to get the RSS feed URL, then fetches the RSS
    feed and finds the episode by matching the episode GUID or title.
    """
    # Extract podcast ID and episode ID from URL
    # URL format: https://podcasts.apple.com/us/podcast/{slug}/id{podcast_id}?i={episode_id}
    parsed = urlparse(url)
    podcast_id_match = re.search(r"/id(\d+)", parsed.path)
    qs = parse_qs(parsed.query)
    episode_id = qs.get("i", [""])[0]

    if not podcast_id_match:
        return ""

    podcast_id = podcast_id_match.group(1)

    # Use iTunes Lookup API to get RSS feed URL
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            resp = client.get(
                f"https://itunes.apple.com/lookup?id={podcast_id}",
            )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""

        feed_url = results[0].get("feedUrl", "")
        if not feed_url:
            return ""

        # Fetch RSS feed and find episode
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            resp = client.get(feed_url)
        if resp.status_code != 200:
            return ""

        return _find_episode_audio_in_rss(resp.text, episode_id)
    except Exception as exc:
        logger.debug("Apple audio URL lookup failed: %s", exc)
        return ""


def _find_episode_audio_in_rss(rss_text: str, episode_id: str) -> str:
    """Find the audio enclosure URL for a specific episode in RSS XML."""
    items = re.findall(r"<item>(.*?)</item>", rss_text, re.DOTALL)
    for item in items:
        # Match by episode ID in GUID or link
        if episode_id and episode_id in item:
            enclosure = re.search(r'<enclosure[^>]+url="([^"]+)"', item)
            if enclosure:
                return enclosure.group(1).replace("&amp;", "&")
        # Fallback: match by title (if episode_id not in item)
        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
        if title_m and episode_id:
            # Sometimes the episode ID appears in the GUID
            guid_m = re.search(r"<guid[^>]*>(.*?)</guid>", item, re.DOTALL)
            if guid_m and episode_id in guid_m.group(1):
                enclosure = re.search(r'<enclosure[^>]+url="([^"]+)"', item)
                if enclosure:
                    return enclosure.group(1).replace("&amp;", "&")
    return ""


# ── Spotify audio URL ────────────────────────────────────────────────────

def _find_spotify_audio_url(url: str) -> str:
    """Find the audio URL for a Spotify episode via cross-platform search.

    Most podcasts are cross-published. If the Spotify episode is also on
    Apple Podcasts, we can find the audio URL via iTunes Lookup → RSS.
    We use the podcast title from defuddle.md metadata to search iTunes.
    """
    # Get episode metadata from defuddle.md
    metadata = _fetch_defuddle_md_metadata(url)
    title = metadata.get("title", "")
    author = metadata.get("author", "")

    if not title:
        return ""

    # Search iTunes for the same podcast
    search_term = title.strip()
    if author:
        # Use author (podcast show name) for better matching
        search_term = author.strip()

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            # Search iTunes podcast directory
            resp = client.get(
                "https://itunes.apple.com/search",
                params={
                    "term": search_term,
                    "media": "podcast",
                    "limit": 5,
                },
            )
            if resp.status_code != 200:
                return ""

            data = resp.json()
            results = data.get("results", [])
            if not results:
                return ""

            # Try each result's feed URL
            for result in results:
                feed_url = result.get("feedUrl", "")
                if not feed_url:
                    continue

                # Fetch RSS feed
                rss_resp = client.get(feed_url)
                if rss_resp.status_code != 200:
                    continue

                # Search for episode matching the title
                audio = _find_episode_audio_by_title(
                    rss_resp.text, title,
                )
                if audio:
                    logger.info(
                        "Found cross-platform audio for Spotify episode '%s' "
                        "via iTunes podcast '%s'",
                        title, result.get("collectionName", ""),
                    )
                    return audio

    except Exception as exc:
        logger.debug("Spotify cross-platform search failed: %s", exc)

    return ""


def _find_episode_audio_by_title(rss_text: str, target_title: str) -> str:
    """Find an episode's audio URL in RSS XML by matching title (fuzzy)."""
    items = re.findall(r"<item>(.*?)</item>", rss_text, re.DOTALL)

    # Normalize: remove special characters for comparison
    def normalize(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower()).strip()

    target_norm = normalize(target_title)

    for item in items:
        title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
        if not title_m:
            continue
        item_title = title_m.group(1).strip()
        item_norm = normalize(item_title)

        # Check for substring match (either direction)
        if (
            target_norm and item_norm
            and (
                target_norm in item_norm
                or item_norm in target_norm
            )
        ):
            enclosure = re.search(r'<enclosure[^>]+url="([^"]+)"', item)
            if enclosure:
                return enclosure.group(1).replace("&amp;", "&")

    return ""


# ── Supadata transcription ───────────────────────────────────────────────

def _get_supadata_key() -> str:
    """Get the Supadata API key from environment."""
    import os
    return os.environ.get("SUPADATA_API_KEY", "")


def _supadata_transcribe_audio(audio_url: str, api_key: str) -> str:
    """Transcribe audio URL via Supadata API with async job polling.

    Returns the transcript text, or empty string if transcription fails
    or times out.
    """
    try:
        with httpx.Client(timeout=30) as client:
            # Submit transcription job
            resp = client.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": audio_url, "text": "true", "mode": "auto"},
                headers={"x-api-key": api_key},
            )

            if resp.status_code == 200:
                # Synchronous response
                data = resp.json()
                return _extract_supadata_content(data)
            elif resp.status_code == 202:
                # Async job — poll for completion
                job_data = resp.json()
                job_id = job_data.get("jobId", "")
                if not job_id:
                    return ""
                return _poll_supadata_job(job_id, api_key)
            else:
                logger.debug(
                    "Supadata returned %d for audio URL: %s",
                    resp.status_code, resp.text[:200],
                )
                return ""
    except Exception as exc:
        logger.debug("Supadata transcription failed: %s", exc)
        return ""


def _poll_supadata_job(job_id: str, api_key: str, max_polls: int = 60) -> str:
    """Poll Supadata async job until completion."""
    for _i in range(max_polls):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"https://api.supadata.ai/v1/transcript/{job_id}",
                    headers={"x-api-key": api_key},
                )
            if resp.status_code == 200:
                data = resp.json()
                content = _extract_supadata_content(data)
                if content and len(content) > 100:
                    return content
                # Still processing
                status = data.get("status", "")
                if status in ("failed", "error"):
                    logger.debug("Supadata job %s failed: %s", job_id, status)
                    return ""
            elif resp.status_code == 202:
                pass  # Still processing
            else:
                logger.debug("Supadata poll returned %d", resp.status_code)
                return ""
        except Exception as exc:
            logger.debug("Supadata poll error: %s", exc)
        time.sleep(5)
    logger.debug("Supadata job %s timed out after %d polls", job_id, max_polls)
    return ""


def _extract_supadata_content(data: dict) -> str:
    """Extract transcript text from Supadata response."""
    content = data.get("content", "")
    if isinstance(content, list):
        # Chunked content: join segments
        parts = []
        for segment in content:
            text = segment.get("text", "") if isinstance(segment, dict) else str(segment)
            parts.append(text)
        return " ".join(parts)
    return str(content) if content else ""


# ── Import httpx ─────────────────────────────────────────────────────────

try:
    import httpx
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    logger.warning("httpx not available — podcast extractor disabled")
