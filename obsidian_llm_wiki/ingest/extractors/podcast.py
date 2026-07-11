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
import os
import re
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from defusedxml import ElementTree

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.supadata_utils import (
    supadata_rate_limit,
    track_supadata_call,
    validate_supadata_key,
)
from obsidian_llm_wiki.ingest.transcript_resolver import (
    TranscriptResult,
    assemblyai_transcribe_url,
    fetch_public_transcript,
    get_assemblyai_key,
    load_transcript_cache,
    save_transcript_cache,
    validate_assemblyai_key,
)

logger = logging.getLogger("obswiki.ingest.extractors.podcast")

__all__ = [
    "extract_spotify",
    "extract_apple_podcast",
    "extract_generic_podcast",
    "validate_supadata_key",
    "validate_assemblyai_key",
]


@dataclass(frozen=True)
class EpisodeAsset:
    """Canonical public episode media discovered from a podcast RSS feed."""

    title: str = ""
    audio_url: str = ""
    transcript_url: str = ""
    transcript_type: str = ""
    transcript_language: str = ""
    guid: str = ""

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


@register_extractor(lambda _parsed, raw: _is_rss_feed(raw))
def extract_podcast_rss(raw_url: str) -> SourceDoc:
    """Extract the most recent transcript-bearing episode from a public feed."""
    return _extract_podcast(raw_url, platform="rss")


# ── Core extraction ─────────────────────────────────────────────────────

def _extract_podcast(raw_url: str, platform: str = "generic") -> SourceDoc:
    """Extract podcast episode with transcript.

    Strategy:
      1. defuddle.md for metadata (title, description, date, host)
      2. Resolve canonical RSS episode / enclosure / podcast:transcript tag
      3. Local transcript cache
      4. Publisher-provided RSS transcript artifact
      5. Supadata for platforms/media it supports
      6. AssemblyAI remote URL transcription
      7. Local Whisper only as a final acquisition-dependent fallback
      8. Metadata-only source marked as transcript unavailable
    """
    errors: list[str] = []

    # Step 1: Get metadata via defuddle.md
    metadata = _fetch_defuddle_md_metadata(raw_url)
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    author = metadata.get("author", "")
    date = metadata.get("published", "")

    # Step 2: Resolve a canonical RSS episode. Spotify is cross-platform
    # discovered through iTunes/RSS; Apple already exposes its show identity.
    asset = _resolve_episode_asset(raw_url, platform, metadata)
    title = title or asset.title
    audio_url = asset.audio_url
    # RSS GUIDs are only guaranteed unique within one feed. Pair with the
    # canonical enclosure URL to avoid cross-show cache collisions.
    cache_identity = (
        f"{asset.guid}|{audio_url}"
        if asset.guid and audio_url
        else asset.guid or audio_url or raw_url
    )

    # Step 3: Cache is authoritative for previously acquired transcripts.
    transcript_result = load_transcript_cache(cache_identity)
    if transcript_result:
        logger.info("Transcript cache hit for %s", raw_url)

    # Step 4: Publisher-provided artifacts from Podcasting 2.0 RSS.
    if transcript_result is None and asset.transcript_url:
        transcript_result = fetch_public_transcript(
            asset.transcript_url,
            mime_type=asset.transcript_type,
            language=asset.transcript_language,
        )
        if transcript_result:
            save_transcript_cache(cache_identity, transcript_result)

    # Step 4b: Some publishers expose a full transcript in the episode page
    # but omit the Podcasting 2.0 RSS tag. defuddle.md retains that body.
    if transcript_result is None:
        transcript_result = _publisher_transcript_from_metadata(metadata, raw_url)
        if transcript_result:
            save_transcript_cache(cache_identity, transcript_result)

    # Step 5: AssemblyAI is the primary remote media provider. It fetches the
    # public RSS enclosure itself, so the pipeline host does not download it.
    if transcript_result is None and audio_url:
        assemblyai_key = get_assemblyai_key()
        if assemblyai_key:
            transcript_result = assemblyai_transcribe_url(audio_url, assemblyai_key)
            if transcript_result:
                save_transcript_cache(cache_identity, transcript_result)
            else:
                errors.append("assemblyai: no transcript returned")
        else:
            logger.debug("No ASSEMBLYAI_API_KEY — skipping AssemblyAI")

    # Step 6: Supadata remains the second remote fallback, including its
    # platform-specialized media handling when AssemblyAI has no result.
    if transcript_result is None and audio_url:
        supadata_key = _get_supadata_key()
        if supadata_key:
            transcript = _supadata_transcribe_audio(audio_url, supadata_key)
            if transcript:
                transcript_result = TranscriptResult(
                    text=transcript,
                    provider="supadata_remote_url",
                    artifact_url=audio_url,
                )
                save_transcript_cache(cache_identity, transcript_result)
            else:
                errors.append("supadata: no transcript returned")
        else:
            logger.debug("No SUPADATA_API_KEY — skipping Supadata")

    # Step 7: Last resort. This downloads from the pipeline host and may be
    # blocked by the origin, so it deliberately follows all remote resolvers.
    if transcript_result is None and audio_url:
        whisper_transcript = _whisper_fallback_transcribe(audio_url)
        if whisper_transcript:
            transcript_result = TranscriptResult(
                text=whisper_transcript,
                provider="local_faster_whisper",
                artifact_url=audio_url,
            )
            save_transcript_cache(cache_identity, transcript_result)
        else:
            errors.append("whisper: fallback unavailable or failed")

    # Step 4: Build content
    parts: list[str] = []
    if author:
        parts.append(f"Host: {author}")
    if date:
        parts.append(f"Published: {date}")
    if audio_url:
        parts.append(f"Audio: {audio_url}")
    if transcript_result:
        parts.append("")
        parts.append("## Transcript")
        parts.append("")
        parts.append(f"Transcript source: {transcript_result.provider}")
        parts.append("")
        parts.append(transcript_result.text)
    elif description:
        parts.append("")
        parts.append("## Episode Description")
        parts.append("")
        parts.append(description)
    else:
        parts.extend([
            "",
            "## Transcript Status",
            "",
            "Transcript unavailable: no publisher artifact or configured "
            "remote/local transcription provider produced usable text.",
        ])

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
                metadata["body"] = text[fm_end + 3:].strip()
        else:
            metadata["body"] = text
        return metadata
    except Exception as exc:
        logger.debug("defuddle.md failed for %s: %s", url, exc)
        return {}


def _publisher_transcript_from_metadata(
    metadata: dict[str, str],
    source_url: str,
) -> TranscriptResult | None:
    """Use a publisher's explicit transcript section before generated ASR."""
    body = metadata.get("body", "")
    heading = re.search(
        r"(?im)^#{1,3}\s+(?:full\s+)?transcript\s*$",
        body,
    )
    if not heading:
        return None
    transcript = body[heading.end():].strip()
    if len(transcript) < 200:
        return None
    return TranscriptResult(
        text=transcript,
        provider="publisher_episode_page",
        artifact_url=source_url,
    )


# ── Canonical RSS episode discovery ─────────────────────────────────────


def _resolve_episode_asset(
    raw_url: str,
    platform: str,
    metadata: dict[str, str],
) -> EpisodeAsset:
    """Resolve an episode's public RSS enclosure and transcript artifact."""
    if platform == "rss":
        return _find_episode_asset_in_rss(_fetch_rss_text(raw_url), allow_first=True)
    if platform == "apple":
        return _find_apple_episode_asset(raw_url, metadata.get("title", ""))
    if platform == "spotify":
        return _find_spotify_episode_asset(raw_url, metadata)
    return EpisodeAsset()


def _fetch_rss_text(feed_url: str) -> str:
    """Fetch an RSS feed without treating a failure as a hard extraction error."""
    if not feed_url:
        return ""
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = client.get(feed_url)
        if response.status_code == 200:
            return response.text
    except httpx.HTTPError as exc:
        logger.debug("RSS fetch failed for %s: %s", feed_url, exc)
    return ""


def _local_name(tag: str) -> str:
    """Return an XML local name regardless of namespace declaration style."""
    return tag.rsplit("}", maxsplit=1)[-1].split(":")[-1]


def _item_text(item, local_name: str) -> str:
    for child in item.iter():
        if _local_name(child.tag) == local_name and child.text:
            return child.text.strip()
    return ""


def _normalise_title(value: str) -> str:
    return re.sub(r"[^\w\s]", "", value.lower()).strip()


def _find_episode_asset_in_rss(
    rss_text: str,
    *,
    episode_id: str = "",
    target_title: str = "",
    allow_first: bool = False,
) -> EpisodeAsset:
    """Resolve one RSS item, including its Podcasting 2.0 transcript tag."""
    if not rss_text:
        return EpisodeAsset()
    try:
        root = ElementTree.fromstring(rss_text)
    except (ElementTree.ParseError, ValueError):
        return EpisodeAsset()

    target_normalized = _normalise_title(target_title)
    fallback_item = None
    for item in root.iter():
        if _local_name(item.tag) != "item":
            continue
        fallback_item = fallback_item or item
        item_xml = ElementTree.tostring(item, encoding="unicode")
        item_title = _item_text(item, "title")
        title_normalized = _normalise_title(item_title)
        id_match = bool(episode_id and episode_id in item_xml)
        title_match = bool(
            target_normalized
            and title_normalized
            and (
                target_normalized in title_normalized
                or title_normalized in target_normalized
            )
        )
        if id_match or title_match:
            return _episode_asset_from_item(item)

    if allow_first and fallback_item is not None:
        return _episode_asset_from_item(fallback_item)
    return EpisodeAsset()


def _episode_asset_from_item(item) -> EpisodeAsset:
    """Extract enclosure, GUID, and optional podcast:transcript attributes."""
    audio_url = ""
    transcript_url = ""
    transcript_type = ""
    transcript_language = ""
    for child in item.iter():
        local_name = _local_name(child.tag)
        if local_name == "enclosure" and not audio_url:
            audio_url = child.attrib.get("url", "").replace("&amp;", "&")
        if local_name == "transcript" and not transcript_url:
            transcript_url = child.attrib.get("url", "")
            transcript_type = child.attrib.get("type", "")
            transcript_language = child.attrib.get("language", "")

    return EpisodeAsset(
        title=_item_text(item, "title"),
        audio_url=audio_url,
        transcript_url=transcript_url,
        transcript_type=transcript_type,
        transcript_language=transcript_language,
        guid=_item_text(item, "guid"),
    )


# ── Apple Podcasts audio URL via iTunes Lookup + RSS ────────────────────

def _find_apple_episode_asset(url: str, target_title: str = "") -> EpisodeAsset:
    """Resolve Apple episode media and publisher transcript from canonical RSS."""
    # URL format: https://podcasts.apple.com/us/podcast/{slug}/id{podcast_id}?i={episode_id}
    parsed = urlparse(url)
    podcast_id_match = re.search(r"/id(\d+)", parsed.path)
    episode_id = parse_qs(parsed.query).get("i", [""])[0]
    if not podcast_id_match:
        return EpisodeAsset()

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = client.get(
                f"https://itunes.apple.com/lookup?id={podcast_id_match.group(1)}",
            )
        if response.status_code != 200:
            return EpisodeAsset()
        results = response.json().get("results", [])
        feed_url = results[0].get("feedUrl", "") if results else ""
        by_storefront_id = _find_episode_asset_in_rss(
            _fetch_rss_text(feed_url), episode_id=episode_id,
        )
        if by_storefront_id.audio_url or by_storefront_id.transcript_url:
            return by_storefront_id
        # Apple storefront episode IDs are frequently absent from the RSS
        # GUID/link. Defuddle's episode title is the safe fallback.
        return _find_episode_asset_in_rss(
            _fetch_rss_text(feed_url), target_title=target_title,
        )
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Apple RSS lookup failed: %s", exc)
        return EpisodeAsset()


def _find_apple_audio_url(url: str) -> str:
    """Backward-compatible Apple enclosure helper."""
    return _find_apple_episode_asset(url).audio_url


def _find_episode_audio_in_rss(rss_text: str, episode_id: str) -> str:
    """Backward-compatible RSS enclosure helper."""
    return _find_episode_asset_in_rss(rss_text, episode_id=episode_id).audio_url


# ── Spotify audio URL ────────────────────────────────────────────────────

def _find_spotify_episode_asset(
    url: str,
    metadata: dict[str, str] | None = None,
) -> EpisodeAsset:
    """Resolve Spotify via cross-platform iTunes discovery and canonical RSS."""
    metadata = metadata or _fetch_defuddle_md_metadata(url)
    title = metadata.get("title", "")
    author = metadata.get("author", "")
    if not title:
        return EpisodeAsset()

    search_term = author.strip() or title.strip()
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
        ) as client:
            response = client.get(
                "https://itunes.apple.com/search",
                params={"term": search_term, "media": "podcast", "limit": 5},
            )
        if response.status_code != 200:
            return EpisodeAsset()

        for result in response.json().get("results", []):
            asset = _find_episode_asset_in_rss(
                _fetch_rss_text(result.get("feedUrl", "")),
                target_title=title,
            )
            if asset.audio_url or asset.transcript_url:
                logger.info(
                    "Found cross-platform RSS episode for Spotify '%s' via '%s'",
                    title,
                    result.get("collectionName", ""),
                )
                return asset
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Spotify cross-platform search failed: %s", exc)
    return EpisodeAsset()


def _find_spotify_audio_url(url: str) -> str:
    """Backward-compatible Spotify enclosure helper."""
    return _find_spotify_episode_asset(url).audio_url


def _find_episode_audio_by_title(rss_text: str, target_title: str) -> str:
    """Backward-compatible title-match enclosure helper."""
    return _find_episode_asset_in_rss(rss_text, target_title=target_title).audio_url


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
        supadata_rate_limit()
        with httpx.Client(timeout=30) as client:
            # Submit transcription job
            resp = client.get(
                "https://api.supadata.ai/v1/transcript",
                params={"url": audio_url, "text": "true", "mode": "auto"},
                headers={"x-api-key": api_key},
            )

            track_supadata_call(resp)

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
            elif resp.status_code == 429:
                logger.warning(
                    "Supadata rate limited (429) for audio URL — "
                    "will try Whisper fallback"
                )
                return ""
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
            supadata_rate_limit()
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"https://api.supadata.ai/v1/transcript/{job_id}",
                    headers={"x-api-key": api_key},
                )
            track_supadata_call(resp)
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


# ── Whisper fallback transcription ─────────────────────────────────────────

def _whisper_fallback_transcribe(audio_url: str) -> str:
    """Transcribe audio URL using local faster-whisper as a last resort.

    Downloads the audio file to a temp directory, transcribes with
    faster_whisper.WhisperModel('base', device='cpu', compute_type='int8'),
    and cleans up the temp file.

    Returns the transcript text, or empty string if:
      - faster_whisper is not installed (logs a warning, doesn't raise)
      - audio download fails
      - transcription fails

    This is the last resort after Supadata and defuddle.md metadata.
    """
    # Check if faster_whisper is installed
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning(
            "faster_whisper is not installed — Whisper fallback unavailable. "
            "Install with: pip install faster-whisper"
        )
        return ""

    # Download audio to temp file
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False,
        ) as tmp:
            with httpx.Client(
                timeout=120, follow_redirects=True,
                headers={"User-Agent": BROWSER_HEADERS["User-Agent"]},
            ) as client:
                resp = client.get(audio_url)
                resp.raise_for_status()
                tmp.write(resp.content)
            tmp_path = tmp.name

        logger.info(
            "Whisper fallback: transcribing audio from %s (%d bytes)",
            audio_url,
            os.path.getsize(tmp_path) if tmp_path else 0,
        )

        # Transcribe with faster-whisper
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(tmp_path)
        transcript_parts = [
            segment.text.strip() for segment in segments if segment.text.strip()
        ]
        transcript = " ".join(transcript_parts)

        if not transcript:
            logger.warning("Whisper transcription produced no text for %s", audio_url)
            return ""

        logger.info(
            "Whisper fallback: transcribed %d chars from %s",
            len(transcript), audio_url,
        )
        return transcript

    except Exception as exc:
        logger.warning("Whisper fallback failed for %s: %s", audio_url, exc)
        return ""

    finally:
        if tmp_path is not None:
            with suppress(OSError):
                os.unlink(tmp_path)


# ── Import httpx ─────────────────────────────────────────────────────────

try:
    import httpx
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False
    logger.warning("httpx not available — podcast extractor disabled")
