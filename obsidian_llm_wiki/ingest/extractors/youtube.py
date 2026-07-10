"""YouTube extractor — tiered metadata + transcript extraction.

Tier 1: yt-dlp + proxy (multi-client extractor_args)
Tier 2: YouTube Innertube API (no-auth, works from any IP)
Tier 3: youtube-transcript-api for captions
Tier 4: oEmbed (title + author only)

Accepts description-only content when the video has no transcript
but has a substantive description (≥ 200 chars total content).

Dependencies (optional): ``yt-dlp`` and ``youtube-transcript-api``.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import quote

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
from obsidian_llm_wiki.ingest.proxy import get_proxy_url, ytdlp_proxy_arg, make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.youtube")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import yt_dlp  # noqa: F401
    from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

# ── Innertube (always available — pure httpx, no deps) ──────────────────

_INNERTUBE_KEY = "AIzaSyAO_FJ2UqR_toL8vFg1h9NJm7b0G1hJC6w"

# ── Domain matching ─────────────────────────────────────────────────────

_YOUTUBE_HOSTS = frozenset((
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
))


def _is_youtube(parsed, raw: str) -> bool:
    """Match YouTube URLs."""
    return bool(parsed.hostname and parsed.hostname.lower() in _YOUTUBE_HOSTS)


# ── Registration ─────────────────────────────────────────────────────────
# Always register — innertube + oEmbed work without yt-dlp/transcript deps.

@register_extractor(_is_youtube)
def extract_youtube(raw_url: str) -> SourceDoc:
    """Extract metadata + transcript from a YouTube video.

    Tries tiers in order, accumulating whatever content is available.
    Accepts description-only if total content ≥ 200 chars.
    Fail-closed if nothing usable — never falls through to extract_web.
    """
    video_id = _extract_video_id(raw_url)
    if not video_id:
        raise RuntimeError(f"Could not extract video ID from: {raw_url}")

    title = ""
    description = ""
    channel = ""
    transcript_text = ""

    # ── Tier 1: yt-dlp + proxy (if available) ────────────────────────
    if _DEPS_AVAILABLE:
        title, description, channel = _try_ytdlp(raw_url)
        if not title:
            logger.warning("yt-dlp metadata failed for '%s' (all clients)", raw_url)

    # ── Tier 2: Innertube API (no deps, always available) ────────────
    if not title or not description:
        innertube = _fetch_innertube(video_id)
        if innertube:
            if not title:
                title = innertube.get("title", "")
            if not channel:
                channel = innertube.get("author", "")
            if not description:
                description = innertube.get("shortDescription", "")
            # Try captions from innertube
            caption_tracks = innertube.get("_caption_tracks", [])
            if caption_tracks and not transcript_text:
                transcript_text = _fetch_innertube_captions(caption_tracks)

    # ── Tier 3: youtube-transcript-api (if available, no proxy needed) ─
    if _DEPS_AVAILABLE and not transcript_text:
        transcript_text = _try_transcript_api(video_id, raw_url)

    # ── Tier 4: oEmbed (title + author only) ─────────────────────────
    if not title:
        oembed = _fetch_oembed(raw_url)
        if oembed:
            title = oembed.get("title", "")
            author = oembed.get("author_name", "")
            if author:
                channel = channel or author
            if not description and title:
                description = f"Video by {author}. Title: {title}"

    # ── Assemble content ─────────────────────────────────────────────
    parts: list[str] = []
    if channel:
        parts.append(f"Channel: {channel}")
    if description:
        parts.append(f"Description:\n{description}")
    if transcript_text:
        parts.append(f"Transcript:\n{transcript_text}")
    else:
        parts.append("Transcript: (not available)")

    content = "\n\n".join(parts)
    stripped = content.strip()

    # ── Quality gate ────────────────────────────────────────────────
    if not title and not transcript_text and not description:
        raise RuntimeError(
            f"YouTube extraction completely failed for '{raw_url}': "
            f"no metadata from any tier. Falling through to web extraction."
        )

    # Accept description-only if total content is substantive (≥ 200 chars).
    # Videos without transcripts still have useful descriptions.
    if len(stripped) < 200:
        raise RuntimeError(
            f"YouTube extraction too thin for '{raw_url}': "
            f"{len(stripped)} chars. Falling through to web extraction."
        )

    if not title:
        title = raw_url

    return SourceDoc(title=title, content=content, url=raw_url)


# ── Tier 1: yt-dlp + proxy ────────────────────────────────────────────────

def _try_ytdlp(url: str) -> tuple[str, str, str]:
    """Try yt-dlp with multiple player clients + proxy. Returns (title, description, channel)."""
    proxy = ytdlp_proxy_arg()
    player_clients = ("android", "ios", "tv_embedded", "web")

    for client in player_clients:
        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        if proxy:
            ydl_opts["proxy"] = proxy
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[possibly-unbound]
                info = ydl.extract_info(url, download=False)
                t = info.get("title", "") or ""
                d = info.get("description", "") or ""
                c = info.get("channel", info.get("uploader", "")) or ""
                if t:
                    return t, d, c
        except Exception as exc:
            logger.debug("yt-dlp client=%s failed for '%s': %s", client, url, exc)
            continue
    return "", "", ""


# ── Tier 2: Innertube API ────────────────────────────────────────────────

def _fetch_innertube(video_id: str) -> dict | None:
    """Fetch video metadata via YouTube Innertube player API.

    No authentication required. Works from any IP (including datacenter).
    Returns dict with title, author, shortDescription, and _caption_tracks.
    """
    payload = {
        "videoId": video_id,
        "context": {
            "client": {
                "clientName": "WEB",
                "clientVersion": "2.20240101.00.00",
            }
        },
    }
    url = f"https://www.youtube.com/youtubei/v1/player?key={_INNERTUBE_KEY}"
    try:
        with httpx.Client(**make_client_kwargs(timeout=15)) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.debug("innertube %d for '%s'", resp.status_code, video_id)
                return None
            data = resp.json()
    except Exception as exc:
        logger.debug("innertube failed for '%s': %s", video_id, exc)
        return None

    vd = data.get("videoDetails", {})
    if not vd:
        return None

    # Extract caption tracks if available
    caption_tracks = []
    caps = data.get("captions", {})
    if caps:
        renderer = caps.get("playerCaptionsTracklistRenderer", {})
        caption_tracks = renderer.get("captionTracks", [])

    result = {
        "title": vd.get("title", ""),
        "author": vd.get("author", ""),
        "shortDescription": vd.get("shortDescription", ""),
        "lengthSeconds": vd.get("lengthSeconds", ""),
        "keywords": vd.get("keywords", []),
        "_caption_tracks": caption_tracks,
    }
    return result


def _fetch_innertube_captions(caption_tracks: list) -> str:
    """Fetch caption text from innertube caption track URLs.

    Returns formatted transcript text, or empty string on failure.
    """
    if not caption_tracks:
        return ""

    # Prefer English, fall back to first track
    track = None
    for t in caption_tracks:
        if t.get("languageCode", "").startswith("en"):
            track = t
            break
    if not track:
        track = caption_tracks[0]

    base_url = track.get("baseUrl", "")
    if not base_url:
        return ""

    try:
        with httpx.Client(**make_client_kwargs(timeout=15)) as client:
            resp = client.get(base_url)
            if resp.status_code != 200:
                return ""
            raw = resp.text

        # Parse XML caption format
        entries = re.findall(
            r'<text start="([\d.]+)"[^>]*>(.*?)</text>', raw, re.DOTALL
        )
        if not entries:
            return ""

        lines = []
        for start, text in entries:
            # Decode HTML entities
            text = text.replace("&amp;", "&").replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            t = float(start)
            lines.append(f"[{t:.0f}s] {text.strip()}")
        return "\n\n".join(lines)
    except Exception as exc:
        logger.debug("innertube caption fetch failed: %s", exc)
        return ""


# ── Tier 3: youtube-transcript-api ───────────────────────────────────────

def _try_transcript_api(video_id: str, raw_url: str) -> str:
    """Try youtube-transcript-api for captions. Returns transcript text or empty."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[possibly-unbound]
        return "\n\n".join(
            f"[{entry.get('start', 0):.0f}s] {entry.get('text', '')}"
            for entry in transcript_list
        )
    except Exception as exc:
        logger.warning("No transcript available for '%s': %s", raw_url, exc)
        return ""


# ── Tier 4: oEmbed ────────────────────────────────────────────────────────

def _fetch_oembed(url: str) -> dict | None:
    """Fetch YouTube oEmbed metadata (no-auth public API)."""
    oembed_url = f"https://www.youtube.com/oembed?url={quote(url, safe='')}&format=json"
    try:
        with httpx.Client(**make_client_kwargs(timeout=10)) as client:
            resp = client.get(oembed_url)
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        logger.debug("oEmbed fetch failed for '%s': %s", url, exc)
    return None


# ── Helpers ─────────────────────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|v/)|youtu\.be/)([a-zA-Z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str:
    """Extract the 11-character video ID from a YouTube URL."""
    match = _VIDEO_ID_RE.search(url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url
    return ""