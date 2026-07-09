"""YouTube extractor — extracts transcript + metadata from YouTube videos.

Dependencies (optional): ``yt-dlp`` and ``youtube-transcript-api``.
Install with: ``pip install okf-pipeline[youtube]``

If either dependency is not installed, this module raises ImportError at
import time, which the registry silently catches — the extractor just
won't be available.

Design contract:
  - Never return a sub-threshold stub. If yt-dlp, transcript, AND oEmbed
    all fail, raise RuntimeError so dispatch falls through to extract_web.
  - oEmbed (no-auth public API) is the last-resort metadata source.
  - yt-dlp uses extractor_args (NOT top-level player_client) for
    multi-client fallback against YouTube bot detection.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS

logger = logging.getLogger("obswiki.ingest.extractors.youtube")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import yt_dlp  # noqa: F401
    from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

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


# ── Registration (only if deps available) ────────────────────────────────

if _DEPS_AVAILABLE:

    @register_extractor(_is_youtube)
    def extract_youtube(raw_url: str) -> SourceDoc:
        """Extract transcript and metadata from a YouTube video.

        Strategy (in order, each can fail gracefully):
          1. yt-dlp for metadata (title, description, channel).
             Tries multiple player clients via extractor_args to bypass
             YouTube bot detection on unattended VPS IPs.
          2. youtube-transcript-api for the transcript text.
          3. YouTube oEmbed API for title + author (no-auth public endpoint).

        If all three produce insufficient content (< 200 chars usable),
        raises RuntimeError so dispatch falls through to extract_web.
        """
        video_id = _extract_video_id(raw_url)
        if not video_id:
            raise RuntimeError(f"Could not extract video ID from: {raw_url}")

        # ── 1. Metadata via yt-dlp (try multiple player clients) ────────
        title = ""
        description = ""
        channel = ""

        # Correct yt-dlp API: extractor_args, NOT top-level player_client.
        player_clients = ("android", "ios", "tv_embedded", "web")

        for client in player_clients:
            ydl_opts: dict = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "extract_flat": False,
                "extractor_args": {"youtube": {"player_client": [client]}},
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[possibly-unbound]
                    info = ydl.extract_info(raw_url, download=False)
                    title = info.get("title", "") or ""
                    description = info.get("description", "") or ""
                    channel = info.get("channel", info.get("uploader", "")) or ""
                if title:
                    break
            except Exception as exc:
                logger.debug("yt-dlp client=%s failed for '%s': %s", client, raw_url, exc)
                continue

        if not title:
            logger.warning("yt-dlp metadata extraction failed for '%s' (all clients)", raw_url)

        # ── 2. Transcript via youtube-transcript-api ────────────────────
        transcript_text = ""
        if video_id:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)  # type: ignore[possibly-unbound]
                transcript_text = "\n\n".join(
                    f"[{entry.get('start', 0):.0f}s] {entry.get('text', '')}"
                    for entry in transcript_list
                )
            except Exception as exc:
                logger.warning("No transcript available for '%s': %s", raw_url, exc)

        # ── 3. oEmbed fallback (no-auth public API) ─────────────────────
        if not title:
            oembed = _fetch_oembed(raw_url)
            if oembed:
                title = oembed.get("title", "")
                author = oembed.get("author_name", "")
                if author:
                    channel = channel or author
                # oEmbed doesn't give description, but we can construct
                # a minimal content stub from title + author.
                if not description and title:
                    description = f"Video by {author}. Title: {title}"

        # ── Assemble content ──────────────────────────────────────────
        parts: list[str] = []
        if channel:
            parts.append(f"Channel: {channel}")
        if description:
            parts.append(f"Description:\n{description}")
        if transcript_text:
            parts.append(f"Transcript:\n{transcript_text}")

        content = "\n\n".join(parts)
        stripped = content.strip()

        # ── Hard failure: no usable content ──────────────────────────
        if not title and not transcript_text and not description:
            raise RuntimeError(
                f"YouTube extraction completely failed for '{raw_url}': "
                f"no metadata, no transcript, no oEmbed (likely bot detection). "
                f"Falling through to web extraction."
            )

        # Reject thin content — YouTube watch pages are cookie-walled,
        # so extract_web will also produce footer chrome. Fail closed.
        if len(stripped) < 200:
            raise RuntimeError(
                f"YouTube extraction too thin for '{raw_url}': "
                f"{len(stripped)} chars. Video may be private, age-restricted, "
                f"or transcript disabled. Falling through to web extraction."
            )

        if not title:
            title = raw_url  # last resort

        return SourceDoc(
            title=title,
            content=content,
            url=raw_url,
        )


# ── oEmbed helper ───────────────────────────────────────────────────────

def _fetch_oembed(url: str) -> dict | None:
    """Fetch YouTube oEmbed metadata (no-auth public API).

    Returns dict with title, author_name, thumbnail_url, etc.
    Returns None on any failure.
    """
    oembed_url = f"https://www.youtube.com/oembed?url={quote(url, safe='')}&format=json"
    try:
        with httpx.Client(timeout=10, headers=BROWSER_HEADERS) as client:
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
    # Bare ID (11 chars, alphanumeric + -_)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url
    return ""