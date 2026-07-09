"""YouTube extractor — extracts transcript + metadata from YouTube videos.

Dependencies (optional): ``yt-dlp`` and ``youtube-transcript-api``.
Install with: ``pip install okf-pipeline[youtube]``

If either dependency is not installed, this module raises ImportError at
import time, which the registry silently catches — the extractor just
won't be available.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

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

        Strategy:
          1. yt-dlp for metadata (title, description, channel, duration).
          2. youtube-transcript-api for the transcript text.

        If no transcript is available, returns the description as content.
        """
        video_id = _extract_video_id(raw_url)
        if not video_id:
            raise RuntimeError(f"Could not extract video ID from: {raw_url}")

        # ── Metadata via yt-dlp ──────────────────────────────────────
        title = raw_url
        description = ""
        channel = ""

        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(raw_url, download=False)
                title = info.get("title", raw_url) or raw_url
                description = info.get("description", "") or ""
                channel = info.get("channel", info.get("uploader", "")) or ""
        except Exception as exc:
            logger.warning("yt-dlp metadata extraction failed for '%s': %s", raw_url, exc)

        # ── Transcript via youtube-transcript-api ────────────────────
        transcript_text = ""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = "\n\n".join(
                f"[{entry.get('start', 0):.0f}s] {entry.get('text', '')}"
                for entry in transcript_list
            )
        except Exception as exc:
            logger.warning("No transcript available for '%s': %s", raw_url, exc)

        # ── Assemble content ──────────────────────────────────────────
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
        if not content.strip():
            raise RuntimeError(f"Could not extract any content from: {raw_url}")

        return SourceDoc(
            title=title,
            content=content,
            url=raw_url,
        )


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