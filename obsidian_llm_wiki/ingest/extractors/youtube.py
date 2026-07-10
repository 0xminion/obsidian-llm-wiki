"""YouTube transcript extraction via TranscriptAPI.com.

TranscriptAPI provides reliable, credit-based YouTube transcript extraction.
API docs: https://transcriptapi.com/docs/api
Endpoint: GET https://transcriptapi.com/api/v2/youtube/transcript
Auth: x-api-key header with the user's API key.
"""

from __future__ import annotations

import logging
import os

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.youtube")

__all__ = ["extract_youtube_video"]

_TRANSCRIPT_API_KEY = os.environ.get("TRANSCRIPT_API_KEY", "").strip() or None
_TRANSCRIPT_API_BASE = "https://transcriptapi.com"
_TRANSCRIPT_API_VERSION = "v2"


def _video_id(url: str) -> str | None:
    """Extract 11-char video ID from any YouTube URL."""
    import re
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.strip("/")[:11]
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        for param in (parsed.query or "").split("&"):
            k, _, v = param.partition("=")
            if k == "v":
                return v[:11]
        for seg in parsed.path.split("/"):
            if len(seg) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", seg):
                return seg
    return None


@register_extractor(lambda parsed, raw: bool(_video_id(raw) if raw.startswith("http") else _video_id(raw)))
def extract_youtube_video(raw_url: str) -> SourceDoc:
    """Extract transcript from a YouTube video via TranscriptAPI.com.

    Strategy:
      1. TranscriptAPI.com v2 API (requires API key in TRANSCRIPT_API_KEY env var)
      2. Fail with clear message if no API key is set

    This replaces the previous yt-dlp-based extractor as the primary path.
    yt-dlp remains available as a fallback if TranscriptAPI fails.

    Raises:
        RuntimeError: If TRANSCRIPT_API_KEY is not set or API call fails.
    """
    if not _TRANSCRIPT_API_KEY:
        raise RuntimeError(
            "TRANSCRIPT_API_KEY not set — set it in your .env or environment. "
            "Get your key at https://transcriptapi.com"
        )

    video_id = _video_id(raw_url)
    if not video_id:
        raise RuntimeError(f"Could not extract YouTube video ID from: {raw_url}")

    api_url = (
        f"{_TRANSCRIPT_API_BASE}/api/{_TRANSCRIPT_API_VERSION}/youtube/transcript"
        f"?videoId={video_id}&languages=en"
    )

    errors: list[str] = []

    # Try transcript API
    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
        ) as client:
            resp = client.get(
                api_url,
                headers={
                    "x-api-key": _TRANSCRIPT_API_KEY,
                    "Accept": "application/json",
                    "User-Agent": "obsidian-llm-wiki/3.0",
                },
            )

        if resp.status_code == 401 or resp.status_code == 403:
            raise RuntimeError(
                f"TranscriptAPI authentication failed ({resp.status_code}) — "
                "check your TRANSCRIPT_API_KEY at https://transcriptapi.com"
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"No transcript available for video {video_id} on TranscriptAPI"
            )
        resp.raise_for_status()
        data = resp.json()

        # Parse TranscriptAPI response shape
        # Returns: {"transcript": [{"text": "...", "start": 0.0, "duration": 5.12}, ...]}
        transcript_segments = data.get("transcript", [])
        if isinstance(transcript_segments, dict):
            transcript_segments = transcript_segments.get("transcript", [])
        if not transcript_segments:
            raise RuntimeError(f"Empty transcript returned for {video_id}")

        # Build plain text: concatenate all segment text
        lines: list[str] = []
        for seg in transcript_segments:
            text = seg.get("text", "").strip()
            if text:
                lines.append(text)

        content = " ".join(lines)
        title = data.get("videoTitle", "") or data.get("title", "") or raw_url

        if not content or len(content) < 200:
            raise RuntimeError(
                f"Transcript too short ({len(content)} chars) for {video_id}"
            )

        logger.info(
            "TranscriptAPI: extracted %d chars for %s (%s)",
            len(content), video_id, raw_url,
        )
        return SourceDoc(title=title, content=content, url=raw_url)

    except Exception as exc:
        errors.append(f"transcript_api: {exc}")

    raise RuntimeError(
        f"YouTube transcript extraction failed for {raw_url}: " +
        "; ".join(errors)
    )
