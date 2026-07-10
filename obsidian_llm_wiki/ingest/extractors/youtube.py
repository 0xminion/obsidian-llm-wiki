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

_TRANSCRIPT_API_BASE = "https://transcriptapi.com"
_TRANSCRIPT_API_VERSION = "v2"


def _get_api_key() -> str | None:
    """Read the TranscriptAPI key from env at call time, not import time.

    The .env file is loaded by the CLI after module import, so a module-level
    read would miss it. This function reads the env var on each call.
    """
    return os.environ.get("TRANSCRIPT_API_KEY", "").strip() or None


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
    api_key = _get_api_key()
    if not api_key:
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
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                    "User-Agent": "obsidian-llm-wiki/3.0",
                },
            )

        if resp.status_code == 401 or resp.status_code == 403:
            # Distinguish auth failure from billing/plan issues
            try:
                body = resp.json()
                detail = body.get("detail", {})
                if isinstance(detail, dict):
                    reason = detail.get("reason", "")
                    message = detail.get("message", "")
                    if reason == "no_active_paid_plan":
                        raise RuntimeError(
                            f"TranscriptAPI: no active paid plan ({message}). "
                            "Upgrade at https://transcriptapi.com/billing"
                        )
                    if message:
                        raise RuntimeError(
                            f"TranscriptAPI auth failed ({resp.status_code}): {message}"
                        )
            except (ValueError, KeyError):
                pass
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

    # Fallback 1: Invidious API (metadata + description, no transcript)
    try:
        from obsidian_llm_wiki.ingest.alt_source import extract_via_invidious
        source = extract_via_invidious(raw_url)
        logger.info("Invidious fallback: extracted %d chars for %s", len(source.content), raw_url)
        return source
    except Exception as exc:
        errors.append(f"invidious: {exc}")

    # Fallback 2: oEmbed metadata (title + author only, minimal)
    try:
        source = _extract_oembed(raw_url)
        if source:
            logger.info("oEmbed fallback: extracted metadata for %s", raw_url)
            return source
    except Exception as exc:
        errors.append(f"oembed: {exc}")

    raise RuntimeError(
        f"YouTube transcript extraction failed for {raw_url}: " +
        "; ".join(errors)
    )


def _extract_oembed(youtube_url: str) -> SourceDoc | None:
    """Fetch video metadata via YouTube oEmbed API (no auth required).

    Returns title + author + thumbnail description. Minimal content,
    but better than nothing when all other methods fail.
    """
    from urllib.parse import quote
    import httpx

    oembed_url = f"https://www.youtube.com/oembed?url={quote(youtube_url, safe='')}&format=json"
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        resp = client.get(oembed_url)
        if resp.status_code != 200:
            return None
        data = resp.json()

    title = data.get("title", "") or youtube_url
    author = data.get("author_name", "") or ""
    thumbnail = data.get("thumbnail_url", "") or ""

    content_parts = [
        f"Title: {title}",
        f"Channel: {author}",
    ]
    if thumbnail:
        content_parts.append(f"Thumbnail: {thumbnail}")
    content_parts.extend([
        "",
        "Note: Full transcript unavailable (no active TranscriptAPI plan).",
        "Only video metadata was extracted. Upgrade at https://transcriptapi.com/billing",
    ])

    content = "\n".join(content_parts)
    if len(content) < 100:
        return None

    return SourceDoc(title=title, content=content, url=youtube_url)
