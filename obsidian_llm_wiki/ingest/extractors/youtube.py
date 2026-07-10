"""YouTube/media transcript extraction via Supadata API.

Supadata provides transcript extraction for YouTube, TikTok, Instagram,
X (Twitter), Facebook, and public video files. It can fetch existing
transcripts (mode=native) or generate them with AI (mode=auto/generate).

API docs: https://docs.supadata.ai/get-transcript
Endpoint: GET https://api.supadata.ai/v1/transcript
Auth: x-api-key header
Pricing: 1 native transcript = 1 credit, 1 generated minute = 2 credits

The API supports async job processing for long videos (HTTP 202 + job ID polling).
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.youtube")

__all__ = ["extract_youtube_video"]

_SUPADATA_API_BASE = "https://api.supadata.ai/v1"
_POLL_INTERVAL = 3  # seconds between job status checks
_POLL_MAX_ATTEMPTS = 40  # ~2 minutes max polling


def _get_api_key() -> str | None:
    """Read the Supadata API key from env at call time (not import time)."""
    return os.environ.get("SUPADATA_API_KEY", "").strip() or None


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


def _is_supported_media_url(url: str) -> bool:
    """Check if URL is a supported media platform for Supadata."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return any(h in host for h in (
            "youtube.com", "youtu.be", "tiktok.com",
            "instagram.com", "x.com", "twitter.com",
            "facebook.com", "fb.watch",
        ))
    except Exception:
        return False


@register_extractor(lambda parsed, raw: raw.startswith("http") and bool(_video_id(raw)))
def extract_youtube_video(raw_url: str) -> SourceDoc:
    """Extract transcript from a YouTube video via Supadata API.

    Strategy:
      1. Supadata API (mode=auto — try native transcript, fallback to AI generation)
      2. Invidious API (metadata + description fallback)
      3. oEmbed metadata (title + channel only, last resort)

    Raises:
        RuntimeError: If SUPADATA_API_KEY is not set or all methods fail.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "SUPADATA_API_KEY not set — set it in your .env or environment. "
            "Get your key at https://dash.supadata.ai/organizations/api-key"
        )

    errors: list[str] = []

    # ── Primary: Supadata API ────────────────────────────────────────
    try:
        source = _supadata_transcript(raw_url, api_key)
        if source:
            logger.info(
                "Supadata: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"supadata: {exc}")

    # ── Fallback 1: Invidious API (metadata + description, no transcript) ──
    try:
        from obsidian_llm_wiki.ingest.alt_source import extract_via_invidious
        source = extract_via_invidious(raw_url)
        logger.info("Invidious fallback: extracted %d chars for %s", len(source.content), raw_url)
        return source
    except Exception as exc:
        errors.append(f"invidious: {exc}")

    # ── Fallback 2: oEmbed metadata (title + author only, minimal) ──
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


def _supadata_transcript(url: str, api_key: str) -> SourceDoc | None:
    """Fetch transcript via Supadata API with async job support.

    Returns None if no transcript available (not an error — caller falls through).
    Raises RuntimeError on auth/billing errors.
    """

    params = {
        "url": url,
        "text": "true",  # plain text transcript
        "mode": "auto",  # try native, fallback to AI generation
    }

    with httpx.Client(
        **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=True),
    ) as client:
        resp = client.get(
            f"{_SUPADATA_API_BASE}/transcript",
            params=params,
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
            },
        )

    # Auth errors
    if resp.status_code == 401:
        raise RuntimeError(
            "Supadata API key invalid — check SUPADATA_API_KEY at "
            "https://dash.supadata.ai/organizations/api-key"
        )
    if resp.status_code == 402:
        raise RuntimeError(
            "Supadata: insufficient credits — check your billing at "
            "https://dash.supadata.ai"
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "Supadata: video is private/restricted/age-gated"
        )

    # Async job — poll for results
    if resp.status_code == 202:
        job_data = resp.json()
        job_id = job_data.get("jobId")
        if not job_id:
            raise RuntimeError("Supadata returned 202 but no jobId")
        return _poll_supadata_job(job_id, api_key)

    resp.raise_for_status()
    data = resp.json()

    # Check for transcript unavailable (206)
    if resp.status_code == 206:
        raise RuntimeError(
            "Supadata: no transcript available for this video"
        )

    return _parse_supadata_response(data, url)


def _poll_supadata_job(job_id: str, api_key: str) -> SourceDoc | None:
    """Poll Supadata job status until completed or failed."""
    with httpx.Client(
        **make_client_kwargs(timeout=30, follow_redirects=True),
    ) as client:
        for attempt in range(_POLL_MAX_ATTEMPTS):
            time.sleep(_POLL_INTERVAL)
            resp = client.get(
                f"{_SUPADATA_API_BASE}/transcript/{job_id}",
                headers={"x-api-key": api_key, "Accept": "application/json"},
            )
            if resp.status_code == 404:
                raise RuntimeError(f"Supadata job {job_id} expired")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "completed":
                result = data.get("result", data)
                return _parse_supadata_response(result, data.get("url", ""))
            if status == "failed":
                error = data.get("error", "unknown error")
                raise RuntimeError(f"Supadata job {job_id} failed: {error}")
            # queued or active — keep polling
            logger.debug("Supadata job %s: %s (attempt %d)", job_id, status, attempt + 1)

    raise RuntimeError(
        f"Supadata job {job_id} timed out after {_POLL_MAX_ATTEMPTS * _POLL_INTERVAL}s"
    )


def _parse_supadata_response(data: dict, url: str) -> SourceDoc | None:
    """Parse Supadata transcript response into SourceDoc."""
    # Response format: {"content": "transcript text...", "lang": "en", ...}
    # Or for chunked: {"content": [{"text": "...", "start": 0, "duration": 5}, ...]}
    content_raw = data.get("content", "")
    lang = data.get("lang", "")

    if isinstance(content_raw, list):
        # Chunked format — join text segments
        text = " ".join(
            seg.get("text", "").strip()
            for seg in content_raw
            if seg.get("text")
        )
    elif isinstance(content_raw, str):
        text = content_raw.strip()
    else:
        text = ""

    if not text or len(text) < 200:
        raise RuntimeError(
            f"Supadata transcript too short ({len(text)} chars) — "
            "video may have no speech"
        )

    # Title from response or fetch via oEmbed
    title = data.get("title", "") or ""
    if not title:
        # Supadata doesn't always return a title — fetch via oEmbed
        title = _fetch_youtube_title(url)
    if not title:
        vid = _video_id(url)
        title = f"YouTube video {vid}" if vid else url

    # Add language note if non-English
    if lang and lang != "en":
        text = f"[Transcript language: {lang}]\n\n{text}"

    return SourceDoc(title=title, content=text, url=url)


def _fetch_youtube_title(url: str) -> str:
    """Fetch video title via YouTube oEmbed API (no auth required).

    Returns empty string if the fetch fails.
    """
    from urllib.parse import quote
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url={quote(url, safe='')}&format=json"
    )
    try:
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            resp = client.get(oembed_url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("title", "") or ""
    except Exception:
        pass
    return ""


def _extract_oembed(youtube_url: str) -> SourceDoc | None:
    """Fetch video metadata via YouTube oEmbed API (no auth required)."""
    from urllib.parse import quote

    oembed_url = f"https://www.youtube.com/oembed?url={quote(youtube_url, safe='')}&format=json"
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        resp = client.get(oembed_url)
        if resp.status_code != 200:
            return None
        data = resp.json()

    title = data.get("title", "") or youtube_url
    author = data.get("author_name", "") or ""

    content_parts = [
        f"Title: {title}",
        f"Channel: {author}",
        "",
        "Note: Full transcript unavailable "
        "(Supadata API key not configured or insufficient credits).",
        "Only video metadata was extracted.",
    ]

    content = "\n".join(content_parts)
    if len(content) < 100:
        return None

    return SourceDoc(title=title, content=content, url=youtube_url)
