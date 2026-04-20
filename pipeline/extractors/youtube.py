"""YouTube video transcript extraction.

Chain: TranscriptAPI → Supadata → yt-dlp + faster-whisper.
Falls back to metadata-only on total failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from urllib.parse import quote

from pipeline.config import Config
from pipeline.models import ExtractedSource, SourceType
from pipeline.extractors._shared import (
    _curl_get,
    _curl_post_json,
    _run,
    _extract_youtube_video_id,
    extract_title,
    transcribe_with_whisper,
    ExtractionError,
)

log = logging.getLogger(__name__)


def extract_youtube(url: str, cfg: Config) -> ExtractedSource:
    """Extract YouTube video transcript.

    Chain: TranscriptAPI → Supadata → yt-dlp + faster-whisper.
    Falls back to metadata-only on total failure.
    """
    video_id = _extract_youtube_video_id(url)
    timeout = cfg.extract_timeout

    # Fetch metadata from YouTube oEmbed
    title = ""
    author = ""
    meta_json = _curl_get(
        f"https://www.youtube.com/oembed?url={quote(url, safe='')}&format=json",
        timeout=timeout,
    )
    if meta_json:
        try:
            meta = json.loads(meta_json)
            title = meta.get("title", "")
            author = meta.get("author_name", "")
        except (json.JSONDecodeError, KeyError):
            pass

    # Try transcript extraction chain
    transcript = _try_youtube_transcript(url, video_id, cfg)

    if not transcript or len(transcript) < 50:
        log.error("YouTube transcript extraction failed for %s", video_id)
        raise ExtractionError(
            f"YouTube transcript extraction failed for {url} (video ID: {video_id}). "
            f"TranscriptAPI, Supadata, and whisper all failed or returned <50 chars. "
            f"Check API keys and whisper installation. "
            f"NEVER accept metadata-only for YouTube."
        )
    else:
        content = transcript

    return ExtractedSource(
        url=url,
        title=title or url,
        content=content,
        type=SourceType.YOUTUBE,
        author=author,
    )


def _try_youtube_transcript(url: str, video_id: str, cfg: Config) -> str:
    """Try TranscriptAPI → Supadata → Whisper fallback chain."""
    timeout = cfg.extract_timeout

    # 1) TranscriptAPI (primary) — MUST pass full URL
    if cfg.transcript_api_key:
        try:
            api_url = (
                f"https://transcriptapi.com/api/v2/youtube/transcript"
                f"?video_url={quote(url, safe='')}&format=text&include_timestamp=true&send_metadata=true"
            )
            resp = _curl_get(
                api_url,
                headers={"Authorization": f"Bearer {cfg.transcript_api_key}"},
                timeout=timeout,
            )
            if resp and len(resp) > 50:
                try:
                    data = json.loads(resp)
                    return data.get("transcript", data.get("content", resp))
                except json.JSONDecodeError:
                    return resp
        except (subprocess.TimeoutExpired, Exception) as e:
            log.debug("TranscriptAPI failed: %s", e)

    # 2) Supadata (fallback)
    if cfg.supadata_api_key:
        try:
            resp = _curl_post_json(
                "https://api.supadata.ai/v1/youtube/transcript",
                data={"video_url": f"https://www.youtube.com/watch?v={video_id}",
                      "format": "text"},
                headers={"x-api-key": cfg.supadata_api_key},
                timeout=timeout,
            )
            if resp and len(resp) > 50:
                try:
                    data = json.loads(resp)
                    return data.get("transcript", data.get("content", resp))
                except json.JSONDecodeError:
                    return resp
        except (subprocess.TimeoutExpired, Exception) as e:
            log.debug("Supadata failed: %s", e)

    # 3) yt-dlp + faster-whisper (last resort)
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_audio = f.name

        try:
            dl = _run(
                ["yt-dlp", "-x", "--audio-format", "mp3",
                 "--max-filesize", "200M", "-o", tmp_audio, url],
                timeout=120,
            )
            if dl.returncode != 0 or not os.path.exists(tmp_audio):
                return ""

            text = transcribe_with_whisper(tmp_audio, cfg.whisper_language)
            return text if len(text) > 50 else ""
        finally:
            if os.path.exists(tmp_audio):
                os.unlink(tmp_audio)
    except (subprocess.TimeoutExpired, Exception) as e:
        log.debug("Whisper fallback failed: %s", e)
        return ""
