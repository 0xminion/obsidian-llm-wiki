"""Cache-first transcript acquisition for public podcast media.

The resolver deliberately prefers publisher-provided artifacts over generated
ASR output. Remote-URL transcription providers fetch the audio themselves,
which avoids downloading public media from the pipeline host when that host is
blocked or challenged. Local Whisper remains the final fallback in the podcast
extractor because it must acquire the media locally.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.url_safety import stream_with_validated_redirects

logger = logging.getLogger("obswiki.ingest.transcript_resolver")

_ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2/transcript"
_MIN_TRANSCRIPT_CHARS = 200
_MAX_PUBLIC_TRANSCRIPT_BYTES = 10_000_000

__all__ = [
    "TranscriptResult",
    "assemblyai_transcribe_url",
    "fetch_public_transcript",
    "get_assemblyai_key",
    "load_transcript_cache",
    "save_transcript_cache",
    "validate_assemblyai_key",
]


@dataclass(frozen=True)
class TranscriptResult:
    """Normalized transcript artifact plus its acquisition provenance."""

    text: str
    provider: str
    artifact_url: str = ""
    language: str = ""
    timed: bool = False
    speaker_labeled: bool = False


def get_assemblyai_key() -> str:
    """Return the configured AssemblyAI key without logging it."""
    return os.environ.get("ASSEMBLYAI_API_KEY", "").strip()


def _cache_dir() -> Path:
    vault_path = os.environ.get("VAULT_PATH", "").strip()
    if vault_path:
        return Path(vault_path).expanduser().resolve() / "04-Wiki" / ".llmwiki" / "transcripts"
    return Path.home() / ".llmwiki" / "transcripts"


def _cache_key(identity: str) -> str:
    normalized = identity.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_transcript_cache(identity: str) -> TranscriptResult | None:
    """Load a previously acquired transcript by canonical episode identity."""
    if not identity:
        return None
    path = _cache_dir() / f"{_cache_key(identity)}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    text = str(payload.get("text", "")).strip()
    if len(text) < _MIN_TRANSCRIPT_CHARS:
        return None
    return TranscriptResult(
        text=text,
        provider=str(payload.get("provider", "cache")),
        artifact_url=str(payload.get("artifact_url", "")),
        language=str(payload.get("language", "")),
        timed=bool(payload.get("timed", False)),
        speaker_labeled=bool(payload.get("speaker_labeled", False)),
    )


def save_transcript_cache(identity: str, result: TranscriptResult) -> None:
    """Persist a usable transcript and immutable provenance for re-use."""
    if not identity or len(result.text.strip()) < _MIN_TRANSCRIPT_CHARS:
        return

    cache_path = _cache_dir() / f"{_cache_key(identity)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(result),
        "identity": identity,
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)


def _strip_timed_transcript(text: str) -> str:
    """Normalize VTT/SRT/plain/HTML transcript content into readable markdown."""
    cleaned = html.unescape(text).replace("\r\n", "\n")
    cleaned = re.sub(r"(?im)^WEBVTT[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?m)^\d+\s*$", "", cleaned)
    cleaned = re.sub(
        r"(?m)^\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}\s+-->.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_json_transcript(payload: Any) -> str:
    """Extract common transcript fields from publisher JSON artifacts."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return "\n".join(_extract_json_transcript(item) for item in payload)
    if not isinstance(payload, dict):
        return ""

    for key in ("text", "transcript", "content", "body"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _extract_json_transcript(value)
    for key in ("segments", "items", "captions", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return _extract_json_transcript(value)
    return ""


def fetch_public_transcript(
    transcript_url: str,
    *,
    mime_type: str = "",
    language: str = "",
) -> TranscriptResult | None:
    """Fetch a publisher-provided RSS/podcast transcript artifact.

    This supports the Podcasting 2.0 ``podcast:transcript`` formats: plain
    text, HTML, WebVTT, SRT, and JSON. It does not attempt authenticated or
    DRM-protected sources.
    """
    if not transcript_url:
        return None

    try:
        with httpx.Client(
            **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
            headers={"User-Agent": BROWSER_HEADERS["User-Agent"], "Accept": "*/*"},
        ) as client, stream_with_validated_redirects(client, transcript_url) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > _MAX_PUBLIC_TRANSCRIPT_BYTES:
                return None
            effective_type = (mime_type or response.headers.get("content-type", "")).lower()
            body = bytearray()
            for chunk in response.iter_bytes():
                if len(body) + len(chunk) > _MAX_PUBLIC_TRANSCRIPT_BYTES:
                    return None
                body.extend(chunk)
            raw_text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
            json_payload = json.loads(raw_text) if "json" in effective_type else None
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Publisher transcript fetch failed for %s: %s", transcript_url, exc)
        return None

    if "json" in effective_type:
        try:
            raw_text = _extract_json_transcript(json_payload)
        except (json.JSONDecodeError, ValueError):
            return None

    text = _strip_timed_transcript(raw_text)
    if len(text) < _MIN_TRANSCRIPT_CHARS:
        return None
    timed = any(marker in effective_type for marker in ("vtt", "subrip", "srt"))
    return TranscriptResult(
        text=text,
        provider="rss_podcast_transcript",
        artifact_url=transcript_url,
        language=language,
        timed=timed,
    )


def validate_assemblyai_key(api_key: str | None = None) -> bool:
    """Validate an AssemblyAI key without creating a transcription job."""
    key = (api_key or get_assemblyai_key()).strip()
    if not key:
        return False
    try:
        with httpx.Client(**make_client_kwargs(timeout=10)) as client:
            response = client.get(
                _ASSEMBLYAI_BASE_URL,
                params={"limit": 1},
                headers={"Authorization": key},
            )
    except httpx.HTTPError as exc:
        logger.warning("AssemblyAI key validation request failed: %s", exc)
        return False
    return response.status_code == 200


def assemblyai_transcribe_url(
    audio_url: str,
    api_key: str | None = None,
    *,
    poll_interval_seconds: float = 3.0,
    max_polls: int = 120,
) -> TranscriptResult | None:
    """Submit a public audio URL to AssemblyAI and return completed text.

    AssemblyAI fetches the remote audio itself. This avoids downloading media
    from the pipeline host, but the source URL must still be publicly reachable
    to AssemblyAI; it cannot bypass authentication, DRM, or a source that blocks
    its own fetcher.
    """
    key = (api_key or get_assemblyai_key()).strip()
    if not key or not audio_url:
        return None

    headers = {"Authorization": key, "Content-Type": "application/json"}
    request = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "auto_chapters": False,
        "punctuate": True,
        "format_text": True,
    }

    try:
        with httpx.Client(**make_client_kwargs(timeout=30)) as client:
            response = client.post(_ASSEMBLYAI_BASE_URL, headers=headers, json=request)
            if response.status_code in (401, 403):
                logger.warning("AssemblyAI rejected the configured API key.")
                return None
            response.raise_for_status()
            transcript_id = str(response.json().get("id", ""))
            if not transcript_id:
                logger.warning("AssemblyAI accepted the request without a transcript id.")
                return None

            for _ in range(max_polls):
                time.sleep(poll_interval_seconds)
                status_response = client.get(
                    f"{_ASSEMBLYAI_BASE_URL}/{transcript_id}",
                    headers={"Authorization": key},
                )
                status_response.raise_for_status()
                payload = status_response.json()
                status = str(payload.get("status", "")).lower()
                if status == "completed":
                    text = str(payload.get("text", "")).strip()
                    if len(text) < _MIN_TRANSCRIPT_CHARS:
                        return None
                    return TranscriptResult(
                        text=text,
                        provider="assemblyai_remote_url",
                        artifact_url=audio_url,
                        language=str(payload.get("language_code", "")),
                        speaker_labeled=bool(payload.get("utterances")),
                    )
                if status == "error":
                    logger.warning(
                        "AssemblyAI could not transcribe remote media: %s",
                        payload.get("error", "unknown error"),
                    )
                    return None
    except httpx.HTTPError as exc:
        logger.warning("AssemblyAI remote transcription failed: %s", exc)
        return None

    logger.warning("AssemblyAI transcription timed out after %d polls.", max_polls)
    return None
