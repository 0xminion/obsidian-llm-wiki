"""Cache-first transcript acquisition for public podcast media.

The resolver deliberately prefers publisher-provided artifacts over generated
ASR output. Remote-URL transcription providers fetch the audio themselves,
which avoids downloading public media from the pipeline host when that host is
blocked or challenged. Local Whisper remains the final fallback in the podcast
extractor because it must acquire the media locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.runtime import writes_allowed
from obsidian_llm_wiki.ingest.url_safety import stream_with_validated_redirects

logger = logging.getLogger("obswiki.ingest.transcript_resolver")

_ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2/transcript"
_MIN_TRANSCRIPT_CHARS = 200
_MAX_PUBLIC_TRANSCRIPT_BYTES = 10_000_000
_PUBLIC_NORMALIZATION_VERSION = 2
_PODCAST_ADMISSION_VERSION = 1
_PUBLIC_PROVIDERS = frozenset({
    "rss_podcast_transcript", "publisher_episode_page", "publisher_transcript_link",
})

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
    # Document identity, extracted before normalization; never inferred from speech.
    document_title: str = ""


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

    # Older public HTML normalization could silently delete literal '<' and
    # subsequent prose. Reacquire those artifacts instead of serving corruption.
    # Unaffected ASR caches remain reusable; this is not a blanket paid-ASR retry.
    if (
        payload.get("provider") == "rss_podcast_transcript"
        and payload.get("normalization_version") != _PUBLIC_NORMALIZATION_VERSION
    ):
        return None
    # Earlier public caches lack document identity and may contain a mismatched
    # artifact or inline link padding. Normalization v2 alone cannot prove safe
    # admission. Miss without mutation; unaffected ASR caches remain reusable.
    if (
        payload.get("provider") in _PUBLIC_PROVIDERS
        and payload.get("podcast_admission_version") != _PODCAST_ADMISSION_VERSION
    ):
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
        document_title=str(payload.get("document_title", "")),
    )


def save_transcript_cache(identity: str, result: TranscriptResult) -> None:
    """Persist a usable transcript and immutable provenance for re-use."""
    if not writes_allowed() or not identity or len(result.text.strip()) < _MIN_TRANSCRIPT_CHARS:
        return

    cache_path = _cache_dir() / f"{_cache_key(identity)}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(result),
        "identity": identity,
        "normalization_version": _PUBLIC_NORMALIZATION_VERSION,
        "podcast_admission_version": _PODCAST_ADMISSION_VERSION,
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fd, temporary = tempfile.mkstemp(dir=cache_path.parent, prefix=".transcript-", suffix=".tmp")
    tmp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class _TranscriptDocumentParser(HTMLParser):
    """Read actual document title/H1 elements without interpreting spoken prose."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[str] = []
        self.parts: list[str] = []
        self.heading = ""
        self.hidden = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.hidden:
            return
        if tag in {"script", "style", "template", "noscript"}:
            self.hidden = tag
        elif tag in {"title", "h1"} and not self.heading:
            self.heading = tag
            self.parts = []

    def handle_data(self, data: str) -> None:
        if self.heading and not self.hidden:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden:
                self.hidden = ""
        elif tag == self.heading:
            self.titles.append(" ".join("".join(self.parts).split()))
            self.heading = ""


class _TranscriptHTMLParser(HTMLParser):
    """Decode character references as data, never as a second layer of markup."""

    _BLOCK_TAGS = {"article", "div", "p", "section", "li", "br", "hr", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")


def _strip_timed_transcript(text: str) -> str:
    """Normalize VTT/SRT/plain/HTML transcript content into readable markdown."""
    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"(?im)^WEBVTT[^\n]*\n", "", cleaned)
    cleaned = re.sub(r"(?m)^\d+\s*$", "", cleaned)
    cleaned = re.sub(
        r"(?m)^\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}\s+-->.*$",
        "",
        cleaned,
    )
    # WebVTT word timestamps are cue syntax, not HTML tags. Remove only this
    # recognized syntax before parsing; escaped timestamp literals remain data.
    cleaned = re.sub(r"<(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3}>", "", cleaned)
    parser = _TranscriptHTMLParser()
    parser.feed(cleaned)
    parser.close()
    cleaned = "".join(parser.parts)
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

    # Parse structure before normalization merges headers and speech. The
    # resulting title is evidence only: it never changes the transcript body.
    document = _TranscriptDocumentParser()
    document.feed(raw_text)
    document.close()
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
        document_title="\n".join(dict.fromkeys(document.titles)),
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
