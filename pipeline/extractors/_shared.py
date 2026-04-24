"""Shared utilities for content extractors.

Contains: subprocess wrappers, curl helpers, title extraction,
URL pattern matching, challenge page detection, and validation.
"""

from __future__ import annotations

import json
import logging
import re
import ipaddress
import socket
import subprocess
from typing import Optional
from urllib.parse import urlparse

log = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction fails and must not fall back to metadata-only.

    Use for YouTube transcripts and podcast audio — these sources MUST have
    full content, never just title/description metadata.
    """
    pass


# ─── Subprocess / HTTP Helpers ────────────────────────────────────────────────

def _run(args: list[str], timeout: int = 45, check: bool = False,
         input_data: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout. Returns CompletedProcess."""
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        input=input_data,
    )


def _validate_url(url: str) -> bool:
    """Validate externally fetched URLs to reduce SSRF exposure.

    Checks:
      1. Well-formed URL with http/https scheme
      2. No embedded credentials
      3. Host is not localhost or an internal/reserved IP literal, including
         alternate IPv4 encodings accepted by curl/yt-dlp (integer/octal/hex).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    if parsed.username or parsed.password:
        return False

    hostname = parsed.hostname or ""
    if not hostname:
        return False

    host = hostname.strip().lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return False

    if _host_is_blocked_address(host):
        return False

    if _host_resolves_to_blocked_address(host):
        return False

    return True


def _host_is_blocked_address(host: str) -> bool:
    """Return True for non-public IP literals, including legacy IPv4 forms."""
    candidates = {host}
    parsed = _parse_ipv4_weird(host)
    if parsed:
        candidates.add(parsed)

    for candidate in candidates:
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False

def _host_resolves_to_blocked_address(host: str) -> bool:
    """Return True when DNS resolution exposes non-public addresses.

    Hostname-only checks are not enough: public-looking names can resolve to
    loopback, RFC1918, link-local, multicast, or otherwise non-public IPs. For
    extraction we fail closed if DNS cannot be resolved.
    """
    if _host_is_blocked_address(host):
        return True
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError):
        return True
    if not infos:
        return True
    for info in infos:
        sockaddr = info[4]
        if not sockaddr or _host_is_blocked_address(sockaddr[0]):
            return True
    return False


def _parse_ipv4_weird(host: str) -> str:
    """Normalize IPv4 forms curl accepts: decimal int, hex/octal, short dotted."""
    if ":" in host:
        return ""
    parts = host.split(".")
    if not all(parts) or len(parts) > 4:
        return ""
    values: list[int] = []
    try:
        for part in parts:
            base = 10
            raw = part.lower()
            if raw.startswith("0x"):
                base = 16
            elif len(raw) > 1 and raw.startswith("0"):
                base = 8
            values.append(int(raw, base))
    except ValueError:
        return ""
    if len(values) == 1:
        value = values[0]
        if not 0 <= value <= 0xFFFFFFFF:
            return ""
        return str(ipaddress.IPv4Address(value))
    if any(not 0 <= value <= 255 for value in values):
        return ""
    while len(values) < 4:
        values.append(0)
    return ".".join(str(v) for v in values)


def _curl_get(url: str, headers: Optional[dict] = None, timeout: int = 45) -> str:
    """GET via curl (not Python urllib — urllib gets 403)."""
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    args = ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https", "--max-time", str(timeout)]
    if headers:
        for k, v in headers.items():
            args.extend(["-H", f"{k}: {v}"])
    args.append(url)
    result = _run(args, timeout=timeout + 5)
    return result.stdout.strip()


def _curl_post_json(url: str, data: dict, headers: Optional[dict] = None,
                    timeout: int = 45) -> str:
    """POST JSON via curl."""
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    args = ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https", "--max-time", str(timeout), "-X", "POST",
            "-H", "Content-Type: application/json"]
    if headers:
        for k, v in headers.items():
            args.extend(["-H", f"{k}: {v}"])
    args.extend(["-d", json.dumps(data)])
    args.append(url)
    result = _run(args, timeout=timeout + 5)
    return result.stdout.strip()


# ─── Title Extraction ────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from text for clean titles."""
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    return text.strip()


def extract_title(content: str) -> str:
    """Extract a title from content text.

    Strategy:
      1. Find first # heading (skip "Original content")
      2. Fallback to first non-empty line (max 120 chars)
    """
    if not content:
        return ""

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.lstrip("# ").startswith("Original content"):
            title = stripped.lstrip("# ").strip()
            if len(title) > 5:
                return _strip_markdown(title[:120])

    # Fallback: first non-empty, non-URL, non-image line
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("http", "!", "[")):
            continue
        if len(stripped) > 20:
            return _strip_markdown(stripped[:120])

    # Last resort: first non-empty line
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped:
            return _strip_markdown(stripped[:120])

    return ""


# ─── URL Patterns ────────────────────────────────────────────────────────────

_YT_PATTERNS = re.compile(
    r"(?:youtube\.com|youtu\.be|youtube-nocookie\.com)"
)
_PODCAST_PATTERNS = re.compile(
    r"(?:podcasts\.apple\.com|open\.spotify\.com/(?:show|episode)|"
    r"spotify\.com/(?:show|episode)|podcasts\.google\.com|"
    r"pca\.st|podbay\.fm|overcast\.fm|pocketcasts\.com|"
    r"castbox\.fm|podbean\.com|anchor\.fm|feeds\.[a-z]+|"
    r"podlink\.com|pod\.link|buzzsprout\.com|libsyn\.com|"
    r"transistor\.fm|simplecast\.com|megaphone\.fm|acast\.com|"
    r"podchaser\.com|podcastaddict\.com|podcastindex\.org|"
    r"redcircle\.com|podigee\.com|spreaker\.com|audioboom\.com|"
    r"omnycontent\.com|chtbl\.com|art19\.com|captivate\.fm|"
    r"fireside\.fm|rss\.com|podomatic\.com|"
    r"/(?:feed|rss|podcast)(?:\.|$))"
)
_TWITTER_PATTERNS = re.compile(
    r"(?:x\.com|twitter\.com)/"
)
_ARXIV_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/\d{4}\.\d{4,5}"
)
_YT_VIDEO_ID_PATTERNS = [
    re.compile(r"[?&]v=([a-zA-Z0-9_-]{11})"),
    re.compile(r"youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(r"shorts/([a-zA-Z0-9_-]{11})"),
    re.compile(r"embed/([a-zA-Z0-9_-]{11})"),
]


def _extract_youtube_video_id(url: str) -> str:
    """Extract 11-char YouTube video ID from URL."""
    for pat in _YT_VIDEO_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    # Fallback: find any 11-char alphanumeric sequence in known path/query segments
    for segment in url.split("/"):
        m = re.search(r"[a-zA-Z0-9_-]{11}", segment)
        if m:
            return m.group(0)
    return ""


def _extract_arxiv_paper_id(url: str) -> str:
    """Extract arxiv paper ID (e.g. 2503.03312) from URL."""
    m = re.search(r"(\d{4}\.\d{4,5})", url)
    return m.group(1) if m else ""


# ─── Cloudflare / Challenge Detection ────────────────────────────────────────

_CHALLENGE_PATTERNS = [
    re.compile(r"Just a moment\.\.\.", re.IGNORECASE),
    re.compile(r"Checking your browser", re.IGNORECASE),
    re.compile(r"cf-browser-verification", re.IGNORECASE),
    re.compile(r"attention required.*cloudflare", re.IGNORECASE),
    re.compile(r"enable javascript and cookies", re.IGNORECASE),
    re.compile(r"verify you are human", re.IGNORECASE),
    re.compile(r"Ray ID:", re.IGNORECASE),
    re.compile(r"_cf_chl_opt", re.IGNORECASE),
]


def _is_challenge_page(content: str) -> bool:
    """Detect Cloudflare/anti-bot challenge pages masquerading as content."""
    if not content:
        return False
    content_lower = content[:20].lower()
    if not content_lower.startswith("<!doctype") and not content_lower.startswith("<html"):
        return False
    for pattern in _CHALLENGE_PATTERNS:
        if pattern.search(content[:2000]):
            return True
    return False


# ─── Extraction Validation ───────────────────────────────────────────────────

def validate_extraction(content: str) -> tuple[bool, str]:
    """Validate extracted content quality. Returns (is_valid, reason)."""
    if not content:
        return False, "empty content"

    content_stripped = content.strip()

    if len(content_stripped) < 5:
        return False, f"too short ({len(content_stripped)} chars)"

    if _is_challenge_page(content_stripped):
        return False, "Cloudflare challenge page"

    failure_indicators = [
        "Content extraction failed",
        "Extraction failed with error",
        "This site can't be reached",
        "ERR_CONNECTION_REFUSED",
        "ERR_NAME_NOT_RESOLVED",
    ]
    for indicator in failure_indicators:
        if indicator in content_stripped[:500]:
            return False, f"failure indicator: {indicator}"

    return True, "ok"


# ─── Whisper Transcription ───────────────────────────────────────────────────

def transcribe_with_whisper(audio_file: str, language: str = "") -> str:
    """Transcribe audio file with local faster-whisper."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        kwargs = {"language": language} if language else {}
        segments, _info = model.transcribe(audio_file, **kwargs)
        return " ".join(s.text for s in segments)
    except ImportError:
        return ""


# ─── AssemblyAI Transcription ────────────────────────────────────────────────

def transcribe_assemblyai(audio_file: str, api_key: str, timeout: int = 45) -> str:
    """Upload audio to AssemblyAI and poll for transcription result."""
    api_url = "https://api.assemblyai.com"

    # Step 1: Upload
    upload_result = _run(
        ["curl", "-s", "-X", "POST", f"{api_url}/v2/upload",
         "-H", f"Authorization: Bearer {api_key}",
         "-H", "Content-Type: application/octet-stream",
         "--data-binary", f"@{audio_file}",
         "--max-time", str(min(timeout, 300))],
        timeout=timeout + 10,
    )
    if upload_result.returncode != 0:
        return ""
    try:
        upload_url = json.loads(upload_result.stdout).get("upload_url", "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not upload_url:
        return ""

    # Step 2: Submit transcript request
    submit_data = json.dumps({
        "audio_url": upload_url,
        "speech_models": ["universal-2"],
        "punctuate": True,
        "format_text": True,
    })
    submit_result = _run(
        ["curl", "-s", "-X", "POST", f"{api_url}/v2/transcript",
         "-H", f"Authorization: Bearer {api_key}",
         "-H", "Content-Type: application/json",
         "-d", submit_data,
         "--max-time", "30"],
        timeout=35,
    )
    if submit_result.returncode != 0:
        return ""
    try:
        transcript_id = json.loads(submit_result.stdout).get("id", "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not transcript_id:
        return ""

    # Step 3: Poll until complete
    import time
    for _ in range(120):  # max 10 minutes
        poll_result = _run(
            ["curl", "-s", f"{api_url}/v2/transcript/{transcript_id}",
             "-H", f"Authorization: Bearer {api_key}",
             "--max-time", "10"],
            timeout=15,
        )
        if poll_result.returncode != 0:
            return ""
        try:
            poll_data = json.loads(poll_result.stdout)
        except json.JSONDecodeError:
            return ""

        status = poll_data.get("status", "")
        if status == "completed":
            return poll_data.get("text", "")
        elif status == "error":
            return ""
        else:
            time.sleep(5)

    return ""
