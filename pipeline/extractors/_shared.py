"""Shared utilities for content extractors.

Contains: subprocess wrappers, curl helpers, title extraction,
URL pattern matching, challenge page detection, and validation.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
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


def _curl_resolve_args(url: str) -> list[str] | None:
    """Pin curl to a public resolved IP to reduce DNS rebinding TOCTOU.

    Returns None when a safe public pin cannot be established; callers must
    fail closed instead of letting curl perform its own DNS lookup.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    for info in infos:
        ip = info[4][0]
        try:
            if not _host_is_blocked_address(ip):
                return ["--resolve", f"{host}:{port}:{ip}"]
        except ValueError:
            continue
    return None


def _curl_header_config(headers: Optional[dict]) -> str:
    if not headers:
        return ""
    lines = []
    for k, v in headers.items():
        key = str(k).replace("\n", "").replace("\r", "")
        value = str(v).replace("\n", "").replace("\r", "")
        lines.append(f'header = "{key}: {value}"')
    return "\n".join(lines) + "\n"


def _curl_get(url: str, headers: Optional[dict] = None, timeout: int = 45) -> str:
    """GET via curl without exposing secret headers in argv."""
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    args = ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https", "--max-time", str(timeout)]
    resolve_args = _curl_resolve_args(url)
    if resolve_args is None:
        log.warning("Blocked URL after unsafe DNS pinning result: %s", url[:80])
        return ""
    args.extend(resolve_args)
    input_config = None
    if headers:
        args.extend(["--config", "-"])
        input_config = _curl_header_config(headers)
    args.append(url)
    result = _run(args, timeout=timeout + 5, input_data=input_config)
    return result.stdout.strip()


def _curl_post_json(url: str, data: dict, headers: Optional[dict] = None,
                    timeout: int = 45) -> str:
    """POST JSON via curl without exposing secret headers in argv."""
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    args = ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https", "--max-time", str(timeout), "-X", "POST"]
    resolve_args = _curl_resolve_args(url)
    if resolve_args is None:
        log.warning("Blocked URL after unsafe DNS pinning result: %s", url[:80])
        return ""
    args.extend(resolve_args)
    config_lines = ['header = "Content-Type: application/json"']
    if headers:
        config_lines.append(_curl_header_config(headers).strip())
    args.extend(["--config", "-"])
    args.extend(["-d", json.dumps(data)])
    args.append(url)
    result = _run(args, timeout=timeout + 5, input_data="\n".join(line for line in config_lines if line) + "\n")
    return result.stdout.strip()


# ─── Title Extraction ────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from text for clean titles."""
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    return text.strip()


def extract_title(content: str, fallback_title: str = "") -> str:
    """Extract a title from content text.

    Strategy:
      1. If content starts with HTML tag → return fallback_title
      2. Find first # heading (skip "Original content")
      3. Find first ## heading if no # heading (ar5iv papers etc.)
      4. Fallback: first non-empty line that LOOKS LIKE a title
      5. Last resort: return fallback_title
    """
    if not content:
        return fallback_title or ""

    stripped = content.lstrip()[:50]
    if stripped.startswith(("<!DOCTYPE", "<html", "<HTML", "<head>", "<HEAD>")):
        return fallback_title or ""

    # Strip ar5iv footnote noise from the first lines
    clean = re.sub(r"††[^\n]*", "", content)
    # Strip inline LaTeX footnote marks (\[...\]) and footnotes from headings
    for i, line in enumerate(clean.split("\n")):
        if line.startswith("#"):
            clean = clean.replace(line, re.sub(r"\\\[.*?\\\]", "", line))
            break

    for line in clean.split("\n"):
        s = line.strip()
        if s.startswith("# ") and not s.lstrip("# ").startswith("Original content"):
            title = s.lstrip("# ").strip()
            if len(title) > 5:
                return _strip_markdown(title[:120])

    # Also check ## headings (ar5iv papers use ## for main title)
    for line in clean.split("\n"):
        s = line.strip()
        if s.startswith("## "):
            title = s.lstrip("#").strip()
            if 10 <= len(title) <= 120:
                lower = title.lower()
                if lower not in {"abstract", "introduction", "background", "methods", "results", "discussion", "conclusion", "references", "acknowledgements", "acknowledgments"} and not lower.startswith("acknowledg"):
                    return _strip_markdown(title)

    _UI_NOISE = re.compile(
        r"(?im)^\s*(?:get\s+(?:the\s+)?app|sign\s+(?:up|in)|follow|like|save|share|login|"
        r"subscribe|join|bookmark|press\s+enter|click\s+to\s+view|image\s+in\s+full"
        r"|load(?:ing)?\s+(?:more|comments)|terms\s+.*\s+use|privacy\s+.*\s+policy|"
        r"\d+\s+claps?|respond|comment)"
    )
    for line in clean.split("\n"):
        s = line.strip()
        if not s or s.startswith(("http", "!", "[", "#")):
            continue
        if _UI_NOISE.search(s):
            continue
        # Title heuristics (titles may end in ? but not . or sentence-like)
        if len(s) < 10 or len(s) > 80:
            continue
        if s[0].islower() or s[0] in ("'", '"', "“"):
            continue
        # Reject if it ends with period (sentence, not title)
        if s[-1] == ".":
            continue
        return _strip_markdown(s[:120])

    return fallback_title or ""


def _extract_html_title(html: str, fallback: str = "") -> str:
    """Extract title from raw HTML using <title> or og:title.

    Returns fallback if neither found.
    """
    if not html:
        return fallback

    # 1. <title> tag
    m = re.search(r"<\s*title\s*>\s*(.+?)\s*<\s*/\s*title\s*>", html, re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
        if t and len(t) > 2:
            return _strip_html(t)[:120]

    # 2. og:title
    m = re.search(r'<\s*meta\s+[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
        if t and len(t) > 2:
            return _strip_html(t)[:120]
    # og:title can also be content-first
    m = re.search(r'<\s*meta\s+[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']', html, re.IGNORECASE | re.DOTALL)
    if m:
        t = m.group(1).strip()
        if t and len(t) > 2:
            return _strip_html(t)[:120]

    return fallback


def _strip_html(text: str) -> str:
    """Remove simple HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _url_to_title(url: str) -> str:
    """Convert a URL path slug into a readable title.

    Handles apostrophes in URL slugs (e.g. medium.com/.../here-s-why-it-s-hard).
    """
    from urllib.parse import urlparse, unquote
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        # Get the last non-empty path segment
        segments = [s for s in path.split("/") if s]
        if segments:
            slug = segments[-1]
            # Strip extensions
            slug = slug.rsplit(".", 1)[0] if "." in slug else slug
            # Strip trailing Medium-style hash (e.g. 8684d9a77e11)
            slug = re.sub(r"-[a-f0-9]{11,}$", "", slug)
            # Words from slug parts, preserving apostrophes
            # "here-s-why-it-s-hard" → split on "-" then convert "s" back to "'s"
            raw_parts = re.split(r"[-_]+", slug)
            words = []
            for i, w in enumerate(raw_parts):
                if w == "s" and words:
                    words[-1] = words[-1] + "'s"   # "here" + "s" → "here's"
                elif w == "re" and words:
                    words[-1] = words[-1] + "'re"  # "we" + "re" → "we're"
                elif w == "m" and words:
                    words[-1] = words[-1] + "'m"    # "i" + "m" → "i'm"
                elif w == "ve" and words:
                    words[-1] = words[-1] + "'ve"   # "they" + "ve" → "they've"
                elif w == "d" and words:
                    words[-1] = words[-1] + "'d"
                elif w == "ll" and words:
                    words[-1] = words[-1] + "'ll"
                elif w == "t" and words and words[-1].lower() in ("won", "can"):
                    words[-1] = words[-1] + "'t"
                else:
                    words.append(w)
            return " ".join(words).title()[:120]
    except Exception:
        pass
    return ""

def _is_cloudflare_html(content: str) -> bool:
    """Detect Cloudflare challenge/error pages."""
    if not content:
        return False
    first_5k = content[:5000].lower()
    markers = [
        "<!doctype html", "<html", "checking your browser",
        "enable javascript", "please wait while we check",
        "cloudflare", "ddos protection",
    ]
    return any(m in first_5k for m in markers[:4]) or first_5k.count("<") > first_5k.count(">") + 50


# ── Title noise detectors ──────────────────────────────────────────────────────

_MEDIUM_TITLE_PATTERNS = [
    re.compile(r"^\s*Get\s+(?:the\s+)?app\s*$", re.IGNORECASE),
    re.compile(r"^\s*Press\s+enter\s+or\s+click\s+to\s+view\s*$", re.IGNORECASE),
    re.compile(r"^\s*Sign\s+[Uu]p\s*$"),
    re.compile(r"^\s*Sign\s+[Ii]n\s*$"),
]

_CLOUDFLARE_TITLES = {
    "your privacy, your choice",
    "just a moment...",
    "are you human?",
    "error 1020",
    "access denied",
    "attention required! | cloudflare",
    "checking your browser before accessing",
    "please wait while your request is being verified",
}

_ARCHIVE_NAV_TITLES = {
    "about", "- about", "blog", "terms of use", "privacy policy",
    "wayback machine",
}


def _is_ui_noise_title(title: str) -> bool:
    """Check if a title is Medium UI noise, Cloudflare, GDPR, archive.org nav, etc."""
    if not title:
        return True
    t = title.strip().lower()
    for pat in _MEDIUM_TITLE_PATTERNS:
        if pat.match(t):
            return True
    if t in _CLOUDFLARE_TITLES or t in _ARCHIVE_NAV_TITLES:
        return True
    if any(m in t for m in ("checking your browser", "just a moment", "your privacy, your choice")):
        return True
    return False


# ─── URL Patterns ────────────────────────────────────────────────────────────
def _is_archive_wrapper(content: str) -> bool:
    """Detect archive.org Wayback Machine wrapper pages (not real article content).

    When defuddle processes an archive.org snapshot, it sometimes parses the
    archive navigation bar (About, Blog, Terms of Use links) instead of the
    actual article.  This detects those wrapper pages so we can fall back to
    Camoufox or use the URL-derived title.
    """
    if not content:
        return False
    text = content[:2000].lower()
    archive_refs = text.count("archive.org") + text.count("wayback machine")
    nav_markers = sum(1 for m in ["about", "blog", "terms of use", "privacy policy"] if m in text[:500])
    return archive_refs >= 3 or nav_markers >= 2


# ─── URL Patterns ────────────────────────────────────────────────────────────

_YT_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
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


_JS_RENDERED_PATTERNS = [
    re.compile(r"medium\.com", re.IGNORECASE),
    re.compile(r"substack\.com", re.IGNORECASE),
    re.compile(r"nature\.com", re.IGNORECASE),
    re.compile(r"akjournals\.com", re.IGNORECASE),
    re.compile(r"blog\.monad\.xyz", re.IGNORECASE),
    re.compile(r"blogs\.law\.ox\.ac\.uk", re.IGNORECASE),
    re.compile(r"oddchain\.com", re.IGNORECASE),
    re.compile(r"thetokendispatch\.com", re.IGNORECASE),
    re.compile(r"sciencedirect\.com", re.IGNORECASE),
    re.compile(r"springer\.com", re.IGNORECASE),
    re.compile(r"ieee\.org", re.IGNORECASE),
    re.compile(r"acm\.org", re.IGNORECASE),
    re.compile(r"jstor\.org", re.IGNORECASE),
]

def _should_use_camoufox_first(url: str) -> bool:
    """Return True for JS-rendered / Cloudflare-walled domains."""
    if not url:
        return False
    return any(p.search(url) for p in _JS_RENDERED_PATTERNS)

def _is_youtube_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return parsed.scheme in {"http", "https"} and host in _YT_ALLOWED_HOSTS


def _canonical_youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _extract_youtube_video_id(url: str) -> str:
    """Extract 11-char YouTube video ID only from validated YouTube URLs."""
    if not _is_youtube_url(url):
        return ""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower().rstrip(".") in {"youtu.be", "www.youtu.be"}:
        segment = parsed.path.strip("/").split("/", 1)[0]
        if re.fullmatch(r"[a-zA-Z0-9_-]{11}", segment):
            return segment
    for pat in _YT_VIDEO_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
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
    """Upload audio to AssemblyAI and poll for transcription result.

    Secret-bearing headers are passed through curl config on stdin, never argv.
    """
    api_url = "https://api.assemblyai.com"
    auth_cfg = _curl_header_config({"Authorization": f"Bearer {api_key}"})

    upload_url_endpoint = f"{api_url}/v2/upload"
    upload_args = [
        "curl", "-s", "-X", "POST", upload_url_endpoint,
        "--config", "-",
        "--data-binary", f"@{audio_file}",
        "--max-time", str(min(timeout, 300)),
    ]
    upload_resolve = _curl_resolve_args(upload_url_endpoint)
    if upload_resolve is None:
        return ""
    upload_args.extend(upload_resolve)
    upload_result = _run(
        upload_args,
        timeout=timeout + 10,
        input_data=auth_cfg + 'header = "Content-Type: application/octet-stream"\n',
    )
    if upload_result.returncode != 0:
        return ""
    try:
        upload_url = json.loads(upload_result.stdout).get("upload_url", "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not upload_url:
        return ""

    submit_data = json.dumps({
        "audio_url": upload_url,
        "speech_models": ["universal-2"],
        "punctuate": True,
        "format_text": True,
    })
    submit_url = f"{api_url}/v2/transcript"
    submit_args = [
        "curl", "-s", "-X", "POST", submit_url,
        "--config", "-",
        "-d", submit_data,
        "--max-time", "30",
    ]
    submit_resolve = _curl_resolve_args(submit_url)
    if submit_resolve is None:
        return ""
    submit_args.extend(submit_resolve)
    submit_result = _run(
        submit_args,
        timeout=35,
        input_data=auth_cfg + 'header = "Content-Type: application/json"\n',
    )
    if submit_result.returncode != 0:
        return ""
    try:
        transcript_id = json.loads(submit_result.stdout).get("id", "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    if not transcript_id:
        return ""

    import time
    for _ in range(120):  # max 10 minutes
        poll_url = f"{api_url}/v2/transcript/{transcript_id}"
        poll_args = ["curl", "-s", poll_url, "--config", "-", "--max-time", "10"]
        poll_resolve = _curl_resolve_args(poll_url)
        if poll_resolve is None:
            return ""
        poll_args.extend(poll_resolve)
        poll_result = _run(poll_args, timeout=15, input_data=auth_cfg)
        if poll_result.returncode != 0:
            return ""
        try:
            poll_data = json.loads(poll_result.stdout)
        except json.JSONDecodeError:
            return ""

        status = poll_data.get("status", "")
        if status == "completed":
            return poll_data.get("text", "")
        if status == "error":
            return ""
        time.sleep(5)

    return ""


# ─── Quality Scoring ───────────────────────────────────────────────────────────

def score_defuddle(body: str) -> float:
    """Paragraph count / total lines ratio for defuddle output.

    A single paragraph of continuous text is ideal (ratio ≈ 1.0).
    Many short lines (nav menus, footers) drive the score down.
    """
    if not body:
        return 0.0
    lines = body.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return 0.0
    # paragraphs = blocks separated by blank lines
    paragraphs = 1
    prev_blank = False
    for line in lines:
        if line.strip():
            if prev_blank:
                paragraphs += 1
            prev_blank = False
        else:
            prev_blank = True
    return min(paragraphs / len(non_empty), 1.0)


def score_youtube(word_count: int, duration_min: int) -> float:
    """Words/min density (ideal = 120 WPM)."""
    if duration_min <= 0:
        return 0.0
    wpm = word_count / duration_min
    return max(0.0, min(1.0, 1.0 - abs(wpm - 120) / 120))


def score_web(body: str) -> float:
    """Text-to-noise ratio after stripping nav/footer markers.

    Only removes lines that are *entirely* noise markers, so inline
    words like "navigator" or "cookies" are never stripped.
    """
    if not body:
        return 0.0
    noise = re.compile(
        r"(?im)^\s*(?:navigation|footer|privacy\s+policy|terms\s+of\s+use|cookie\s+notice|advertisements?)\s*$"
    )
    cleaned = noise.sub("", body)
    cleaned_stripped = re.sub(r"\s+", "", cleaned)
    body_stripped = re.sub(r"\s+", "", body)
    if not body_stripped:
        return 0.0
    return min(len(cleaned_stripped) / len(body_stripped), 1.0)


def score_pdf(body: str, page_estimate: int) -> float:
    """Chars per page estimate."""
    if page_estimate <= 0:
        return 0.0
    chars_per_page = len(body) / page_estimate
    # ideal ~ 3000 chars/page
    return max(0.0, min(1.0, chars_per_page / 3000))


def score_podcast(word_count: int, audio_sec: int) -> float:
    """Ideal 150 WPM, penalize if <50 words."""
    if audio_sec <= 0:
        return 0.0
    wpm = (word_count / audio_sec) * 60
    if word_count < 50:
        return 0.0
    return max(0.0, min(1.0, 1.0 - abs(wpm - 150) / 150))


__all__ = [
    "ExtractionError",
    "_run",
    "_curl_get",
    "_curl_post_json",
    "_validate_url",
    "_strip_markdown",
    "extract_title",
    "_extract_youtube_video_id",
    "_extract_arxiv_paper_id",
    "_is_challenge_page",
    "validate_extraction",
    "transcribe_with_whisper",
    "transcribe_assemblyai",
    "score_defuddle",
    "score_youtube",
    "score_web",
    "score_pdf",
    "score_podcast",
]
