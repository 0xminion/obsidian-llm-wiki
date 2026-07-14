"""YouTube transcript extraction via yt-dlp and AssemblyAI.

Strategy:
  1. yt-dlp subtitle extraction — downloads auto-generated or manual subtitles
  2. AssemblyAI remote-URL transcription — submits the YouTube URL to AssemblyAI
  3. Invidious API (metadata + description fallback)
  4. oEmbed metadata (title + channel only, last resort)

AssemblyAI API docs: https://www.assemblyai.com/docs/api-v2/transcript
Endpoint: POST https://api.assemblyai.com/v2/transcript
Auth: Authorization header with API key
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote, urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

logger = logging.getLogger("obswiki.ingest.youtube")

__all__ = ["extract_youtube_video"]

_ASSEMBLYAI_API_BASE = "https://api.assemblyai.com/v2"
_POLL_INTERVAL = 5  # seconds between job status checks
_POLL_MAX_ATTEMPTS = 120  # ~10 minutes max polling


def _get_assemblyai_key() -> str | None:
    """Read the AssemblyAI API key from env at call time."""
    return os.environ.get("ASSEMBLYAI_API_KEY", "").strip() or None


def _video_id(url: str) -> str | None:
    """Extract 11-char video ID from any YouTube URL."""
    import re
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


@register_extractor(lambda parsed, raw: raw.startswith("http") and bool(_video_id(raw)))
def extract_youtube_video(raw_url: str) -> SourceDoc:
    """Extract transcript from a YouTube video.

    Strategy:
      1. yt-dlp subtitle extraction (downloads auto-generated or manual subtitles)
      2. AssemblyAI remote-URL transcription (submits YouTube URL to AAI)
      3. Invidious API (metadata + description fallback)
      4. oEmbed metadata (title + channel only, last resort)

    Raises:
        RuntimeError: If all methods fail.
    """
    errors: list[str] = []

    # ── Primary: yt-dlp subtitle extraction ──────────────────────────
    try:
        source = _ytdlp_transcript(raw_url)
        if source:
            logger.info(
                "yt-dlp: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"yt-dlp: {exc}")

    # ── Fallback 1: AssemblyAI remote-URL transcription ──────────────
    aai_key = _get_assemblyai_key()
    if aai_key:
        try:
            source = _assemblyai_transcript(raw_url, aai_key)
            if source:
                logger.info(
                    "AssemblyAI: extracted %d chars for %s",
                    len(source.content), raw_url,
                )
                return source
        except Exception as exc:
            errors.append(f"assemblyai: {exc}")
    else:
        errors.append("assemblyai: API key not set")

    # ── Fallback 2: Invidious API (metadata + description, no transcript) ──
    try:
        from obsidian_llm_wiki.ingest.alt_source import extract_via_invidious
        source = extract_via_invidious(raw_url)
        logger.info("Invidious fallback: extracted %d chars for %s", len(source.content), raw_url)
        return source
    except Exception as exc:
        errors.append(f"invidious: {exc}")

    # ── Fallback 3: oEmbed metadata (title + author only, minimal) ──
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


# ── yt-dlp subtitle extraction ──────────────────────────────────────────


def _ytdlp_transcript(url: str) -> SourceDoc | None:
    """Extract transcript via yt-dlp subtitle download.

    yt-dlp can download auto-generated subtitles and convert them to plain text.
    This requires the yt-dlp CLI to be installed.
    """
    ytdlp = shutil.which("yt-dlp") or shutil.which("youtube-dl")
    if not ytdlp:
        raise RuntimeError("yt-dlp not found — install with: pip install yt-dlp")

    vid = _video_id(url)
    if not vid:
        raise RuntimeError(f"Could not extract video ID from {url}")

    # Use yt-dlp to download subtitles (auto-generated + manual)
    # --write-auto-sub: download auto-generated subtitles
    # --sub-lang en: prefer English subtitles
    # --skip-download: don't download the video
    # --convert-sub vtt: convert to VTT format
    tmp_dir: str | None = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="ytdlp_")
        cmd = [
            ytdlp,
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en,en-US,en-GB",
            "--sub-format", "vtt/srt",
            "--convert-sub", "vtt",
            "--no-playlist",
            "-o", os.path.join(tmp_dir, "%(title)s.%(ext)s"),
        ]

        # Route yt-dlp through the residential proxy / Tailscale exit node
        # when configured. YouTube blocks requests from datacenter IPs;
        # a residential IP or Tailscale exit node bypasses the bot check.
        proxy = (
            os.environ.get("RESIDENTIAL_PROXY_URL", "").strip()
            or os.environ.get("HTTPS_PROXY", "").strip()
        )
        if proxy:
            cmd.extend(["--proxy", proxy])
        cmd.append(url)

        env = os.environ.copy()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env,
        )

        # Find subtitle files
        sub_files = []
        if tmp_dir:
            for f in os.listdir(tmp_dir):
                if f.endswith((".vtt", ".srt", ".ass", ".json3")):
                    sub_files.append(os.path.join(tmp_dir, f))

        if not sub_files:
            # yt-dlp may have failed silently — check stderr
            if proc.returncode != 0:
                stderr = proc.stderr.strip()[:200]
                raise RuntimeError(f"yt-dlp exited {proc.returncode}: {stderr}")
            raise RuntimeError("yt-dlp: no subtitle files downloaded")

        # Parse the first available subtitle file
        sub_path = sorted(sub_files)[0]
        text = _parse_subtitle_file(sub_path)

        if not text or len(text) < 200:
            raise RuntimeError(
                f"yt-dlp transcript too short ({len(text)} chars) — "
                "video may have no speech or no subtitles"
            )

        # Get title from yt-dlp output or subtitle filename
        title = _fetch_youtube_title(url)
        if not title:
            # Try to extract from filename
            title = os.path.basename(sub_path).rsplit(".", 2)[0]
        if not title:
            title = f"YouTube video {vid}"

        return SourceDoc(title=title, content=text, url=url)

    finally:
        if tmp_dir:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_subtitle_file(path: str) -> str:
    """Parse VTT/SRT subtitle file into plain text."""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    lines = raw.split("\n")
    text_parts: list[str] = []
    seen_lines: set[str] = set()

    for line in lines:
        line = line.strip()
        # Skip VTT header
        if line.startswith("WEBVTT"):
            continue
        # Skip timestamp lines (00:00:01.000 --> 00:00:05.000)
        if "-->" in line:
            continue
        # Skip cue identifiers and numbers
        if line.isdigit():
            continue
        # Skip empty lines
        if not line:
            continue
        # Skip VTT styling/positioning blocks (multi-line cues)
        if line.startswith("NOTE") or line.startswith("STYLE") or line.startswith("REGION"):
            continue
        # Skip pure VTT tag lines like <c>, but NOT content with inline tags
        if (
            line.startswith("<")
            and line.endswith(">")
            and not any(
                c.isalpha()
                for c in line[1:-1]
                if c not in "c/."
            )
        ):
            continue

        # Strip HTML tags from subtitle text
        import re
        clean = re.sub(r"<[^>]+>", "", line)
        clean = clean.strip()
        if clean and clean not in seen_lines:
            seen_lines.add(clean)
            text_parts.append(clean)

    return " ".join(text_parts)


# ── AssemblyAI remote-URL transcription ─────────────────────────────────


def _assemblyai_transcript(url: str, api_key: str) -> SourceDoc | None:
    """Submit a YouTube URL to AssemblyAI for remote-URL transcription.

    AssemblyAI can fetch media from a public URL and transcribe it.
    First tries yt-dlp to get a direct audio stream URL. If that fails
    (YouTube bot detection, no cookies), submits the YouTube URL directly
    to AssemblyAI — their service can fetch from YouTube on their servers.
    """
    # Try to get a direct audio stream URL via yt-dlp
    audio_url = _get_youtube_audio_url(url)

    if not audio_url:
        # yt-dlp failed (bot detection, no cookies). AssemblyAI cannot
        # fetch from YouTube URLs directly — it needs an actual audio
        # file URL. Try downloading audio locally and uploading.
        audio_url = _download_and_upload_audio(url, api_key)
        if not audio_url:
            raise RuntimeError(
                "Could not get audio URL via yt-dlp or local download; "
                "AssemblyAI requires an audio file URL"
            )

    # Submit to AssemblyAI
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "audio_url": audio_url,
        "language_code": "en",  # default to English
    }

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{_ASSEMBLYAI_API_BASE}/transcript",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        transcript_data = resp.json()

    transcript_id = transcript_data.get("id")
    if not transcript_id:
        raise RuntimeError("AssemblyAI: no transcript ID returned")

    # Poll for completion
    with httpx.Client(timeout=30) as client:
        for attempt in range(_POLL_MAX_ATTEMPTS):
            time.sleep(_POLL_INTERVAL)
            resp = client.get(
                f"{_ASSEMBLYAI_API_BASE}/transcript/{transcript_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "completed":
                text = data.get("text", "")
                if not text or len(text) < 200:
                    raise RuntimeError(
                        f"AssemblyAI transcript too short ({len(text)} chars)"
                    )
                title = _fetch_youtube_title(url) or f"YouTube video {_video_id(url)}"
                return SourceDoc(title=title, content=text, url=url)

            if status == "error":
                error = data.get("error", "unknown error")
                raise RuntimeError(f"AssemblyAI transcription failed: {error}")

            logger.debug(
                "AssemblyAI transcript %s: %s (attempt %d)",
                transcript_id, status, attempt + 1,
            )

    raise RuntimeError(
        f"AssemblyAI transcript {transcript_id} timed out after "
        f"{_POLL_MAX_ATTEMPTS * _POLL_INTERVAL}s"
    )


def _get_youtube_audio_url(url: str) -> str | None:
    """Get a direct audio stream URL from YouTube via yt-dlp."""
    ytdlp = shutil.which("yt-dlp") or shutil.which("youtube-dl")
    if not ytdlp:
        return None

    cmd = [
        ytdlp,
        "-f", "bestaudio",
        "-g",  # print direct URL only
        "--no-playlist",
    ]

    # Route through residential proxy / Tailscale exit node when configured
    proxy = (
        os.environ.get("RESIDENTIAL_PROXY_URL", "").strip()
        or os.environ.get("HTTPS_PROXY", "").strip()
    )
    if proxy:
        cmd.extend(["--proxy", proxy])
    cmd.append(url)

    env = os.environ.copy()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=env,
        )
        if proc.returncode == 0:
            url_out = proc.stdout.strip().split("\n")[0]
            if url_out.startswith("http"):
                return url_out
    except Exception:
        pass
    return None


# ── Metadata helpers ────────────────────────────────────────────────────


def _fetch_youtube_title(url: str) -> str:
    """Fetch video title via YouTube oEmbed API (no auth required).

    Returns empty string if the fetch fails.
    """
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url={quote(url, safe='')}&format=json"
    )
    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            from obsidian_llm_wiki.ingest.url_safety import get_with_validated_redirects
            resp = get_with_validated_redirects(client, oembed_url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("title", "") or ""
    except Exception:
        pass
    return ""


def _download_and_upload_audio(url: str, api_key: str) -> str | None:
    """Download audio via yt-dlp and upload to AssemblyAI's upload endpoint.

    When YouTube blocks direct URL extraction (bot detection), download
    the audio locally with cookies and upload it to AssemblyAI's upload
    URL for transcription.
    """
    ytdlp = shutil.which("yt-dlp") or shutil.which("youtube-dl")
    if not ytdlp:
        return None

    tmp_dir: str | None = None
    try:
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="ytdlp_audio_")
        output_path = os.path.join(tmp_dir, "audio.%(ext)s")

        env = os.environ.copy()

        # Try with cookies if configured
        cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE", "")
        cmd = [
            ytdlp,
            "-f", "bestaudio",
            "-x", "--audio-format", "mp3",
            "--no-playlist",
            "-o", output_path,
            "--no-warnings",
        ]
        # Route through residential proxy / Tailscale exit node when configured
        proxy = (
            os.environ.get("RESIDENTIAL_PROXY_URL", "").strip()
            or os.environ.get("HTTPS_PROXY", "").strip()
        )
        if proxy:
            cmd.extend(["--proxy", proxy])
        if cookies_file and os.path.isfile(cookies_file):
            cmd.extend(["--cookies", cookies_file])
        cmd.append(url)

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
        )
        if proc.returncode != 0:
            logger.warning("yt-dlp audio download failed: %s", proc.stderr[:200])
            return None

        # Find the downloaded audio file
        audio_files = [
            os.path.join(tmp_dir, f)
            for f in os.listdir(tmp_dir)
            if f.endswith((".mp3", ".m4a", ".webm", ".ogg", ".wav"))
        ]
        if not audio_files:
            return None

        audio_path = audio_files[0]
        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            return None

        # Upload to AssemblyAI's upload endpoint
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {"Authorization": api_key}

        with open(audio_path, "rb") as f, httpx.Client(timeout=300) as client:
            resp = client.post(
                upload_url,
                headers=headers,
                content=f,
            )
            resp.raise_for_status()
            data = resp.json()

        upload_result = data.get("upload_url", "")
        if upload_result:
            logger.info("Uploaded audio to AssemblyAI: %s", upload_result)
            return upload_result

    except Exception as exc:
        logger.warning("Audio download+upload failed: %s", exc)
    finally:
        if tmp_dir:
            import shutil as _shutil
            _shutil.rmtree(tmp_dir, ignore_errors=True)

    return None


def _extract_oembed(youtube_url: str) -> SourceDoc | None:
    """Fetch video metadata via YouTube oEmbed API (no auth required)."""
    oembed_url = f"https://www.youtube.com/oembed?url={quote(youtube_url, safe='')}&format=json"
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        from obsidian_llm_wiki.ingest.url_safety import get_with_validated_redirects
        resp = get_with_validated_redirects(client, oembed_url)
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
        "(yt-dlp and AssemblyAI could not extract subtitles).",
        "Only video metadata was extracted.",
    ]

    content = "\n".join(content_parts)
    if len(content) < 100:
        return None

    return SourceDoc(title=title, content=content, url=youtube_url)
