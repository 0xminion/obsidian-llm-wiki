"""Web content extraction with Cloudflare detection and fallback chain.

Chain: defuddle → curl+liteparse → defuddle --json → archive.org.
Detects and retries on Cloudflare challenge pages.
Handles arxiv specially via alphaxiv.org.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from pipeline.config import Config
from pipeline.models import ExtractedSource, SourceType
from pipeline.extractors._shared import (
    _curl_get,
    _run,
    _ARXIV_PATTERN,
    _extract_arxiv_paper_id,
    _is_challenge_page,
    extract_title,
)

log = logging.getLogger(__name__)


def extract_web(url: str, cfg: Config, source_type: SourceType = SourceType.WEB) -> ExtractedSource:
    """Extract web content via defuddle CLI with curl fallback and retry logic.

    Chain: defuddle → curl+liteparse → defuddle --json → archive.org
    Detects and retries on Cloudflare challenge pages.
    """
    timeout = cfg.extract_timeout
    max_retries = cfg.max_retries

    content = ""
    for attempt in range(max_retries):
        content = _extract_web_content(url, timeout, attempt=attempt)

        if not content:
            log.warning("Web extraction attempt %d returned empty for %s", attempt + 1, url)
            continue

        if _is_challenge_page(content):
            log.warning("Web extraction attempt %d got Cloudflare challenge for %s", attempt + 1, url)
            content = ""
            continue

        if len(content.strip()) < 20:
            log.warning("Web extraction attempt %d too short (%d chars) for %s", attempt + 1, len(content), url)
            content = ""
            continue

        break  # Success

    # If all retries failed, try archive.org
    if not content or _is_challenge_page(content):
        content = _try_archive_extract(url, timeout)
        if content:
            log.info("Archive.org extraction succeeded for %s", url)

    # Final fallback: Camoufox headless browser
    if not content or _is_challenge_page(content):
        content = _try_camoufox(url, timeout)
        if content:
            log.info("Camoufox extraction succeeded for %s", url)

    if not content:
        log.warning("All web extraction methods failed for %s", url)
        content = f"URL: {url}\n\nNote: Content extraction failed (all methods exhausted)."

    title = extract_title(content)

    return ExtractedSource(
        url=url,
        title=title or url,
        content=content,
        type=source_type,
    )


def _extract_web_content(url: str, timeout: int = 45, attempt: int = 0) -> str:
    """Extract web content. Tries defuddle → curl fallback.

    attempt > 0 rotates user-agents for retry.
    Handles arxiv specially via alphaxiv.org.
    """
    # Arxiv special handling
    if _ARXIV_PATTERN.search(url):
        paper_id = _extract_arxiv_paper_id(url)
        if paper_id:
            # Try arxiv HTML first
            html_url = f"https://arxiv.org/html/{paper_id}v1"
            content = _try_defuddle(html_url, timeout)
            if content and len(content) > 500:
                return content

            # Try alphaxiv full text
            content = _curl_get(
                f"https://www.alphaxiv.org/abs/{paper_id}.md",
                timeout=timeout,
            )
            if content and len(content) > 500:
                return content

            # Try alphaxiv overview
            content = _curl_get(
                f"https://www.alphaxiv.org/overview/{paper_id}.md",
                timeout=timeout,
            )
            if content and len(content) > 200 and "No intermediate report" not in content:
                return content

    # Standard defuddle extraction
    content = _try_defuddle(url, timeout)
    if content and len(content) > 100 and not _is_challenge_page(content):
        return content

    # Fallback: curl + liteparse (rotates user-agent on retry)
    content = _try_curl_extract(url, timeout, attempt=attempt)
    if content and len(content) > 200 and not _is_challenge_page(content):
        return content

    # Last resort: defuddle --json
    content = _try_defuddle_json(url, timeout)
    if content and len(content) > 200 and not _is_challenge_page(content):
        return content

    return ""


def _try_defuddle(url: str, timeout: int = 45) -> str:
    """Try defuddle parse --markdown URL -o tmpfile."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            tmpfile = f.name
        try:
            result = _run(
                ["defuddle", "parse", "--markdown", url, "-o", tmpfile],
                timeout=timeout,
            )
            if result.returncode == 0 and os.path.exists(tmpfile) and os.path.getsize(tmpfile) > 0:
                return Path(tmpfile).read_text(encoding="utf-8", errors="replace")
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _try_defuddle_json(url: str, timeout: int = 45) -> str:
    """Try defuddle parse --json URL."""
    try:
        result = _run(
            ["defuddle", "parse", "--json", url],
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            content = data.get("content", "")
            if content and len(content) > 200:
                return content
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return ""


def _try_curl_extract(url: str, timeout: int = 45, attempt: int = 0) -> str:
    """Try liteparse: curl download → liteparse parse --format text.

    Rotates user-agents on retry to bypass simple bot detection.
    """
    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
    ]
    ua = user_agents[attempt % len(user_agents)]
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            tmpfile = f.name
        try:
            dl = _run(
                ["curl", "-sL", "--max-time", str(timeout),
                 "-H", f"User-Agent: {ua}",
                 "-H", "Accept: text/html,application/xhtml+xml",
                 "-H", "Accept-Language: en-US,en;q=0.9",
                 url, "-o", tmpfile],
                timeout=timeout + 5,
            )
            if dl.returncode == 0 and os.path.exists(tmpfile) and os.path.getsize(tmpfile) > 0:
                parse = _run(
                    ["liteparse", "parse", "--format", "text", tmpfile],
                    timeout=timeout,
                )
                if parse.returncode == 0 and parse.stdout:
                    return parse.stdout[:5000]
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _try_archive_extract(url: str, timeout: int = 45) -> str:
    """Try archive.org Wayback Machine as last resort.

    Fetches the most recent archived version of the page.
    """
    from datetime import datetime
    archive_url = f"https://web.archive.org/web/{datetime.now().year}/{url}"
    try:
        content = _try_defuddle(archive_url, timeout)
        if content and len(content) > 200:
            # Strip archive.org header
            if "Wayback Machine" in content[:500]:
                for marker in ["<!DOCTYPE", "<html", "<article", "<main"]:
                    idx = content.find(marker)
                    if idx > 0:
                        content = content[idx:]
                        break
            return content
    except Exception as e:
        log.debug("Archive.org extract failed: %s", e)
    return ""


def _try_camoufox(url: str, timeout: int = 45) -> str:
    """Try Camoufox headless browser for JS-heavy / anti-bot pages.

    Final fallback in the extraction chain. Uses AsyncCamoufox to render
    the page and extract visible text.
    """
    try:
        from camoufox import AsyncCamoufox
    except ImportError:
        log.debug("Camoufox not installed, skipping browser fallback")
        return ""

    import asyncio

    async def _fetch() -> str:
        async with AsyncCamoufox(headless=True) as browser:
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            # Wait for JS-rendered content
            await asyncio.sleep(3)
            text = await page.evaluate("() => document.body.innerText")
            return text or ""

    try:
        text = asyncio.run(_fetch())
        if text and len(text.strip()) > 200:
            log.info("Camoufox extraction succeeded for %s (%d chars)", url, len(text))
            return text
    except Exception as e:
        log.debug("Camoufox extract failed: %s", e)
    return ""
