"""Web content extraction with Cloudflare detection and fallback chain.

Chain: defuddle -> curl+liteparse -> defuddle --json -> archive.org -> Camoufox.
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
    _is_archive_wrapper,
    _run,
    _url_to_title,
    _ARXIV_PATTERN,
    _extract_arxiv_paper_id,
    _is_challenge_page,
    _validate_url,
    extract_title,
    validate_extraction,
)

log = logging.getLogger(__name__)


def extract_web(url: str, cfg: Config, source_type: SourceType = SourceType.WEB) -> ExtractedSource:
    """Extract web content via defuddle CLI with curl fallback and retry logic.

    Chain: defuddle -> curl+liteparse -> defuddle --json -> archive.org -> Camoufox
    Detects and retries on Cloudflare challenge pages.
    """
    from pipeline.extractors._shared import _extract_html_title, _is_cloudflare_html
    from pipeline.extractors._shared import _is_ui_noise_title as _garbage_title
    from pipeline.extractors._shared import _should_use_camoufox_first

    timeout = cfg.extract_timeout
    max_retries = cfg.max_retries

    # -- Step 0: JS-rendered domains -> Camoufox FIRST ----------------------
    content = ""
    camoufox_title = ""
    if _should_use_camoufox_first(url):
        log.info("JS-rendered site detected for %s; trying Camoufox first", url)
        cfx_text, cfx_title = _try_camoufox_with_title(url, timeout)
        if cfx_text and len(cfx_text.strip()) > 100:
            content = cfx_text
            if cfx_title and not _garbage_title(cfx_title):
                camoufox_title = cfx_title

    # -- Step 0b: Fetch title if Camoufox did not succeed ---------------------
    fallback_title = ""
    if not content:
        try:
            quick_html = _curl_get(url, timeout=min(timeout, 10))
            if quick_html and not _is_cloudflare_html(quick_html):
                fallback_title = _extract_html_title(quick_html, fallback="")
                if _garbage_title(fallback_title):
                    fallback_title = ""
        except Exception:
            pass
        if not fallback_title:
            fallback_title = _url_to_title(url)
    else:
        fallback_title = camoufox_title or _url_to_title(url)

    # ── Step 1: Main extraction chain ────────────────────────────────────────────
    if not content:
        content = ""
        camoufox_title = ""  # populated if Camoufox saves us
        for attempt in range(max_retries):
            content = _extract_web_content(url, timeout, attempt=attempt)

            if not content:
                continue
            if _is_challenge_page(content):
                content = ""
                continue
            if len(content.strip()) < 20:
                content = ""
                continue
            break

    # ── Step 2: archive.org fallback ─────────────────────────────────────────────
    if not content or _is_challenge_page(content):
        archive_content = _try_archive_extract(url, timeout)
        if archive_content:
            archive_title = extract_title(archive_content, fallback_title="")
            if archive_content and not _is_archive_wrapper(archive_content):
                content = archive_content
                if archive_title and not _garbage_title(archive_title):
                    fallback_title = archive_title
            else:
                # archive.org wrapper — try Camoufox
                log.info("Archive.org wrapper detected, trying Camoufox for %s", url)
                cfx_text, cfx_title = _try_camoufox_with_title(url, timeout)
                if cfx_text:
                    content = cfx_text
                    if cfx_title and not _garbage_title(cfx_title):
                        camoufox_title = cfx_title

    # ── Step 3: Final Camoufox fallback (if content still empty) ────────────────
    if not content or _is_challenge_page(content):
        cfx_text, cfx_title = _try_camoufox_with_title(url, timeout)
        if cfx_text:
            content = cfx_text
            if cfx_title and not _garbage_title(cfx_title):
                camoufox_title = cfx_title

    # ── Step 4: Reject raw HTML garbage ──────────────────────────────────────────
    if content and content.strip().startswith(("<!DOCTYPE", "<!doctype html", "<html", "<HTML")):
        log.warning("All extractors returned raw HTML for %s, rejecting", url)
        content = ""

    if not content:
        log.warning("All web extraction methods failed for %s", url)
        content = f"URL: {url}\n\nNote: Content extraction failed (all methods exhausted)."
    else:
        # Run quality validation
        is_valid, reason = validate_extraction(content)
        if not is_valid:
            log.warning("Final content validation failed for %s: %s", url, reason)
            content = f"URL: {url}\n\nNote: Content extraction failed ({reason})."

    # ── Title extraction ─────────────────────────────────────────────────────────
    # Pipeline: Camoufox title → HTML title → content-based (H1/H2) → URL slug
    first_body_line = content.strip().split("\n")[0][:80].lower() if content else ""
    _BODY_NOISE = re.compile(
        r"^\s*(?:press\s+enter\s+or\s+click\s+to\s+view|get\s+(?:the\s+)?app|sign\s+up|"
        r"sign\s+in|loading\s+more|please\s+wait)"
    )

    title = ""

    # 1. Camoufox title (highest fidelity)
    if camoufox_title and not _garbage_title(camoufox_title):
        title = camoufox_title

    # 2. Content-based: H1 heading or first good line (even if starts with ##)
    if not title and not _BODY_NOISE.search(first_body_line):
        from pipeline.extractors._shared import extract_title as _extract_title
        content_title = _extract_title(content, fallback_title="")
        if content_title:
            is_error_msg = any(m in content_title.lower() for m in (
                "request could not be satisfied", "403 error", "access denied",
                "error 1020", "attention required", "blocked", "checking your browser",
            ))
            # Reject only if it looks like a body sentence (no title-like brevity/structure)
            looks_like_body = (
                len(content_title) > 120
                or content_title[0].islower()
                or content_title.startswith(("'", '"', "“"))
                or content_title.endswith((".",))  # period = sentence
            )
            if not is_error_msg and not looks_like_body and not _garbage_title(content_title):
                title = content_title

    # 3. Final fallback pipeline
    if not title or _garbage_title(title):
        if fallback_title and not _garbage_title(fallback_title):
            title = fallback_title
        else:
            title = _url_to_title(url)

    # 4. Clean up: strip site suffixes like "| Site Name" from HTML titles
    title = re.sub(r"\s*[|]\s*(?:Home\s+-\s+)?[^|]{3,80}$", "", title).strip()

    return ExtractedSource(
        url=url,
        title=title or fallback_title or url,
        content=content,
        type=source_type,
    )



def _extract_web_content(url: str, timeout: int = 45, attempt: int = 0) -> str:
    """Extract web content. Tries defuddle -> curl fallback.

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


def _strip_leading_teasers(content: str) -> str:
    """Strip Medium/Substack/Substack nav teasers from the top of markdown.

    These are injected at the top of the article by lazy-load scripts and
    defuddle sometimes grabs them as the first content line.
    """
    if not content:
        return content
    lines = content.split("\n")
    result = []
    skipped = 0
    for line in lines:
        raw = line.strip()
        # skip Medium/Substack "Join the conversation" teasers
        if re.match(
            r"^\*?Have\s+thoughts\s+on\s+this\s+topic\?.*Join\s+.*(?:conversation|discussion).*\*?",
            raw, re.IGNORECASE,
        ):
            skipped += 1
            continue
        # skip "More From:" nav markers (Oxford Law, Medium, etc.)
        if raw.lower() in ("more from:", "more from"):
            skipped += 1
            continue
        # skip bare author-link lines like "[ Terence Cassar ](url)"
        if re.match(r"^\[\s*[^\]]+\s*\]\s*\([^)]+\)$", raw):
            skipped += 1
            continue
        # skip empty lines and horizontal rules after a teaser was found
        if skipped >= 1 and (not raw or raw == "---"):
            continue
        # once we hit real content, keep everything after this point
        result.append(line)
    return "\n".join(result).strip()


def _try_defuddle_direct(url: str, timeout: int = 45) -> str:
    """Try defuddle directly against a URL (uses defuddle's own HTTP fetch).

    This works for sites like X/Twitter where defuddle's internal fetch
    handles cookies/session better than a bare curl download.
    """
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as out:
            tmpfile = out.name
        try:
            result = _run(
                ["defuddle", "parse", "--markdown", url, "-o", tmpfile],
                timeout=timeout,
            )
            if result.returncode == 0 and os.path.exists(tmpfile) and os.path.getsize(tmpfile) > 0:
                text = Path(tmpfile).read_text(encoding="utf-8", errors="replace")
                if not _is_challenge_page(text) and len(text.strip()) > 50:
                    return _strip_leading_teasers(text)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _try_defuddle(url: str, timeout: int = 45) -> str:
    """Try defuddle against safely downloaded HTML, not a live URL."""
    # First: try defuddle direct (better for X/Twitter and other auth walls)
    direct = _try_defuddle_direct(url, timeout)
    if direct:
        return direct
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as src, \
             tempfile.NamedTemporaryFile(suffix=".md", delete=False) as out:
            srcfile = src.name
            tmpfile = out.name
        try:
            dl = _run(
                ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https",
                 "--max-time", str(timeout), url, "-o", srcfile],
                timeout=timeout + 5,
            )
            if dl.returncode != 0 or not os.path.exists(srcfile) or os.path.getsize(srcfile) == 0:
                return ""
            result = _run(
                ["defuddle", "parse", "--markdown", srcfile, "-o", tmpfile],
                timeout=timeout,
            )
            if result.returncode == 0 and os.path.exists(tmpfile) and os.path.getsize(tmpfile) > 0:
                text = Path(tmpfile).read_text(encoding="utf-8", errors="replace")
                return _strip_leading_teasers(text)
        finally:
            for path in (srcfile, tmpfile):
                if os.path.exists(path):
                    os.unlink(path)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def _try_defuddle_json(url: str, timeout: int = 45) -> str:
    """Try defuddle JSON output against safely downloaded HTML."""
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            srcfile = f.name
        try:
            dl = _run(
                ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https",
                 "--max-time", str(timeout), url, "-o", srcfile],
                timeout=timeout + 5,
            )
            if dl.returncode != 0 or not os.path.exists(srcfile) or os.path.getsize(srcfile) == 0:
                return ""
            result = _run(
                ["defuddle", "parse", "--json", srcfile],
                timeout=timeout,
            )
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                content = data.get("content", "")
                if content and len(content) > 200:
                    return content
        finally:
            if os.path.exists(srcfile):
                os.unlink(srcfile)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return ""


def _try_curl_extract(url: str, timeout: int = 45, attempt: int = 0) -> str:
    """Try liteparse: curl download -> liteparse parse --format text.

    Rotates user-agents on retry to bypass simple bot detection.
    """
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
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
                ["curl", "-s", "--max-redirs", "0", "--proto", "=http,https", "--max-time", str(timeout),
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
                    result = parse.stdout
                    # liteparse returns raw HTML -> detect and reject it
                    if result.strip().startswith(("<!DOCTYPE", "<!doctype html", "<html", "<HTML")):
                        log.warning("liteparse returned raw HTML for %s, rejecting", url)
                        return ""
                    return result[:5000]
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
    if not _validate_url(url):
        log.warning("Blocked potentially unsafe URL: %s", url[:80])
        return ""
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
    except (subprocess.SubprocessError, ConnectionError, TimeoutError, OSError) as e:
        log.debug("Archive.org extract failed: %s", e)
    return ""


def _try_camoufox_with_title(url: str, timeout: int = 45) -> tuple[str, str]:
    """Try Camoufox headless browser and return (text, document.title).

    Used when we need BOTH content AND a real title.
    """
    if not _validate_url(url):
        return "", ""
    try:
        from camoufox import AsyncCamoufox
    except ImportError:
        return "", ""

    import asyncio

    async def _fetch() -> tuple[str, str]:
        async with AsyncCamoufox(headless=True) as browser:
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                await asyncio.sleep(2)
                text = await page.evaluate("() => document.body ? document.body.innerText : ''")
                title = await page.evaluate("() => document.title")
                return (text or "").strip(), (title or "").strip()[:120]
            except Exception:
                return "", ""

    try:
        return asyncio.run(_fetch())
    except (ConnectionError, TimeoutError, OSError, RuntimeError):
        return "", ""


def _try_camoufox(url: str, timeout: int = 45) -> str:
    """Try Camoufox headless browser for JS-heavy / anti-bot pages.

    Final fallback in the extraction chain. Uses AsyncCamoufox to render
    the page and extract visible text.
    """
    text, _title = _try_camoufox_with_title(url, timeout)
    return text

