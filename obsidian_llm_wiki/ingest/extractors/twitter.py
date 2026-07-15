"""Twitter/X post and article extractor via defuddle.

Extraction strategy — X Articles and tweets MUST go through defuddle:

  1. **defuddle CLI** (local, ``npx defuddle parse <url> --md``) — primary.
     Same engine as defuddle.md but runs locally.  Proxy env vars are
     stripped because Node.js fetch() does not support SOCKS proxies.

  2. **defuddle.md** (hosted service at https://defuddle.md/<url>) — fallback.
     Renders JS-heavy X pages server-side.  May return JS stubs for
     auth-walled /article/ URLs.

  3. **trafilatura** via ``extract_web`` — last resort for non-JS pages.

**DO NOT route X URLs through VxTwitter, direct HTTP, or browser_navigate.**
These do NOT work for X Articles:

  - VxTwitter (api.vxtwitter.com) only handles ``/status/`` tweet IDs, not
    ``/article/`` URLs — returns 404.
  - Direct HTTP fetch of x.com returns a JS-rendered React shell with no
    article content (the page requires client-side rendering + authentication).
  - Browser navigation hits X's login wall — articles are gated behind an
    authenticated session.
  - Wayback Machine archives the JS shell, not the rendered article content.

**X Article auth wall (July 2026):** X started requiring authentication to
view ``/article/`` URLs.  defuddle.md's server-side renderer gets a
``"JavaScript is not available"`` stub from X instead of the article body.
When this happens, the extractor detects the stub and falls through to the
defuddle CLI, then to ``extract_web``.  If all three fail, the URL is logged
to the failed URLs ledger.

For ``/status/`` URLs (regular tweets), defuddle.md still works — X serves
tweet content server-side for unauthenticated requests.
"""

from __future__ import annotations

import logging

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.twitter")

__all__ = ["extract_twitter"]

# X/Twitter pages that carry no article content — detect and reject these
# so the extractor falls through to the next strategy instead of returning
# a content-free stub to the pipeline.
_STUB_MARKERS = (
    "JavaScript is not available",
    "Something went wrong",
    "cookie wall",
)


def _is_twitter_url(parsed, raw: str) -> bool:
    """Match Twitter/X URLs."""
    host = (parsed.hostname or "").lower()
    return host in ("x.com", "twitter.com", "www.x.com", "www.twitter.com")


@register_extractor(_is_twitter_url)
def extract_twitter(raw_url: str) -> SourceDoc:
    """Extract content from a Twitter/X post or article.

    Strategy (see module docstring for rationale):
      1. defuddle.md hosted service — renders JS-heavy X pages
      2. defuddle CLI (local) — same engine, local fallback
      3. trafilatura via extract_web — last resort

    DO NOT add VxTwitter, direct HTTP, or browser_navigate fallbacks —
    they do not work for X Articles (see module docstring).

    Raises:
        RuntimeError: If all extraction strategies fail.
    """
    errors: list[str] = []

    # ── Primary: Defuddle CLI (local) ─────────────────────────────────
    # The local defuddle CLI is the most reliable path — it renders JS-
    # heavy X pages with full article content.  Proxy env vars are
    # stripped (Node.js can't handle SOCKS proxies natively).
    try:
        source = _extract_via_defuddle(raw_url)
        if source:
            logger.info(
                "Defuddle CLI: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"defuddle_cli: {exc}")

    # ── Fallback: Defuddle.md (hosted service) ────────────────────────
    try:
        source = _extract_via_defuddle_md(raw_url)
        if source:
            logger.info(
                "Defuddle.md fallback: extracted %d chars for %s",
                len(source.content), raw_url,
            )
            return source
    except Exception as exc:
        errors.append(f"defuddle_md: {exc}")

    # ── Last resort: web extraction (trafilatura) ─────────────────────
    try:
        from obsidian_llm_wiki.ingest.web import extract_web
        return extract_web(raw_url)
    except Exception as exc:
        errors.append(f"web: {exc}")

    raise RuntimeError(
        f"Twitter extraction failed for {raw_url}: " + "; ".join(errors)
    )


def _extract_via_defuddle_md(url: str) -> SourceDoc | None:
    """Extract full content via defuddle.md web service.

    defuddle.md is a hosted version of defuddle that renders JS-heavy pages
    (including X Articles) and returns clean markdown with YAML frontmatter.
    URL format: https://defuddle.md/<original-url>
    """
    # Build defuddle.md URL
    # Strip https:// from the original URL
    stripped = url.replace("https://", "").replace("http://", "")
    defuddle_url = f"https://defuddle.md/{stripped}"

    with httpx.Client(
        **make_client_kwargs(timeout=DEFAULT_TIMEOUT, follow_redirects=False),
        headers={
            "User-Agent": BROWSER_HEADERS["User-Agent"],
            "Accept": "text/html",
        },
    ) as client:
        from obsidian_llm_wiki.ingest.url_safety import get_with_validated_redirects
        resp = get_with_validated_redirects(client, defuddle_url)

    if resp.status_code != 200:
        logger.debug("defuddle.md returned %d for %s", resp.status_code, url)
        return None

    text = resp.text.strip()
    if not text or len(text) < 100:
        return None

    # Detect X/Twitter JavaScript-disabled stubs and error pages —
    # defuddle.md renders the page shell but X requires JS for article
    # content, returning a stub like:
    #   ---
    #   title: "JavaScript is not available."
    #   site: "X (formerly Twitter)"
    #   ---
    # This passes the 100-char gate but carries no article content.
    # The module-level _STUB_MARKERS tuple is used for detection.
    if any(marker.lower() in text.lower() for marker in _STUB_MARKERS):
        logger.debug("defuddle.md returned JS stub/error for %s — skipping", url)
        return None

    # Parse frontmatter (defuddle.md returns YAML frontmatter + markdown body)
    title = ""
    content = text

    if text.startswith("---"):
        fm_end = text.find("---", 3)
        if fm_end > 0:
            fm_text = text[3:fm_end].strip()
            content = text[fm_end + 3:].strip()

            # Parse YAML frontmatter manually (avoid yaml dependency)
            for line in fm_text.split("\n"):
                if line.startswith("title:"):
                    title = line[6:].strip().strip('"').strip("'")
                elif line.startswith("author:"):
                    pass  # Author available but not needed in SourceDoc
                elif line.startswith("word_count:"):
                    pass  # Available but not needed

    if not title:
        # Try first # heading
        for line in content.split("\n", 5):
            if line.startswith("# "):
                title = line[2:].strip()
                break
    if not title:
        title = url

    # Strip cover image markdown from content start
    if content.startswith("!["):
        content = content.split("\n", 1)[-1].lstrip() if "\n" in content else content

    if not content or len(content.strip()) < 50:
        return None

    return SourceDoc(title=title, content=content.strip(), url=url)


def _extract_via_defuddle(url: str) -> SourceDoc | None:
    """Fallback: extract via defuddle CLI directly."""
    import os
    import shutil
    import subprocess

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return None

    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--md"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", url, "--md"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""
    # Strip SOCKS proxy env vars — Node.js fetch() does not support SOCKS
    # proxies natively and will throw "socket hang up" if HTTPS_PROXY is set
    # to a socks5h:// URL.  defuddle CLI needs direct or HTTP proxy access.
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "ALL_PROXY", "all_proxy"):
        env.pop(key, None)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, env=env,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None

        output = proc.stdout.strip()
        if len(output) < 50:
            return None

        # Extract title from first # heading
        lines = output.split("\n", 3)
        title = ""
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()

        # If no heading, try defuddle --json for metadata
        if not title:
            title = _defuddle_metadata_title(url)

        if not title:
            title = url

        # Strip image markdown from content
        content = output
        if content.startswith("!["):
            content = content.split("\n", 1)[-1].lstrip() if "\n" in content else content

        if len(content) < 50:
            return None

        return SourceDoc(title=title, content=content, url=url)

    except Exception:
        return None


def _defuddle_metadata_title(url: str) -> str:
    """Fetch page title via defuddle --json."""
    import json
    import os
    import shutil
    import subprocess

    defuddle_path = shutil.which("defuddle") or shutil.which("npx")
    if not defuddle_path:
        return ""

    if "defuddle" in defuddle_path:
        cmd = [defuddle_path, "parse", url, "--json"]
    else:
        cmd = [defuddle_path, "defuddle", "parse", url, "--json"]

    env = os.environ.copy()
    env["NODE_EXTRA_CA_CERTS"] = ""
    # Strip SOCKS proxy env vars — Node.js fetch() does not support SOCKS
    # proxies natively and will throw "socket hang up" if HTTPS_PROXY is set
    # to a socks5h:// URL.  defuddle CLI needs direct or HTTP proxy access.
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "ALL_PROXY", "all_proxy"):
        env.pop(key, None)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=30, env=env,
        )
        if proc.returncode != 0:
            return ""
        data = json.loads(proc.stdout)
        return (data.get("title") or "").strip()
    except Exception:
        return ""
