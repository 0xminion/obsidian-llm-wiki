"""Clippings quality gate — determines whether a clipping needs extraction.

Scans 02-Clippings/ for markdown files and evaluates each:
  * Passes gate (body > threshold AND has title) → ready for synthesis
  * Fails gate → skip (too short or missing title)
"""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

__all__ = ["check_clipping", "collect_clippings"]


def check_clipping(path: Path, config: Config) -> tuple[bool, SourceDoc | None]:
    """Evaluate a single clipping file against the quality gate."""
    raw = safe_read_file(path)
    if not raw.strip():
        return False, None

    meta, body = parse_frontmatter(raw)
    title = _extract_title(meta, body)

    if not title:
        return False, None

    body_stripped = body.strip()
    if len(body_stripped) < config.clipping_min_body_chars:
        return False, None

    url = meta.get("source_url") or meta.get("url") or ""
    return True, SourceDoc(title=title, content=body_stripped, url=url)


def collect_clippings(config: Config) -> list[tuple[Path, SourceDoc]]:
    """Scan 02-Clippings/ and return all clippings that pass the quality gate."""
    clippings_dir = config.clippings_dir
    if not clippings_dir.exists():
        return []

    passed: list[tuple[Path, SourceDoc]] = []
    for f in sorted(clippings_dir.iterdir()):
        if f.suffix != ".md" or not f.is_file():
            continue
        ok, source = check_clipping(f, config)
        if ok and source is not None:
            passed.append((f, source))

    return passed


def _extract_title(meta: dict, body: str) -> str:
    """Extract a title from frontmatter or body."""
    import re

    title = (meta.get("title") or "").strip()
    if title:
        return title

    source = (meta.get("source") or "").strip()
    if source:
        return source

    source_url = (meta.get("source_url") or meta.get("url") or "").strip()
    if source_url:
        derived = _title_from_url(source_url)
        if derived:
            return derived

    h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:80].rstrip()

    return ""


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a URL path."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return ""

    segments = [
        s for s in path.split("/")
        if s and not s.endswith((".html", ".htm", ".php", ".asp"))
    ]
    if not segments:
        return ""

    last = segments[-1]
    for ext in (".html", ".htm", ".php", ".asp", ".aspx", ".md"):
        if last.endswith(ext):
            last = last[: -len(ext)]
            break
    return last.replace("-", " ").replace("_", " ").strip()
