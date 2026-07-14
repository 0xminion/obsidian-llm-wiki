"""YAML frontmatter, wikilinks, tags, and file I/O utilities.

Extracted from render/obsidian.py. This module is the single home for these
helpers — render/obsidian.py imports them from here and re-exports the names
for its existing callers.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "slugify",
    "make_wikilink",
    "sanitize_tag",
    "build_frontmatter",
    "parse_frontmatter",
    "extract_links",
    "extract_wikilinks",
    "safe_read_file",
    "atomic_write",
    "timestamp",
]


def slugify(text: str) -> str:
    """Convert arbitrary text to a filename-safe slug."""
    cleaned = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    slug = cleaned.strip("-").lower()
    return slug if slug else "untitled"


def make_wikilink(slug: str, alias: str | None = None) -> str:
    """Build an Obsidian wikilink ``[[slug]]`` or ``[[slug|alias]]``."""
    if alias and alias != slug:
        return f"[[{slug}|{alias}]]"
    return f"[[{slug}]]"


def sanitize_tag(tag: str) -> str:
    """Sanitize a single tag for Obsidian compatibility.

    Obsidian tags cannot contain spaces. Replace spaces with hyphens.
    Also strips leading/trailing whitespace and removes special chars
    that break YAML parsing.
    """
    tag = (tag or "").strip()
    tag = re.sub(r"\s+", "-", tag)
    tag = re.sub(r"[#\"'`,;()\[\]{}]", "", tag)
    return tag


def build_frontmatter(fm_dict: dict[str, Any]) -> str:
    """Serialize a dict to a ``---``-delimited YAML frontmatter block.

    Tags are sanitized: spaces → hyphens, special chars removed.
    """
    if "tags" in fm_dict and isinstance(fm_dict["tags"], list):
        fm_dict = dict(fm_dict)
        fm_dict["tags"] = [
            sanitize_tag(t) for t in fm_dict["tags"]
            if t and str(t).strip()
        ]

    dumped = yaml.dump(
        fm_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip()
    return f"---\n{dumped}\n---\n"


_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from ``raw``.  Returns (meta, body).

    Handles edge cases the old ``partition``-based approach missed:
      * Body starting immediately after closing ``---`` (no leading newline)
      * No trailing newline after the closing ``---``
      * A YAML scalar (non-dict) frontmatter block
    """
    if not raw.startswith("---\n"):
        return {}, raw
    match = _FM_RE.match(raw)
    if not match:
        return {}, raw
    yaml_block, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(meta, dict):
        meta = {}
    body = body.lstrip("\n")
    return meta, body


# Standard markdown link: [text](url). Excludes images ![alt](url).
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)]*)\)")

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def extract_links(body: str) -> list[tuple[str, str]]:
    """Extract standard markdown ``[text](url)`` links from ``body``.

    Returns a list of ``(text, url)`` tuples in document order. For Obsidian
    ``[[wikilinks]]`` use :func:`extract_wikilinks` — the two link syntaxes
    are deliberately separate functions.
    """
    return [(m.group(1), m.group(2)) for m in _LINK_RE.finditer(body)]


def extract_wikilinks(body: str) -> list[tuple[str, str]]:
    """Extract all ``[[wikilinks]]`` from a markdown body.

    Returns a list of (slug, alias) tuples. Alias is empty when no
    alias is present.
    """
    links: list[tuple[str, str]] = []
    for m in _WIKILINK_RE.finditer(body):
        slug = m.group(1).strip()
        alias = (m.group(2) or "").strip()
        links.append((slug, alias))
    return links


def safe_read_file(path: str | Path) -> str:
    """Read a file, returning empty string on error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def atomic_write(path: str | Path, content: str) -> None:
    """Write content atomically using a temp file + rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(p.parent), prefix=p.name + ".", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        # Persist the directory entry too: rename is atomic, but not durable
        # across a power loss until the containing directory is flushed.
        if hasattr(os, "O_DIRECTORY"):
            dir_fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        # BaseException so the temp file is cleaned up even on
        # KeyboardInterrupt/SystemExit.
        with suppress(OSError):
            os.unlink(tmp)
        raise


def timestamp() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Keep _sanitize_tag as an alias for backward compat
_sanitize_tag = sanitize_tag
