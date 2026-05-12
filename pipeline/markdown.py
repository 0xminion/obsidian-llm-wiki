"""Markdown parsing and manipulation helpers.

Ported from llm-wiki-compiler/src/utils/markdown.ts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# ── Slugify ────────────────────────────────────────────────────────────


def slugify(title: str) -> str:
    """Convert a title to a filename-safe slug.

    Unicode-aware: keeps letters/numbers from any script, strips punctuation.
    Port of the TS slugify() using \\p{L}\\p{N}.
    """
    # Remove apostrophes/smart quotes
    cleaned = title.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    # Keep Unicode letters, numbers, spaces, hyphens
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    # Replace whitespace runs with single hyphen
    cleaned = re.sub(r"\s+", "-", cleaned)
    # Collapse hyphen runs
    cleaned = re.sub(r"-+", "-", cleaned)
    # Trim leading/trailing hyphens
    cleaned = cleaned.strip("-")
    return cleaned.lower()


# ── Frontmatter ────────────────────────────────────────────────────────


def build_frontmatter(fields: dict[str, Any]) -> str:
    """Build a YAML frontmatter block from key-value pairs."""
    dumped = yaml.dump(fields, width=float("inf"), allow_unicode=True,
                       default_flow_style=False).rstrip()
    return f"---\n{dumped}\n---"


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown. Returns (meta, body).

    Uses partition() to avoid regex issues with --- inside code blocks.
    """
    if not content.startswith("---\n"):
        return {}, content

    _prefix, sep, rest = content.partition("---\n")
    if not sep:
        return {}, content

    remaining, sep2, body = rest.partition("\n---")
    if not sep2:
        # Single --- line, no closing ---
        return {}, content

    try:
        meta = yaml.safe_load(remaining)
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}

    # Strip leading newline after frontmatter closing ---
    body = body.removeprefix("\n")

    return meta or {}, body


# ── File I/O ───────────────────────────────────────────────────────────


def atomic_write(file_path: str | Path, content: str) -> None:
    """Atomically write a file: write to .tmp, then rename."""
    fp = Path(file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = fp.with_suffix(fp.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.rename(fp)


def safe_read_file(file_path: str | Path) -> str:
    """Read a file, returning empty string if it doesn't exist."""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def validate_wiki_page(content: str) -> bool:
    """Check that a wiki page has non-empty content and valid frontmatter."""
    if not content or not content.strip():
        return False
    meta, body = parse_frontmatter(content)
    if not meta.get("title"):
        return False
    return body.strip()


# ── Citations ──────────────────────────────────────────────────────────

# Matches ^[...] markers
_CITATION_PATTERN = re.compile(r"\^\[([^\]]+)\]")

# Matches file.md, file.md:1-3, file.md#L1-L3
_SPAN_PATTERN = re.compile(
    r"^(?P<file>[^:#]+)"
    r"(?:(?::(?P<colon_start>\d+)(?:-(?P<colon_end>\d+))?)"
    r"|(?:#L(?P<hash_start>\d+)(?:-L(?P<hash_end>\d+))?))?$"
)


def extract_citations(body: str) -> list[str]:
    """Extract unique source filenames from ^[filename] citations."""
    from pipeline.models import SourceSpan

    spans: list[SourceSpan] = []
    for match in _CITATION_PATTERN.finditer(body):
        raw = match.group(1)
        for part in raw.split(","):
            trimmed = part.strip()
            if not trimmed:
                continue
            parsed = _parse_span_entry(trimmed)
            if parsed:
                spans.append(parsed)

    seen: set[str] = set()
    result: list[str] = []
    for s in spans:
        if s.file and s.file not in seen:
            seen.add(s.file)
            result.append(s.file)
    return result


def extract_claim_citations(body: str) -> list[ClaimCitation]:  # noqa: F821
    """Extract claim-level citations from markdown body."""
    from pipeline.models import ClaimCitation

    citations: list[ClaimCitation] = []
    for match in _CITATION_PATTERN.finditer(body):
        raw = match.group(1)
        spans = _parse_citation_entries(raw)
        if spans:
            citations.append(ClaimCitation(raw=raw, spans=spans))
    return citations


def _parse_citation_entries(inner: str) -> list[SourceSpan]:  # noqa: F821
    """Parse the inside of ^[...] into one or more SourceSpan entries."""
    from pipeline.models import SourceSpan

    spans: list[SourceSpan] = []
    for part in inner.split(","):
        trimmed = part.strip()
        if not trimmed:
            continue
        span = _parse_span_entry(trimmed)
        if span is not None:
            spans.append(span)
    return spans


def _parse_span_entry(entry: str) -> SourceSpan | None:  # noqa: F821
    """Parse a single citation entry: file.md / file.md:1-3 / file.md#L1-L3."""
    from pipeline.models import SourceSpan

    match = _SPAN_PATTERN.match(entry)
    if not match:
        return SourceSpan(file=entry)

    file = match.group("file")
    col_start = match.group("colon_start")
    col_end = match.group("colon_end")
    hash_start = match.group("hash_start")
    hash_end = match.group("hash_end")

    start = col_start or hash_start
    end = col_end or hash_end

    if start is None:
        return SourceSpan(file=file)

    start_line = int(start)
    end_line = int(end) if end else start_line

    if start_line < 1 or end_line < start_line:
        return None

    return SourceSpan(file=file, lines=(start_line, end_line))


def is_malformed_citation_entry(entry: str) -> bool:
    """Detect malformed citation entries (bad line ranges, broken format)."""
    trimmed = entry.strip()
    if not trimmed:
        return True
    if ":" not in trimmed and "#" not in trimmed:
        return False
    match = _SPAN_PATTERN.match(trimmed)
    if not match:
        return True
    col_start = match.group("colon_start")
    col_end = match.group("colon_end")
    hash_start = match.group("hash_start")
    hash_end = match.group("hash_end")
    start = col_start or hash_start
    if start is None:
        return False
    end = col_end or hash_end
    start_line = int(start)
    end_line = int(end) if end else start_line
    return start_line < 1 or end_line < start_line
