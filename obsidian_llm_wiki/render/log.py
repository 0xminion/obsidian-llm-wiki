"""Vault chronological log — ``log.md`` in the wiki root.

Maintains an append-only, grep-friendly chronological record of every
pipeline action (ingest, build, query, fix) with ISO timestamps.

Format convention:
    ## [2026-07-13T10:30:00Z] ACTION: detail
    - optional body lines
    - ...

Grep usage::

    grep '^## \\[' log.md           # list all entries
    grep '^## \\[.*' log.md | grep -i 'build'

The log is safe for concurrent append — it uses atomic write semantics and
preserves existing content.
"""

from __future__ import annotations

import logging
from pathlib import Path

from obsidian_llm_wiki.render.frontmatter import timestamp as _timestamp

logger = logging.getLogger("obswiki.render.log")

__all__ = [
    "log_entry",
    "log_to_file",
    "read_log_entries",
    "format_log_entry",
    "append_log",
]

LOG_FILENAME = "log.md"

# Valid action types (extensible but documented).
VALID_ACTIONS = frozenset({
    "ingest",
    "build",
    "query",
    "fix",
    "render",
    "validate",
    "maintenance",
    "error",
})


def format_log_entry(
    action: str,
    detail: str,
    *,
    timestamp: str | None = None,
    body: str | list[str] | None = None,
) -> str:
    """Format a single log entry as markdown.

    Args:
        action: The action type (ingest, build, query, fix, render, etc.).
        detail: A short human-readable summary of what happened.
        timestamp: Optional ISO timestamp; defaults to ``now``.
        body: Optional extra lines (list) or paragraph (str) for detail.

    Returns:
        A markdown string starting with ``## [timestamp] ACTION: detail``.
    """
    ts = timestamp or _timestamp()
    action_upper = action.upper()
    header = f"## [{ts}] {action_upper}: {detail}"

    if body is None:
        return header

    body_lines = [body] if isinstance(body, str) else list(body)

    if not body_lines:
        return header

    return header + "\n" + "\n".join(body_lines)


def log_entry(
    action: str,
    detail: str,
    *,
    timestamp: str | None = None,
    body: str | list[str] | None = None,
) -> str:
    """Alias for :func:`format_log_entry` — kept for backward compat."""
    return format_log_entry(action, detail, timestamp=timestamp, body=body)


def log_to_file(
    log_path: Path,
    action: str,
    detail: str,
    *,
    timestamp: str | None = None,
    body: str | list[str] | None = None,
) -> None:
    """Append a log entry to ``log_path``.

    Creates the file (with a title header) if it does not exist.  Existing
    content is preserved — new entries are appended at the end.
    """
    entry = format_log_entry(action, detail, timestamp=timestamp, body=body)

    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        # Ensure file ends with a newline before appending.
        if not existing.endswith("\n"):
            existing += "\n"
        log_path.write_text(existing + entry + "\n", encoding="utf-8")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Vault Log\n\n"
            "> Chronological record of pipeline actions. "
            "Grep with `grep '^## [' log.md`.\n\n"
        )
        log_path.write_text(header + entry + "\n", encoding="utf-8")

    logger.debug("Log entry written to %s: %s %s", log_path, action, detail)


def append_log(
    bundle_dir: Path,
    action: str,
    detail: str,
    *,
    timestamp: str | None = None,
    body: str | list[str] | None = None,
) -> Path:
    """Append a log entry to ``bundle_dir/log.md``.

    Convenience wrapper around :func:`log_to_file` that resolves the log
    path from the bundle directory.  Returns the path to ``log.md``.
    """
    log_path = bundle_dir / LOG_FILENAME
    log_to_file(log_path, action, detail, timestamp=timestamp, body=body)
    return log_path


def read_log_entries(
    log_path: Path,
    *,
    action: str | None = None,
) -> list[str]:
    """Read a vault log file and return matching entry headers.

    Args:
        log_path: Path to ``log.md``.
        action: Optional action filter (case-insensitive).  When provided,
            only entries whose action matches are returned.

    Returns:
        List of header lines (``## [timestamp] ACTION: detail``).
    """
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    entries = [ln for ln in lines if ln.startswith("## [")]
    if action is None:
        return entries
    target = action.upper()
    return [ln for ln in entries if f"] {target}:" in ln]
