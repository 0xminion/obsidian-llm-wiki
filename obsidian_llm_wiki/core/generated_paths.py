"""Filesystem boundary for generated artifacts, backups and rollback.

Reject symlinks (even in-root aliases) rather than treating their targets as
owned files. These checks handle pre-existing layouts, not adversarial races:
path-based operations still have a check/use window. No multi-file power-loss
atomicity is claimed; render transactions provide in-process rollback only.
"""

from __future__ import annotations

import re
from pathlib import Path

_SHARD_BASENAME = re.compile(r"(?:nodes|edges)-(?:b[0-9]{3}-)?[0-9]{3,}\.json\Z")


def checked_generated_path(path: Path, root: Path | None = None) -> Path:
    """Validate lexical and resolved containment before filesystem access.

    Return the original spelling so callers retain their public path contracts.
    The root itself cannot be a symlink; ancestors above it are caller-owned.
    Without an explicitly authorized root, check every ancestor. Low-level
    metadata APIs must not silently authorize a symlink by resolving it first.
    """
    absolute_path = path.absolute()
    root = root if root is not None else Path(absolute_path.anchor)
    absolute_root = root.absolute()
    try:
        relative = absolute_path.relative_to(absolute_root)
        if ".." in relative.parts or ".." in absolute_root.parts:
            raise ValueError("parent traversal")
        cursor = absolute_root
        if cursor.is_symlink():
            raise ValueError("symlink root")
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("symlink component")
        absolute_path.resolve().relative_to(absolute_root.resolve())
    except (ValueError, RuntimeError, OSError) as exc:
        raise ValueError(f"Unsafe generated path {path} under {root}: {exc}") from exc
    return path


def checked_metadata_directory(directory: Path, root: Path | None = None) -> Path:
    """Preflight metadata before startup reads, SQLite opens or error metrics.

    Inspect names, not contents: every immediate metadata entry (including
    SQLite sidecars) and legacy cache JSON input must be safe before migration.
    Do not recursively walk graph shards/backups or the source corpus.
    """
    checked_generated_path(directory, root)
    if directory.is_dir():
        for path in directory.iterdir():
            checked_generated_path(path, directory)
        cache = directory / "cache"
        if cache.is_dir():
            for path in cache.glob("*.json"):
                checked_generated_path(path, directory)
    return directory


def is_graph_shard_basename(value: object) -> bool:
    """An untrusted manifest may name only the generated shard namespace."""
    return isinstance(value, str) and _SHARD_BASENAME.fullmatch(value) is not None
