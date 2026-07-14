"""Validated source-file identities for vault and cache paths."""

from __future__ import annotations

from pathlib import Path

__all__ = ["source_file_path", "validate_source_filename"]


def validate_source_filename(value: object) -> str:
    """Return a safe source filename or reject traversal and ambiguous inputs."""
    if not isinstance(value, str) or not value or len(value) > 200:
        raise ValueError("Source filename must be a non-empty bounded string")
    if value != Path(value).name or "\\" in value or ":" in value or not value.endswith(".md"):
        raise ValueError("Source filename must be a basename ending in .md")
    if value.startswith(".") or any(ord(char) < 32 for char in value):
        raise ValueError("Source filename contains unsupported characters")
    return value


def source_file_path(directory: Path, filename: object) -> Path:
    """Resolve a validated source file path and prove it stays under ``directory``."""
    safe_name = validate_source_filename(filename)
    root = directory.resolve()
    candidate = (root / safe_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # defensive: validation should make this unreachable
        raise ValueError(f"Source path escapes {directory}") from exc
    return candidate
