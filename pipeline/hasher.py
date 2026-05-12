"""Source file hashing for change detection.

Ported from llm-wiki-compiler/src/compiler/hasher.ts.

Computes SHA-256 hashes of source files and compares them against
previously stored state to determine which files need recompilation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline.models import SourceChange, SourceStatus, WikiState


def hash_file(file_path: str | Path) -> str:
    """Read a file and compute its SHA-256 hash."""
    content = Path(file_path).read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_content(content: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_changes(sources_dir: str | Path, prev_state: WikiState) -> list[SourceChange]:
    """Scan the sources directory and compare hashes against previous state.

    Returns a list of SourceChange entries describing each file's status.
    """
    sp = Path(sources_dir)
    changes: list[SourceChange] = []

    # Current files
    current_files: set[str] = set()
    if sp.exists():
        for f in sp.iterdir():
            if f.suffix == ".md" and f.is_file():
                current_files.add(f.name)
                status = _classify_file(f, prev_state)
                changes.append(SourceChange(file=f.name, status=status))

    # Deleted files — present in state but missing from disk
    for filename in prev_state.sources:
        if filename not in current_files:
            changes.append(SourceChange(file=filename, status=SourceStatus.DELETED))

    return changes


def _classify_file(file_path: Path, prev_state: WikiState) -> SourceStatus:
    """Classify a single source file as new, changed, or unchanged."""
    file_hash = hash_file(file_path)
    prev = prev_state.sources.get(file_path.name)
    if not prev:
        return SourceStatus.NEW
    if prev.hash != file_hash:
        return SourceStatus.CHANGED
    return SourceStatus.UNCHANGED
