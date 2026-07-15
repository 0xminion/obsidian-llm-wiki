"""State persistence + change detection (slim port).

Reads/writes ``.llmwiki/state.json`` for incremental compilation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from obsidian_llm_wiki.core.models import SourceChange, SourceStatus, WikiState
from obsidian_llm_wiki.render.obsidian import atomic_write, safe_read_file

__all__ = [
    "read_state",
    "write_state",
    "update_source_state",
    "remove_source_state",
    "hash_file",
    "hash_content",
    "detect_changes",
]


def hash_file(file_path: str | Path) -> str:
    """SHA-256 hash of a file (UTF-8 or raw bytes)."""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    except UnicodeDecodeError:
        raw = Path(file_path).read_bytes()
        return hashlib.sha256(raw).hexdigest()


def hash_content(content: str) -> str:
    """SHA-256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def detect_changes(sources_dir: str | Path, prev_state: WikiState) -> list[SourceChange]:
    """Scan sources directory and compare hashes against previous state."""
    sp = Path(sources_dir)
    changes: list[SourceChange] = []
    current_files: set[str] = set()

    if sp.exists():
        for f in sp.iterdir():
            if f.suffix == ".md" and f.is_file():
                current_files.add(f.name)
                status = _classify(f, prev_state)
                changes.append(SourceChange(file=f.name, status=status))

    for filename in prev_state.sources:
        if filename not in current_files:
            changes.append(SourceChange(file=filename, status=SourceStatus.DELETED))

    return changes


def _classify(file_path: Path, prev_state: WikiState) -> SourceStatus:
    """Classify a source file as new, changed, or unchanged."""
    file_hash = hash_file(file_path)
    prev = prev_state.sources.get(file_path.name)
    if not prev:
        return SourceStatus.NEW
    if prev.hash != file_hash:
        return SourceStatus.CHANGED
    return SourceStatus.UNCHANGED


def read_state(state_file: str | Path) -> WikiState:
    """Read persisted wiki state from disk."""
    from obsidian_llm_wiki.core.models import SourceState

    raw = safe_read_file(state_file)
    if not raw:
        return WikiState()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return WikiState()

    if not isinstance(data, dict):
        return WikiState()
    raw_sources = data.get("sources", {})
    if not isinstance(raw_sources, dict):
        return WikiState()
    sources: dict[str, SourceState] = {}
    for filename, entry in raw_sources.items():
        if not isinstance(entry, dict):
            continue
        sources[filename] = SourceState(
            hash=entry.get("hash", ""),
            concepts=entry.get("concepts", []),
            compiled_at=entry.get("compiled_at") or entry.get("compiledAt"),
        )
    return WikiState(sources=sources)


def write_state(state_file: str | Path, state: WikiState) -> None:
    """Persist wiki state to disk atomically."""
    data = {
        "sources": {
            filename: {
                "hash": entry.hash,
                "concepts": entry.concepts,
                "compiled_at": entry.compiled_at,
            }
            for filename, entry in state.sources.items()
        }
    }
    atomic_write(Path(state_file), json.dumps(data, indent=2, ensure_ascii=False))


def update_source_state(
    state: WikiState, filename: str, file_hash: str, concepts: list[str], compiled_at: str
) -> None:
    """Update or create a source entry in the state."""
    from obsidian_llm_wiki.core.models import SourceState

    state.sources[filename] = SourceState(
        hash=file_hash, concepts=concepts, compiled_at=compiled_at
    )


def remove_source_state(state: WikiState, filename: str) -> None:
    """Remove a source entry from the state."""
    state.sources.pop(filename, None)
