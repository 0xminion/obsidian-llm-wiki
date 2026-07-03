"""Wiki state persistence for incremental compilation.

Ported from obsidian-llm-wiki/src/utils/state.ts and src/compiler/source-state.ts.

Reads/writes .llmwiki/state.json for source hash tracking, concept
ownership, and change detection across compilation runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.okf_markdown import safe_read_file
from pipeline.okf_models import SourceState, WikiState


def read_state(state_file: str | Path) -> WikiState:
    """Read the persisted wiki state from disk."""
    sf = Path(state_file)
    raw = safe_read_file(sf)
    if not raw:
        return WikiState()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return WikiState()

    sources: dict[str, SourceState] = {}
    for filename, entry in data.get("sources", {}).items():
        sources[filename] = SourceState(
            hash=entry.get("hash", ""),
            concepts=entry.get("concepts", []),
            compiled_at=entry.get("compiledAt") or entry.get("compiled_at"),
        )

    return WikiState(sources=sources)


def write_state(state_file: str | Path, state: WikiState) -> None:
    """Persist wiki state to disk atomically."""
    from pipeline.okf_markdown import atomic_write

    data: dict[str, object] = {
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
    state.sources[filename] = SourceState(
        hash=file_hash,
        concepts=concepts,
        compiled_at=compiled_at,
    )


def remove_source_state(state: WikiState, filename: str) -> None:
    """Remove a source entry from the state."""
    state.sources.pop(filename, None)
