"""Synthesis cache — persists per-source SourceSynthesis as JSON.

The cache lives at ``.llmwiki/cache/<source_filename>.json``.  Each file
contains the full SourceSynthesis for one source, serialised via
``source_synthesis_to_dict`` and deserialised via
``source_synthesis_from_dict``.

This is what makes incremental builds correct: unchanged sources reuse
their cached synthesis, so the rendered corpus is always complete.
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    SourceSynthesis,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.render.obsidian import atomic_write, safe_read_file

logger = logging.getLogger("obswiki.core.cache")

__all__ = [
    "synthesis_cache_dir",
    "synthesis_cache_path",
    "save_synthesis",
    "load_synthesis",
    "load_all_cached_syntheses",
    "delete_cached_synthesis",
]


def synthesis_cache_dir(cache_root: Path) -> Path:
    """Return the synthesis cache directory under ``cache_root``.

    ``cache_root`` is typically ``config.llmwiki_dir``.
    """
    return cache_root / "cache"


def synthesis_cache_path(cache_root: Path, source_file: str) -> Path:
    """Return the cache JSON path for a given source filename."""
    return synthesis_cache_dir(cache_root) / f"{source_file}.json"


def save_synthesis(
    synthesis: SourceSynthesis,
    cache_root: Path,
    source_file: str,
) -> None:
    """Persist a SourceSynthesis to the cache, keyed by source filename."""
    synthesis.source_file = source_file
    data = source_synthesis_to_dict(synthesis)
    path = synthesis_cache_path(cache_root, source_file)
    atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))


def load_synthesis(cache_root: Path, source_file: str) -> SourceSynthesis | None:
    """Load a cached SourceSynthesis for a source filename.

    Returns ``None`` if the cache file doesn't exist or is corrupt.
    """
    path = synthesis_cache_path(cache_root, source_file)
    raw = safe_read_file(path)
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt synthesis cache for '%s' — will recompile.", source_file)
        return None
    synth = source_synthesis_from_dict(data)
    synth.source_file = source_file
    return synth


def load_all_cached_syntheses(cache_root: Path) -> dict[str, SourceSynthesis]:
    """Load all cached syntheses from ``cache_root/cache/``.

    Returns a dict mapping source filename → SourceSynthesis.
    Corrupt entries are logged and skipped.
    """
    cache_dir = synthesis_cache_dir(cache_root)
    if not cache_dir.is_dir():
        return {}

    result: dict[str, SourceSynthesis] = {}
    for f in sorted(cache_dir.glob("*.json")):
        source_file = f.stem  # e.g. "article.md.json" → "article.md"
        raw = safe_read_file(f)
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt synthesis cache for '%s' — skipping.", source_file)
            continue
        synth = source_synthesis_from_dict(data)
        synth.source_file = source_file
        result[source_file] = synth
    return result


def delete_cached_synthesis(cache_root: Path, source_file: str) -> None:
    """Remove a cached synthesis file (called when a source is deleted)."""
    path = synthesis_cache_path(cache_root, source_file)
    with suppress(OSError):
        path.unlink(missing_ok=True)
