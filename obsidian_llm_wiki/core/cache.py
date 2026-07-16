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
from hashlib import sha256
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    SourceSynthesis,
    _concept_from_dict,
    concept_note_to_dict,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.source_files import validate_source_filename
from obsidian_llm_wiki.render.obsidian import atomic_write, safe_read_file

logger = logging.getLogger("obswiki.core.cache")
_MAX_CACHE_FILENAME_BYTES = 200

__all__ = [
    "synthesis_cache_dir",
    "synthesis_cache_path",
    "save_synthesis",
    "load_synthesis",
    "load_all_cached_syntheses",
    "delete_cached_synthesis",
    "load_resynthesis_overlay",
    "save_resynthesis_overlay",
]


def synthesis_cache_dir(cache_root: Path) -> Path:
    """Return the synthesis cache directory under ``cache_root``.

    ``cache_root`` is typically ``config.llmwiki_dir``.
    """
    return cache_root / "cache"


def synthesis_cache_path(cache_root: Path, source_file: str) -> Path:
    """Return a filesystem-safe cache path for a source filename.

    Source filenames are user-visible and may legitimately approach ext4's
    255-byte component limit. Atomic writes append a temporary suffix, so
    using the source filename verbatim can fail only at cache persistence.
    Long names use a stable digest; the original filename remains in the
    serialized ``source_file`` field for reverse lookup.
    """
    filename = validate_source_filename(source_file)
    candidate = f"{filename}.json"
    if len(candidate.encode("utf-8")) > _MAX_CACHE_FILENAME_BYTES:
        candidate = f"sha256-{sha256(filename.encode('utf-8')).hexdigest()}.json"
    return synthesis_cache_dir(cache_root) / candidate


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
    if not isinstance(data, dict):
        logger.warning("Invalid synthesis cache for '%s' — will recompile.", source_file)
        return None
    try:
        synth = source_synthesis_from_dict(data)
    except (AttributeError, KeyError, TypeError, ValueError):
        logger.warning("Invalid synthesis cache for '%s' — will recompile.", source_file)
        return None
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
        fallback_source_file = f.stem  # e.g. "article.md.json" → "article.md"
        raw = safe_read_file(f)
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt synthesis cache for '%s' — skipping.", fallback_source_file)
            continue
        if not isinstance(data, dict):
            logger.warning("Invalid synthesis cache for '%s' — skipping.", fallback_source_file)
            continue
        try:
            synth = source_synthesis_from_dict(data)
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning("Invalid synthesis cache for '%s' — skipping.", fallback_source_file)
            continue
        source_file = synth.source_file or fallback_source_file
        synth.source_file = source_file
        result[source_file] = synth
    return result


def delete_cached_synthesis(cache_root: Path, source_file: str) -> None:
    """Remove a cached synthesis file (called when a source is deleted)."""
    path = synthesis_cache_path(cache_root, source_file)
    with suppress(OSError):
        path.unlink(missing_ok=True)


# ── Resynthesis overlay ─────────────────────────────────────────────────
#
# Incremental concept re-synthesis rewrites a concept's body coherently when a
# new source references it. The per-source caches still hold the pre-rewrite
# concepts (they were saved before resynthesis ran), so without a persistence
# layer the rewritten body would revert to the mechanically-merged version on
# the very next build. The overlay stores the rewritten concepts by slug and
# is re-applied to the merged bundle each run; an entry is invalidated when a
# freshly compiled source extracts the same slug (its resynthesis supersedes).


def _overlay_path(cache_root: Path) -> Path:
    return synthesis_cache_dir(cache_root) / "_resynthesis_overlay.json"


def load_resynthesis_overlay(cache_root: Path) -> dict[str, ConceptNote]:
    """Load persisted resynthesized concepts, keyed by slug."""
    raw = safe_read_file(_overlay_path(cache_root))
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt resynthesis overlay — ignoring.")
        return {}
    concepts = data.get("concepts", {})
    if not isinstance(concepts, dict):
        return {}
    overlay: dict[str, ConceptNote] = {}
    for slug, concept_dict in concepts.items():
        if isinstance(concept_dict, dict):
            overlay[slug] = _concept_from_dict(concept_dict)
    return overlay


def save_resynthesis_overlay(
    cache_root: Path,
    overlay: dict[str, ConceptNote],
) -> None:
    """Persist resynthesized concepts (full rewrite of the overlay file)."""
    data = {
        "concepts": {
            slug: concept_note_to_dict(concept)
            for slug, concept in overlay.items()
        },
    }
    atomic_write(
        _overlay_path(cache_root),
        json.dumps(data, indent=2, ensure_ascii=False),
    )
