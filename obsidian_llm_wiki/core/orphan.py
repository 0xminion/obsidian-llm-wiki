"""Orphan management — marks exclusively-owned concepts when sources are deleted.

When a source file is removed, concepts that were *only* owned by that source
get an ``orphaned: true`` frontmatter flag.  Concepts shared with other live
sources are preserved.

This is a clean reimplementation of the legacy ``pipeline/orphan.py`` logic,
using the new package's models and render helpers — no cross-package imports.
"""

from __future__ import annotations

import logging
from pathlib import Path

from obsidian_llm_wiki.core.models import WikiState
from obsidian_llm_wiki.render.obsidian import (
    atomic_write,
    build_frontmatter,
    parse_frontmatter,
    safe_read_file,
)

logger = logging.getLogger("obswiki.core.orphan")

__all__ = [
    "mark_orphaned_concepts",
    "orphan_page",
    "find_exclusively_owned_concepts",
]


def find_exclusively_owned_concepts(
    deleted_source_file: str,
    state: WikiState,
) -> list[str]:
    """Return concept slugs owned *only* by ``deleted_source_file``.

    A concept is exclusively owned if no other source in ``state`` claims it.
    """
    src_state = state.sources.get(deleted_source_file)
    if not src_state:
        return []

    owned = set(src_state.concepts)
    if not owned:
        return []

    # Collect concepts owned by other (still-live) sources.
    shared: set[str] = set()
    for filename, other_state in state.sources.items():
        if filename == deleted_source_file:
            continue
        shared.update(other_state.concepts)

    return sorted(owned - shared)


def mark_orphaned_concepts(
    concepts_dir: Path,
    deleted_source_file: str,
    state: WikiState,
) -> list[str]:
    """Mark exclusively-owned concepts as orphaned in their frontmatter.

    Args:
        concepts_dir: Path to the ``concepts/`` directory.
        deleted_source_file: Filename of the deleted source.
        state: Current wiki state (must still contain the deleted source's entry).

    Returns:
        List of concept slugs that were marked orphaned.
    """
    exclusive = find_exclusively_owned_concepts(deleted_source_file, state)
    if not exclusive:
        return []

    orphaned: list[str] = []
    reason = f"Source deleted: {deleted_source_file}"
    for slug in exclusive:
        if orphan_page(concepts_dir, slug, reason):
            orphaned.append(slug)

    if orphaned:
        logger.info(
            "Orphaned %d concept(s) from deleted source '%s': %s",
            len(orphaned), deleted_source_file, ", ".join(orphaned),
        )

    return orphaned


def orphan_page(concepts_dir: Path, slug: str, reason: str) -> bool:
    """Add ``orphaned: true`` to a concept page's frontmatter.

    Returns ``True`` if the page was updated, ``False`` if it was already
    orphaned or the file doesn't exist.
    """
    page_path = concepts_dir / f"{slug}.md"
    if not page_path.exists():
        return False

    raw = safe_read_file(page_path)
    meta, body = parse_frontmatter(raw)

    if meta.get("orphaned"):
        return False  # Already marked

    meta["orphaned"] = True
    meta["orphaned_reason"] = reason

    new_content = build_frontmatter(meta) + "\n" + body
    atomic_write(page_path, new_content)
    return True
