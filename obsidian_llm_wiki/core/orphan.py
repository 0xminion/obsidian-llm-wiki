"""Orphan management — marks exclusively-owned concepts when sources are deleted.

When a source file is removed, concepts that were *only* owned by that source
get an ``orphaned: true`` frontmatter flag.  Concepts shared with other live
sources are preserved.

This is a clean reimplementation of the legacy ``pipeline/orphan.py`` logic,
using the new package's models and render helpers — no cross-package imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from obsidian_llm_wiki.core.generated_paths import checked_generated_path
from obsidian_llm_wiki.core.models import WikiState, _is_safe_slug
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
    if not _is_safe_slug(slug):
        logger.warning("Refusing unsafe orphan identity: %r", slug)
        return False
    page_path = concepts_dir / f"{slug}.md"
    bundle_dir = concepts_dir.parent
    try:
        checked_generated_path(page_path, bundle_dir)
    except ValueError:
        logger.warning("Refusing unsafe orphan path: %s", page_path)
        return False
    if not page_path.is_file():
        return False

    raw = safe_read_file(page_path)
    meta, body = parse_frontmatter(raw)

    if meta.get("orphaned"):
        return False  # Already marked

    meta["orphaned"] = True
    meta["orphaned_reason"] = reason

    new_content = build_frontmatter(meta) + "\n" + body
    from obsidian_llm_wiki.render import obsidian as renderer

    # Orphan metadata is a program mutation, not a user edit. Advance ownership
    # only if the exact previous bytes were already trusted and unreviewed.
    # Human/unknown notes keep their prior trust state, even though we annotate
    # their lifecycle metadata and preserve a recoverable backup.
    transaction = renderer._ACTIVE_RENDER_TRANSACTION.get()
    own_transaction = transaction is None
    token = None
    if own_transaction:
        transaction = renderer._RenderTransaction(bundle_dir, lazy=True)
        token = renderer._ACTIVE_RENDER_TRANSACTION.set(transaction)
    try:
        trusted = not renderer._page_is_protected(page_path, raw, bundle_dir)
        transaction.before_change(page_path)
        renderer._backup_generated_page(page_path, bundle_dir)
        checked_generated_path(page_path, bundle_dir)
        atomic_write(page_path, new_content)
        transaction.record_write(page_path, new_content)
        if trusted:
            renderer._remember_generated_page(page_path, new_content, bundle_dir)
            if own_transaction:
                renderer._write_generated_artifact(
                    bundle_dir / ".llmwiki" / "render-page-hashes.json",
                    json.dumps(transaction.page_hashes, sort_keys=True), bundle_dir,
                )
    except BaseException:
        if own_transaction:
            transaction.rollback()
        raise
    finally:
        if token is not None:
            renderer._ACTIVE_RENDER_TRANSACTION.reset(token)
    if own_transaction:
        transaction.commit()
    return True
