"""Dependency management for incremental compilation.

Tracks concept ownership across sources, determines which sources are
affected by changes, and identifies frozen slugs (concepts that must not
be deleted because they are shared with live sources).

Ported from llm-wiki-compiler source-state logic.
"""

from __future__ import annotations

import logging

from pipeline.models import SourceChange, SourceStatus, WikiState

logger = logging.getLogger("llmwiki.deps")


# ── Shared concepts ─────────────────────────────────────────────────────


def find_shared_concepts(source_file: str, state: WikiState) -> set[str]:
    """Find concepts shared between *source_file* and any other source.

    Useful for dedup/orphan logic: when a source is removed, concepts that
    are shared with other sources must not be deleted.

    Args:
        source_file: The filename of the source to check (e.g. ``"article.md"``).
        state: The current wiki state.

    Returns:
        Set of concept names that appear in *source_file* AND at least one
        other source.
    """
    source_entry = state.sources.get(source_file)
    if source_entry is None:
        return set()

    source_concepts = set(source_entry.concepts)
    if not source_concepts:
        return set()

    shared: set[str] = set()

    for other_file, other_entry in state.sources.items():
        if other_file == source_file:
            continue
        other_concepts = set(other_entry.concepts)
        overlap = source_concepts & other_concepts
        shared.update(overlap)

    if shared:
        logger.debug(
            "Source '%s' shares %d concepts with other sources",
            source_file,
            len(shared),
        )

    return shared


# ── Affected sources ────────────────────────────────────────────────────


def find_affected_sources(
    state: WikiState,
    changes: list[SourceChange],
) -> list[str]:
    """Find sources whose concepts overlap with changed sources.

    When a source changes, other sources that share concepts with it may
    need recompilation (e.g., because concept pages need merging with
    updated information).

    Args:
        state: The current wiki state.
        changes: List of detected source changes (from ``hasher.detect_changes``).

    Returns:
        Sorted list of source filenames that are *indirectly* affected
        by changed sources (excluding the changed sources themselves).
    """
    # Collect concepts from all changed/new sources
    affected_concepts: set[str] = set()
    changed_files: set[str] = set()

    for change in changes:
        if change.status in (SourceStatus.NEW, SourceStatus.CHANGED):
            changed_files.add(change.file)
            entry = state.sources.get(change.file)
            if entry is not None:
                affected_concepts.update(entry.concepts)

    if not affected_concepts:
        return []

    # Find sources that share concepts with changed sources
    affected_sources: set[str] = set()

    for filename, entry in state.sources.items():
        if filename in changed_files:
            continue  # Already being processed directly
        source_concepts = set(entry.concepts)
        if source_concepts & affected_concepts:
            affected_sources.add(filename)

    if affected_sources:
        logger.info(
            "%d sources indirectly affected by %d changed sources",
            len(affected_sources),
            len(changed_files),
        )

    return sorted(affected_sources)


# ── Frozen slugs ────────────────────────────────────────────────────────


def find_frozen_slugs(
    state: WikiState,
    changes: list[SourceChange],
) -> set[str]:
    """Find concept slugs that must NOT be deleted.

    When a source is deleted, its concepts would normally be deleted too.
    However, if another *live* source also owns one of those concepts,
    that concept is *frozen* — it must be preserved because the live source
    still needs it.

    Args:
        state: The current wiki state.
        changes: List of detected source changes.

    Returns:
        Set of concept names that are owned by deleted sources but ALSO
        owned by at least one live (non-deleted) source.
    """
    # Partition sources into deleted and live
    deleted_files: set[str] = set()
    for change in changes:
        if change.status == SourceStatus.DELETED:
            deleted_files.add(change.file)

    if not deleted_files:
        return set()

    # Collect concepts from deleted sources
    deleted_concepts: set[str] = set()
    for df in deleted_files:
        entry = state.sources.get(df)
        if entry is not None:
            deleted_concepts.update(entry.concepts)

    if not deleted_concepts:
        return set()

    # Collect concepts from live (non-deleted) sources
    live_concepts: set[str] = set()
    for filename, entry in state.sources.items():
        if filename not in deleted_files:
            live_concepts.update(entry.concepts)

    # Frozen = overlap between deleted and live
    frozen = deleted_concepts & live_concepts

    if frozen:
        logger.info(
            "%d frozen slugs: concepts from deleted sources that are "
            "still owned by live sources",
            len(frozen),
        )
        logger.debug("Frozen slugs: %s", sorted(frozen))

    return frozen
