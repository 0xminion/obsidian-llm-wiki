"""Orphan management (deterministic, no LLM).

Ported from llm-wiki-compiler/src/compiler/orphan.ts.

Handles:
  - Marking concepts as orphaned when their exclusive owner source is deleted.
  - Preserving shared concepts (multiple sources contribute to same concept).
  - Cleanup: frozen slugs with no remaining owners.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.markdown import atomic_write, build_frontmatter, parse_frontmatter, safe_read_file
from pipeline.models import WikiState

# ── Public API ──────────────────────────────────────────────────────────


def mark_orphaned(root_dir: str | Path, source_file: str, state: WikiState) -> None:
    """When a source is deleted: orphan exclusively-owned concepts.

    Concepts shared across multiple sources are preserved (they are just
    no longer linked to the deleted source).

    Args:
        root_dir: Wiki root directory.
        source_file: Filename of the deleted source.
        state: Current WikiState (will be modified in-place).
    """
    root = Path(root_dir)
    concepts_dir = root / "concepts"

    if not concepts_dir.is_dir():
        return

    # Get concepts owned by this source
    source_state = state.sources.get(source_file)
    if not source_state:
        return

    owned_concepts = set(source_state.concepts)
    if not owned_concepts:
        return

    for slug in owned_concepts:
        # Check if any OTHER source also owns this concept
        other_owners = _find_other_owners(state, slug, source_file)
        if not other_owners:
            # Exclusive ownership — orphan the page
            orphan_page(root_dir, slug, f"Source deleted: {source_file}")
        # else: shared concept, keep active (just remove this source from state)


def orphan_page(root_dir: str | Path, slug: str, reason: str) -> None:
    """Add 'orphaned: true' to a concept page's frontmatter.

    Args:
        root_dir: Wiki root directory.
        slug: Concept page slug.
        reason: Human-readable reason for orphaning (stored in frontmatter).
    """
    root = Path(root_dir)
    page_path = root / "concepts" / f"{slug}.md"

    if not page_path.exists():
        return

    raw = safe_read_file(page_path)
    meta, body = parse_frontmatter(raw)

    if meta.get("orphaned"):
        return  # Already orphaned

    meta["orphaned"] = True
    meta["orphaned_reason"] = reason

    new_content = build_frontmatter(meta) + "\n" + body
    atomic_write(page_path, new_content)


def orphan_unowned_frozen_pages(root_dir: str | Path, frozen_slugs: list[str]) -> None:
    """Cleanup: frozen slugs with no remaining owners → orphan.

    A "frozen" slug is one that has been flagged for review because its
    owning source was deleted but it might be shared. If after review no
    other source claims it, orphan it.

    Args:
        root_dir: Wiki root directory.
        frozen_slugs: List of slugs to check for orphaning.
    """
    root = Path(root_dir)
    concepts_dir = root / "concepts"

    if not concepts_dir.is_dir():
        return

    frozen_set = set(frozen_slugs)

    for mdfile in concepts_dir.glob("*.md"):
        slug = mdfile.stem
        if slug not in frozen_set:
            continue

        raw = safe_read_file(mdfile)
        meta, _body = parse_frontmatter(raw)

        if meta.get("orphaned"):
            continue  # Already orphaned

        # No remaining owners → orphan
        orphan_page(root_dir, slug, "No remaining source owners (frozen cleanup)")


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_other_owners(state: WikiState, slug: str, exclude_file: str) -> list[str]:
    """Find source files (other than exclude_file) that own a concept slug."""
    others: list[str] = []
    for filename, source_state in state.sources.items():
        if filename == exclude_file:
            continue
        if slug in source_state.concepts:
            others.append(filename)
    return others
