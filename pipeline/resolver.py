"""Bidirectional wikilink injection (deterministic, no LLM).

Ported from llm-wiki-compiler/src/compiler/resolver.ts.

Two-pass approach:
  Pass 1 (outbound): changed pages get [[wikilinks]] for any concept title they mention.
  Pass 2 (inbound): ALL pages scanned for mentions of NEW concept titles.

Complexity: O(changed * total_concepts) — reverse-match for position preservation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from pipeline.markdown import (
    atomic_write,
    build_frontmatter,
    parse_frontmatter,
    safe_read_file,
    slugify,
)

# ── Data structures ────────────────────────────────────────────────────


class TitleEntry(NamedTuple):
    slug: str
    title: str


# ── Core ────────────────────────────────────────────────────────────────


def resolve_links(root_dir: str | Path, changed_slugs: list[str], new_slugs: list[str]) -> int:
    """Run bidirectional wikilink injection and return number of pages modified.

    Args:
        root_dir: Path to the wiki root (contains concepts/, sources/, etc.).
        changed_slugs: List of slugs for pages that were just compiled/changed.
        new_slugs: List of slugs for newly created concept pages.

    Returns:
        Count of pages that were modified with injected wikilinks.
    """
    root = Path(root_dir)
    concepts_dir = root / "concepts"
    if not concepts_dir.is_dir():
        return 0

    changed_set = set(changed_slugs)
    new_set = set(new_slugs)

    # ── Build title index from all concept .md files ──
    title_index: dict[str, TitleEntry] = {}  # lower_title -> TitleEntry
    slug_index: dict[str, TitleEntry] = {}   # slug -> TitleEntry

    for mdfile in concepts_dir.glob("*.md"):
        raw = safe_read_file(mdfile)
        meta, _body = parse_frontmatter(raw)
        title = meta.get("title")
        slug = meta.get("slug") or mdfile.stem
        if title:
            entry = TitleEntry(slug=slug, title=title)
            slug_index[slug] = entry
            title_index[title.lower()] = entry

    if not title_index:
        return 0

    # Collect all concept titles sorted longest-first (avoids partial matches)
    concept_titles = sorted(title_index.keys(), key=len, reverse=True)

    modified_count = 0

    # ── Pass 1: outbound links for changed pages ──
    for slug in changed_set:
        page_path = concepts_dir / f"{slug}.md"
        if not page_path.exists():
            continue
        raw = safe_read_file(page_path)
        if not raw.strip():
            continue

        modified, new_raw = _inject_outbound_links(raw, slug, concept_titles,
                                                    title_index, slug_index)
        if modified:
            atomic_write(page_path, new_raw)
            modified_count += 1

    # ── Pass 2: inbound links for NEW concept titles across ALL pages ──
    if new_set:
        new_entries = {slug: entry for slug, entry in slug_index.items()
                       if slug in new_set}
        for mdfile in concepts_dir.glob("*.md"):
            slug = mdfile.stem
            if slug in changed_set:
                # Already handled in pass 1
                continue
            raw = safe_read_file(mdfile)
            if not raw.strip():
                continue

            modified, new_raw = _inject_outbound_links(raw, slug, concept_titles,
                                                        title_index, slug_index,
                                                        only_new_titles=_build_new_title_set(new_entries))
            if modified:
                atomic_write(mdfile, new_raw)
                modified_count += 1

    return modified_count


# ── Helpers ─────────────────────────────────────────────────────────────


def _build_new_title_set(new_entries: dict[str, TitleEntry]) -> set[str]:
    """Build set of lowercased titles for new concept entries."""
    return {entry.title.lower() for entry in new_entries.values()}


def _inject_outbound_links(
    raw: str,
    own_slug: str,
    concept_titles: list[str],
    title_index: dict[str, TitleEntry],
    slug_index: dict[str, TitleEntry],
    only_new_titles: set[str] | None = None,
) -> tuple[bool, str]:
    """Inject [[wikilinks]] for concept titles mentioned in the body.

    Returns (modified: bool, new_text: str).
    """
    meta, body = parse_frontmatter(raw)
    if not body.strip():
        return False, raw

    # Extract existing wikilinks to avoid double-linking
    existing_links = _extract_existing_links(body)

    # Find code blocks and citation markers to skip
    skip_ranges = _find_skip_ranges(body)

    modified = False
    new_body = body

    for lower_title in concept_titles:
        if only_new_titles is not None and lower_title not in only_new_titles:
            continue

        entry = title_index[lower_title]
        if entry.slug == own_slug:
            continue  # Don't link to self

        # Skip if already linked
        if entry.slug in existing_links:
            continue

        # Try to find and replace occurrences (case-insensitive, word-boundary)
        new_body, count = _replace_title_occurrences(
            new_body, entry.title, entry.slug, skip_ranges
        )
        if count > 0:
            modified = True

    if not modified:
        return False, raw

    # Reassemble: frontmatter + body
    fm = build_frontmatter(meta)
    result = fm + "\n" + new_body if fm else raw[:0] + new_body
    return True, result


# ── Regex helpers ───────────────────────────────────────────────────────

_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")


def _extract_existing_links(body: str) -> set[str]:
    """Extract slugs from existing [[wikilinks]] in body."""
    slugs: set[str] = set()
    for match in _LINK_RE.finditer(body):
        target = match.group(1).strip()
        slugs.add(target)
        slugs.add(slugify(target))
    return slugs


_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`", re.MULTILINE)
_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_CITATION_RE = re.compile(r"\^\[[^\]]+\]")


def _find_skip_ranges(body: str) -> list[tuple[int, int]]:
    """Find (start, end) character ranges to skip: code blocks, inline code, headers, citations."""
    ranges: list[tuple[int, int]] = []

    for pattern in [_CODE_BLOCK_RE, _HEADER_RE, _CITATION_RE]:
        for match in pattern.finditer(body):
            ranges.append((match.start(), match.end()))

    return ranges


def _replace_title_occurrences(
    body: str,
    title: str,
    slug: str,
    skip_ranges: list[tuple[int, int]],
) -> tuple[str, int]:
    """Replace occurrences of `title` with `[[slug|title]]` using word-boundary matching.

    Returns (new_body, count).
    """
    # Build a word-boundary regex (case-insensitive)
    escaped = re.escape(title)
    pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

    replacements: list[tuple[int, int, str]] = []

    for match in pattern.finditer(body):
        start, end = match.start(), match.end()
        # Check if inside a skip range
        if any(s <= start < e for s, e in skip_ranges):
            continue
        # Check if this match exactly matches an existing [[...]] — shouldn't
        # happen for new links, but be defensive
        if body[max(0, start - 2):start] == "[[":
            continue
        link_text = f"[[{slug}|{title}]]"
        replacements.append((start, end, link_text))

    if not replacements:
        return body, 0

    # Apply replacements in reverse order to preserve positions
    result = body
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]

    return result, len(replacements)
