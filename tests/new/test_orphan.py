"""Tests for obsidian_llm_wiki.core.orphan — orphan management."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.models import SourceState, WikiState
from obsidian_llm_wiki.core.orphan import (
    find_exclusively_owned_concepts,
    mark_orphaned_concepts,
    orphan_page,
)
from obsidian_llm_wiki.render.obsidian import build_frontmatter, parse_frontmatter, safe_read_file


def _make_concept_file(concepts_dir: Path, slug: str, title: str = "") -> Path:
    """Write a minimal concept page to disk."""
    concepts_dir.mkdir(parents=True, exist_ok=True)
    path = concepts_dir / f"{slug}.md"
    fm = {"type": "Concept", "title": title or slug, "tags": ["t"]}
    body = f"# {title or slug}\n\nContent here."
    path.write_text(build_frontmatter(fm) + "\n" + body)
    return path


def test_find_exclusively_owned_concepts():
    """Concepts only in the deleted source are returned."""
    state = WikiState(sources={
        "deleted.md": SourceState(hash="h", concepts=["a", "b", "c"]),
        "live.md": SourceState(hash="h", concepts=["b", "d"]),
    })
    result = find_exclusively_owned_concepts("deleted.md", state)
    assert result == ["a", "c"]


def test_find_exclusively_owned_concepts_all_shared():
    """When all concepts are shared, nothing is exclusively owned."""
    state = WikiState(sources={
        "deleted.md": SourceState(hash="h", concepts=["a", "b"]),
        "live.md": SourceState(hash="h", concepts=["a", "b"]),
    })
    assert find_exclusively_owned_concepts("deleted.md", state) == []


def test_find_exclusively_owned_concepts_no_state():
    """When the deleted source has no state, returns empty."""
    state = WikiState()
    assert find_exclusively_owned_concepts("missing.md", state) == []


def test_find_exclusively_owned_concepts_no_concepts():
    """When the deleted source has no concepts, returns empty."""
    state = WikiState(sources={
        "deleted.md": SourceState(hash="h", concepts=[]),
    })
    assert find_exclusively_owned_concepts("deleted.md", state) == []


def test_orphan_page_marks_frontmatter(tmp_path: Path):
    """orphan_page adds orphaned: true to frontmatter."""
    _make_concept_file(tmp_path, "my-concept", "My Concept")
    result = orphan_page(tmp_path, "my-concept", "Source deleted: test.md")
    assert result is True

    raw = safe_read_file(tmp_path / "my-concept.md")
    meta, body = parse_frontmatter(raw)
    assert meta["orphaned"] is True
    assert "Source deleted: test.md" in meta["orphaned_reason"]
    assert "# My Concept" in body  # body preserved


def test_orphan_page_idempotent(tmp_path: Path):
    """orphan_page returns False if already orphaned."""
    _make_concept_file(tmp_path, "my-concept", "My Concept")
    orphan_page(tmp_path, "my-concept", "reason 1")
    result = orphan_page(tmp_path, "my-concept", "reason 2")
    assert result is False

    raw = safe_read_file(tmp_path / "my-concept.md")
    meta, _ = parse_frontmatter(raw)
    assert "reason 1" in meta["orphaned_reason"]


def test_orphan_page_missing_file(tmp_path: Path):
    """orphan_page returns False for missing file."""
    assert orphan_page(tmp_path, "nonexistent", "reason") is False


def test_mark_orphaned_concepts_exclusive_only(tmp_path: Path):
    """Only exclusively-owned concepts are marked orphaned."""
    _make_concept_file(tmp_path, "exclusive", "Exclusive")
    _make_concept_file(tmp_path, "shared", "Shared")

    state = WikiState(sources={
        "deleted.md": SourceState(hash="h", concepts=["exclusive", "shared"]),
        "live.md": SourceState(hash="h", concepts=["shared"]),
    })

    orphaned = mark_orphaned_concepts(tmp_path, "deleted.md", state)
    assert orphaned == ["exclusive"]

    # Shared concept should NOT be orphaned.
    shared_raw = safe_read_file(tmp_path / "shared.md")
    shared_meta, _ = parse_frontmatter(shared_raw)
    assert "orphaned" not in shared_meta

    # Exclusive concept should be orphaned.
    excl_raw = safe_read_file(tmp_path / "exclusive.md")
    excl_meta, _ = parse_frontmatter(excl_raw)
    assert excl_meta["orphaned"] is True


def test_mark_orphaned_concepts_no_concepts(tmp_path: Path):
    """mark_orphaned_concepts returns empty list when source has no concepts."""
    state = WikiState(sources={
        "deleted.md": SourceState(hash="h", concepts=[]),
    })
    result = mark_orphaned_concepts(tmp_path, "deleted.md", state)
    assert result == []
