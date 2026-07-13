"""Tests for render_vault stale cleanup, empty vault, and graph export.

Covers:
  - P0: Stale file cleanup (non-reviewed deleted, reviewed preserved,
    orphaned preserved, index.md preserved)
  - P1: Empty SynthesisBundle + empty sources dict — no crash, stale cleaned
  - P1: graph.json and graph.mmd generation in render_vault
"""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.obsidian import render_vault

# ── Helpers ──────────────────────────────────────────────────────────────


def _write_md(path: Path, body: str = "", *, reviewed: bool = False,
              orphaned: bool = False) -> None:
    """Write a markdown page with optional frontmatter flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---"]
    fm_lines.append("title: " + path.stem)
    if reviewed:
        fm_lines.append("reviewed: true")
    if orphaned:
        fm_lines.append("orphaned: true")
    fm_lines.append("---")
    fm_lines.append("")
    fm_lines.append(f"# {path.stem}")
    if body:
        fm_lines.append("")
        fm_lines.append(body)
    path.write_text("\n".join(fm_lines) + "\n", encoding="utf-8")


# ── P0: Stale file cleanup ──────────────────────────────────────────────


def test_stale_concept_cleanup_deletes_non_reviewed(tmp_path: Path):
    """A non-reviewed, non-orphaned concept file not in the new bundle is deleted."""
    stale_concept = tmp_path / "concepts" / "old-concept.md"
    _write_md(stale_concept, body="Old generated content")

    # New bundle has a different concept only.
    bundle = SynthesisBundle(
        concepts=[ConceptNote(title="New Concept", slug="new-concept", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert not stale_concept.exists()
    assert (tmp_path / "concepts" / "new-concept.md").exists()


def test_stale_concept_cleanup_preserves_reviewed(tmp_path: Path):
    """A reviewed concept file not in the new bundle is preserved."""
    reviewed_concept = tmp_path / "concepts" / "reviewed-old.md"
    _write_md(reviewed_concept, body="Human curated content", reviewed=True)

    bundle = SynthesisBundle(
        concepts=[ConceptNote(title="New", slug="new", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert reviewed_concept.exists()
    content = reviewed_concept.read_text(encoding="utf-8")
    assert "reviewed: true" in content
    assert "Human curated content" in content


def test_stale_concept_cleanup_preserves_orphaned(tmp_path: Path):
    """An orphaned concept file (orphaned: true) not in the new bundle is preserved."""
    orphaned_concept = tmp_path / "concepts" / "orphaned-old.md"
    _write_md(orphaned_concept, body="Orphaned concept", orphaned=True)

    bundle = SynthesisBundle(
        concepts=[ConceptNote(title="New", slug="new", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert orphaned_concept.exists()


def test_stale_concept_cleanup_preserves_index_md(tmp_path: Path):
    """index.md in concepts/ is never deleted by stale cleanup."""
    index_file = tmp_path / "concepts" / "index.md"
    _write_md(index_file, body="Concept index")

    bundle = SynthesisBundle(
        concepts=[ConceptNote(title="New", slug="new", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert index_file.exists()


def test_stale_entry_cleanup_deletes_non_reviewed(tmp_path: Path):
    """A non-reviewed entry file not in the new bundle is deleted."""
    stale_entry = tmp_path / "entries" / "old-entry.md"
    _write_md(stale_entry, body="Old entry content")

    bundle = SynthesisBundle(
        sources=[SourceSynthesis(source_title="New Source", source_summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert not stale_entry.exists()


def test_stale_entry_cleanup_preserves_reviewed(tmp_path: Path):
    """A reviewed entry file not in the new bundle is preserved."""
    reviewed_entry = tmp_path / "entries" / "reviewed-old.md"
    _write_md(reviewed_entry, body="Curated entry", reviewed=True)

    bundle = SynthesisBundle(
        sources=[SourceSynthesis(source_title="New Source", source_summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert reviewed_entry.exists()
    assert "reviewed: true" in reviewed_entry.read_text(encoding="utf-8")


def test_stale_moc_cleanup_deletes_non_reviewed(tmp_path: Path):
    """A non-reviewed MoC file not in the new bundle is deleted."""
    stale_moc = tmp_path / "mocs" / "old-moc.md"
    _write_md(stale_moc, body="Old MoC content")

    bundle = SynthesisBundle(
        maps=[MapOfContent(title="New MoC", slug="new-moc", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert not stale_moc.exists()
    assert (tmp_path / "mocs" / "new-moc.md").exists()


def test_stale_moc_cleanup_preserves_reviewed(tmp_path: Path):
    """A reviewed MoC file not in the new bundle is preserved."""
    reviewed_moc = tmp_path / "mocs" / "reviewed-old.md"
    _write_md(reviewed_moc, body="Curated MoC", reviewed=True)

    bundle = SynthesisBundle(
        maps=[MapOfContent(title="New", slug="new", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert reviewed_moc.exists()
    assert "reviewed: true" in reviewed_moc.read_text(encoding="utf-8")


def test_stale_moc_cleanup_preserves_index_md(tmp_path: Path):
    """index.md in mocs/ is never deleted by stale cleanup."""
    index_file = tmp_path / "mocs" / "index.md"
    _write_md(index_file, body="MoC index")

    bundle = SynthesisBundle(
        maps=[MapOfContent(title="New", slug="new", summary="New")],
    )
    render_vault(tmp_path, bundle, {})

    assert index_file.exists()


def test_stale_cleanup_comprehensive(tmp_path: Path):
    """Comprehensive stale cleanup: pre-existing files (reviewed, orphaned,
    non-reviewed) across concepts/entries/mocs, then render_vault with a
    bundle that omits those slugs.

    Asserts:
      - non-reviewed deleted
      - reviewed preserved
      - orphaned preserved
      - index.md preserved
    """
    # ── Pre-existing files ──────────────────────────────────────────
    # Concepts: reviewed, orphaned, and stale non-reviewed
    reviewed_concept = tmp_path / "concepts" / "reviewed-concept.md"
    _write_md(reviewed_concept, body="Curated concept", reviewed=True)

    orphaned_concept = tmp_path / "concepts" / "orphaned-concept.md"
    _write_md(orphaned_concept, body="Orphaned", orphaned=True)

    stale_concept = tmp_path / "concepts" / "stale-concept.md"
    _write_md(stale_concept, body="Should be deleted")

    concepts_index = tmp_path / "concepts" / "index.md"
    _write_md(concepts_index, body="Concepts index")

    # Entries: reviewed and stale
    reviewed_entry = tmp_path / "entries" / "reviewed-entry.md"
    _write_md(reviewed_entry, body="Curated entry", reviewed=True)

    stale_entry = tmp_path / "entries" / "stale-entry.md"
    _write_md(stale_entry, body="Should be deleted")

    # MoCs: reviewed, stale, index
    reviewed_moc = tmp_path / "mocs" / "reviewed-moc.md"
    _write_md(reviewed_moc, body="Curated MoC", reviewed=True)

    stale_moc = tmp_path / "mocs" / "stale-moc.md"
    _write_md(stale_moc, body="Should be deleted")

    mocs_index = tmp_path / "mocs" / "index.md"
    _write_md(mocs_index, body="MoCs index")

    # ── Render with a bundle that has DIFFERENT slugs ───────────────
    bundle = SynthesisBundle(
        concepts=[
            ConceptNote(title="Fresh Concept", slug="fresh-concept", summary="Fresh"),
        ],
        sources=[
            SourceSynthesis(source_title="Fresh Source", source_summary="Fresh"),
        ],
        maps=[
            MapOfContent(title="Fresh MoC", slug="fresh-moc", summary="Fresh"),
        ],
    )
    render_vault(tmp_path, bundle, {})

    # ── Assertions ──────────────────────────────────────────────────
    # Non-reviewed stale files deleted
    assert not stale_concept.exists(), "stale concept should be deleted"
    assert not stale_entry.exists(), "stale entry should be deleted"
    assert not stale_moc.exists(), "stale MoC should be deleted"

    # Reviewed files preserved
    assert reviewed_concept.exists(), "reviewed concept should be preserved"
    assert reviewed_entry.exists(), "reviewed entry should be preserved"
    assert reviewed_moc.exists(), "reviewed MoC should be preserved"

    # Orphaned concept preserved
    assert orphaned_concept.exists(), "orphaned concept should be preserved"

    # index.md files preserved
    assert concepts_index.exists(), "concepts/index.md should be preserved"
    assert mocs_index.exists(), "mocs/index.md should be preserved"

    # New bundle files exist
    assert (tmp_path / "concepts" / "fresh-concept.md").exists()
    assert (tmp_path / "mocs" / "fresh-moc.md").exists()
    assert (tmp_path / "index.md").exists()  # bundle root index


# ── P1: Empty vault render ───────────────────────────────────────────────


def test_render_vault_empty_bundle_no_crash(tmp_path: Path):
    """render_vault with an empty SynthesisBundle and empty sources dict
    does not crash and produces a root index.md."""
    bundle = SynthesisBundle()
    written = render_vault(tmp_path, bundle, {})

    # Root index.md should be created.
    assert (tmp_path / "index.md").exists()
    # Directories should exist.
    assert (tmp_path / "sources").is_dir()
    assert (tmp_path / "entries").is_dir()
    assert (tmp_path / "concepts").is_dir()
    assert (tmp_path / "mocs").is_dir()
    # written list should include the index.
    assert any("index.md" in p for p in written)


def test_render_vault_empty_bundle_cleans_stale_non_reviewed(tmp_path: Path):
    """Empty bundle + empty sources: stale non-reviewed files are cleaned."""
    stale_concept = tmp_path / "concepts" / "stale.md"
    _write_md(stale_concept, body="Should be deleted")

    stale_entry = tmp_path / "entries" / "stale.md"
    _write_md(stale_entry, body="Should be deleted")

    stale_moc = tmp_path / "mocs" / "stale.md"
    _write_md(stale_moc, body="Should be deleted")

    # Reviewed files should be preserved even with empty bundle.
    reviewed_concept = tmp_path / "concepts" / "reviewed.md"
    _write_md(reviewed_concept, body="Keep me", reviewed=True)

    bundle = SynthesisBundle()
    render_vault(tmp_path, bundle, {})

    assert not stale_concept.exists()
    assert not stale_entry.exists()
    assert not stale_moc.exists()
    assert reviewed_concept.exists()
    assert "reviewed: true" in reviewed_concept.read_text(encoding="utf-8")


# ── P1: graph.json and graph.mmd generation in render_vault ──────────────


def test_render_vault_generates_graph_json(tmp_path: Path):
    """render_vault produces .llmwiki/graph.json with valid JSON."""
    from obsidian_llm_wiki.synth.dedupe import merge_bundle

    synth = SourceSynthesis(
        source_title="Test Source",
        source_summary="Summary",
        concepts=[
            ConceptNote(title="Concept A", slug="concept-a", summary="A",
                        tags=["test"]),
            ConceptNote(title="Concept B", slug="concept-b", summary="B",
                        tags=["test"]),
        ],
        maps=[MapOfContent(title="Test MoC", slug="test-moc", summary="MoC",
                           concept_slugs=["concept-a", "concept-b"])],
    )
    bundle = merge_bundle([synth])
    sources = {"test-source.md": SourceDoc(title="Test Source", content="Content")}

    render_vault(tmp_path, bundle, sources)

    graph_json = tmp_path / ".llmwiki" / "graph.json"
    assert graph_json.exists()
    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data
    assert "mocs" in data
    # Should have concept nodes.
    concept_nodes = [n for n in data["nodes"] if n["type"] == "concept"]
    assert len(concept_nodes) == 2
    slugs = {n["id"] for n in concept_nodes}
    assert "concept-a" in slugs
    assert "concept-b" in slugs


def test_render_vault_generates_graph_mmd(tmp_path: Path):
    """render_vault produces .llmwiki/graph.mmd with a Mermaid diagram."""
    from obsidian_llm_wiki.synth.dedupe import merge_bundle

    synth = SourceSynthesis(
        source_title="Test Source",
        source_summary="Summary",
        concepts=[
            ConceptNote(title="Concept A", slug="concept-a", summary="A"),
            ConceptNote(title="Concept B", slug="concept-b", summary="B"),
        ],
        maps=[MapOfContent(title="Test MoC", slug="test-moc", summary="MoC",
                           concept_slugs=["concept-a", "concept-b"])],
    )
    bundle = merge_bundle([synth])
    sources = {"test-source.md": SourceDoc(title="Test Source", content="Content")}

    render_vault(tmp_path, bundle, sources)

    graph_mmd = tmp_path / ".llmwiki" / "graph.mmd"
    assert graph_mmd.exists()
    content = graph_mmd.read_text(encoding="utf-8")
    assert content.startswith("graph LR")
    # Should contain the MoC subgraph.
    assert "subgraph" in content


def test_render_vault_graph_export_with_empty_bundle(tmp_path: Path):
    """render_vault with empty bundle still generates valid (empty) graph files."""
    bundle = SynthesisBundle()
    render_vault(tmp_path, bundle, {})

    graph_json = tmp_path / ".llmwiki" / "graph.json"
    graph_mmd = tmp_path / ".llmwiki" / "graph.mmd"
    assert graph_json.exists()
    assert graph_mmd.exists()

    data = json.loads(graph_json.read_text(encoding="utf-8"))
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["mocs"] == []

    mmd_content = graph_mmd.read_text(encoding="utf-8")
    assert mmd_content.strip().startswith("graph LR")
