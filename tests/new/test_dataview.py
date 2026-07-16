"""Tests for render.dataview — Dataview/Bases view generators."""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceProvenance,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.dataview import (
    render_concepts_by_confidence_view,
    render_contradictions_by_status_view,
    render_dataview_views,
    render_mocs_by_count_view,
    render_sources_by_freshness_view,
)
from obsidian_llm_wiki.synth.dedupe import merge_bundle


def _make_bundle() -> SynthesisBundle:
    """Build a test bundle with concepts of varying confidence and MoCs."""
    synth = SourceSynthesis(
        source_title="Paper A",
        source_summary="Summary A",
        source_tags=["ml"],
        source_file="paper-a.md",
        concepts=[
            ConceptNote(
                title="High Confidence", slug="high-confidence",
                summary="A", tags=["ml"], confidence=0.95,
            ),
            ConceptNote(
                title="Low Confidence", slug="low-confidence",
                summary="B", tags=["ml"], confidence=0.3,
            ),
            ConceptNote(
                title="Medium Confidence", slug="medium-confidence",
                summary="C", tags=["ai"], confidence=0.6,
            ),
        ],
        maps=[
            MapOfContent(
                title="ML Topic", slug="ml-topic", summary="MOC",
                concept_slugs=["high-confidence", "low-confidence"],
            ),
            MapOfContent(
                title="AI Topic", slug="ai-topic", summary="MOC2",
                concept_slugs=["medium-confidence"],
            ),
        ],
    )
    return merge_bundle([synth])


# ── Concepts by confidence ──────────────────────────────────────────────


def test_concepts_by_confidence_has_dataview_block():
    view = render_concepts_by_confidence_view(_make_bundle())
    assert "```dataview" in view
    assert "FROM \"concepts\"" in view
    assert "SORT confidence DESC" in view


def test_concepts_by_confidence_static_table_sorted():
    view = render_concepts_by_confidence_view(_make_bundle())
    assert "| Concept | Confidence | Tags |" in view
    # High confidence should appear before low confidence in the static table.
    high_pos = view.index("[[high-confidence|")
    low_pos = view.index("[[low-confidence|")
    assert high_pos < low_pos


def test_concepts_by_confidence_includes_all_concepts():
    bundle = _make_bundle()
    view = render_concepts_by_confidence_view(bundle)
    for concept in bundle.concepts:
        assert f"[[{concept.slug}|" in view


# ── MoCs by count ───────────────────────────────────────────────────────


def test_mocs_by_count_has_dataview_block():
    view = render_mocs_by_count_view(_make_bundle())
    assert "```dataview" in view
    assert "FROM \"mocs\"" in view
    assert "SORT length(concept_slugs)" in view


def test_mocs_by_count_static_table_sorted():
    view = render_mocs_by_count_view(_make_bundle())
    assert "| MoC | Concepts | Tags |" in view
    # ml-topic has 2 concepts, ai-topic has 1 — ml-topic should come first.
    ml_pos = view.index("[[ml-topic|")
    ai_pos = view.index("[[ai-topic|")
    assert ml_pos < ai_pos


def test_mocs_by_count_includes_all_mocs():
    bundle = _make_bundle()
    view = render_mocs_by_count_view(bundle)
    for moc in bundle.maps:
        assert f"[[{moc.slug}|" in view


# ── Contradictions by status ────────────────────────────────────────────


def test_contradictions_by_status_empty_when_no_file():
    view = render_contradictions_by_status_view(Path("/nonexistent/path.json"))
    assert "# Contradictions by Status" in view
    assert "No contradictions detected" in view
    assert "```dataview" in view


def test_contradictions_by_status_with_records(tmp_path: Path):
    store_path = tmp_path / "contradictions.json"
    store_path.write_text(json.dumps({
        "records": [
            {"id": "c1", "summary": "Conflict A", "status": "detected",
             "sources": [], "evidence": []},
            {"id": "c2", "summary": "Conflict B", "status": "resolved",
             "sources": [], "evidence": []},
        ],
        "source_revisions": [],
    }), encoding="utf-8")
    view = render_contradictions_by_status_view(store_path)
    assert "# Contradictions by Status" in view
    assert "### detected" in view
    assert "### resolved" in view
    assert "Conflict A" in view
    assert "Conflict B" in view


# ── Sources by freshness ────────────────────────────────────────────────


def test_sources_by_freshness_has_dataview_block():
    sources = {"paper-a.md": SourceDoc(title="Paper A", content="x")}
    view = render_sources_by_freshness_view(sources, _make_bundle())
    assert "```dataview" in view
    assert "FROM \"sources\"" in view
    assert "SORT timestamp DESC" in view


def test_sources_by_freshness_static_table():
    sources = {
        "paper-a.md": SourceDoc(
            title="Paper A",
            content="x",
            url="https://a.com",
            provenance=SourceProvenance(retrieved_at="2026-01-02T00:00:00Z"),
        ),
        "paper-b.md": SourceDoc(
            title="Paper B",
            content="y",
            url="https://b.com",
            provenance=SourceProvenance(retrieved_at="2026-01-03T00:00:00Z"),
        ),
    }
    view = render_sources_by_freshness_view(sources, _make_bundle())
    assert "| Source | Retrieved |" in view
    # paper-b has a newer retrieved_at so it should appear first.
    b_pos = view.index("[[paper-b|")
    a_pos = view.index("[[paper-a|")
    assert b_pos < a_pos


def test_sources_by_freshness_missing_provenance():
    """Sources without provenance.retrieved_at display '—'."""
    sources = {"paper.md": SourceDoc(title="Paper", content="x")}
    view = render_sources_by_freshness_view(sources, _make_bundle())
    assert "| — |" in view


# ── Top-level orchestrator ───────────────────────────────────────────────


def test_render_dataview_views_creates_all_files(tmp_path: Path):
    bundle = _make_bundle()
    sources = {
        "paper-a.md": SourceDoc(title="Paper A", content="Content A"),
    }
    written = render_dataview_views(tmp_path, bundle, sources)

    # Should create 4 view files + index.md = 5 files.
    assert len(written) == 5
    views_dir = tmp_path / "views"
    assert (views_dir / "concepts-by-confidence.md").exists()
    assert (views_dir / "mocs-by-count.md").exists()
    assert (views_dir / "contradictions-by-status.md").exists()
    assert (views_dir / "sources-by-freshness.md").exists()
    assert (views_dir / "index.md").exists()
    for path in views_dir.glob("*.md"):
        assert path.read_text(encoding="utf-8").startswith("---\n")


def test_render_dataview_views_uses_default_contradiction_path(tmp_path: Path):
    """When contradictions_path is None, the default .llmwiki path is used."""
    bundle = _make_bundle()
    sources = {"paper-a.md": SourceDoc(title="Paper A", content="x")}
    # No contradictions.json — should still produce the view without error.
    render_dataview_views(tmp_path, bundle, sources)
    contra = (tmp_path / "views" / "contradictions-by-status.md").read_text()
    assert "No contradictions detected" in contra


def test_render_dataview_views_index_has_links(tmp_path: Path):
    bundle = _make_bundle()
    sources = {"paper-a.md": SourceDoc(title="Paper A", content="x")}
    render_dataview_views(tmp_path, bundle, sources)
    idx = (tmp_path / "views" / "index.md").read_text()
    assert "[[concepts-by-confidence" in idx
    assert "[[mocs-by-count" in idx
    assert "[[contradictions-by-status" in idx
    assert "[[sources-by-freshness" in idx
