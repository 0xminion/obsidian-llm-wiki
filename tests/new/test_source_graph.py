"""Tests for render.source_graph — source dependency graph export."""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.source_graph import (
    build_source_dependency_dict,
    build_source_dependency_mermaid,
    export_source_dependency_graph,
)
from obsidian_llm_wiki.synth.dedupe import merge_bundle


def _make_bundle() -> SynthesisBundle:
    """Build a test bundle with two sources contributing overlapping concepts."""
    synth_a = SourceSynthesis(
        source_title="Source A",
        source_summary="Summary A",
        source_tags=["ml"],
        source_file="source-a.md",
        concepts=[
            ConceptNote(
                title="Concept A", slug="concept-a", summary="A summary",
                tags=["ml"], confidence=0.9,
            ),
            ConceptNote(
                title="Concept B", slug="concept-b", summary="B summary",
                tags=["ml"], confidence=0.8,
            ),
        ],
    )
    synth_b = SourceSynthesis(
        source_title="Source B",
        source_summary="Summary B",
        source_tags=["ai"],
        source_file="source-b.md",
        concepts=[
            ConceptNote(
                title="Concept B", slug="concept-b", summary="B summary",
                tags=["ml"], confidence=0.85,
            ),
            ConceptNote(
                title="Concept C", slug="concept-c", summary="C summary",
                tags=["ai"], confidence=0.7,
            ),
        ],
    )
    return merge_bundle([synth_a, synth_b])


# ── JSON builder ─────────────────────────────────────────────────────────


def test_build_source_dependency_dict_has_metadata():
    graph = build_source_dependency_dict(_make_bundle())
    meta = graph["metadata"]
    assert meta["source_count"] == 2
    assert meta["concept_count"] >= 3
    assert meta["edge_count"] == 4  # 2 per source


def test_build_source_dependency_dict_source_nodes():
    graph = build_source_dependency_dict(_make_bundle())
    source_ids = [s["id"] for s in graph["nodes"]["sources"]]
    assert "source-a" in source_ids
    assert "source-b" in source_ids


def test_build_source_dependency_dict_concept_nodes():
    graph = build_source_dependency_dict(_make_bundle())
    concept_ids = [c["id"] for c in graph["nodes"]["concepts"]]
    assert "concept-a" in concept_ids
    assert "concept-b" in concept_ids
    assert "concept-c" in concept_ids


def test_build_source_dependency_dict_edges():
    graph = build_source_dependency_dict(_make_bundle())
    edges = graph["edges"]
    # source-a → concept-a, source-a → concept-b
    # source-b → concept-b, source-b → concept-c
    assert {"source": "source-a", "target": "concept-a", "type": "contributed_to"} in edges
    assert {"source": "source-a", "target": "concept-b", "type": "contributed_to"} in edges
    assert {"source": "source-b", "target": "concept-b", "type": "contributed_to"} in edges
    assert {"source": "source-b", "target": "concept-c", "type": "contributed_to"} in edges


def test_build_source_dependency_dict_concept_convergence():
    """concept-b is contributed to by both sources — should have two edges."""
    graph = build_source_dependency_dict(_make_bundle())
    edges_to_b = [e for e in graph["edges"] if e["target"] == "concept-b"]
    assert len(edges_to_b) == 2


# ── Mermaid builder ─────────────────────────────────────────────────────


def test_build_source_dependency_mermaid_starts_with_graph_lr():
    mmd = build_source_dependency_mermaid(_make_bundle())
    assert mmd.startswith("graph LR")


def test_build_source_dependency_mermaid_has_edges():
    mmd = build_source_dependency_mermaid(_make_bundle())
    assert "contributed_to" in mmd
    assert "-->" in mmd


def test_build_source_dependency_mermaid_has_source_nodes():
    mmd = build_source_dependency_mermaid(_make_bundle())
    assert "source_a" in mmd
    assert "source_b" in mmd


# ── File export ──────────────────────────────────────────────────────────


def test_export_source_dependency_graph_creates_files(tmp_path: Path):
    bundle = _make_bundle()
    out = tmp_path / "dep-graph"
    written = export_source_dependency_graph(bundle, out)
    assert len(written) == 2
    json_path = out / "source-dependency-graph.json"
    mmd_path = out / "source-dependency-graph.mmd"
    assert json_path.exists()
    assert mmd_path.exists()

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metadata" in raw
    assert "nodes" in raw
    assert "edges" in raw
    assert raw["metadata"]["source_count"] == 2


def test_export_source_dependency_graph_creates_output_dir(tmp_path: Path):
    """Output directory is created when it does not exist."""
    bundle = _make_bundle()
    out = tmp_path / "deep" / "nested" / "graph"
    export_source_dependency_graph(bundle, out)
    assert (out / "source-dependency-graph.json").exists()


def test_export_source_dependency_graph_empty_bundle(tmp_path: Path):
    """Empty bundle produces zero-count graph without error."""
    bundle = SynthesisBundle()
    out = tmp_path / "dep"
    written = export_source_dependency_graph(bundle, out)
    assert len(written) == 2
    raw = json.loads((out / "source-dependency-graph.json").read_text(encoding="utf-8"))
    assert raw["metadata"]["source_count"] == 0
    assert raw["metadata"]["edge_count"] == 0
