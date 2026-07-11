"""Tests for render.graph_export — JSON and Mermaid graph export."""

from __future__ import annotations

import json
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.render.graph_export import (
    export_graph,
    export_graph_json,
    export_graph_mermaid,
)


def _make_bundle() -> SynthesisBundle:
    """Build a test bundle with concepts, MoC, and relationships."""
    c1 = ConceptNote(
        title="Bitcoin", slug="bitcoin", summary="Digital currency",
        tags=["crypto", "blockchain"], confidence=0.9,
        related=[ConceptLink(slug="mining", relation="enables", display="Mining")],
    )
    c2 = ConceptNote(
        title="Mining", slug="mining", summary="Proof of work mining",
        tags=["crypto"], confidence=0.8,
        related=[ConceptLink(slug="bitcoin", relation="related_to")],
    )
    c3 = ConceptNote(
        title="Lightning Network", slug="lightning-network",
        summary="Layer 2 scaling", tags=["crypto", "scaling"],
        confidence=0.7,
    )
    moc = MapOfContent(
        title="Cryptocurrency", slug="cryptocurrency", summary="Crypto concepts",
        concept_slugs=["bitcoin", "mining"],
    )
    source = SourceSynthesis(
        source_title="Crypto Paper", source_summary="About crypto",
        source_file="crypto-paper.md",
        concepts=[c1, c2, c3],
        maps=[moc],
    )
    return SynthesisBundle(
        sources=[source],
        concepts=[c1, c2, c3],
        maps=[moc],
    )


# ── JSON export ──────────────────────────────────────────────────────────


def test_export_graph_json_writes_valid_json(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))

    assert "nodes" in data
    assert "edges" in data
    assert "mocs" in data


def test_export_graph_json_nodes_include_concepts(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    concept_nodes = [n for n in data["nodes"] if n["type"] == "concept"]

    assert len(concept_nodes) == 3
    slugs = {n["id"] for n in concept_nodes}
    assert "bitcoin" in slugs
    assert "mining" in slugs
    assert "lightning-network" in slugs


def test_export_graph_json_nodes_include_mocs(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    moc_nodes = [n for n in data["nodes"] if n["type"] == "moc"]

    assert len(moc_nodes) == 1
    assert moc_nodes[0]["id"] == "cryptocurrency"


def test_export_graph_json_nodes_include_sources(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    source_nodes = [n for n in data["nodes"] if n["type"] == "source"]

    assert len(source_nodes) == 1
    assert source_nodes[0]["label"] == "Crypto Paper"


def test_export_graph_json_edges_include_concept_links(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    # Filter out MoC→concept "contains" edges
    rel_edges = [e for e in data["edges"] if e["relation"] != "contains"]

    # bitcoin→mining (enables) should be present
    bitcoin_edges = [e for e in rel_edges if e["source"] == "bitcoin"]
    assert any(e["target"] == "mining" and e["relation"] == "enables" for e in bitcoin_edges)


def test_export_graph_json_edges_include_moc_membership(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    contains_edges = [e for e in data["edges"] if e["relation"] == "contains"]

    assert any(e["source"] == "cryptocurrency" and e["target"] == "bitcoin" for e in contains_edges)
    assert any(e["source"] == "cryptocurrency" and e["target"] == "mining" for e in contains_edges)


def test_export_graph_json_mocs_metadata(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["mocs"]) == 1
    assert data["mocs"][0]["id"] == "cryptocurrency"
    assert data["mocs"][0]["concept_count"] == 2


def test_export_graph_json_node_has_moc_field(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    bitcoin_node = next(n for n in data["nodes"] if n["id"] == "bitcoin")
    assert bitcoin_node["moc"] == "cryptocurrency"

    orphan_node = next(n for n in data["nodes"] if n["id"] == "lightning-network")
    assert orphan_node["moc"] == ""


def test_export_graph_json_bidirectional_detection(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.json"
    export_graph_json(bundle, output)

    data = json.loads(output.read_text(encoding="utf-8"))
    rel_edges = [e for e in data["edges"] if e["relation"] != "contains"]
    # bitcoin↔mining: bitcoin has enables→mining, mining has related_to→bitcoin
    # So the edge should be bidirectional=True
    bidir_edges = [e for e in rel_edges if e.get("bidirectional")]
    assert len(bidir_edges) >= 1


# ── Mermaid export ───────────────────────────────────────────────────────


def test_export_graph_mermaid_writes_diagram(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.mmd"
    export_graph_mermaid(bundle, output)

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert content.startswith("graph LR")


def test_export_graph_mermaid_has_subgraphs_for_mocs(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.mmd"
    export_graph_mermaid(bundle, output)

    content = output.read_text(encoding="utf-8")
    assert "subgraph" in content
    assert "cryptocurrency" in content.lower() or "Cryptocurrency" in content


def test_export_graph_mermaid_has_typed_edges(tmp_path: Path):
    bundle = _make_bundle()
    output = tmp_path / "graph.mmd"
    export_graph_mermaid(bundle, output)

    content = output.read_text(encoding="utf-8")
    assert "enables" in content


# ── Combined export ──────────────────────────────────────────────────────


def test_export_graph_writes_both_files(tmp_path: Path):
    bundle = _make_bundle()
    output_dir = tmp_path / "graph"
    export_graph(bundle, output_dir)

    assert (output_dir / "graph.json").exists()
    assert (output_dir / "graph.mmd").exists()


def test_export_graph_empty_bundle(tmp_path: Path):
    """Empty bundle should produce valid (empty) graph files."""
    bundle = SynthesisBundle()
    output_dir = tmp_path / "graph"
    export_graph(bundle, output_dir)

    data = json.loads((output_dir / "graph.json").read_text())
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["mocs"] == []
