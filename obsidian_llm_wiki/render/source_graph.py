"""Source dependency graph export — which sources contributed to which concepts.

Produces two artifacts:
  * ``source-dependency-graph.json`` — structured JSON with nodes (sources,
    concepts) and edges (source → concept contributions).
  * ``source-dependency-graph.mmd``   — Mermaid diagram for Obsidian embedding.

The graph captures provenance: for each ``SourceSynthesis`` in the bundle,
its concept slugs are linked to the source node.  When the same concept is
derived from multiple sources, the graph shows convergence — the basis for
trust/confidence assessment.

All rendering is deterministic — no LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import islice
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import SynthesisBundle

logger = logging.getLogger("obswiki.render.source_graph")

# Budgets aligned with main graph_export thresholds.
_SOURCE_GRAPH_MONOLITH_NODES = 10_000
_SOURCE_GRAPH_MONOLITH_EDGES = 20_000
_SOURCE_GRAPH_SHARD_SIZE = 5_000

__all__ = [
    "export_source_dependency_graph",
    "build_source_dependency_dict",
    "build_source_dependency_mermaid",
]

SOURCE_GRAPH_JSON = "source-dependency-graph.json"
SOURCE_GRAPH_MMD = "source-dependency-graph.mmd"


def export_source_dependency_graph(
    bundle: SynthesisBundle,
    output_dir: Path,
    *,
    shard_size: int = _SOURCE_GRAPH_SHARD_SIZE,
    monolith_node_threshold: int = _SOURCE_GRAPH_MONOLITH_NODES,
    monolith_edge_threshold: int = _SOURCE_GRAPH_MONOLITH_EDGES,
) -> list[str]:
    """Export the source dependency graph as JSON + Mermaid.

    Args:
        bundle: The merged SynthesisBundle.
        output_dir: Directory to write the graph files into.  Created if it
            does not exist.

    Returns:
        List of file paths written.
    """
    written: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    from obsidian_llm_wiki.render.obsidian import _write_generated_artifact

    # JSON
    json_path = output_dir / SOURCE_GRAPH_JSON
    graph = build_source_dependency_dict(bundle)
    node_count = len(bundle.sources) + len(bundle.concepts)
    edge_count = sum(len(s.concepts) for s in bundle.sources)
    if (
        node_count <= monolith_node_threshold
        and edge_count <= monolith_edge_threshold
    ):
        if _write_generated_artifact(
            json_path,
            json.dumps(graph, indent=2, ensure_ascii=False),
            output_dir.parent,
        ):
            written.append(str(json_path))

        # Mermaid
        mmd_path = output_dir / SOURCE_GRAPH_MMD
        if _write_generated_artifact(
            mmd_path,
            build_source_dependency_mermaid(bundle),
            output_dir.parent,
        ):
            written.append(str(mmd_path))
    else:
        written.extend(
            _export_sharded_source_graph(
                graph,
                json_path,
                shard_size=shard_size,
                bundle_dir=output_dir.parent,
            )
        )

    logger.info(
        "Source dependency graph exported: %d sources, %d concepts, %d edges → %s",
        graph["metadata"].get("source_count", 0),
        graph["metadata"].get("concept_count", 0),
        graph["metadata"].get("edge_count", 0),
        output_dir,
    )
    return written


# ── Builders ────────────────────────────────────────────────────────────


def _slugify_source_id(raw_id: str) -> str:
    """Slugify a source file or title for use as a graph node ID."""
    from obsidian_llm_wiki.render.frontmatter import slugify

    return slugify(raw_id.removesuffix(".md"))


def _mermaid_safe_id(slug: str) -> str:
    """Convert a slug to a Mermaid-safe node identifier."""
    return slug.replace("-", "_")


def _mermaid_safe_label(text: str) -> str:
    """Escape a label for Mermaid node display."""
    text = text.replace('"', "'")
    text = text.replace("[", "‹").replace("]", "›")
    return text


def build_source_dependency_dict(bundle: SynthesisBundle) -> dict[str, Any]:
    """Build the source→concept dependency graph as a dict.

    Structure::

        {
          "metadata": {"source_count", "concept_count", "edge_count"},
          "nodes": {
            "sources": [{"id", "label", "source_file", "concept_count"}],
            "concepts": [{"id", "label", "confidence"}],
          },
          "edges": [{"source", "target", "type": "contributed_to"}]
        }
    """
    # Collect which sources contributed to which concepts.
    # Each SourceSynthesis has source_title/source_file + a list of concepts.
    source_nodes: list[dict[str, Any]] = []
    concept_node_set: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    canonical_slugs = {concept.slug for concept in bundle.concepts}
    for synthesis in bundle.sources:
        raw_id = synthesis.source_file or synthesis.source_title
        if not raw_id:
            continue
        source_id = _slugify_source_id(raw_id)
        concept_slugs = [c.slug for c in synthesis.concepts if c.slug in canonical_slugs]
        source_nodes.append({
            "id": source_id,
            "label": synthesis.source_title or raw_id,
            "source_file": synthesis.source_file or "",
            "concept_count": len(concept_slugs),
        })
        for concept in synthesis.concepts:
            if concept.slug not in canonical_slugs:
                continue
            if concept.slug not in concept_node_set:
                concept_node_set[concept.slug] = {
                    "id": concept.slug,
                    "label": concept.title,
                    "confidence": concept.confidence,
                }
            edges.append({
                "source": source_id,
                "target": concept.slug,
                "type": "contributed_to",
            })

    return {
        "metadata": {
            "source_count": len(source_nodes),
            "concept_count": len(concept_node_set),
            "edge_count": len(edges),
        },
        "nodes": {
            "sources": source_nodes,
            "concepts": list(concept_node_set.values()),
        },
        "edges": edges,
    }


def build_source_dependency_mermaid(bundle: SynthesisBundle) -> str:
    """Build a Mermaid diagram of source→concept contributions.

    Filters edges to concepts present in the canonical bundle and suppresses
    stale source-local concepts.
    """
    lines: list[str] = ["graph LR"]
    canonical_slugs = {concept.slug for concept in bundle.concepts}

    # Source nodes as a subgraph.
    source_ids: list[str] = []
    for synthesis in bundle.sources:
        raw_id = synthesis.source_file or synthesis.source_title
        if not raw_id:
            continue
        sid = _mermaid_safe_id(_slugify_source_id(raw_id))
        label = _mermaid_safe_label(synthesis.source_title or raw_id)
        lines.append(f'  {sid}[("{label}")]')
        source_ids.append(sid)

    # Concept nodes.
    for concept in bundle.concepts:
        cid = _mermaid_safe_id(concept.slug)
        label = _mermaid_safe_label(concept.title)
        lines.append(f'  {cid}["{label}"]')

    # Edges: source → contributed_to → concept.
    seen: set[tuple[str, str]] = set()
    for synthesis in bundle.sources:
        raw_id = synthesis.source_file or synthesis.source_title
        if not raw_id:
            continue
        sid = _mermaid_safe_id(_slugify_source_id(raw_id))
        for concept in synthesis.concepts:
            cid = _mermaid_safe_id(concept.slug)
            if concept.slug not in canonical_slugs:
                continue
            key = (sid, cid)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {sid} -->|contributed_to| {cid}")

    return "\n".join(lines) + "\n"


def _export_sharded_source_graph(
    graph: dict[str, Any],
    output_path: Path,
    *,
    shard_size: int,
    bundle_dir: Path,
) -> list[str]:
    """Write deterministic, bounded source graph node/edge shards plus manifest."""
    from obsidian_llm_wiki.render.obsidian import (
        _delete_generated_artifact,
        _write_generated_artifact,
    )

    def _chunks(items):
        while chunk := list(islice(items, shard_size)):
            yield chunk

    def _bucket(entity_id: str) -> int:
        digest = hashlib.blake2b(
            entity_id.encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") % 64

    output_dir = output_path.parent / f"{output_path.stem}-shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    keep: set[Path] = set()

    nodes = graph.get("nodes", {})
    node_records: list[dict[str, Any]] = []
    for node in nodes.get("sources", []):
        node_records.append({"kind": "source", **node})
    for node in nodes.get("concepts", []):
        node_records.append({"kind": "concept", **node})

    node_buckets: dict[int, list[dict[str, Any]]] = {}
    for node in node_records:
        node_buckets.setdefault(_bucket(node["id"]), []).append(node)
    node_shards: list[dict[str, Any]] = []
    for bucket in sorted(node_buckets):
        records = sorted(node_buckets[bucket], key=lambda item: str(item["id"]))
        for part, shard_nodes in enumerate(_chunks(iter(records))):
            path = output_dir / f"nodes-b{bucket:03d}-{part:03d}.json"
            content = json.dumps({"kind": "nodes", "nodes": shard_nodes}, ensure_ascii=False)
            if _write_generated_artifact(path, content, bundle_dir):
                written.append(str(path))
            keep.add(path)
            node_shards.append({"path": path.name, "count": len(shard_nodes)})

    edge_buckets: dict[int, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []):
        edge_buckets.setdefault(_bucket(str(edge["source"])), []).append(edge)
    edge_shards: list[dict[str, Any]] = []
    for bucket in sorted(edge_buckets):
        records = sorted(
            edge_buckets[bucket],
            key=lambda item: (
                str(item["source"]),
                str(item["target"]),
                str(item.get("relation", "")),
            ),
        )
        for part, shard_edges in enumerate(_chunks(iter(records))):
            path = output_dir / f"edges-b{bucket:03d}-{part:03d}.json"
            content = json.dumps({"kind": "edges", "edges": shard_edges}, ensure_ascii=False)
            if _write_generated_artifact(path, content, bundle_dir):
                written.append(str(path))
            keep.add(path)
            edge_shards.append({"path": path.name, "count": len(shard_edges)})

    manifest_path = output_dir / "manifest.json"
    manifest_content = json.dumps(
        {
            "kind": "source-dependency-graph",
            "node_shards": node_shards,
            "edge_shards": edge_shards,
            "node_count": len(node_records),
            "edge_count": len(graph.get("edges", [])),
        },
        ensure_ascii=False,
    )
    if _write_generated_artifact(manifest_path, manifest_content, bundle_dir):
        written.append(str(manifest_path))
    keep.add(manifest_path)

    for stale in output_dir.glob("*.json"):
        if stale not in keep:
            _delete_generated_artifact(stale, bundle_dir)

    return written
