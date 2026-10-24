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

from obsidian_llm_wiki.core.generated_paths import checked_generated_path, is_graph_shard_basename
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
    checked_generated_path(output_dir, output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    from obsidian_llm_wiki.render.obsidian import _write_generated_artifact

    # JSON
    json_path = output_dir / SOURCE_GRAPH_JSON
    stale_shards = _previous_source_graph_shards(json_path)
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
        from obsidian_llm_wiki.render.obsidian import _delete_generated_artifact

        for stale in sorted(stale_shards):
            if _delete_generated_artifact(stale, output_dir.parent):
                written.append(str(stale))
    else:
        written.extend(
            _export_sharded_source_graph(
                graph,
                json_path,
                shard_size=shard_size,
                bundle_dir=output_dir.parent,
                stale_shards=stale_shards,
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
    """Canonical source/concept identities and validated contribution edges."""
    from obsidian_llm_wiki.render.graph_export import _graph_ids
    ids = _graph_ids(bundle)
    canonical = {c.slug: c for c in bundle.concepts}
    source_nodes = []
    concepts = {}
    edges = []
    for synthesis in bundle.sources:
        raw = synthesis.source_file or synthesis.source_title
        if not raw:
            continue
        source_id = ids[("source", raw)]
        slugs = list(dict.fromkeys(c.slug for c in synthesis.concepts if c.slug in canonical))
        source_nodes.append({"id": source_id, "label": synthesis.source_title or raw,
                             "source_file": synthesis.source_file, "concept_count": len(slugs)})
        for slug in slugs:
            concept = canonical[slug]
            cid = ids[("concept", slug)]
            concepts[cid] = {"id": cid, "label": concept.title, "confidence": concept.confidence}
            edges.append({"source": source_id, "target": cid, "type": "contributed_to"})
    return {"metadata": {"source_count": len(source_nodes), "concept_count": len(concepts),
                         "edge_count": len(edges)},
            "nodes": {"sources": source_nodes, "concepts": list(concepts.values())},
            "edges": edges}


def build_source_dependency_mermaid(bundle: SynthesisBundle) -> str:
    """Bounded, escaped source graph with IDs shared with JSON."""
    from obsidian_llm_wiki.render.graph_export import _mermaid_safe_id, _mermaid_safe_label
    if len(bundle.sources) + len(bundle.concepts) > 120:
        return 'graph LR\n  summary["Source graph exceeds 120 nodes; use JSON export"]\n'
    graph = build_source_dependency_dict(bundle)
    if len(graph["edges"]) > 240:
        return 'graph LR\n  summary["Source graph exceeds 240 edges; use JSON export"]\n'
    lines = ["graph LR"]
    for node in graph["nodes"]["sources"]:
        lines.append(f'  {_mermaid_safe_id(node["id"])}[("{_mermaid_safe_label(node["label"])}")]')
    for node in graph["nodes"]["concepts"]:
        lines.append(f'  {_mermaid_safe_id(node["id"])}["{_mermaid_safe_label(node["label"])}"]')
    for edge in graph["edges"]:
        sid, cid = (_mermaid_safe_id(edge[key]) for key in ("source", "target"))
        lines.append(f'  {sid} -->|contributed_to| {cid}')
    return "\n".join(lines) + "\n"


def _export_sharded_source_graph(
    graph: dict[str, Any],
    output_path: Path,
    *,
    shard_size: int,
    bundle_dir: Path,
    stale_shards: set[Path],
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
    checked_generated_path(output_dir, bundle_dir)
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

    summary_path = output_path.with_suffix(".mmd")
    summary = (
        "graph LR\n"
        f'  source_graph_manifest["Sharded source graph: {len(node_records)} nodes / '
        f'{len(graph.get("edges", []))} edges"]\n'
    )
    if _write_generated_artifact(summary_path, summary, bundle_dir):
        written.append(str(summary_path))
    if _delete_generated_artifact(output_path, bundle_dir):
        written.append(str(output_path))

    for stale in sorted(stale_shards):
        if stale not in keep and _delete_generated_artifact(stale, bundle_dir):
            written.append(str(stale))

    return written


def _previous_source_graph_shards(output_path: Path) -> set[Path]:
    """Load prior shard paths from the manifest, with one legacy fallback scan."""
    output_dir = output_path.parent / f"{output_path.stem}-shards"
    manifest_path = output_dir / "manifest.json"
    checked_generated_path(manifest_path, output_path.parent.parent)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            records: list[dict[str, Any]] = []
            for key in ("node_shards", "edge_shards"):
                raw_records = manifest.get(key, [])
                if isinstance(raw_records, list):
                    records.extend(
                        record for record in raw_records if isinstance(record, dict)
                    )
            return {
                output_dir / str(record["path"])
                for record in records
                if is_graph_shard_basename(record.get("path"))
            } | {manifest_path}
    if not output_dir.is_dir():
        return set()
    return {path for path in output_dir.glob("*.json")
            if path.name == "manifest.json" or is_graph_shard_basename(path.name)}
