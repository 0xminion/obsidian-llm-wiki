"""Graph visualization export — JSON and Mermaid formats.

Exports the knowledge graph from a SynthesisBundle as:
  - JSON (compatible with Obsidian graph view and D3.js)
  - Mermaid graph diagram (for Obsidian embedding)

JSON format:
    {
      "nodes": [
        {"id": "slug", "label": "title", "type": "concept|moc|source",
         "tags": [...], "confidence": 0.8, "moc": "moc-slug"}
      ],
      "edges": [
        {"source": "slug-a", "target": "slug-b",
         "relation": "depends_on", "bidirectional": true}
      ],
      "mocs": [
        {"id": "moc-slug", "label": "title", "concept_count": 5}
      ]
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import SynthesisBundle

logger = logging.getLogger("obswiki.render.graph_export")
_GRAPH_BUCKETS = 64

__all__ = [
    "export_graph_json",
    "export_graph_mermaid",
    "export_graph",
]


def export_graph_json(bundle: SynthesisBundle, output_path: Path) -> bool:
    """Export the knowledge graph as JSON.

    Produces a structure with ``nodes``, ``edges``, and ``mocs`` arrays,
    compatible with Obsidian's graph view and D3.js force-directed layouts.
    """
    graph = _build_graph_dict(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from obsidian_llm_wiki.render.obsidian import _write_generated_artifact

    changed = _write_generated_artifact(
        output_path,
        json.dumps(graph, indent=2, ensure_ascii=False),
        output_path.parent.parent,
    )
    logger.info(
        "Graph JSON exported: %d nodes, %d edges, %d MoCs → %s",
        len(graph["nodes"]),
        len(graph["edges"]),
        len(graph["mocs"]),
        output_path,
    )
    return changed


def export_graph_mermaid(bundle: SynthesisBundle, output_path: Path) -> bool:
    """Export the knowledge graph as a Mermaid diagram.

    Produces a ``graph LR`` Mermaid diagram showing concept-to-concept
    relationships with typed edge labels. Suitable for Obsidian embedding
    via ```mermaid code blocks.
    """
    mermaid = _build_mermaid(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from obsidian_llm_wiki.render.obsidian import _write_generated_artifact

    changed = _write_generated_artifact(
        output_path, mermaid, output_path.parent.parent
    )
    logger.info("Graph Mermaid exported → %s", output_path)
    return changed


def export_graph(
    bundle: SynthesisBundle,
    output_dir: Path,
    *,
    shard_size: int = 5_000,
    monolith_threshold: int = 10_000,
) -> list[str]:
    """Export both JSON and Mermaid graph files to ``output_dir``.

    Convenience function that writes ``graph.json`` and ``graph.mmd``.
    """
    if shard_size < 1 or monolith_threshold < 1:
        raise ValueError("graph shard limits must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    total_nodes = len(bundle.concepts) + len(bundle.maps) + len(bundle.sources)
    if total_nodes <= monolith_threshold:
        written: list[str] = []
        graph_json = output_dir / "graph.json"
        graph_mmd = output_dir / "graph.mmd"
        if export_graph_json(bundle, graph_json):
            written.append(str(graph_json))
        if export_graph_mermaid(bundle, graph_mmd):
            written.append(str(graph_mmd))
        _remove_stale_shards(output_dir, keep=set())
        return written
    return _export_sharded_graph(bundle, output_dir, shard_size=shard_size)


def _iter_graph_nodes(bundle: SynthesisBundle) -> Iterator[dict[str, Any]]:
    concept_to_moc = {
        slug: moc.slug for moc in bundle.maps for slug in moc.concept_slugs
    }
    for concept in sorted(bundle.concepts, key=lambda item: item.slug):
        yield {
            "id": concept.slug,
            "label": concept.title,
            "type": "concept",
            "tags": list(concept.tags),
            "confidence": concept.confidence,
            "moc": concept_to_moc.get(concept.slug, ""),
        }
    for moc in sorted(bundle.maps, key=lambda item: item.slug):
        yield {
            "id": moc.slug,
            "label": moc.title,
            "type": "moc",
            "tags": list(moc.tags),
            "confidence": 1.0,
            "moc": "",
            "concept_count": len(moc.concept_slugs),
        }
    for source in sorted(
        bundle.sources, key=lambda item: item.source_file or item.source_title
    ):
        raw_id = source.source_file or source.source_title
        yield {
            "id": _slugify_source_id(raw_id),
            "label": source.source_title,
            "type": "source",
            "tags": list(source.source_tags),
            "confidence": 1.0,
            "moc": "",
        }


def _iter_graph_edges(bundle: SynthesisBundle) -> Iterator[dict[str, Any]]:
    concept_by_slug = {concept.slug: concept for concept in bundle.concepts}
    for concept in sorted(bundle.concepts, key=lambda item: item.slug):
        local_seen: set[tuple[str, str]] = set()
        for link in concept.related:
            local_key = (link.slug, link.relation)
            if local_key in local_seen:
                continue
            local_seen.add(local_key)
            target = concept_by_slug.get(link.slug)
            has_reverse = bool(
                target
                and any(
                    reverse.slug == concept.slug and reverse.relation == link.relation
                    for reverse in target.related
                )
            )
            if has_reverse and concept.slug > link.slug:
                continue
            yield {
                "source": concept.slug,
                "target": link.slug,
                "relation": link.relation,
                "bidirectional": has_reverse,
            }
    for moc in sorted(bundle.maps, key=lambda item: item.slug):
        for slug in sorted(set(moc.concept_slugs)):
            yield {
                "source": moc.slug,
                "target": slug,
                "relation": "contains",
                "bidirectional": False,
            }


def _chunks(items: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    while chunk := list(islice(items, size)):
        yield chunk


def _graph_bucket(entity_id: str) -> int:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _GRAPH_BUCKETS


def _export_sharded_graph(
    bundle: SynthesisBundle,
    output_dir: Path,
    *,
    shard_size: int,
) -> list[str]:
    from obsidian_llm_wiki.render.obsidian import (
        _delete_generated_artifact,
        _write_generated_artifact,
    )

    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    keep: set[Path] = set()
    node_buckets: dict[int, list[dict[str, Any]]] = {}
    for node in _iter_graph_nodes(bundle):
        node_buckets.setdefault(_graph_bucket(str(node["id"])), []).append(node)
    node_shards: list[dict[str, Any]] = []
    node_count = 0
    for bucket in sorted(node_buckets):
        records = sorted(node_buckets[bucket], key=lambda item: str(item["id"]))
        for part, nodes in enumerate(_chunks(iter(records), shard_size)):
            path = graph_dir / f"nodes-b{bucket:03d}-{part:03d}.json"
            content = json.dumps({"kind": "nodes", "nodes": nodes}, ensure_ascii=False)
            if _write_generated_artifact(path, content, output_dir.parent):
                written.append(str(path))
            keep.add(path)
            node_count += len(nodes)
            node_shards.append({"path": path.name, "count": len(nodes)})

    edge_buckets: dict[int, list[dict[str, Any]]] = {}
    for edge in _iter_graph_edges(bundle):
        edge_buckets.setdefault(_graph_bucket(str(edge["source"])), []).append(edge)
    edge_shards: list[dict[str, Any]] = []
    edge_count = 0
    for bucket in sorted(edge_buckets):
        records = sorted(
            edge_buckets[bucket],
            key=lambda item: (
                str(item["source"]),
                str(item["target"]),
                str(item["relation"]),
            ),
        )
        for part, edges in enumerate(_chunks(iter(records), shard_size)):
            path = graph_dir / f"edges-b{bucket:03d}-{part:03d}.json"
            content = json.dumps({"kind": "edges", "edges": edges}, ensure_ascii=False)
            if _write_generated_artifact(path, content, output_dir.parent):
                written.append(str(path))
            keep.add(path)
            edge_count += len(edges)
            edge_shards.append({"path": path.name, "count": len(edges)})

    manifest_path = graph_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "node_count": node_count,
        "edge_count": edge_count,
        "shard_size": shard_size,
        "shards": node_shards,
        "edge_shards": edge_shards,
    }
    if _write_generated_artifact(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False),
        output_dir.parent,
    ):
        written.append(str(manifest_path))
    keep.add(manifest_path)

    summary_path = output_dir / "graph.mmd"
    summary = (
        "graph LR\n"
        f'  graph_manifest["Sharded graph: {node_count} nodes / {edge_count} edges"]\n'
    )
    if _write_generated_artifact(summary_path, summary, output_dir.parent):
        written.append(str(summary_path))
    keep.add(summary_path)
    monolith = output_dir / "graph.json"
    if _delete_generated_artifact(monolith, output_dir.parent):
        written.append(str(monolith))
    _remove_stale_shards(output_dir, keep=keep)
    return written


def _remove_stale_shards(output_dir: Path, *, keep: set[Path]) -> None:
    from obsidian_llm_wiki.render.obsidian import _delete_generated_artifact

    graph_dir = output_dir / "graph"
    if not graph_dir.is_dir():
        return
    for path in graph_dir.glob("*.json"):
        if path not in keep:
            _delete_generated_artifact(path, output_dir.parent)


# ── Internal builders ───────────────────────────────────────────────────


def _slugify_source_id(raw_id: str) -> str:
    """Slugify a source file or title for use as a graph node ID.

    Uses the shared slugify so graph node IDs stay consistent with the
    wikilink slugs the renderer emits. Only a trailing .md is stripped —
    a blanket replace would mangle names like ``notes.mdx``.
    """
    from obsidian_llm_wiki.render.frontmatter import slugify

    return slugify(raw_id.removesuffix(".md"))


def _mermaid_safe_id(slug: str) -> str:
    """Convert a slug to a Mermaid-safe node identifier."""
    return slug.replace("-", "_")


def _mermaid_safe_label(text: str) -> str:
    """Escape a label for Mermaid node display."""
    # Replace characters that break Mermaid syntax.
    text = text.replace('"', "'")
    # Brackets break subgraph/node syntax — replace with angle brackets.
    text = text.replace("[", "‹").replace("]", "›")
    return text


def _build_graph_dict(bundle: SynthesisBundle) -> dict[str, Any]:
    """Build the graph dict from a SynthesisBundle."""
    # Build MoC lookup: concept_slug → moc_slug
    concept_to_moc: dict[str, str] = {}
    for moc in bundle.maps:
        for slug in moc.concept_slugs:
            concept_to_moc[slug] = moc.slug

    # Nodes: concepts
    nodes: list[dict[str, Any]] = []
    for concept in bundle.concepts:
        nodes.append({
            "id": concept.slug,
            "label": concept.title,
            "type": "concept",
            "tags": list(concept.tags),
            "confidence": concept.confidence,
            "moc": concept_to_moc.get(concept.slug, ""),
        })

    # Nodes: MoCs
    for moc in bundle.maps:
        nodes.append({
            "id": moc.slug,
            "label": moc.title,
            "type": "moc",
            "tags": list(moc.tags),
            "confidence": 1.0,
            "moc": "",
        })

    # Nodes: sources — slugify the ID for consistency with wikilinks.
    for source in bundle.sources:
        raw_id = source.source_file or source.source_title
        source_slug = _slugify_source_id(raw_id)
        nodes.append({
            "id": source_slug,
            "label": source.source_title,
            "type": "source",
            "tags": list(source.source_tags),
            "confidence": 1.0,
            "moc": "",
        })

    # Edges: concept-to-concept (from related links).
    # Index lookups keep this O(concepts + edges) — a linear concept scan per
    # edge and a linear edge rescan per duplicate would make large corpora
    # quadratic.
    concept_by_slug = {c.slug: c for c in bundle.concepts}
    edges: list[dict[str, Any]] = []
    edge_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

    for concept in bundle.concepts:
        for link in concept.related:
            edge_key = (concept.slug, link.slug, link.relation)
            reverse_key = (link.slug, concept.slug, link.relation)
            if edge_key in edge_by_key:
                continue
            existing_reverse = edge_by_key.get(reverse_key)
            if existing_reverse is not None:
                # Edge already added in the other direction — mark it.
                existing_reverse["bidirectional"] = True
                continue

            # Check if reverse edge exists.
            target_concept = concept_by_slug.get(link.slug)
            has_reverse = False
            if target_concept:
                has_reverse = any(
                    r.slug == concept.slug for r in target_concept.related
                )

            edge = {
                "source": concept.slug,
                "target": link.slug,
                "relation": link.relation,
                "bidirectional": has_reverse,
            }
            edges.append(edge)
            edge_by_key[edge_key] = edge

    # Edges: MoC-to-concept (membership)
    for moc in bundle.maps:
        for slug in moc.concept_slugs:
            edges.append({
                "source": moc.slug,
                "target": slug,
                "relation": "contains",
                "bidirectional": False,
            })

    # MoC metadata
    mocs: list[dict[str, Any]] = [
        {
            "id": moc.slug,
            "label": moc.title,
            "concept_count": len(moc.concept_slugs),
        }
        for moc in bundle.maps
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "mocs": mocs,
    }


def _build_mermaid(bundle: SynthesisBundle) -> str:
    """Build a Mermaid graph diagram from a SynthesisBundle."""
    lines: list[str] = ["graph LR"]
    concept_by_slug = {c.slug: c for c in bundle.concepts}

    # MoC nodes with subgraph grouping
    for moc in bundle.maps:
        moc_id = _mermaid_safe_id(moc.slug)
        moc_label = _mermaid_safe_label(moc.title)
        lines.append(f"  subgraph {moc_id}[{moc_label}]")

        for slug in moc.concept_slugs:
            concept = concept_by_slug.get(slug)
            if concept:
                label = _mermaid_safe_label(concept.title)
                lines.append(f'    {_mermaid_safe_id(slug)}["{label}"]')
            else:
                lines.append(f"    {_mermaid_safe_id(slug)}[{_mermaid_safe_label(slug)}]")

        lines.append("  end")

    # Orphan concepts (not in any MoC) as standalone nodes
    moced_slugs: set[str] = set()
    for moc in bundle.maps:
        moced_slugs.update(moc.concept_slugs)

    for concept in bundle.concepts:
        if concept.slug not in moced_slugs:
            label = _mermaid_safe_label(concept.title)
            lines.append(f'  {_mermaid_safe_id(concept.slug)}["{label}"]')

    # Edges
    seen_edges: set[tuple[str, str]] = set()
    for concept in bundle.concepts:
        for link in concept.related:
            source_id = _mermaid_safe_id(concept.slug)
            target_id = _mermaid_safe_id(link.slug)
            edge_key = (source_id, target_id)
            reverse_key = (target_id, source_id)

            if edge_key in seen_edges or reverse_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            relation = link.relation or "related_to"
            # Mermaid edge with label
            if relation in ("related_to",):
                lines.append(f"  {source_id} --- {target_id}")
            else:
                lines.append(f'  {source_id} -->|{relation}| {target_id}')

    return "\n".join(lines) + "\n"
