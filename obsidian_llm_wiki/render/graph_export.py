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

from obsidian_llm_wiki.core.generated_paths import checked_generated_path, is_graph_shard_basename
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
    checked_generated_path(output_path, output_path.parent.parent)
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
    checked_generated_path(output_path, output_path.parent.parent)
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
    checked_generated_path(output_dir, output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_shards = _previous_graph_shards(output_dir)
    total_nodes = len(bundle.concepts) + len(bundle.maps) + len(bundle.sources)
    total_edges = sum(1 for _ in _iter_graph_edges(bundle))
    if total_nodes + total_edges <= monolith_threshold:
        written: list[str] = []
        graph_json = output_dir / "graph.json"
        graph_mmd = output_dir / "graph.mmd"
        if export_graph_json(bundle, graph_json):
            written.append(str(graph_json))
        if export_graph_mermaid(bundle, graph_mmd):
            written.append(str(graph_mmd))
        _remove_stale_shards(output_dir, keep=set(), stale=stale_shards)
        return written
    return _export_sharded_graph(
        bundle,
        output_dir,
        shard_size=shard_size,
        stale_shards=stale_shards,
    )


def _iter_graph_nodes(bundle: SynthesisBundle) -> Iterator[dict[str, Any]]:
    ids = _graph_ids(bundle)
    live = {c.slug for c in bundle.concepts}
    concept_to_moc = {slug: ids[("moc", moc.slug)] for moc in bundle.maps
                      for slug in moc.concept_slugs if slug in live}
    for concept in sorted(bundle.concepts, key=lambda item: item.slug):
        yield {"id": ids[("concept", concept.slug)], "label": concept.title,
               "type": "concept", "tags": list(concept.tags),
               "confidence": concept.confidence, "moc": concept_to_moc.get(concept.slug, "")}
    for moc in sorted(bundle.maps, key=lambda item: item.slug):
        yield {"id": ids[("moc", moc.slug)], "label": moc.title, "type": "moc",
               "tags": list(moc.tags), "confidence": 1.0, "moc": "",
               "concept_count": len(set(moc.concept_slugs) & live)}
    for source in sorted(bundle.sources, key=lambda item: item.source_file or item.source_title):
        raw = source.source_file or source.source_title
        yield {"id": ids[("source", raw)], "label": source.source_title,
               "type": "source", "tags": list(source.source_tags), "confidence": 1.0, "moc": ""}


def _iter_graph_edges(bundle: SynthesisBundle) -> Iterator[dict[str, Any]]:
    ids = _graph_ids(bundle)
    concept_by_slug = {concept.slug: concept for concept in bundle.concepts}
    symmetric = {"related_to", "contrasts_with", "complements", "competes_with", "equivalent_to"}
    # A sparse hub has many incoming edges. Scanning its adjacency for each
    # reverse lookup makes that linear-size graph quadratic to export.
    edge_keys = {(concept.slug, link.slug, link.relation)
                 for concept in bundle.concepts for link in concept.related}
    for concept in sorted(bundle.concepts, key=lambda item: item.slug):
        local_seen: set[tuple[str, str]] = set()
        for link in concept.related:
            local_key = (link.slug, link.relation)
            target = concept_by_slug.get(link.slug)
            if not target or link.slug == concept.slug or local_key in local_seen:
                continue
            local_seen.add(local_key)
            has_reverse = link.relation in symmetric and (
                link.slug, concept.slug, link.relation
            ) in edge_keys
            if has_reverse and concept.slug > link.slug:
                continue
            yield {"source": ids[("concept", concept.slug)], "target": ids[("concept", link.slug)],
                   "relation": link.relation, "bidirectional": has_reverse}
    for moc in sorted(bundle.maps, key=lambda item: item.slug):
        for slug in sorted(set(moc.concept_slugs) & concept_by_slug.keys()):
            yield {"source": ids[("moc", moc.slug)], "target": ids[("concept", slug)],
                   "relation": "contains", "bidirectional": False}


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
    stale_shards: set[Path],
) -> list[str]:
    from obsidian_llm_wiki.render.obsidian import (
        _delete_generated_artifact,
        _write_generated_artifact,
    )

    graph_dir = output_dir / "graph"
    checked_generated_path(graph_dir, output_dir.parent)
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
    _remove_stale_shards(output_dir, keep=keep, stale=stale_shards)
    return written


def _previous_graph_shards(output_dir: Path) -> set[Path]:
    """Read the previous manifest; glob only once for legacy migration."""
    graph_dir = output_dir / "graph"
    manifest_path = graph_dir / "manifest.json"
    checked_generated_path(manifest_path, output_dir.parent)
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            records: list[dict[str, Any]] = []
            for key in ("shards", "edge_shards"):
                raw_records = manifest.get(key, [])
                if isinstance(raw_records, list):
                    records.extend(
                        record for record in raw_records if isinstance(record, dict)
                    )
            return {
                graph_dir / str(record["path"])
                for record in records
                if is_graph_shard_basename(record.get("path"))
            } | {manifest_path}
    if not graph_dir.is_dir():
        return set()
    return {path for path in graph_dir.glob("*.json")
            if path.name == "manifest.json" or is_graph_shard_basename(path.name)}


def _remove_stale_shards(
    output_dir: Path,
    *,
    keep: set[Path],
    stale: set[Path],
) -> None:
    from obsidian_llm_wiki.render.obsidian import _delete_generated_artifact

    for path in sorted(stale):
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


def _graph_ids(bundle: SynthesisBundle) -> dict[tuple[str, str], str]:
    """Keep legacy IDs when unique; qualify every collision by type/identity."""
    records = [("concept", c.slug, c.slug) for c in bundle.concepts]
    records += [("moc", m.slug, m.slug) for m in bundle.maps]
    records += [("source", s.source_file or s.source_title,
                 _slugify_source_id(s.source_file or s.source_title)) for s in bundle.sources]
    counts: dict[str, int] = {}
    for _kind, _raw, base in records:
        counts[base] = counts.get(base, 0) + 1
    return {(kind, raw): base if counts[base] == 1 else f"{kind}:{raw}"
            for kind, raw, base in records}


def _mermaid_safe_id(slug: str) -> str:
    """Stable bounded IDs, including keywords, punctuation and Unicode."""
    return "n_" + hashlib.sha256(slug.encode("utf-8")).hexdigest()


def _mermaid_safe_label(text: str) -> str:
    """Quote-safe Mermaid entity text; labels cannot inject diagram syntax."""
    text = " ".join(str(text).split())[:240]
    for old, new in (("#", "#35;"), ("&", "#38;"), ('"', "#quot;"),
                     ("<", "#60;"), (">", "#62;"), ("|", "#124;"),
                     ("[", "#91;"), ("]", "#93;")):
        text = text.replace(old, new)
    return text


def _build_graph_dict(bundle: SynthesisBundle) -> dict[str, Any]:
    """The same validated graph records in both monolithic and sharded mode."""
    nodes = list(_iter_graph_nodes(bundle))
    return {"nodes": nodes, "edges": list(_iter_graph_edges(bundle)),
            "mocs": [{"id": n["id"], "label": n["label"], "concept_count": n["concept_count"]}
                     for n in nodes if n["type"] == "moc"]}


def _build_mermaid(bundle: SynthesisBundle) -> str:
    """Bounded escaped diagram over the same validated graph as JSON."""
    if len(bundle.concepts) + len(bundle.maps) + len(bundle.sources) > 120:
        return 'graph LR\n  summary["Graph exceeds 120 nodes; use JSON export"]\n'
    graph = _build_graph_dict(bundle)
    if len(graph["edges"]) > 240:
        return 'graph LR\n  summary["Graph exceeds 240 edges; use JSON export"]\n'
    ids = _graph_ids(bundle)
    lines = ["graph LR"]
    for moc in bundle.maps:
        mid = _mermaid_safe_id(ids[("moc", moc.slug)])
        lines.extend([f'  subgraph group_{mid}["{_mermaid_safe_label(moc.title)}"]',
                      f'    {mid}["{_mermaid_safe_label(moc.title)}"]', "  end"])
    for node in graph["nodes"]:
        if node["type"] != "moc":
            nid = _mermaid_safe_id(node["id"])
            lines.append(f'  {nid}["{_mermaid_safe_label(node["label"])}"]')
    for edge in graph["edges"]:
        source_id, target_id = (_mermaid_safe_id(edge[key]) for key in ("source", "target"))
        if edge["relation"] == "related_to":
            lines.append(f"  {source_id} --- {target_id}")
        else:
            label = _mermaid_safe_label(edge["relation"])
            lines.append(f'  {source_id} -->|"{label}"| {target_id}')
    return "\n".join(lines) + "\n"
