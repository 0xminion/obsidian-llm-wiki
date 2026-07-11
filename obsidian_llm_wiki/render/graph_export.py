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

import json
import logging
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import SynthesisBundle

logger = logging.getLogger("obswiki.render.graph_export")

__all__ = [
    "export_graph_json",
    "export_graph_mermaid",
    "export_graph",
]


def export_graph_json(bundle: SynthesisBundle, output_path: Path) -> None:
    """Export the knowledge graph as JSON.

    Produces a structure with ``nodes``, ``edges``, and ``mocs`` arrays,
    compatible with Obsidian's graph view and D3.js force-directed layouts.
    """
    graph = _build_graph_dict(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Graph JSON exported: %d nodes, %d edges, %d MoCs → %s",
        len(graph["nodes"]),
        len(graph["edges"]),
        len(graph["mocs"]),
        output_path,
    )


def export_graph_mermaid(bundle: SynthesisBundle, output_path: Path) -> None:
    """Export the knowledge graph as a Mermaid diagram.

    Produces a ``graph LR`` Mermaid diagram showing concept-to-concept
    relationships with typed edge labels. Suitable for Obsidian embedding
    via ```mermaid code blocks.
    """
    mermaid = _build_mermaid(bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mermaid, encoding="utf-8")
    logger.info("Graph Mermaid exported → %s", output_path)


def export_graph(bundle: SynthesisBundle, output_dir: Path) -> None:
    """Export both JSON and Mermaid graph files to ``output_dir``.

    Convenience function that writes ``graph.json`` and ``graph.mmd``.
    """
    export_graph_json(bundle, output_dir / "graph.json")
    export_graph_mermaid(bundle, output_dir / "graph.mmd")


# ── Internal builders ───────────────────────────────────────────────────


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

    # Nodes: sources
    for source in bundle.sources:
        source_slug = source.source_file or source.source_title
        nodes.append({
            "id": source_slug,
            "label": source.source_title,
            "type": "source",
            "tags": list(source.source_tags),
            "confidence": 1.0,
            "moc": "",
        })

    # Edges: concept-to-concept (from related links)
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for concept in bundle.concepts:
        for link in concept.related:
            edge_key = (concept.slug, link.slug, link.relation)
            reverse_key = (link.slug, concept.slug, link.relation)
            if edge_key in seen_edges or reverse_key in seen_edges:
                # Edge already added — mark as bidirectional.
                for e in edges:
                    if (
                        e["source"] == link.slug
                        and e["target"] == concept.slug
                        and e["relation"] == link.relation
                    ):
                        e["bidirectional"] = True
                        break
                continue

            # Check if reverse edge exists.
            target_concept = next(
                (c for c in bundle.concepts if c.slug == link.slug), None
            )
            has_reverse = False
            if target_concept:
                has_reverse = any(
                    r.slug == concept.slug for r in target_concept.related
                )

            edges.append({
                "source": concept.slug,
                "target": link.slug,
                "relation": link.relation,
                "bidirectional": has_reverse,
            })
            seen_edges.add(edge_key)

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

    # MoC nodes with subgraph grouping
    for moc in bundle.maps:
        lines.append(f"  subgraph {moc.slug.replace('-', '_')}[" + moc.title + "]")

        for slug in moc.concept_slugs:
            concept = next(
                (c for c in bundle.concepts if c.slug == slug), None
            )
            if concept:
                label = concept.title.replace('"', "'")
                lines.append(f"    {slug.replace('-', '_')}[\"{label}\"]")
            else:
                lines.append(f"    {slug.replace('-', '_')}[{slug}]")

        lines.append("  end")

    # Orphan concepts (not in any MoC) as standalone nodes
    moced_slugs: set[str] = set()
    for moc in bundle.maps:
        moced_slugs.update(moc.concept_slugs)

    for concept in bundle.concepts:
        if concept.slug not in moced_slugs:
            label = concept.title.replace('"', "'")
            lines.append(f"  {concept.slug.replace('-', '_')}[\"{label}\"]")

    # Edges
    seen_edges: set[tuple[str, str]] = set()
    for concept in bundle.concepts:
        for link in concept.related:
            source_id = concept.slug.replace("-", "_")
            target_id = link.slug.replace("-", "_")
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
