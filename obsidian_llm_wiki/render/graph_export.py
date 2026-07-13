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
