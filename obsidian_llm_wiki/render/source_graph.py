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

import json
import logging
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import SynthesisBundle

logger = logging.getLogger("obswiki.render.source_graph")

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

    # JSON
    json_path = output_dir / SOURCE_GRAPH_JSON
    graph = build_source_dependency_dict(bundle)
    json_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written.append(str(json_path))

    # Mermaid
    mmd_path = output_dir / SOURCE_GRAPH_MMD
    mmd_path.write_text(build_source_dependency_mermaid(bundle), encoding="utf-8")
    written.append(str(mmd_path))

    logger.info(
        "Source dependency graph exported: %d sources, %d concepts, %d edges → %s",
        graph["metadata"]["source_count"],
        graph["metadata"]["concept_count"],
        graph["metadata"]["edge_count"],
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

    for synthesis in bundle.sources:
        raw_id = synthesis.source_file or synthesis.source_title
        if not raw_id:
            continue
        source_id = _slugify_source_id(raw_id)
        concept_slugs = [c.slug for c in synthesis.concepts]
        source_nodes.append({
            "id": source_id,
            "label": synthesis.source_title or raw_id,
            "source_file": synthesis.source_file or "",
            "concept_count": len(concept_slugs),
        })
        for concept in synthesis.concepts:
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

    # Also include concepts in the bundle that weren't from any synthesis
    # (e.g. merged concepts) — they still appear as concept nodes.
    for concept in bundle.concepts:
        if concept.slug not in concept_node_set:
            concept_node_set[concept.slug] = {
                "id": concept.slug,
                "label": concept.title,
                "confidence": concept.confidence,
            }

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
    """Build a Mermaid diagram of source→concept contributions."""
    lines: list[str] = ["graph LR"]

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
            key = (sid, cid)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {sid} -->|contributed_to| {cid}")

    return "\n".join(lines) + "\n"
