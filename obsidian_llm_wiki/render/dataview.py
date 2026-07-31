"""Dataview / Obsidian Bases view generators.

Produces markdown files containing Dataview and Bases queries that render
as live views inside Obsidian (when the Dataview plugin is installed) or as
static fallback tables otherwise.

Views generated:
  * ``views/concepts-by-confidence.md`` — concepts sorted by confidence.
  * ``views/mocs-by-count.md``         — MoCs sorted by concept count.
  * ``views/contradictions-by-status.md`` — contradiction records by status.
  * ``views/sources-by-freshness.md``  — sources sorted by retrieval time.

All views are deterministic — no LLM calls.  The SynthesisBundle (and an
optional ContradictionStore) are the single inputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import SynthesisBundle

logger = logging.getLogger("obswiki.render.dataview")

__all__ = [
    "render_dataview_views",
    "render_concepts_by_confidence_view",
    "render_mocs_by_count_view",
    "render_contradictions_by_status_view",
    "render_sources_by_freshness_view",
]

VIEWS_DIR = "views"


def _view_document(title: str, body: str) -> str:
    """Wrap a generated view in ordinary Obsidian frontmatter."""
    return f"---\ntype: View\ntitle: {title}\n---\n\n{body}"


# ── Individual view renderers ──────────────────────────────────────────


def render_concepts_by_confidence_view(bundle: SynthesisBundle) -> str:
    """Concepts sorted by confidence (descending).

    Includes both a Dataview block (live, for plugin users) and a static
    fallback markdown table (visible without any plugins).
    """
    lines: list[str] = [
        "# Concepts by Confidence",
        "",
        "> Concepts sorted from highest to lowest confidence.",
        "",
        "## Dataview",
        "",
        "```dataview",
        "TABLE",
        "  confidence AS \"Confidence\",",
        "  tags AS \"Tags\"",
        'FROM "concepts"',
        "WHERE type = \"Concept\"",
        "SORT confidence DESC",
        "```",
        "",
        "## Static View",
        "",
        "| Concept | Confidence | Tags |",
        "|---------|------------|------|",
    ]
    for c in sorted(bundle.concepts, key=lambda x: (-x.confidence, x.slug)):
        tags = ", ".join(c.tags) if c.tags else "—"
        lines.append(f"| [[{c.slug}|{c.title}]] | {c.confidence:.2f} | {tags} |")
    lines.append("")
    return "\n".join(lines)


def render_mocs_by_count_view(bundle: SynthesisBundle) -> str:
    """MoCs sorted by concept count (descending)."""
    lines: list[str] = [
        "# Maps of Content by Concept Count",
        "",
        "> MoCs sorted by number of contained concepts.",
        "",
        "## Dataview",
        "",
        "```dataview",
        "TABLE",
        "  length(concept_slugs) AS \"Concepts\",",
        "  tags AS \"Tags\"",
        'FROM "mocs"',
        "WHERE type = \"Map of Content\"",
        "SORT length(concept_slugs) DESC",
        "```",
        "",
        "## Static View",
        "",
        "| MoC | Concepts | Tags |",
        "|-----|----------|------|",
    ]
    for m in sorted(bundle.maps, key=lambda x: (-len(x.concept_slugs), x.slug)):
        tags = ", ".join(m.tags) if m.tags else "—"
        lines.append(
            f"| [[{m.slug}|{m.title}]] | {len(m.concept_slugs)} | {tags} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_contradictions_by_status_view(
    contradictions_path: Path | None = None,
) -> str:
    """Contradiction records grouped by status.

    Reads ``contradictions.json`` from the ``.llmwiki`` directory.  When the
    file does not exist (no contradictions have been detected), the view is
    still generated with an empty-state message.
    """
    lines: list[str] = [
        "# Contradictions by Status",
        "",
        "> Detected factual contradictions grouped by lifecycle status.",
        "",
        "## Dataview",
        "",
        "```dataview",
        "TABLE",
        "  status AS \"Status\",",
        "  summary AS \"Summary\",",
        "  sources AS \"Sources\"",
        'FROM "contradictions"',
        "SORT status ASC",
        "```",
        "",
        "## Static View",
        "",
    ]
    records = _load_contradiction_records(contradictions_path)
    if not records:
        lines.extend(
            ["*No contradictions detected yet.*", ""]
        )
        return "\n".join(lines)

    # Group by status.
    by_status: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_status.setdefault(r.get("status", "detected"), []).append(r)
    for status in sorted(by_status):
        lines.append(f"### {status}")
        lines.append("")
        lines.append("| ID | Summary |")
        lines.append("|----|---------|")
        for r in by_status[status]:
            lines.append(f"| {r.get('id', '—')} | {r.get('summary', '—')} |")
        lines.append("")
    return "\n".join(lines)


def render_sources_by_freshness_view(
    sources: dict[str, Any],
    bundle: SynthesisBundle,
) -> str:
    """Sources sorted by freshness (retrieval timestamp, newest first).

    Uses the ``provenance.retrieved_at`` field from ``SourceDoc`` when
    available, falling back to the frontmatter ``timestamp`` field of the
    rendered source page.  When neither is present, the source is listed
    last with an ``—`` marker.
    """
    lines: list[str] = [
        "# Sources by Freshness",
        "",
        "> Sources sorted by retrieval/recency timestamp (newest first).",
        "",
        "## Dataview",
        "",
        "```dataview",
        "TABLE",
        "  timestamp AS \"Generated\",",
        "  source_type AS \"Type\",",
        "  url AS \"URL\"",
        'FROM "sources"',
        "WHERE type = \"Source\"",
        "SORT timestamp DESC",
        "```",
        "",
        "## Static View",
        "",
        "| Source | Retrieved | Type | URL |",
        "|--------|-----------|------|-----|",
    ]
    # Build a lookup of source_title → provenance.retrieved_at.
    source_info: list[tuple[str, str, str, str, str]] = []
    for filename, doc in sources.items():
        retrieved = ""
        if hasattr(doc, "provenance") and doc.provenance.retrieved_at:
            retrieved = doc.provenance.retrieved_at
        source_type = getattr(doc, "source_type", "") or "—"
        url = getattr(doc, "url", "") or "—"
        stem = filename[:-3] if filename.endswith(".md") else filename
        source_info.append((doc.title, retrieved, source_type, url, stem))
    # Sort: retrieved desc, with missing values last.
    source_info.sort(key=lambda x: (x[1] or "", x[0]), reverse=True)
    for title, retrieved, source_type, url, stem in source_info:
        display_retrieved = retrieved or "—"
        lines.append(
            f"| [[{stem}|{title}]] | {display_retrieved} | {source_type} | {url} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Top-level orchestrator ──────────────────────────────────────────────


def _shard_static_rows(
    *,
    base_name: str,
    title: str,
    intro: list[str],
    table_header: list[str],
    rows: list[str],
    shard_size: int,
) -> tuple[str, dict[str, str]]:
    shards: dict[str, str] = {}
    links: list[str] = []
    for index, start in enumerate(range(0, len(rows), shard_size), 1):
        chunk = rows[start:start + shard_size]
        filename = f"{base_name}-{index:05d}.md"
        shard_title = f"{title} · {index}"
        shards[filename] = _view_document(
            shard_title,
            "\n".join([f"# {shard_title}", "", *table_header, *chunk, ""]),
        )
        links.append(
            f"- [[{Path(filename).stem}|Items {start + 1}–{start + len(chunk)}]]"
        )
    root = "\n".join(
        [
            f"# {title}",
            "",
            *intro,
            "",
            f"## Static View ({len(rows)} items)",
            "",
            *links,
            "",
        ]
    )
    return root, shards


def render_dataview_views(
    bundle_dir: Path,
    bundle: SynthesisBundle,
    sources: dict[str, Any],
    contradictions_path: Path | None = None,
    *,
    shard_size: int = 2_000,
) -> list[str]:
    """Generate all Dataview/Bases view files in ``bundle_dir/views/``.

    Args:
        bundle_dir: The wiki root directory (e.g. vault/04-Wiki).
        bundle: The merged SynthesisBundle.
        sources: Dict mapping source filename → SourceDoc.
        contradictions_path: Optional path to ``contradictions.json``.
            When None, ``bundle_dir/.llmwiki/contradictions.json`` is used.

    Returns:
        List of file paths that were written.
    """
    views_dir = bundle_dir / VIEWS_DIR
    views_dir.mkdir(parents=True, exist_ok=True)

    if contradictions_path is None:
        contradictions_path = bundle_dir / ".llmwiki" / "contradictions.json"

    written: list[str] = []
    if shard_size < 1:
        raise ValueError("report shard_size must be positive")
    views: list[tuple[str, str]] = []
    shard_documents: dict[str, str] = {}

    try:
        if len(bundle.concepts) > shard_size:
            concept_rows = [
                f"| [[{concept.slug}|{concept.title}]] | {concept.confidence:.2f} | "
                f"{', '.join(concept.tags) or '—'} |"
                for concept in sorted(
                    bundle.concepts, key=lambda item: (-item.confidence, item.slug)
                )
            ]
            root, shards = _shard_static_rows(
                base_name="concepts-by-confidence",
                title="Concepts by Confidence",
                intro=[
                    "> Concepts sorted from highest to lowest confidence.",
                    "",
                    "```dataview",
                    'TABLE confidence AS "Confidence", tags AS "Tags"',
                    'FROM "concepts"',
                    "SORT confidence DESC",
                    "```",
                ],
                table_header=[
                    "| Concept | Confidence | Tags |",
                    "|---------|------------|------|",
                ],
                rows=concept_rows,
                shard_size=shard_size,
            )
            views.append(("concepts-by-confidence.md", root))
            shard_documents.update(shards)
        else:
            views.append((
                "concepts-by-confidence.md",
                render_concepts_by_confidence_view(bundle),
            ))
        if len(bundle.maps) > shard_size:
            moc_rows = [
                f"| [[{moc.slug}|{moc.title}]] | {len(moc.concept_slugs)} | "
                f"{', '.join(moc.tags) or '—'} |"
                for moc in sorted(
                    bundle.maps, key=lambda item: (-len(item.concept_slugs), item.slug)
                )
            ]
            root, shards = _shard_static_rows(
                base_name="mocs-by-count",
                title="Maps of Content by Concept Count",
                intro=["> MoCs sorted by number of contained concepts."],
                table_header=[
                    "| MoC | Concepts | Tags |",
                    "|-----|----------|------|",
                ],
                rows=moc_rows,
                shard_size=shard_size,
            )
            views.append(("mocs-by-count.md", root))
            shard_documents.update(shards)
        else:
            views.append((
                "mocs-by-count.md",
                render_mocs_by_count_view(bundle),
            ))
        views.append((
            "contradictions-by-status.md",
            render_contradictions_by_status_view(contradictions_path),
        ))
        if len(sources) > shard_size:
            source_rows: list[tuple[str, str]] = []
            for filename, source in sources.items():
                retrieved = getattr(source.provenance, "retrieved_at", "")
                source_type = getattr(source, "source_type", "") or "—"
                url = getattr(source, "url", "") or "—"
                stem = filename.removesuffix(".md")
                row = (
                    f"| [[{stem}|{source.title}]] | {retrieved or '—'} | "
                    f"{source_type} | {url} |"
                )
                source_rows.append((retrieved, row))
            root, shards = _shard_static_rows(
                base_name="sources-by-freshness",
                title="Sources by Freshness",
                intro=["> Sources sorted by retrieval/recency timestamp."],
                table_header=[
                    "| Source | Retrieved | Type | URL |",
                    "|--------|-----------|------|-----|",
                ],
                rows=[row for _retrieved, row in sorted(source_rows, reverse=True)],
                shard_size=shard_size,
            )
            views.append(("sources-by-freshness.md", root))
            shard_documents.update(shards)
        else:
            views.append((
                "sources-by-freshness.md",
                render_sources_by_freshness_view(sources, bundle),
            ))
    except Exception as exc:
        logger.debug("Dataview views skipped: %s", exc)
        return written

    from obsidian_llm_wiki.render.obsidian import _write_generated_artifact

    for filename, content in views:
        path = views_dir / filename
        title = content.splitlines()[0].removeprefix("# ")
        if _write_generated_artifact(
            path, _view_document(title, content), bundle_dir
        ):
            written.append(str(path))

    for filename, content in sorted(shard_documents.items()):
        path = views_dir / filename
        if _write_generated_artifact(path, content, bundle_dir):
            written.append(str(path))

    shard_prefixes = (
        "concepts-by-confidence-",
        "mocs-by-count-",
        "sources-by-freshness-",
    )
    for path in views_dir.glob("*.md"):
        if path.name in shard_documents:
            continue
        if any(path.name.startswith(prefix) for prefix in shard_prefixes):
            from obsidian_llm_wiki.render.obsidian import _delete_generated_artifact

            if _delete_generated_artifact(path, bundle_dir):
                written.append(str(path))

    # Write a views index.
    idx = views_dir / "index.md"
    idx_lines: list[str] = [
        "# Dataview Views",
        "",
        "> Live views powered by the Dataview plugin / Obsidian Bases.",
        "",
        "- [[concepts-by-confidence|Concepts by Confidence]]",
        "- [[mocs-by-count|MoCs by Concept Count]]",
        "- [[contradictions-by-status|Contradictions by Status]]",
        "- [[sources-by-freshness|Sources by Freshness]]",
        "",
    ]
    if _write_generated_artifact(
        idx,
        _view_document("Dataview Views", "\n".join(idx_lines)),
        bundle_dir,
    ):
        written.append(str(idx))

    logger.info("Dataview views generated: %d files → %s", len(written), views_dir)
    return written


# ── Helpers ────────────────────────────────────────────────────────────


def _load_contradiction_records(
    path: Path | None,
) -> list[dict[str, Any]]:
    """Load contradiction records from a JSON store file.

    Returns an empty list when the file is missing or unreadable.
    """
    if path is None or not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not load contradiction store %s: %s", path, exc)
        return []
    if not isinstance(raw, dict):
        return []
    return raw.get("records", [])
