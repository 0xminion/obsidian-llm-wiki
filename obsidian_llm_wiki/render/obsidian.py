"""Obsidian vault renderer — pure functions from SynthesisBundle to markdown.

Produces Obsidian-flavoured markdown with:
  * YAML frontmatter (type, title, tags, aliases, timestamp)
  * Wikilinks ([[slug]] and [[slug|alias]])
  * Per-directory structure (sources/, entries/, concepts/, mocs/)
  * Per-directory index.md and bundle-root index.md

All rendering is deterministic — no LLM calls.  The SynthesisBundle is the
single input; the output is a complete vault directory tree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    ConceptType,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)

# Shared helpers live in their own modules; the names are re-exported here
# (see __all__) because many callers import them from render.obsidian.
from obsidian_llm_wiki.render.bilingual import (
    ensure_english_first_bilingual as _ensure_english_first_bilingual,  # noqa: F401 — re-exported
)
from obsidian_llm_wiki.render.bilingual import (
    is_chinese as _is_chinese,
)
from obsidian_llm_wiki.render.bilingual import (
    moc_needs_bilingual_headings as _moc_needs_bilingual_headings,
)
from obsidian_llm_wiki.render.bilingual import (
    normalize_bilingual_titles_and_slugs as _normalize_bilingual_titles_and_slugs,
)
from obsidian_llm_wiki.render.crossrefs import (
    build_cross_ref_diagram as _build_cross_ref_diagram,
)
from obsidian_llm_wiki.render.crossrefs import (
    build_moc_cross_ref_diagram as _build_moc_cross_ref_diagram,
)
from obsidian_llm_wiki.render.frontmatter import (
    atomic_write,
    build_frontmatter,
    extract_links,
    make_wikilink,
    parse_frontmatter,
    safe_read_file,
    slugify,
)
from obsidian_llm_wiki.render.frontmatter import (
    sanitize_tag as _sanitize_tag,  # noqa: F401 — re-exported for callers
)
from obsidian_llm_wiki.render.frontmatter import (
    timestamp as _timestamp,
)

logger = logging.getLogger("obswiki.render.obsidian")

__all__ = [
    "render_vault",
    "render_entry_page",
    "render_concept_page",
    "render_moc_page",
    "render_source_page",
    "render_bundle_index",
    "render_directory_index",
    "build_frontmatter",
    "parse_frontmatter",
    "extract_links",
    "safe_read_file",
    "atomic_write",
    "slugify",
    "make_wikilink",
]


# ── Page renderers ──────────────────────────────────────────────────────


def render_source_page(source: SourceDoc, timestamp: str | None = None) -> str:
    """Render a Source-type page (raw content for provenance).

    Avoids duplicate headings: if the content already starts with the title
    (as a # heading or plain text), the body heading is skipped.
    """
    ts = timestamp or _timestamp()
    fm = {
        "type": ConceptType.SOURCE.value,
        "title": source.title,
        "url": source.url or "",
        "timestamp": ts,
    }
    content = source.content.strip()
    title_clean = source.title.strip()

    # Check if content already starts with the title as a heading or plain text
    starts_with_heading = content.startswith(f"# {title_clean}")
    starts_with_plain = content.startswith(title_clean)

    if starts_with_heading:
        # Content already has the heading — don't duplicate
        body = content
    elif starts_with_plain:
        # Content starts with title text (not as heading) — add heading, strip duplicate
        body = f"# {source.title}\n\n{content[len(title_clean):].lstrip()}"
    else:
        body = f"# {source.title}\n\n{content}"
    return f"{build_frontmatter(fm)}\n{body}"


def render_entry_page(
    synthesis: SourceSynthesis,
    source_slug: str,
    concept_slugs: list[str],
    timestamp: str | None = None,
) -> str:
    """Render an Entry-type page (synthesis of one source)."""
    ts = timestamp or _timestamp()
    fm = {
        "type": ConceptType.ENTRY.value,
        "title": synthesis.source_title,
        "tags": synthesis.source_tags,
        "timestamp": ts,
    }

    parts: list[str] = [f"# {synthesis.source_title}", ""]

    if synthesis.source_summary:
        parts.extend([synthesis.source_summary, ""])

    if synthesis.key_points:
        parts.extend(["## Key Findings", ""])
        for point in synthesis.key_points:
            parts.append(f"- {point}")
        parts.append("")

    if concept_slugs:
        parts.extend(["## Linked Concepts", ""])
        for slug in concept_slugs:
            parts.append(f"- {make_wikilink(slug)}")
        parts.append("")

    if synthesis.open_questions:
        parts.extend(["## Open Questions", ""])
        for q in synthesis.open_questions:
            parts.append(f"- {q}")
        parts.append("")

    parts.extend([
        "## Source",
        "",
        f"- {make_wikilink(source_slug, synthesis.source_title)}",
    ])

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"


def render_concept_page(
    concept: ConceptNote,
    timestamp: str | None = None,
    all_concepts: dict[str, ConceptNote] | None = None,
) -> str:
    """Render a Concept-type page (evergreen atomic note).

    Args:
        concept: The concept to render.
        timestamp: Optional ISO timestamp (defaults to now).
        all_concepts: Optional dict of slug→ConceptNote for cross-reference discovery.
            When provided, the page includes a '关联图谱 / Cross-References' section
            with typed edges discovered by comparing this concept's related slugs
            against the other direction (bidirectional edges inferred).
    """
    ts = timestamp or _timestamp()
    fm: dict[str, Any] = {
        "type": ConceptType.CONCEPT.value,
        "title": concept.title,
        "tags": concept.tags,
        "timestamp": ts,
        "confidence": concept.confidence,
        "provenance": concept.provenance,
    }
    if concept.aliases:
        fm["aliases"] = concept.aliases
    if concept.related:
        fm["relations"] = [
            {"target": r.slug, "type": r.relation, "display": r.display or r.slug}
            for r in concept.related
        ]

    parts: list[str] = [f"# {concept.title}", ""]

    if concept.summary:
        parts.extend([concept.summary, ""])

    for section in concept.sections:
        parts.extend([f"## {section.heading}", ""])
        if section.prose:
            parts.extend([section.prose, ""])
        elif section.points:
            for point in section.points:
                parts.append(f"- {point}")
            parts.append("")

    if concept.claims:
        parts.extend(["## Claims", ""])
        for claim in concept.claims:
            parts.append(f"- {claim.text}")
        parts.append("")

    if concept.related:
        parts.extend(["## Related Concepts", ""])
        for link in concept.related:
            display = link.display or link.slug
            parts.append(f"- {make_wikilink(link.slug, display)} — `{link.relation}`")
        parts.append("")

    # ── 关联图谱 / Cross-References ──────────────────────────────────
    # Typed-edge relationship graph. Shown when all_concepts is provided.
    # Renders inside a markdown code block (```text) for monospace display
    # with copy icon in Obsidian, matching the user's screenshot format.
    if all_concepts and concept.related:
        cross_ref_lines = _build_cross_ref_diagram(concept, all_concepts)
        if cross_ref_lines:
            parts.extend(["## Cross-References / 关联图谱", ""])
            parts.append("```text")
            parts.extend(cross_ref_lines)
            parts.append("```")
            parts.append("")

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"




def render_moc_page(
    moc: MapOfContent,
    timestamp: str | None = None,
    all_concepts: dict[str, ConceptNote] | None = None,
    cross_lingual_links: dict[str, list[tuple[str, float, str]]] | None = None,
) -> str:
    """Render a Map of Content page.

    Args:
        moc: The MOC to render.
        timestamp: Optional ISO timestamp.
        all_concepts: Optional slug→ConceptNote dict. When provided, the MOC
            displays concept language badges and cross-lingual aliases, grouping
            concepts that share the same semantic meaning across languages under
            a unified entry. Also enables 关联图谱 cross-references.
        cross_lingual_links: Optional dict from embedding.find_cross_lingual_links.
            Maps slug → list of (target_slug, score, display). Used to show
            cross-lingual concept pairs in the MoC.
    """
    ts = timestamp or _timestamp()
    fm = {
        "type": ConceptType.MOC.value,
        "title": moc.title,
        "tags": moc.tags,
        "timestamp": ts,
    }

    parts: list[str] = [f"# {moc.title}", ""]

    if moc.summary:
        parts.extend([moc.summary, ""])

    if moc.concept_slugs:
        concept_entries = [
            all_concepts.get(slug) if all_concepts else None
            for slug in moc.concept_slugs
        ]
        bilingual_headings = _moc_needs_bilingual_headings(moc, concept_entries)
        concepts_heading = "## Concepts / 概念" if bilingual_headings else "## Concepts"
        parts.extend([concepts_heading, ""])
        for slug in moc.concept_slugs:
            entry = all_concepts.get(slug) if all_concepts else None
            badge = ""
            definition = ""
            if entry:
                if entry.aliases:
                    # Show cross-lingual aliases as language badge
                    zh_alias = next(
                        (a for a in entry.aliases if _is_chinese(a)), None,
                    )
                    if zh_alias:
                        badge = f" · {zh_alias}"
                # Include brief definition from concept summary
                if entry.summary:
                    definition = f" — {entry.summary}"
            parts.append(f"- {make_wikilink(slug)}{badge}{definition}")
        parts.append("")

    # ── 关联图谱 / Cross-References in MoC ──────────────────────────
    # Show relationship diagram between concepts in this MoC
    if all_concepts and moc.concept_slugs:
        moc_concepts = [
            all_concepts[s] for s in moc.concept_slugs
            if s in all_concepts
        ]
        if len(moc_concepts) >= 2:
            diagram_lines = _build_moc_cross_ref_diagram(moc_concepts, all_concepts)
            if diagram_lines:
                parts.extend(["## Cross-References / 关联图谱", ""])
                parts.append("```text")
                parts.extend(diagram_lines)
                parts.append("```")
                parts.append("")

    # ── Cross-lingual links from embedding ──────────────────────────
    # Instead of a separate section, merge cross-lingual concepts into the
    # Concepts list so they appear as part of the same MoC umbrella.
    if cross_lingual_links and moc.concept_slugs:
        existing_slugs = set(moc.concept_slugs)
        added_slugs: list[str] = []
        for slug in moc.concept_slugs:
            if slug in cross_lingual_links:
                for target_slug, _score, display in cross_lingual_links[slug]:
                    if target_slug in existing_slugs:
                        continue  # Already in this MoC
                    existing_slugs.add(target_slug)
                    added_slugs.append(target_slug)
                    target_concept = all_concepts.get(target_slug) if all_concepts else None
                    badge = ""
                    if target_concept and target_concept.confidence < 0.5:
                        badge = " *(low confidence)*"
                    definition = ""
                    if target_concept:
                        definition = (
                            f" — {target_concept.summary}"
                            if target_concept.summary else ""
                        )
                    parts.append(
                        f"- {make_wikilink(target_slug, display)}"
                        f"{badge}{definition} *(cross-lingual link)*"
                    )
        # Don't add a separate section heading

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"





# ── Index renderers ─────────────────────────────────────────────────────


def render_directory_index(
    dir_name: str,
    md_files: list[Path],
    bundle_dir: Path,
) -> str:
    """Render a per-directory ``index.md`` listing all pages in the directory."""
    parts: list[str] = [f"# {dir_name.title()}", ""]

    for f in sorted(md_files, key=lambda p: p.name):
        if f.name in ("index.md", "log.md"):
            continue
        raw = safe_read_file(f)
        meta, _ = parse_frontmatter(raw)
        title = meta.get("title", f.stem)
        parts.append(f"- [[{f.stem}|{title}]]")

    parts.append("")
    return "\n".join(parts)


def render_bundle_index(
    bundle_dir: Path,
    concept_count: int,
    entry_count: int,
    moc_count: int,
    source_count: int,
) -> str:
    """Render the bundle-root ``index.md``."""
    parts: list[str] = [
        "# Knowledge Wiki",
        "",
        f"Generated: {_timestamp().split('T')[0]}",
        "",
        "## Overview",
        "",
        f"- **Sources**: {source_count}",
        f"- **Entries**: {entry_count}",
        f"- **Concepts**: {concept_count}",
        f"- **Maps of Content**: {moc_count}",
        "",
        "## Sections",
        "",
        "- [[sources/index|Sources]]",
        "- [[entries/index|Entries]]",
        "- [[concepts/index|Concepts]]",
        "- [[mocs/index|Maps of Content]]",
        "",
    ]
    return "\n".join(parts)


# ── Full vault renderer ─────────────────────────────────────────────────


def render_vault(
    bundle_dir: Path,
    bundle: SynthesisBundle,
    sources: dict[str, SourceDoc],
    config: Any = None,
) -> list[str]:
    """Render a complete vault from a SynthesisBundle.

    Args:
        bundle_dir: The wiki root directory (e.g. vault/04-Wiki).
        bundle: The merged SynthesisBundle from synth.dedupe.
        sources: Dict mapping source filename → SourceDoc.
        config: Optional pipeline config for threshold settings
            (similarity_dedup_threshold, moc_assignment_threshold).
            When None, defaults are used.

    Returns:
        List of file paths that were written.
    """
    written: list[str] = []
    ts = _timestamp()

    # ── Backlink propagation: ensure bidirectional edges ──────────────
    # This runs inside render_vault so it works even when re-rendering
    # from cache without going through the full pipeline. Semantic dedup
    # and MoC orphan assignment do NOT run here: they mutate the bundle
    # (merging concepts, rewriting slugs), so the pipeline runs them in the
    # synthesis stage — before rendering and before the state write — where
    # their failures are visible instead of swallowed.
    from obsidian_llm_wiki.synth.dedupe import propagate_backlinks
    propagate_backlinks(bundle)

    # Make the language policy deterministic. The synthesis prompt asks Chinese
    # sources to use English-first bilingual titles, but smaller/local models do
    # not always comply. Rendering is the last safe choke point before filenames
    # and wikilinks are written, so normalize here and remap slugs consistently.
    _normalize_bilingual_titles_and_slugs(bundle)

    # Ensure directories exist.
    dirs = {
        "sources": bundle_dir / "sources",
        "entries": bundle_dir / "entries",
        "concepts": bundle_dir / "concepts",
        "mocs": bundle_dir / "mocs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Render source pages ──────────────────────────────────────────
    for filename, source in sources.items():
        page = render_source_page(source, ts)
        path = dirs["sources"] / filename
        atomic_write(path, page)
        written.append(str(path))

    # ── Render entry pages ───────────────────────────────────────────
    # Build a lookup from slugified source_title → actual source filename
    # so the Source wikilink in each entry resolves to the real source note.
    source_filename_lookup: dict[str, str] = {}
    for filename in sources:
        stem = filename[:-3] if filename.endswith(".md") else filename
        source_filename_lookup[slugify(stem)] = stem

    # Also map each synthesis's source_file to the actual source stem.
    for synthesis in bundle.sources:
        if synthesis.source_file:
            sf = synthesis.source_file
            sf_stem = sf[:-3] if sf.endswith(".md") else sf
            title_slug = slugify(synthesis.source_title)
            source_filename_lookup[title_slug] = sf_stem

    # For remaining unmatched, try prefix/substring matching.
    for synthesis in bundle.sources:
        title_slug = slugify(synthesis.source_title)
        if title_slug not in source_filename_lookup:
            for s in source_filename_lookup:
                if s.startswith(title_slug) or title_slug.startswith(s):
                    source_filename_lookup[title_slug] = source_filename_lookup[s]
                    break
            else:
                # Try substring match on the Chinese part of the title
                import re as _re
                zh_match = _re.search(r"[\u4e00-\u9fff]", synthesis.source_title)
                if zh_match:
                    zh_part = synthesis.source_title[zh_match.start():]
                    zh_slug = slugify(zh_part)
                    for s in source_filename_lookup:
                        if zh_slug and (zh_slug in s or s in zh_slug):
                            source_filename_lookup[title_slug] = source_filename_lookup[s]
                            break

    for synthesis in bundle.sources:
        entry_slug = slugify(synthesis.source_title)
        actual_source_stem = source_filename_lookup.get(entry_slug, entry_slug)
        concept_slugs = [c.slug for c in synthesis.concepts]
        page = render_entry_page(synthesis, actual_source_stem, concept_slugs, ts)
        path = dirs["entries"] / f"{entry_slug}.md"
        atomic_write(path, page)
        written.append(str(path))

    # Build concept map for cross-reference linking.
    concept_map: dict[str, ConceptNote] = {
        c.slug: c for c in bundle.concepts
    }

    # ── Cross-lingual embedding links ────────────────────────────────
    # Find semantically similar concepts across languages using embeddings.
    cross_lingual_links: dict[str, list[tuple[str, float, str]]] = {}
    try:
        from obsidian_llm_wiki.synth.embedding import find_cross_lingual_links
        cross_lingual_links = find_cross_lingual_links(bundle.concepts)
        if cross_lingual_links:
            logger.info(
                "Embedding: found %d cross-lingual concept links",
                len(cross_lingual_links),
            )
    except Exception as exc:
        logger.debug("Embedding-based linking skipped: %s", exc)

    # ── Render concept pages ─────────────────────────────────────────
    for concept in bundle.concepts:
        page = render_concept_page(concept, ts, all_concepts=concept_map)
        path = dirs["concepts"] / f"{concept.slug}.md"
        atomic_write(path, page)
        written.append(str(path))

    # ── Render MOC pages ─────────────────────────────────────────────
    for moc in bundle.maps:
        page = render_moc_page(
            moc, ts,
            all_concepts=concept_map,
            cross_lingual_links=cross_lingual_links or None,
        )
        path = dirs["mocs"] / f"{moc.slug}.md"
        atomic_write(path, page)
        written.append(str(path))

    # ── Per-directory index.md ───────────────────────────────────────
    for dir_name, dir_path in dirs.items():
        md_files = [f for f in dir_path.glob("*.md")
                    if f.name not in ("index.md", "log.md")]
        if md_files:
            idx = render_directory_index(dir_name, md_files, bundle_dir)
            idx_path = dir_path / "index.md"
            atomic_write(idx_path, idx)
            written.append(str(idx_path))

    # ── Bundle-root index.md ─────────────────────────────────────────
    source_count = len(sources)
    entry_count = len(bundle.sources)
    concept_count = len(bundle.concepts)
    moc_count = len(bundle.maps)
    bundle_idx = render_bundle_index(
        bundle_dir, concept_count, entry_count, moc_count, source_count
    )
    bundle_idx_path = bundle_dir / "index.md"
    atomic_write(bundle_idx_path, bundle_idx)
    written.append(str(bundle_idx_path))

    # ── Graph visualization export ───────────────────────────────────
    # Export the knowledge graph as JSON (for D3.js / Obsidian graph view)
    # and Mermaid (for Obsidian embedding).
    try:
        from obsidian_llm_wiki.render.graph_export import export_graph
        graph_dir = bundle_dir / ".llmwiki"
        export_graph(bundle, graph_dir)
        written.append(str(graph_dir / "graph.json"))
        written.append(str(graph_dir / "graph.mmd"))
    except Exception as exc:
        logger.debug("Graph export skipped: %s", exc)

    return written
