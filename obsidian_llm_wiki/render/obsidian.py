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

import os
import re
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    ConceptType,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)

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


# ── Utilities ───────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert arbitrary text to a filename-safe slug."""
    cleaned = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    slug = cleaned.strip("-").lower()
    return slug if slug else "untitled"


def make_wikilink(slug: str, alias: str | None = None) -> str:
    """Build an Obsidian wikilink ``[[slug]]`` or ``[[slug|alias]]``.

    The alias is only included when it is non-empty and differs from the slug.
    """
    if alias and alias != slug:
        return f"[[{slug}|{alias}]]"
    return f"[[{slug}]]"


def build_frontmatter(fm_dict: dict[str, Any]) -> str:
    """Serialize a dict to a ``---``-delimited YAML frontmatter block.

    The block ends with a trailing newline so it composes cleanly with a
    body: ``build_frontmatter(fm) + "\\n\\n" + body``.
    """
    dumped = yaml.dump(
        fm_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip()
    return f"---\n{dumped}\n---\n"


_FM_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from ``raw``.  Returns (meta, body).

    Handles edge cases the old ``partition``-based approach missed:
      * Body starting immediately after closing ``---`` (no leading newline)
      * No trailing newline after the closing ``---``
    """
    if not raw.startswith("---\n"):
        return {}, raw
    match = _FM_RE.match(raw)
    if not match:
        return {}, raw
    yaml_block, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(meta, dict):
        meta = {}
    body = body.lstrip("\n")
    return meta, body


# Standard markdown link: [text](url). Excludes images ![alt](url).
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)]*)\)")


def extract_links(body: str) -> list[tuple[str, str]]:
    """Extract standard markdown ``[text](url)`` links from ``body``.

    Returns a list of ``(text, url)`` tuples in document order.
    """
    return [(m.group(1), m.group(2)) for m in _LINK_RE.finditer(body)]


def safe_read_file(path: str | Path) -> str:
    """Read a file as UTF-8, returning ``""`` on any error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def atomic_write(path: str | Path, content: str) -> None:
    """Atomically write ``content`` to ``path`` via temp + os.replace."""
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(fp.parent), prefix=fp.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(content)
        os.replace(tmp_name, fp)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Page renderers ──────────────────────────────────────────────────────


def render_source_page(source: SourceDoc, timestamp: str | None = None) -> str:
    """Render a Source-type page (raw content for provenance)."""
    ts = timestamp or _timestamp()
    fm = {
        "type": ConceptType.SOURCE.value,
        "title": source.title,
        "url": source.url or "",
        "timestamp": ts,
    }
    body = f"# {source.title}\n\n{source.content}"
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
        f"- {make_wikilink(source_slug, 'Source document')}",
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
    if all_concepts and concept.related:
        cross_ref_lines = _build_cross_ref_section(concept, all_concepts)
        if cross_ref_lines:
            parts.extend(["## 关联图谱 / Cross-References", ""])
            parts.extend(cross_ref_lines)
            parts.append("")

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"


def _build_cross_ref_section(
    concept: ConceptNote,
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build the typed-edge cross-reference section for a concept.

    Shows all typed relationships as a structured flow diagram matching the
    screenshot the user shared: node → edge-type → node.

    Example output:
      - 庞氏经济学 (Ponzi Economics) → 三盘理论 (Three-disc Theory)
        type: 衍生自 (derives_from)
    """
    lines: list[str] = []
    for link in concept.related:
        target = all_concepts.get(link.slug)
        if not target:
            continue

        display = link.display or link.slug
        relation_type = link.relation or "related_to"

        # Primary edge: this concept → target
        lines.append(
            f"- {make_wikilink(concept.slug, concept.title)} "
            f"→ {make_wikilink(link.slug, display)}"
        )
        lines.append(f"  type: `{relation_type}`")

        # Reverse edge: did target also link back to this concept?
        # If not, note the unidirectional relationship
        reverse_links = [
            r for r in (target.related or [])
            if r.slug == concept.slug
        ]
        if reverse_links:
            rev_rel = reverse_links[0].relation or "related_to"
            lines.append(f"  ← {make_wikilink(target.slug, target.title)} (`{rev_rel}`)")
        else:
            lines.append(f"  ← (unidirectional — target does not link back)")

    return lines


def render_moc_page(
    moc: MapOfContent,
    timestamp: str | None = None,
    all_concepts: dict[str, ConceptNote] | None = None,
) -> str:
    """Render a Map of Content page.

    Args:
        moc: The MOC to render.
        timestamp: Optional ISO timestamp.
        all_concepts: Optional slug→ConceptNote dict. When provided, the MOC
            displays concept language badges and cross-lingual aliases, grouping
            concepts that share the same semantic meaning across languages under
            a unified entry.
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
        parts.extend(["## Concepts", ""])
        for slug in moc.concept_slugs:
            entry = all_concepts.get(slug) if all_concepts else None
            badge = ""
            if entry and entry.aliases:
                # Show cross-lingual aliases as language badge
                zh_alias = next((a for a in entry.aliases if _is_chinese(a)), None)
                if zh_alias:
                    badge = f" · {zh_alias}"
            parts.append(f"- {make_wikilink(slug)}{badge}")
        parts.append("")

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"


def _is_chinese(text: str) -> bool:
    """Return True if text contains Chinese characters."""
    import re
    return bool(re.search(r"[\u4e00-\u9fff]", text))


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
) -> list[str]:
    """Render a complete vault from a SynthesisBundle.

    Args:
        bundle_dir: The wiki root directory (e.g. vault/04-Wiki).
        bundle: The merged SynthesisBundle from synth.dedupe.
        sources: Dict mapping source filename → SourceDoc.

    Returns:
        List of file paths that were written.
    """
    written: list[str] = []
    ts = _timestamp()

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
    for synthesis in bundle.sources:
        entry_slug = slugify(synthesis.source_title)
        concept_slugs = [c.slug for c in synthesis.concepts]
        page = render_entry_page(synthesis, entry_slug, concept_slugs, ts)
        path = dirs["entries"] / f"{entry_slug}.md"
        atomic_write(path, page)
        written.append(str(path))

    # Build concept map for cross-reference linking.
    # Used by render_concept_page (typed edge cross-refs) and
    # render_moc_page (cross-lingual language badges).
    concept_map: dict[str, ConceptNote] = {
        c.slug: c for c in bundle.concepts
    }

    # ── Render concept pages ─────────────────────────────────────────
    for concept in bundle.concepts:
        page = render_concept_page(concept, ts, all_concepts=concept_map)
        path = dirs["concepts"] / f"{concept.slug}.md"
        atomic_write(path, page)
        written.append(str(path))

    # ── Render MOC pages ─────────────────────────────────────────────
    for moc in bundle.maps:
        page = render_moc_page(moc, ts, all_concepts=concept_map)
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

    return written
