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

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    ConceptType,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
    concept_note_to_dict,
    moc_to_dict,
    normalize_slug,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.text_quality import dedupe_statements

if TYPE_CHECKING:
    from obsidian_llm_wiki.core.registry import RenderDiff, VaultRegistry

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
    build_cross_links as _build_cross_links,
)
from obsidian_llm_wiki.render.crossrefs import (
    build_cross_ref_diagram as _build_cross_ref_diagram,
)
from obsidian_llm_wiki.render.crossrefs import (
    build_moc_cross_links as _build_moc_cross_links,
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
_ACTIVE_RENDER_TRANSACTION: ContextVar[Any] = ContextVar(
    "active_render_transaction", default=None
)


def _render_fingerprint(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"render-v4\x00{raw}".encode()).hexdigest()


def _configured_int(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value


def _diff_render_manifest(
    registry: VaultRegistry | None,
    kind: str,
    manifest: dict[str, tuple[str, str]],
    pending: list[tuple[str, dict[str, tuple[str, str]], RenderDiff]],
    *,
    force: bool = False,
) -> tuple[set[str], tuple[str, ...]]:
    if registry is None:
        return set(manifest), ()
    diff = registry.render_diff(kind, manifest)
    pending.append((kind, manifest, diff))
    return (set(manifest) if force else set(diff.changed_keys)), diff.stale_paths


_PROVENANCE_SCALAR_KEYS = (
    "requested_url",
    "resolved_url",
    "extracted_url",
    "content_type",
    "document_format",
    "retrieved_at",
    "content_sha256",
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


# ── Page renderers ──────────────────────────────────────────────────────


def render_source_page(source: SourceDoc, timestamp: str | None = None) -> str:
    """Render a Source-type page (raw content for provenance).

    Avoids duplicate headings: if the content already starts with the title
    (as a # heading or plain text), the body heading is skipped.
    """
    ts = timestamp or _timestamp()
    fm: dict[str, Any] = {
        "type": ConceptType.SOURCE.value,
        "title": source.title,
        "url": source.url or "",
        "timestamp": ts,
    }
    if source.aliases:
        fm["aliases"] = source.aliases
    if source.tags:
        fm["tags"] = source.tags
    if source.source_type:
        fm["source_type"] = source.source_type
    # Obsidian Properties accepts scalar and list values but warns for nested
    # mappings. Preserve complete provenance in a readable, round-trippable
    # list of strings rather than emitting a YAML object in the property pane.
    provenance = _render_provenance(source)
    if provenance:
        fm["provenance"] = provenance
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


def _render_provenance(source: SourceDoc) -> list[str]:
    """Return compact source provenance entries compatible with Obsidian Properties."""
    values = source.provenance
    entries = [
        f"{key}: {getattr(values, key)}"
        for key in _PROVENANCE_SCALAR_KEYS
        if getattr(values, key)
    ]
    entries.extend(f"extractor_chain: {stage}" for stage in values.extractor_chain if stage)
    entries.extend(f"diagnostics: {message}" for message in values.diagnostics if message)
    return entries


def _write_generated_page(path: Path, page: str, bundle_dir: Path) -> bool:
    """Atomically replace changed generated content while preserving human work.

    Every automatic rewrite snapshots the exact previous bytes first.  A
    byte-identical render is a no-op: it neither makes a backup nor replaces the
    inode.  Reviewed pages retain their body while their generated metadata is
    refreshed.
    """
    transaction = _ACTIVE_RENDER_TRANSACTION.get()
    if path.exists():
        existing = safe_read_file(path)
        try:
            from obsidian_llm_wiki.core.backups import backup_file, list_backups
            from obsidian_llm_wiki.core.review import is_reviewed_page

            if is_reviewed_page(existing):
                generated_meta, _ = parse_frontmatter(page)
                _, reviewed_body = parse_frontmatter(existing)
                generated_meta["reviewed"] = True
                page = f"{build_frontmatter(generated_meta)}\n{reviewed_body.lstrip()}"

            if page == existing or _same_generated_page(page, existing):
                return False
            if transaction is not None:
                transaction.before_change(path)
                for backup in list_backups(path, bundle_dir / ".llmwiki" / "backups"):
                    transaction.before_change(backup)
            backup = backup_file(path, bundle_dir / ".llmwiki" / "backups")
            if transaction is not None:
                transaction.record_created(backup)
        except Exception as exc:
            raise RuntimeError(f"Could not safely replace generated page {path}: {exc}") from exc
    elif transaction is not None:
        transaction.before_change(path)

    atomic_write(path, page)
    if transaction is not None:
        transaction.record_write(path, page)
    return True


def _same_generated_page(left: str, right: str) -> bool:
    """Ignore volatile generation timestamps when semantic output is unchanged."""
    left_meta, left_body = parse_frontmatter(left)
    right_meta, right_body = parse_frontmatter(right)
    left_meta.pop("timestamp", None)
    right_meta.pop("timestamp", None)
    return left_meta == right_meta and left_body == right_body


def _write_generated_artifact(path: Path, content: str, bundle_dir: Path) -> bool:
    """Atomically write a non-page artifact only when its bytes changed."""
    existing = safe_read_file(path) if path.is_file() else None
    if existing == content:
        return False
    transaction = _ACTIVE_RENDER_TRANSACTION.get()
    if transaction is not None:
        transaction.before_change(path)
    atomic_write(path, content)
    if transaction is not None:
        transaction.record_write(path, content)
    return True


def _delete_generated_artifact(path: Path, bundle_dir: Path) -> bool:
    """Delete one stale generated artifact with transaction rollback support."""
    if not path.is_file():
        return False
    if path.suffix.lower() == ".md":
        from obsidian_llm_wiki.core.review import is_reviewed_page

        if is_reviewed_page(safe_read_file(path)):
            return False
    transaction = _ACTIVE_RENDER_TRANSACTION.get()
    if transaction is not None:
        transaction.before_change(path)
        transaction.record_delete(path)
    path.unlink()
    return True


def _safe_page_path(directory: Path, slug: object, title: object) -> Path:
    """Return a normalized page path proven to be inside ``directory``."""
    normalized_slug = normalize_slug(slug, title)
    root = directory.resolve()
    candidate = (directory / f"{normalized_slug}.md").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {directory}: {candidate}") from exc
    return candidate


def _normalize_bundle_output_slugs(bundle: SynthesisBundle) -> None:
    """Defend direct renderer callers that bypass the synthesis JSON parser."""
    for concept in bundle.concepts:
        concept.slug = normalize_slug(concept.slug, concept.title)
        for link in concept.related:
            link.slug = normalize_slug(link.slug)
    for synthesis in bundle.sources:
        for concept in synthesis.concepts:
            concept.slug = normalize_slug(concept.slug, concept.title)
            for link in concept.related:
                link.slug = normalize_slug(link.slug)
    for moc in bundle.maps:
        moc.slug = normalize_slug(moc.slug, moc.title)
        moc.concept_slugs = [normalize_slug(slug) for slug in moc.concept_slugs]


def render_entry_page(
    synthesis: SourceSynthesis,
    source_slug: str,
    concept_slugs: list[str],
    timestamp: str | None = None,
) -> str:
    """Render an Entry-type page (synthesis of one source)."""
    ts = timestamp or _timestamp()
    entry_tags = list(
        dict.fromkeys(
            normalized
            for tag in synthesis.source_tags
            if (normalized := _sanitize_tag(tag).casefold())
        )
    )
    fm = {
        "type": ConceptType.ENTRY.value,
        "title": synthesis.source_title,
        "tags": entry_tags,
        "timestamp": ts,
    }

    parts: list[str] = [f"# {synthesis.source_title}", ""]

    if synthesis.source_summary:
        parts.extend([synthesis.source_summary, ""])

    if synthesis.key_points:
        parts.extend(["## Key Findings", ""])
        for point in dedupe_statements(synthesis.key_points):
            parts.append(f"- {point}")
        parts.append("")

    if concept_slugs:
        parts.extend(["## Linked Concepts", ""])
        for slug in concept_slugs:
            parts.append(f"- {make_wikilink(slug)}")
        parts.append("")

    if synthesis.open_questions:
        parts.extend(["## Open Questions", ""])
        for q in dedupe_statements(synthesis.open_questions):
            parts.append(f"- {q}")
        parts.append("")

    parts.extend([
        "## Source",
        "",
        f"- {make_wikilink(f'sources/{source_slug}', synthesis.source_title)}",
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
    related = [
        relation
        for relation in concept.related
        if all_concepts is None or relation.slug in all_concepts
    ]
    rendered_concept = deepcopy(concept)
    rendered_concept.related = related
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
    if related:
        # Serialize as pipe-separated strings so Obsidian's Properties panel
        # treats ``relations`` as a simple list of strings (no nested-object
        # warning). Format: "slug|relation_type|display_label".
        fm["relations"] = [
            f"{r.slug}|{r.relation}|{r.display or r.slug}" for r in related
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
        evidence_entries: list[tuple[int, str, str, bool]] = []
        # (number, quote, source_file, verified)
        for claim in concept.claims:
            evidence = claim.evidence
            has_quote = evidence is not None and bool(evidence.quote)
            has_source = evidence is not None and bool(evidence.source_file)
            verified = (
                has_quote
                and has_source
                and str(evidence.verification) == "verified"
            )
            unverified_with_quote = (
                has_quote
                and has_source
                and str(evidence.verification) != "verified"
            )
            marker = ""
            if verified or unverified_with_quote:
                marker_number = len(evidence_entries) + 1
                marker = f" [{marker_number}]" if verified else f" [{marker_number}*]"
                evidence_entries.append(
                    (marker_number, evidence.quote, evidence.source_file, verified)
                )
            parts.append(f"- {claim.text}{marker}")
        parts.append("")
        if evidence_entries:
            from obsidian_llm_wiki.core.source_files import validate_source_filename

            parts.extend(["## Evidence", ""])
            for number, quote, source_file, verified in evidence_entries:
                try:
                    safe_source_file = validate_source_filename(source_file)
                    source_stem = Path(safe_source_file).stem
                    source_link = make_wikilink(f"sources/{source_stem}", source_stem)
                except ValueError:
                    source_link = source_file
                suffix = "" if verified else " *(unverified)*"
                parts.extend([f'{number}. \u201c{quote}\u201d{suffix}', f"   \u2014 {source_link}"])
            parts.append("")

    # ── 关联图谱 / Cross-References ──────────────────────────────────
    # The ASCII flow diagram goes inside a ```text code block for monospace
    # display. The cross-link wikilinks are rendered as regular markdown
    # outside the code block so Obsidian treats them as live, clickable
    # links — wikilinks inside code blocks are literal text.
    if all_concepts and related:
        cross_ref_lines = _build_cross_ref_diagram(rendered_concept, all_concepts)
        if cross_ref_lines:
            parts.extend(["## Cross-References / 关联图谱", ""])
            parts.append("```text")
            parts.extend(cross_ref_lines)
            parts.append("```")
            parts.append("")
            # Clickable cross-links outside the code block
            cross_links = _build_cross_links(rendered_concept, all_concepts)
            if cross_links:
                parts.extend(cross_links)
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

    _augment_moc_cross_lingual_members(
        moc, all_concepts, cross_lingual_links,
    )
    # A source-local MoC can retain slugs removed or renamed during canonical
    # concept merge. Never emit a wikilink for a page that will not be rendered.
    concept_slugs = [
        slug
        for slug in moc.concept_slugs
        if all_concepts is None or slug in all_concepts
    ]

    if concept_slugs:
        concept_entries = [
            all_concepts.get(slug) if all_concepts else None
            for slug in concept_slugs
        ]
        bilingual_headings = _moc_needs_bilingual_headings(moc, concept_entries)
        concepts_heading = "## Concepts / 概念" if bilingual_headings else "## Concepts"
        parts.extend([concepts_heading, ""])
        for slug in concept_slugs:
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
    # ASCII flow diagram in a code block + clickable wikilinks outside.
    # Every MoC with ≥2 concepts gets a Cross-References section for
    # structural consistency — even when no inter-concept relations exist
    # yet, a placeholder message is shown instead of silently omitting
    # the section.
    if all_concepts and concept_slugs:
        moc_concepts = [
            all_concepts[s] for s in concept_slugs
            if s in all_concepts
        ]
        if len(moc_concepts) >= 2:
            diagram_lines = _build_moc_cross_ref_diagram(moc_concepts, all_concepts)
            parts.extend(["## Cross-References / 关联图谱", ""])
            if diagram_lines:
                parts.append("```text")
                parts.extend(diagram_lines)
                parts.append("```")
                parts.append("")
                # Clickable cross-links outside the code block
                moc_cross_links = _build_moc_cross_links(moc_concepts, all_concepts)
                if moc_cross_links:
                    parts.extend(moc_cross_links)
                    parts.append("")
            else:
                # No inter-concept relations yet — show placeholder so
                # all MoCs have a consistent section structure.
                parts.extend([
                    "*No cross-references available yet.*",
                    "",
                ])

    body = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n{body}"


def _augment_moc_cross_lingual_members(
    moc: MapOfContent,
    all_concepts: dict[str, ConceptNote] | None,
    cross_lingual_links: dict[str, list[tuple[str, float, str]]] | None,
) -> None:
    """Add validated semantic siblings before rendering the MoC concept list.

    The mutation is deliberate: the graph export and page agree on membership.
    Rebuilds recalculate it from current embeddings, so this adds no hidden
    persisted state.
    """
    if not all_concepts or not cross_lingual_links:
        return

    existing = set(moc.concept_slugs)
    for slug in tuple(moc.concept_slugs):
        for target_slug, _score, _display in cross_lingual_links.get(slug, []):
            if target_slug not in all_concepts or target_slug in existing:
                continue
            moc.concept_slugs.append(target_slug)
            existing.add(target_slug)


def _remap_cross_lingual_links(
    links: dict[str, list[tuple[str, float, str]]],
    slug_map: dict[str, str],
) -> dict[str, list[tuple[str, float, str]]]:
    """Keep pre-render embedding links valid after bilingual slug normalization."""
    if not slug_map:
        return links
    remapped: dict[str, list[tuple[str, float, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for source_slug, targets in links.items():
        source = slug_map.get(source_slug, source_slug)
        for target_slug, score, display in targets:
            target = slug_map.get(target_slug, target_slug)
            if source == target or (source, target) in seen:
                continue
            seen.add((source, target))
            remapped.setdefault(source, []).append((target, score, display))
    return remapped





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


def _render_directory_indexes(
    dir_name: str,
    dir_path: Path,
    items: list[tuple[str, str]],
    bundle_dir: Path,
    *,
    shard_size: int,
) -> list[str]:
    """Write bounded index pages from entity metadata, never a directory scan."""
    if shard_size < 1:
        raise ValueError("index_shard_size must be positive")
    ordered = sorted(items)
    written: list[str] = []
    keep: set[Path] = set()
    index_path = dir_path / "index.md"
    previous_index = safe_read_file(index_path)
    old_shard_stems = set(
        re.findall(r"\[\[(index-[^|\]]+)(?:\|[^\]]*)?\]\]", previous_index)
    )

    if len(ordered) <= shard_size:
        body = [f"# {dir_name.title()}", ""]
        body.extend(f"- [[{slug}|{title}]]" for slug, title in ordered)
        body.append("")
        if _write_generated_page(index_path, "\n".join(body), bundle_dir):
            written.append(str(index_path))
    else:
        shard_links: list[str] = []
        buckets: dict[str, list[tuple[str, str]]] = {}
        for slug, title in ordered:
            bucket = hashlib.blake2b(slug.encode(), digest_size=1).hexdigest()
            buckets.setdefault(bucket, []).append((slug, title))
        for bucket in sorted(buckets):
            records = sorted(buckets[bucket])
            for part, start in enumerate(range(0, len(records), shard_size)):
                chunk = records[start:start + shard_size]
                shard_path = dir_path / f"index-b{bucket}-{part:03d}.md"
                body = [f"# {dir_name.title()} · {bucket.upper()} · {part + 1}", ""]
                body.extend(f"- [[{slug}|{title}]]" for slug, title in chunk)
                body.append("")
                if _write_generated_page(shard_path, "\n".join(body), bundle_dir):
                    written.append(str(shard_path))
                keep.add(shard_path)
                shard_links.append(
                    f"- [[{shard_path.stem}|Bucket {bucket.upper()} · {part + 1}]]"
                )
        index_body = [
            f"# {dir_name.title()}",
            "",
            f"{len(ordered)} items across {len(shard_links)} shards.",
            "",
            *shard_links,
            "",
        ]
        if _write_generated_page(index_path, "\n".join(index_body), bundle_dir):
            written.append(str(index_path))

    actual_index = safe_read_file(index_path)
    actual_shard_stems = set(
        re.findall(r"\[\[(index-[^|\]]+)(?:\|[^\]]*)?\]\]", actual_index)
    )
    keep.update(dir_path / f"{stem}.md" for stem in actual_shard_stems)
    for stale_stem in sorted(old_shard_stems - actual_shard_stems):
        stale = dir_path / f"{stale_stem}.md"
        if stale not in keep and _delete_generated_artifact(stale, bundle_dir):
            written.append(str(stale))
    return written


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


class _RenderTransaction:
    """Restore every pre-existing vault file if a render cannot finish.

    The renderer has several independent output steps, so a per-page atomic
    write alone cannot prevent a later failure from leaving a mixed vault.
    Snapshotting the entire pre-existing tree also lets rollback remove backups
    created by an aborted render while retaining the pre-existing backup set.
    """

    def __init__(self, bundle_dir: Path, *, lazy: bool = False) -> None:
        self.bundle_dir = bundle_dir
        self._lazy = lazy
        self._staging = tempfile.TemporaryDirectory(prefix="obsidian-render-")
        self._staging_dir = Path(self._staging.name)
        self._root_existed = bundle_dir.exists()
        self._original_files: set[Path] = set()
        self._passthrough_files: set[Path] = set()
        self._original_dirs: set[Path] = set()
        self._expected_outputs: dict[Path, str | None] = {}
        if not lazy:
            self._snapshot()

    def before_change(self, path: Path) -> None:
        """Lazily snapshot one path before a renderer mutation."""
        if not self._lazy:
            return
        relative = path.relative_to(self.bundle_dir)
        if relative in self._original_files or relative in self._expected_outputs:
            return
        if path.is_file():
            destination = self._staging_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            self._original_files.add(relative)

    def record_created(self, path: Path) -> None:
        """Track a new append-only artifact such as a page backup."""
        if self._lazy and path.is_file():
            relative = path.relative_to(self.bundle_dir)
            self._expected_outputs[relative] = safe_read_file(path)

    def record_write(self, path: Path, content: str) -> None:
        self._expected_outputs[path.relative_to(self.bundle_dir)] = content

    def record_delete(self, path: Path) -> None:
        self._expected_outputs[path.relative_to(self.bundle_dir)] = None

    def _snapshot(self) -> None:
        if not self._root_existed:
            return
        self._original_dirs.add(Path("."))
        for path in self.bundle_dir.rglob("*"):
            relative = path.relative_to(self.bundle_dir)
            if path.is_dir():
                self._original_dirs.add(relative)
                continue
            if not path.is_file():
                continue
            # Backup files are append-only during a render: existing snapshots
            # are never mutated, only new uniquely named files are added. Keep
            # their names for rollback (which removes new files), but do not
            # copy thousands of historical backups before every render.
            # Copying that immutable archive made a no-op 800-note render take
            # minutes and turns a 50k-note vault into an I/O denial of service.
            if relative.parts[:2] == (".llmwiki", "backups"):
                self._original_files.add(relative)
                self._passthrough_files.add(relative)
                continue
            destination = self._staging_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            self._original_files.add(relative)

    def commit(self) -> None:
        self._staging.cleanup()

    def rollback(self) -> None:
        """Restore the exact pre-render snapshot, including reviewed metadata."""
        if self._lazy:
            self._rollback_lazy()
            return
        if self.bundle_dir.exists():
            for path in sorted(
                (p for p in self.bundle_dir.rglob("*") if p.is_file()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                relative = path.relative_to(self.bundle_dir)
                if relative not in self._original_files:
                    expected = self._expected_outputs.get(relative)
                    is_renderer_backup = relative.parts[:2] == (".llmwiki", "backups")
                    if is_renderer_backup or (
                        expected is not None and safe_read_file(path) == expected
                    ):
                        path.unlink()
                    elif relative in self._expected_outputs:
                        logger.warning("Preserving concurrent edit during rollback: %s", path)

        for relative in self._original_files:
            if relative in self._passthrough_files:
                continue
            source = self._staging_dir / relative
            destination = self.bundle_dir / relative
            if relative in self._expected_outputs:
                expected = self._expected_outputs[relative]
                if expected is None and destination.exists():
                    logger.warning(
                        "Preserving concurrent recreated page during rollback: %s", destination
                    )
                    continue
                if expected is not None and (
                    not destination.exists() or safe_read_file(destination) != expected
                ):
                    logger.warning("Preserving concurrent edit during rollback: %s", destination)
                    continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)

        if self.bundle_dir.exists():
            for directory in sorted(
                (p for p in self.bundle_dir.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                if directory.relative_to(self.bundle_dir) not in self._original_dirs:
                    with suppress(OSError):
                        directory.rmdir()
            if not self._root_existed:
                with suppress(OSError):
                    self.bundle_dir.rmdir()
        self._staging.cleanup()

    def _rollback_lazy(self) -> None:
        """Restore only paths touched by a scalable render transaction."""
        expected_paths = set(self._expected_outputs)
        for relative, expected in sorted(
            self._expected_outputs.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            destination = self.bundle_dir / relative
            original = self._staging_dir / relative
            if relative in self._original_files:
                current_matches = (
                    (expected is None and not destination.exists())
                    or (
                        expected is not None
                        and destination.is_file()
                        and safe_read_file(destination) == expected
                    )
                )
                if not current_matches:
                    logger.warning("Preserving concurrent edit during rollback: %s", destination)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(original, destination)
            elif destination.is_file() and expected is not None:
                if safe_read_file(destination) == expected:
                    destination.unlink()
                else:
                    logger.warning("Preserving concurrent edit during rollback: %s", destination)

        for relative in self._original_files - expected_paths:
            source = self._staging_dir / relative
            destination = self.bundle_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        self._staging.cleanup()


def _render_vault(
    bundle_dir: Path,
    bundle: SynthesisBundle,
    sources: dict[str, SourceDoc],
    config: Any = None,
    *,
    cross_lingual_links: dict[str, list[tuple[str, float, str]]] | None = None,
    render_registry: VaultRegistry | None = None,
    force_render: bool = False,
    preserve_source_filenames: set[str] | None = None,
    protected_stale_paths: set[str] | None = None,
) -> list[str]:
    """Render a complete vault from a SynthesisBundle.

    Args:
        bundle_dir: The wiki root directory (e.g. vault/04-Wiki).
        bundle: The merged SynthesisBundle from synth.dedupe.
        sources: Dict mapping source filename → SourceDoc.
        config: Optional pipeline config for threshold settings
            (similarity_dedup_threshold, moc_assignment_threshold).
            When None, defaults are used.
        cross_lingual_links: Optional precomputed embedding links. Passing
            these keeps pipeline metrics and rendered MoC membership aligned
            without repeating embedding calls.

    Returns:
        List of file paths that were written.
    """
    written: list[str] = []
    stale_files: list[Path] = []
    pending_manifests: list[
        tuple[str, dict[str, tuple[str, str]], RenderDiff]
    ] = []
    ts = _timestamp()

    # ── Backlink propagation ─────────────────────────────────────────
    # NOTE: propagate_backlinks is called in run_pipeline() before rendering.
    # It is NOT called here to avoid double-propagation with incorrect
    # reverse relation types on the second pass. If render_vault is called
    # standalone (e.g. from a test), the caller is responsible for calling
    # propagate_backlinks first.

    # Make the language policy deterministic. The synthesis prompt asks Chinese
    # sources to use English-first bilingual titles, but smaller/local models do
    # not always comply. Rendering is the last safe choke point before filenames
    # and wikilinks are written, so normalize here and remap slugs consistently.
    slug_map = _normalize_bilingual_titles_and_slugs(bundle)
    _normalize_bundle_output_slugs(bundle)
    if cross_lingual_links:
        cross_lingual_links = _remap_cross_lingual_links(cross_lingual_links, slug_map)

    # Ensure directories exist.
    dirs = {
        "sources": bundle_dir / "sources",
        "entries": bundle_dir / "entries",
        "concepts": bundle_dir / "concepts",
        "mocs": bundle_dir / "mocs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Source-local synthesis data can retain cached concepts that semantic
    # deduplication removed from the bundle. Entry pages must only link to
    # concepts this render will materialize.
    current_concept_slugs = {concept.slug for concept in bundle.concepts}

    # ── Render source pages ──────────────────────────────────────────
    from obsidian_llm_wiki.core.source_files import source_file_path

    source_paths = {
        filename: source_file_path(dirs["sources"], filename)
        for filename in sources
    }
    previous_source_manifest = (
        render_registry.render_manifest("source-page")
        if render_registry is not None and preserve_source_filenames
        else {}
    )
    preserved = preserve_source_filenames or set()
    source_manifest: dict[str, tuple[str, str]] = {}
    for filename, path in source_paths.items():
        if filename in preserved and filename in previous_source_manifest:
            source_manifest[filename] = previous_source_manifest[filename]
        else:
            source_manifest[filename] = (
                str(path.relative_to(bundle_dir)),
                _render_fingerprint(asdict(sources[filename])),
            )
    dirty_sources, stale_source_paths = _diff_render_manifest(
        render_registry,
        "source-page",
        source_manifest,
        pending_manifests,
        force=force_render,
    )
    dirty_sources.difference_update(
        filename
        for filename in preserved
        if filename in source_paths and source_paths[filename].is_file()
    )
    stale_files.extend(bundle_dir / path for path in stale_source_paths)
    for filename, source in sources.items():
        if filename not in dirty_sources:
            continue
        page = render_source_page(source, ts)
        path = source_paths[filename]
        if _write_generated_page(path, page, bundle_dir):
            written.append(str(path))

    # ── Render entry pages ───────────────────────────────────────────
    # Remove stale entry files from previous runs.
    from obsidian_llm_wiki.core.review import is_reviewed_page
    if render_registry is None:
        current_entry_slugs = {
            normalize_slug(slugify(s.source_title), s.source_title)
            for s in bundle.sources
        }
        for old_file in dirs["entries"].glob("*.md"):
            if (
                old_file.stem not in current_entry_slugs
                and old_file.name != "index.md"
                and not is_reviewed_page(safe_read_file(old_file))
            ):
                stale_files.append(old_file)

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

    entry_records: list[tuple[str, SourceSynthesis, str, list[str], Path]] = []
    entry_manifest: dict[str, tuple[str, str]] = {}
    for synthesis in bundle.sources:
        entry_slug = normalize_slug(
            slugify(synthesis.source_title), synthesis.source_title
        )
        actual_source_stem = source_filename_lookup.get(entry_slug, entry_slug)
        concept_slugs = [
            concept.slug
            for concept in synthesis.concepts
            if concept.slug in current_concept_slugs
        ]
        path = _safe_page_path(dirs["entries"], entry_slug, synthesis.source_title)
        key = synthesis.source_file or entry_slug
        entry_records.append(
            (key, synthesis, actual_source_stem, concept_slugs, path)
        )
        entry_manifest[key] = (
            str(path.relative_to(bundle_dir)),
            _render_fingerprint(
                {
                    "synthesis": source_synthesis_to_dict(synthesis),
                    "source_stem": actual_source_stem,
                    "concept_slugs": concept_slugs,
                }
            ),
        )
    dirty_entries, stale_entry_paths = _diff_render_manifest(
        render_registry,
        "entry-page",
        entry_manifest,
        pending_manifests,
        force=force_render,
    )
    stale_files.extend(bundle_dir / path for path in stale_entry_paths)
    for key, synthesis, actual_source_stem, concept_slugs, path in entry_records:
        if key not in dirty_entries:
            continue
        page = render_entry_page(synthesis, actual_source_stem, concept_slugs, ts)
        if _write_generated_page(path, page, bundle_dir):
            written.append(str(path))

    # Build concept map for cross-reference linking.
    concept_map: dict[str, ConceptNote] = {
        c.slug: c for c in bundle.concepts
    }

    # ── Cross-lingual embedding links ────────────────────────────────
    # Find semantically similar concepts across languages using embeddings.
    if cross_lingual_links is None:
        try:
            from obsidian_llm_wiki.synth.embedding import find_cross_lingual_links
            cross_lingual_links = find_cross_lingual_links(bundle.concepts)
        except Exception as exc:
            logger.debug("Embedding-based linking skipped: %s", exc)
            cross_lingual_links = {}
    if cross_lingual_links:
        logger.info(
            "Embedding: found %d cross-lingual concept links",
            len(cross_lingual_links),
        )

    # ── Render concept pages ─────────────────────────────────────────
    # First, remove stale concept files from previous runs that are no
    # longer in the current bundle. When the LLM produces different slugs
    # across runs (e.g. "jump-risk" → "jump-risk-prediction-markets"), the
    # old files remain on disk and pollute the vault.
    from obsidian_llm_wiki.core.review import is_reviewed_page

    if render_registry is None:
        for old_file in dirs["concepts"].glob("*.md"):
            if old_file.name == "index.md":
                continue
            if old_file.stem in current_concept_slugs:
                continue
            old_content = safe_read_file(old_file)
            if is_reviewed_page(old_content):
                continue
            # Don't delete orphaned concepts — they're intentionally kept
            # when their source is deleted but the concept still has value.
            old_meta, _ = parse_frontmatter(old_content)
            if old_meta.get("orphaned") is True:
                continue
            stale_files.append(old_file)

    concept_paths = {
        concept.slug: _safe_page_path(
            dirs["concepts"], concept.slug, concept.title
        )
        for concept in bundle.concepts
    }
    concept_manifest = {
        concept.slug: (
            str(concept_paths[concept.slug].relative_to(bundle_dir)),
            _render_fingerprint(
                {
                    "concept": concept_note_to_dict(concept),
                    "related_titles": {
                        link.slug: concept_map[link.slug].title
                        for link in concept.related
                        if link.slug in concept_map
                    },
                }
            ),
        )
        for concept in bundle.concepts
    }
    dirty_concepts, stale_concept_paths = _diff_render_manifest(
        render_registry,
        "concept-page",
        concept_manifest,
        pending_manifests,
        force=force_render,
    )
    for relative in stale_concept_paths:
        old_file = bundle_dir / relative
        old_content = safe_read_file(old_file)
        old_meta, _ = parse_frontmatter(old_content)
        if not is_reviewed_page(old_content) and old_meta.get("orphaned") is not True:
            stale_files.append(old_file)

    for concept in bundle.concepts:
        if concept.slug not in dirty_concepts:
            continue
        page = render_concept_page(concept, ts, all_concepts=concept_map)
        path = concept_paths[concept.slug]
        if _write_generated_page(path, page, bundle_dir):
            written.append(str(path))

    # ── Render MOC pages ─────────────────────────────────────────────
    # Remove stale MoC files from previous runs.
    current_moc_slugs = {m.slug for m in bundle.maps}
    if render_registry is None:
        for old_file in dirs["mocs"].glob("*.md"):
            if (
                old_file.stem not in current_moc_slugs
                and old_file.name != "index.md"
                and not is_reviewed_page(safe_read_file(old_file))
            ):
                stale_files.append(old_file)

    moc_paths = {
        moc.slug: _safe_page_path(dirs["mocs"], moc.slug, moc.title)
        for moc in bundle.maps
    }
    moc_manifest = {
        moc.slug: (
            str(moc_paths[moc.slug].relative_to(bundle_dir)),
            _render_fingerprint(
                {
                    "moc": moc_to_dict(moc),
                    "members": {
                        slug: concept_note_to_dict(concept_map[slug])
                        for slug in moc.concept_slugs
                        if slug in concept_map
                    },
                    "cross_lingual": {
                        slug: (cross_lingual_links or {}).get(slug, [])
                        for slug in moc.concept_slugs
                    },
                }
            ),
        )
        for moc in bundle.maps
    }
    dirty_mocs, stale_moc_paths = _diff_render_manifest(
        render_registry,
        "moc-page",
        moc_manifest,
        pending_manifests,
        force=force_render,
    )
    for relative in stale_moc_paths:
        old_file = bundle_dir / relative
        if not is_reviewed_page(safe_read_file(old_file)):
            stale_files.append(old_file)
    for moc in bundle.maps:
        if moc.slug not in dirty_mocs:
            continue
        page = render_moc_page(
            moc, ts,
            all_concepts=concept_map,
            cross_lingual_links=cross_lingual_links or None,
        )
        path = moc_paths[moc.slug]
        if _write_generated_page(path, page, bundle_dir):
            written.append(str(path))

    # ── Per-directory index.md ───────────────────────────────────────
    index_items = {
        "sources": [
            (source_paths[filename].stem, source.title)
            for filename, source in sources.items()
        ],
        "entries": [
            (path.stem, synthesis.source_title)
            for _key, synthesis, _stem, _slugs, path in entry_records
        ],
        "concepts": [
            (concept.slug, concept.title) for concept in bundle.concepts
        ],
        "mocs": [(moc.slug, moc.title) for moc in bundle.maps],
    }
    index_shard_size = _configured_int(config, "index_shard_size", 2_000)
    for dir_name, dir_path in dirs.items():
        written.extend(
            _render_directory_indexes(
                dir_name,
                dir_path,
                index_items[dir_name],
                bundle_dir,
                shard_size=index_shard_size,
            )
        )

    # ── Bundle-root index.md ─────────────────────────────────────────
    source_count = len(sources)
    entry_count = len(bundle.sources)
    concept_count = len(bundle.concepts)
    moc_count = len(bundle.maps)
    bundle_idx = render_bundle_index(
        bundle_dir, concept_count, entry_count, moc_count, source_count
    )
    bundle_idx_path = bundle_dir / "index.md"
    if _write_generated_page(bundle_idx_path, bundle_idx, bundle_dir):
        written.append(str(bundle_idx_path))

    # ── Graph visualization export ───────────────────────────────────
    # Export the knowledge graph as JSON (for D3.js / Obsidian graph view)
    # and Mermaid (for Obsidian embedding).
    try:
        from obsidian_llm_wiki.render.graph_export import export_graph
        graph_dir = bundle_dir / ".llmwiki"
        written.extend(
            export_graph(
                bundle,
                graph_dir,
                shard_size=_configured_int(config, "graph_shard_size", 5_000),
                monolith_threshold=_configured_int(
                    config, "graph_monolith_threshold", 10_000
                ),
            )
        )
    except Exception as exc:
        raise RuntimeError("Graph export failed during vault render") from exc

    # ── Dataview / Bases views ───────────────────────────────────────
    # Generate live Dataview views (concepts by confidence, MoCs by count,
    # contradictions by status, sources by freshness) for Obsidian users.
    try:
        from obsidian_llm_wiki.render.dataview import render_dataview_views
        written.extend(
            render_dataview_views(
                bundle_dir,
                bundle,
                sources,
                shard_size=_configured_int(config, "report_shard_size", 2_000),
            )
        )
    except Exception as exc:
        raise RuntimeError("Dataview view render failed") from exc

    # ── Source dependency graph ──────────────────────────────────────
    # Export which sources contributed to which concepts (JSON + Mermaid).
    try:
        from obsidian_llm_wiki.render.source_graph import export_source_dependency_graph
        dep_graph_dir = bundle_dir / ".llmwiki"
        dep_graph_paths = export_source_dependency_graph(bundle, dep_graph_dir)
        written.extend(dep_graph_paths)
    except Exception as exc:
        raise RuntimeError("Source dependency graph export failed") from exc

    # ── Vault log.md ─────────────────────────────────────────────────
    # Append only when the render changed an artifact. Otherwise every no-op
    # build grows log.md forever and defeats bounded steady-state writes.
    if written or stale_files:
        try:
            from obsidian_llm_wiki.render.log import append_log

            body_lines = [
                f"- sources: {source_count}",
                f"- entries: {entry_count}",
                f"- concepts: {concept_count}",
                f"- mocs: {moc_count}",
            ]
            if bundle.errors:
                body_lines.append(f"- errors: {len(bundle.errors)}")
            transaction = _ACTIVE_RENDER_TRANSACTION.get()
            expected_log_path = bundle_dir / "log.md"
            if transaction is not None:
                transaction.before_change(expected_log_path)
            log_path = append_log(
                bundle_dir,
                "build",
                f"rendered vault ({source_count} sources, {concept_count} concepts)",
                body=body_lines,
            )
            if transaction is not None:
                transaction.record_write(log_path, safe_read_file(log_path))
            written.append(str(log_path))
        except Exception as exc:
            raise RuntimeError("Vault log append failed") from exc

    for old_file in stale_files:
        relative_old_file = old_file.relative_to(bundle_dir).as_posix()
        if (
            relative_old_file in (protected_stale_paths or set())
            or not old_file.is_file()
            or is_reviewed_page(safe_read_file(old_file))
        ):
            continue
        transaction = _ACTIVE_RENDER_TRANSACTION.get()
        if transaction is not None:
            transaction.before_change(old_file)
            transaction.record_delete(old_file)
        old_file.unlink()
        logger.debug("Removed stale generated page: %s", old_file)

    if render_registry is not None:
        render_registry.commit_render_manifests(pending_manifests)

    return written


def render_vault(
    bundle_dir: Path,
    bundle: SynthesisBundle,
    sources: dict[str, SourceDoc],
    config: Any = None,
    *,
    cross_lingual_links: dict[str, list[tuple[str, float, str]]] | None = None,
    render_registry: VaultRegistry | None = None,
    force_render: bool = False,
    preserve_source_filenames: set[str] | None = None,
    protected_stale_paths: set[str] | None = None,
) -> list[str]:
    """Render a vault atomically with respect to generated output.

    Every pre-existing vault file (including reviewed metadata, indexes, and
    backup history) is restored if any render step raises. Rendering operates
    on a defensive bundle copy because normalization and MoC augmentation are
    renderer-local output preparation, not caller-visible state changes. Stale
    pages are removed only after every output has been written successfully.
    """
    transaction = _RenderTransaction(bundle_dir, lazy=True)
    render_bundle = deepcopy(bundle)
    token = _ACTIVE_RENDER_TRANSACTION.set(transaction)
    try:
        written = _render_vault(
            bundle_dir,
            render_bundle,
            sources,
            config,
            cross_lingual_links=cross_lingual_links,
            render_registry=render_registry,
            force_render=force_render,
            preserve_source_filenames=preserve_source_filenames,
            protected_stale_paths=protected_stale_paths,
        )
    except BaseException:
        try:
            transaction.rollback()
        except BaseException:
            logger.exception("Could not fully roll back failed vault render")
        raise
    finally:
        _ACTIVE_RENDER_TRANSACTION.reset(token)
    transaction.commit()
    return written
