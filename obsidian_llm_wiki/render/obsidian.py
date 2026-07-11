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


def _sanitize_tag(tag: str) -> str:
    """Sanitize a single tag for Obsidian compatibility.

    Obsidian tags cannot contain spaces. Replace spaces with hyphens.
    Also strips leading/trailing whitespace and removes special chars
    that break YAML parsing.
    """
    tag = (tag or "").strip()
    # Replace spaces with hyphens
    tag = re.sub(r"\s+", "-", tag)
    # Remove characters that break Obsidian tags
    tag = re.sub(r"[#\"'`,;()\[\]{}]", "", tag)
    return tag


def build_frontmatter(fm_dict: dict[str, Any]) -> str:
    """Serialize a dict to a ``---``-delimited YAML frontmatter block.

    Tags are sanitized: spaces → hyphens, special chars removed.
    """
    # Sanitize tags if present
    if "tags" in fm_dict and isinstance(fm_dict["tags"], list):
        fm_dict = dict(fm_dict)  # shallow copy to avoid mutating caller's dict
        fm_dict["tags"] = [
            _sanitize_tag(t) for t in fm_dict["tags"]
            if t and str(t).strip()
        ]

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


def _build_cross_ref_diagram(
    concept: ConceptNote,
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build the typed-edge cross-reference section as an ASCII flow diagram.

    Renders inside a code block (```text) so Obsidian shows it as monospace
    with a copy icon. Wikilinks inside code blocks render as literal [[slug]]
    text — the user's screenshot shows them this way intentionally.

    Format matches the user's screenshot:
      注意力价值 (Attention Value)
          ↓ tokenized by
      Pump.fun → 人人可发币 → Meme币爆发
          ↓ amplified by
      KOL/交易所注意力捕获
          ↓ creates
      流动性收割 ↔ 上市即出货

      Cross-links: [[slug]] (descriptor)
                   [[slug]] (descriptor)
    """
    lines: list[str] = []

    # Build the flow diagram from this concept's related edges
    for link in concept.related:
        target = all_concepts.get(link.slug)
        if not target:
            continue

        target_display = link.display or target.slug
        relation_type = link.relation or "related_to"

        # Primary edge: this concept → relation → target
        lines.append(f"{concept.title}")
        lines.append(f"    ↓ {relation_type}")

        # Target line with any sub-relationships
        target_sub = ""
        if target.related:
            sub_links = [
                r for r in target.related
                if r.slug != concept.slug
            ][:2]
            if sub_links:
                sub_parts = []
                for sl in sub_links:
                    sub_target = all_concepts.get(sl.slug)
                    if sub_target:
                        sub_parts.append(sub_target.title)
                if sub_parts:
                    target_sub = " → " + " → ".join(sub_parts)

        # Check for bidirectional edge
        reverse_links = [
            r for r in (target.related or [])
            if r.slug == concept.slug
        ]
        if reverse_links and not target_sub:
            rev_rel = reverse_links[0].relation or "related_to"
            lines.append(
                f"{target.title} ↔ {concept.title} ({rev_rel})"
            )
        elif target_sub:
            lines.append(f"{target_display}{target_sub}")
        else:
            lines.append(f"{target_display}")

    # Cross-links section (wikilinks as literal text inside code block)
    if lines and concept.related:
        cross_link_lines: list[str] = []
        for link in concept.related:
            target = all_concepts.get(link.slug)
            if target:
                descriptor = link.relation or "related"
                cross_link_lines.append(
                    f"  [[{link.slug}]] ({descriptor})"
                )
        if cross_link_lines:
            lines.append("")
            lines.append("Cross-links:")
            lines.extend(cross_link_lines)

    return lines


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


def _build_moc_cross_ref_diagram(
    moc_concepts: list[ConceptNote],
    all_concepts: dict[str, ConceptNote],
) -> list[str]:
    """Build ASCII flow diagram matching concept cross-ref format.

    Format (consistent with _build_cross_ref_diagram):
      Concept A
          ↓ relation
      Concept B → Concept C

      Cross-links:
        [[slug-a]] (relation)
        [[slug-b]] (relation)
    """
    lines: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    moc_slugs = {c.slug for c in moc_concepts}

    for concept in moc_concepts:
        for link in concept.related or []:
            if link.slug not in all_concepts:
                continue
            if link.slug not in moc_slugs:
                continue

            pair_key = tuple(sorted([concept.slug, link.slug]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            target = all_concepts[link.slug]
            relation = link.relation or "related_to"

            # Check if bidirectional
            reverse = any(
                r.slug == concept.slug
                for r in (target.related or [])
            )

            # Flow format: concept on one line, relation on next, target on third
            lines.append(concept.title)
            lines.append(f"    ↓ {relation}")
            if reverse:
                lines.append(f"{target.title} ↔ {concept.title}")
            else:
                lines.append(f"{target.title}")
            lines.append("")

    # Cross-links section with [[slug]] (relation) — same as concepts
    if lines:
        cross_links: list[str] = []
        seen_link_slugs: set[str] = set()
        for concept in moc_concepts:
            for link in concept.related or []:
                if link.slug not in moc_slugs:
                    continue
                pair = tuple(sorted([concept.slug, link.slug]))
                if pair in seen_link_slugs:
                    continue
                seen_link_slugs.add(pair)
                descriptor = link.relation or "related"
                cross_links.append(
                    f"  [[{link.slug}]] ({descriptor})"
                )
        if cross_links:
            if lines and lines[-1] == "":
                lines.pop()
            lines.append("")
            lines.append("Cross-links:")
            lines.extend(cross_links)

    return lines


def _is_chinese(text: str) -> bool:
    """Return True if text contains Chinese characters (but not Japanese)."""
    import re
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_kana = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    return has_cjk and not has_kana


def _has_latin(text: str) -> bool:
    """Return True if text contains Latin alphabet characters."""
    return bool(re.search(r"[A-Za-z]", text or ""))


def _title_from_slug(slug: str) -> str:
    """Convert a slug to a readable English title fallback."""
    words = [w for w in re.split(r"[-_]+", slug or "") if w]
    if not words:
        return "Untitled"
    small = {
        "a", "an", "and", "as", "at", "by", "for", "in",
        "of", "on", "or", "the", "to", "vs", "via",
    }
    titled: list[str] = []
    for idx, word in enumerate(words):
        lower = word.lower()
        if lower in {"usdt", "swift", "tbml", "llm", "ai"}:
            titled.append(lower.upper())
        elif idx > 0 and lower in small:
            titled.append(lower)
        else:
            titled.append(lower.capitalize())
    return " ".join(titled)


def _split_bilingual_title(title: str) -> tuple[str, str] | None:
    """Return (english, chinese) if title already has both languages."""
    title = (title or "").strip()
    if not (_is_chinese(title) and _has_latin(title)):
        return None

    # Desired format: English first, Chinese in parentheses.
    m = re.fullmatch(r"(?P<en>[A-Za-z][^)）]+?)\s*[（(](?P<zh>.*[\u4e00-\u9fff].*)[)）]", title)
    if m:
        return m.group("en").strip(), m.group("zh").strip()

    # Common model failure: Chinese first, English in parentheses.
    m = re.fullmatch(r"(?P<zh>.*[\u4e00-\u9fff].*?)\s*[（(](?P<en>[A-Za-z][^)）]+)[)）]", title)
    if m:
        return m.group("en").strip(), m.group("zh").strip()

    return None


def _english_alias(aliases: list[str]) -> str:
    """Pick the first English alias if one exists."""
    for alias in aliases or []:
        alias = (alias or "").strip()
        if alias and _has_latin(alias) and not _is_chinese(alias):
            return alias
    return ""


def _ensure_english_first_bilingual(
    title: str,
    *,
    slug: str = "",
    aliases: list[str] | None = None,
) -> str:
    """Return English-first bilingual title when title contains Chinese.

    If the model produced a Chinese-only title, derive the English side from an
    English alias first, then from the canonical slug. This is intentionally
    deterministic: filenames and wikilinks must not depend on another LLM call.
    """
    title = (title or "").strip()
    if not _is_chinese(title):
        return title

    existing = _split_bilingual_title(title)
    if existing:
        english, chinese = existing
        return f"{english} ({chinese})"

    english = _english_alias(aliases or []) or _title_from_slug(slug)
    return f"{english} ({title})"


def _bilingual_slug(english_first_title: str) -> str:
    """Filename-safe slug that keeps both English and Chinese title parts."""
    return slugify(english_first_title)


def _english_side(title: str) -> str:
    """Return the English side of an English-first bilingual title."""
    if not title:
        return ""
    existing = _split_bilingual_title(title)
    if existing:
        return existing[0]
    if _has_latin(title):
        return re.split(r"[（(]", title, maxsplit=1)[0].strip()
    return ""


def _normalize_bilingual_titles_and_slugs(bundle: SynthesisBundle) -> None:
    """Normalize Chinese-derived titles/slugs across a bundle in-place."""
    slug_map: dict[str, str] = {}

    # Concepts: normalize title and slug, then remap all relationship targets.
    for concept in bundle.concepts:
        if _is_chinese(concept.title):
            old_slug = concept.slug
            concept.title = _ensure_english_first_bilingual(
                concept.title,
                slug=concept.slug,
                aliases=concept.aliases,
            )
            concept.slug = _bilingual_slug(concept.title)
            if old_slug and old_slug != concept.slug:
                slug_map[old_slug] = concept.slug

    # Source-local concept objects may not be the same instances as bundle.concepts.
    for synthesis in bundle.sources:
        for concept in synthesis.concepts:
            if concept.slug in slug_map:
                concept.slug = slug_map[concept.slug]
            if _is_chinese(concept.title):
                concept.title = _ensure_english_first_bilingual(
                    concept.title,
                    slug=concept.slug,
                    aliases=concept.aliases,
                )
                concept.slug = _bilingual_slug(concept.title)

    def remap_links(concepts: list[ConceptNote]) -> None:
        for concept in concepts:
            for link in concept.related:
                if link.slug in slug_map:
                    link.slug = slug_map[link.slug]

    remap_links(bundle.concepts)
    for synthesis in bundle.sources:
        remap_links(synthesis.concepts)

    # MoCs: normalize titles/slugs and remap concept references.
    for moc in bundle.maps:
        moc.concept_slugs = [slug_map.get(slug, slug) for slug in moc.concept_slugs]
        if _is_chinese(moc.title):
            moc.title = _ensure_english_first_bilingual(moc.title, slug=moc.slug)
            moc.slug = _bilingual_slug(moc.title)

    for synthesis in bundle.sources:
        for moc in synthesis.maps:
            moc.concept_slugs = [slug_map.get(slug, slug) for slug in moc.concept_slugs]
            if _is_chinese(moc.title):
                moc.title = _ensure_english_first_bilingual(moc.title, slug=moc.slug)
                moc.slug = _bilingual_slug(moc.title)

    # Entries from Chinese sources: source titles do not have a canonical English
    # slug, so derive the English side from the first two linked concepts.
    for synthesis in bundle.sources:
        if not _is_chinese(synthesis.source_title):
            continue
        if _has_latin(synthesis.source_title):
            synthesis.source_title = _ensure_english_first_bilingual(
                synthesis.source_title,
                slug=slugify(synthesis.source_title),
            )
            continue
        concept_titles = [
            _english_side(concept.title) or _title_from_slug(concept.slug)
            for concept in synthesis.concepts[:2]
            if concept.slug
        ]
        english = " and ".join(concept_titles) if concept_titles else "Chinese Source"
        synthesis.source_title = f"{english} ({synthesis.source_title})"


def _moc_needs_bilingual_headings(
    moc: MapOfContent,
    concepts: list[ConceptNote | None],
) -> bool:
    """Return True for Chinese or multilingual MoCs."""
    titles = [moc.title, *(c.title for c in concepts if c)]
    has_chinese = any(_is_chinese(t) for t in titles)
    has_english = any(_has_latin(t) for t in titles)
    return has_chinese and has_english


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
    # from cache without going through the full pipeline.
    from obsidian_llm_wiki.synth.dedupe import propagate_backlinks
    propagate_backlinks(bundle)

    # ── Semantic concept dedup ───────────────────────────────────────
    # Merge same-language concepts with high embedding similarity.
    try:
        from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts
        dedup_threshold = (
            config.similarity_dedup_threshold if config else 0.85
        )
        semantic_dedupe_concepts(bundle, threshold=dedup_threshold)
    except Exception as exc:
        logger.debug("Semantic dedup skipped: %s", exc)

    # ── Embedding-based MoC assignment for orphans ───────────────────
    # Assign concepts not in any MoC to the most semantically similar MoC.
    try:
        from obsidian_llm_wiki.synth.dedupe import assign_orphans_to_mocs
        moc_threshold = (
            config.moc_assignment_threshold if config else 0.55
        )
        assign_orphans_to_mocs(bundle, threshold=moc_threshold)
    except Exception as exc:
        logger.debug("MoC orphan assignment skipped: %s", exc)

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
