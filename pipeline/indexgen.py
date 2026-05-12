"""Index + MOC generation (deterministic, no LLM).

Ported from llm-wiki-compiler/src/compiler/indexgen.ts.

Produces:
  wiki/index.md — Alphabetical concept listing
  wiki/MOC.md  — Tag-based Map of Content
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.markdown import atomic_write, parse_frontmatter, safe_read_file
from pipeline.models import PageSummary

# ── Scanning ────────────────────────────────────────────────────────────


def scan_wiki_pages(dir_path: str | Path) -> list[dict]:
    """Scan a directory for .md files with frontmatter.

    Returns a list of dicts: {title, slug, summary, tags, orphaned}.
    """
    dp = Path(dir_path)
    results: list[dict] = []

    if not dp.is_dir():
        return results

    for mdfile in sorted(dp.glob("*.md")):
        raw = safe_read_file(mdfile)
        if not raw.strip():
            continue
        meta, body = parse_frontmatter(raw)
        if not meta.get("title"):
            continue

        # Extract summary: first non-empty paragraph after frontmatter
        summary = _extract_summary(body)

        results.append({
            "title": meta.get("title", ""),
            "slug": meta.get("slug", mdfile.stem),
            "summary": summary,
            "tags": meta.get("tags", []),
            "orphaned": meta.get("orphaned", False),
        })

    return results


def collect_page_summaries(dir_path: str | Path) -> list[PageSummary]:
    """Scan directory and return list of PageSummary models."""
    pages = scan_wiki_pages(dir_path)
    return [
        PageSummary(
            title=p["title"],
            slug=p["slug"],
            summary=p["summary"],
            tags=p["tags"],
        )
        for p in pages
    ]


# ── Index ───────────────────────────────────────────────────────────────


def generate_index(wiki_dir: str | Path, concepts_dir: str | Path,
                   queries_dir: str | Path | None = None) -> Path:
    """Write wiki/index.md with alphabetical concept listing.

    Excludes orphaned pages.

    Returns path to the generated index file.
    """
    root = Path(wiki_dir)
    root.mkdir(parents=True, exist_ok=True)

    concepts = collect_page_summaries(concepts_dir)

    # Exclude orphaned (check via scan for orphaned flag)
    all_pages = scan_wiki_pages(concepts_dir)
    orphaned_slugs = {p["slug"] for p in all_pages if p["orphaned"]}

    active = [c for c in concepts if c.slug not in orphaned_slugs]
    active.sort(key=lambda c: c.title.lower())

    lines: list[str] = []
    lines.append("# Knowledge Wiki")
    lines.append("")
    lines.append("## Concepts")
    lines.append("")

    for c in active:
        tag_str = f" *{', '.join(c.tags)}*" if c.tags else ""
        lines.append(f"- **[[{c.slug}|{c.title}]]** — {c.summary}{tag_str}")

    lines.append("")

    index_path = root / "index.md"
    atomic_write(index_path, "\n".join(lines) + "\n")
    return index_path


# ── MOC ─────────────────────────────────────────────────────────────────


def generate_moc(wiki_dir: str | Path, concepts_dir: str | Path) -> Path:
    """Write wiki/MOC.md: tag-based sections with [[wikilinks]].

    Includes an "Uncategorized" section for tagless concepts.

    Returns path to the generated MOC file.
    """
    root = Path(wiki_dir)
    root.mkdir(parents=True, exist_ok=True)

    concepts = collect_page_summaries(concepts_dir)

    # Group by tags
    tag_groups: dict[str, list[PageSummary]] = {}
    uncategorized: list[PageSummary] = []

    for c in concepts:
        if not c.tags:
            uncategorized.append(c)
            continue
        for tag in c.tags:
            normalized = tag.strip().lower()
            if normalized:
                tag_groups.setdefault(normalized, []).append(c)

    # Sort groups
    sorted_tags = sorted(tag_groups.keys())
    uncategorized.sort(key=lambda c: c.title.lower())

    lines: list[str] = []
    lines.append("# Map of Content")
    lines.append("")
    lines.append("*Auto-generated tag-based navigation.*")
    lines.append("")

    for tag in sorted_tags:
        group = tag_groups[tag]
        # Deduplicate (a concept may appear under multiple tags)
        seen: set[str] = set()
        unique: list[PageSummary] = []
        for c in group:
            if c.slug not in seen:
                seen.add(c.slug)
                unique.append(c)
        unique.sort(key=lambda c: c.title.lower())

        display_tag = tag.title() if tag.islower() else tag
        lines.append(f"## {display_tag}")
        lines.append("")
        for c in unique:
            lines.append(f"- **[[{c.slug}|{c.title}]]** — {c.summary}")
        lines.append("")

    if uncategorized:
        lines.append("## Uncategorized")
        lines.append("")
        for c in uncategorized:
            lines.append(f"- **[[{c.slug}|{c.title}]]** — {c.summary}")
        lines.append("")

    moc_path = root / "MOC.md"
    atomic_write(moc_path, "\n".join(lines) + "\n")
    return moc_path


# ── Helpers ─────────────────────────────────────────────────────────────


def _extract_summary(body: str) -> str:
    """Extract first non-empty paragraph as summary, max ~200 chars."""
    if not body:
        return ""
    paragraphs = body.strip().split("\n\n")
    for para in paragraphs:
        stripped = para.strip()
        # Skip headers, blockquotes, code, and horizontal rules
        if stripped.startswith("#") or stripped.startswith(">"):
            continue
        if stripped.startswith("```") or stripped.startswith("---"):
            continue
        clean = _clean_inline(stripped)
        if clean:
            if len(clean) > 200:
                clean = clean[:197] + "..."
            return clean
    return ""


def _clean_inline(text: str) -> str:
    """Strip wikilinks to just display text, remove formatting markers."""
    # Replace [[slug|display]] → display, [[slug]] → slug
    text = re.sub(r"\[\[(?:[^\]|]+)\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove bold/italic/strikethrough/backtick markers
    text = text.replace("**", "").replace("*", "").replace("__", "")
    text = text.replace("_", "").replace("`", "").replace("~~", "")
    return text.strip()
