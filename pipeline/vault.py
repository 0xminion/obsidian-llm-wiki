"""Vault file operations module.

Handles all filesystem interactions with the Obsidian vault:
  - Writing sources, entries, concepts, MoCs
  - Edge (relationship) management in edges.tsv
  - URL deduplication via url-index.tsv
  - Wiki index rebuild (reindex)
  - Inbox archiving
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading as _threading
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pipeline.config import Config
from pipeline.models import Edge, ExtractedSource, Plan
from pipeline.utils import extract_frontmatter_field as _extract_frontmatter_field
from pipeline.utils import parse_url_file_content, title_to_filename

log = logging.getLogger(__name__)


# ─── Collision Detection ─────────────────────────────────────────────────────

def check_collision(directory: Path, filename: str) -> bool:
    """Return True if safe to write (file does NOT exist), False if collision."""
    target = directory / f"{filename}.md"
    return not target.exists()


def resolve_collision(directory: Path, filename: str) -> str:
    """Return a unique filename by appending -1, -2, etc. if needed."""
    candidate = filename
    counter = 1
    while not check_collision(directory, candidate):
        candidate = f"{filename}-{counter}"
        counter += 1
        if counter > 100:
            # Fallback: timestamp suffix
            return f"{filename}-{int(datetime.now(timezone.utc).timestamp())}"
    return candidate


# ─── YAML Frontmatter Helpers ─────────────────────────────────────────────────


def _format_yaml_value(key: str, value) -> str:
    """Format a single YAML key-value pair."""
    if isinstance(value, list):
        if not value:
            return f"{key}: []"
        escaped_items = []
        for item in value:
            item_str = str(item)
            # Escape items with special YAML chars or newlines
            if re.search(r"[\n:#{}\[\],&*?|>!%@`'\"\\]", item_str) or item_str.startswith("- "):
                escaped_items.append(f'  - "{item_str.replace(chr(34), chr(92)+chr(34))}"')
            else:
                escaped_items.append(f"  - {item_str}")
        items = "\n".join(escaped_items)
        return f"{key}:\n{items}"
    if value is None or value == "":
        return f'{key}: ""'
    if isinstance(value, str) and value.startswith("[[") and value.endswith("]]"):
        return f'{key}: "{value}"'
    # Handle newlines in string values
    if isinstance(value, str) and "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key}: "{escaped}"'
    # Quote strings that contain special YAML chars (but not URLs — they're safe)
    if isinstance(value, str) and re.search(r"[#{}\[\],&*?|>!%@`]", value):
        return f'{key}: "{value}"'
    if isinstance(value, str) and ":" in value and not value.startswith("http"):
        return f'{key}: "{value}"'
    return f"{key}: {value}"



def _build_frontmatter(fields: dict) -> str:
    """Build a YAML frontmatter block from a dict of fields using PyYAML.

    Deterministic output: no anchors (anchors are never used in Obsidian),
    no arbitrary line wrapping, Unicode preserved.
    """
    dumped = yaml.safe_dump(
        fields,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=float("inf"),
        explicit_start=False,
    )
    return "---\n" + dumped.strip() + "\n---\n"


# ─── Write Source ─────────────────────────────────────────────────────────────

def write_source(cfg: Config, source: ExtractedSource) -> Path:
    """Write a source note to cfg.sources_dir.

    Returns the Path of the created file.
    Raises FileExistsError if collision detected (after resolution).
    """
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)

    filename = title_to_filename(source.title)
    filename = resolve_collision(cfg.sources_dir, filename)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    frontmatter = _build_frontmatter({
        "title": source.title,
        "source_url": source.url,
        "source_type": source.type.value if hasattr(source.type, "value") else str(source.type),
        "author": source.author or "",
        "date_captured": now,
        "tags": [],
        "status": "raw",
    })

    body = f"# {source.title}\n\n{source.content}\n"
    content = frontmatter + body

    target = cfg.sources_dir / f"{filename}.md"
    target.write_text(content, encoding="utf-8")
    return target


# ─── Write Entry ──────────────────────────────────────────────────────────────

def write_entry(cfg: Config, plan: Plan, content: str, source_note_name: str | None = None) -> Path:
    """Write an entry note to cfg.entries_dir.

    Returns the Path of the created file.
    """
    cfg.entries_dir.mkdir(parents=True, exist_ok=True)

    filename = title_to_filename(plan.title)
    filename = resolve_collision(cfg.entries_dir, filename)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    source_name = source_note_name or title_to_filename(plan.title)
    source_link = f"[[{source_name}]]"

    frontmatter = _build_frontmatter({
        "title": plan.title,
        "source": source_link,  # links to the Source note (same filename)
        "date_entry": now,
        "status": "draft",
        "template": plan.template.value if hasattr(plan.template, "value") else str(plan.template),
        "tags": plan.tags,
    })

    full_content = frontmatter + content
    # Safety net: agent/review paths may omit the heading — templates always include it
    if not content.strip().startswith("#"):
        full_content = frontmatter + f"# {plan.title}\n\n{content}"

    target = cfg.entries_dir / f"{filename}.md"
    target.write_text(full_content, encoding="utf-8")
    return target


# ─── Write Concept ────────────────────────────────────────────────────────────

def write_concept(cfg: Config, name: str, content: str, sources: list[str]) -> Path:
    """Write a concept note to cfg.concepts_dir.

    Returns the Path of the created file.
    """
    cfg.concepts_dir.mkdir(parents=True, exist_ok=True)

    filename = title_to_filename(name)
    filename = resolve_collision(cfg.concepts_dir, filename)

    frontmatter = _build_frontmatter({
        "title": name,
        "type": "concept",
        "status": "draft",
        "sources": sources,
        "tags": [],
    })

    full_content = frontmatter + content
    if not content.strip().startswith("#"):
        full_content = frontmatter + f"# {name}\n\n{content}"

    target = cfg.concepts_dir / f"{filename}.md"
    target.write_text(full_content, encoding="utf-8")
    return target


# ─── Update MoC ───────────────────────────────────────────────────────────────

def update_moc(cfg: Config, moc_name: str, entry_name: str, description: str) -> None:
    """Append an entry under a topic section in a MoC file.

    Creates the MoC if it doesn't exist.
    """
    cfg.mocs_dir.mkdir(parents=True, exist_ok=True)

    filename = title_to_filename(moc_name)
    moc_path = cfg.mocs_dir / f"{filename}.md"

    entry_line = f"- [[{entry_name}]]: {description}"

    if not moc_path.exists():
        # Create new MoC with basic structure
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content = (
            f"---\ntitle: {moc_name}\ntype: moc\nstatus: draft\ncreated: {today}\ntags: []\n---\n\n"
            f"# {moc_name}\n\n"
            f"## Overview / 概述\n\n"
            f"Map of Content for {moc_name}.\n\n"
            f"---\n\n"
            f"## Entries\n\n"
            f"{entry_line}\n"
        )
        moc_path.write_text(content, encoding="utf-8")
        return

    # MoC exists — try to append under a generic "Entries" section or last section
    existing = moc_path.read_text(encoding="utf-8")

    # Find the last ## heading section or "Entries" section
    # Strategy: find "## Entries" or the last ## heading, append after its content
    lines = existing.split("\n")
    insert_idx = None

    # First try to find "## Entries" section
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") and "entries" in stripped.lower():
            # Found Entries section — find where to insert (before next ## or end)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    insert_idx = j
                    break
            else:
                insert_idx = len(lines)
            break

    # If no Entries section, create one and append entry_line (no insert needed)
    if insert_idx is None:
        lines.append("")
        lines.append("## Entries")
        lines.append("")
        lines.append(entry_line)
        moc_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # Check for duplicate before inserting
    # Use regex to match exact wikilink (not substring like [[AI]] inside [[AI Safety]])
    full_text = "\n".join(lines)
    if not re.search(rf'\[\[{re.escape(entry_name)}(?:\|[^\]]*)?\]\]', full_text):
        lines.insert(insert_idx, entry_line)

    moc_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Edge Management ─────────────────────────────────────────────────────────

# In-memory edge cache to avoid O(N²) reads on every write.
# Thread-safe via lock for concurrent write_edge() calls.

_edge_cache: set[tuple[str, str, str]] | None = None
_edge_cache_path: Path | None = None
_edge_cache_lock = _threading.Lock()


def _load_edge_cache(edges_file: Path) -> set[tuple[str, str, str]]:
    """Load existing edges into an in-memory set for O(1) duplicate checks."""
    global _edge_cache, _edge_cache_path
    with _edge_cache_lock:
        if _edge_cache is not None and _edge_cache_path == edges_file:
            return _edge_cache

        cache: set[tuple[str, str, str]] = set()
        if edges_file.exists():
            for line in edges_file.read_text(encoding="utf-8").strip().split("\n"):
                if line.startswith("source\t") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    cache.add((parts[0], parts[1], parts[2]))
        _edge_cache = cache
        _edge_cache_path = edges_file
        return cache


def clear_edge_cache() -> None:
    """Reset the edge cache. Call after external edge file modifications."""
    global _edge_cache, _edge_cache_path
    with _edge_cache_lock:
        _edge_cache = None
        _edge_cache_path = None


def write_edge(cfg: Config, edge: Edge) -> None:
    """Append an edge to edges.tsv. Skips if duplicate.

    Uses an in-memory cache for O(1) duplicate checks instead of
    re-reading the file on every call. Thread-safe.
    """
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    edges_file = cfg.edges_file

    # Create with header if missing
    if not edges_file.exists():
        edges_file.write_text("source\ttarget\ttype\tdescription\n", encoding="utf-8")

    # Check duplicate via cache (O(1) instead of O(N))
    cache = _load_edge_cache(edges_file)
    edge_key = (edge.source, edge.target, edge.type.value)
    with _edge_cache_lock:
        if edge_key in cache:
            return  # duplicate, skip
        # Append to file
        with edges_file.open("a", encoding="utf-8") as f:
            f.write(edge.to_tsv() + "\n")
        # Update cache
        cache.add(edge_key)


def read_edges(cfg: Config) -> list[Edge]:
    """Read all edges from edges.tsv."""
    edges_file = cfg.edges_file
    if not edges_file.exists():
        return []

    edges = []
    for line in edges_file.read_text(encoding="utf-8").strip().split("\n"):
        if line.startswith("source\t"):
            continue  # skip header
        edge = Edge.from_tsv(line)
        if edge:
            edges.append(edge)
    return edges


# ─── URL Deduplication ────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Normalize URL: strip protocol and trailing slash."""
    s = re.sub(r"^https?://", "", url)
    s = s.rstrip("/")
    return s


def url_exists(cfg: Config, url: str) -> bool:
    """Check if a URL is already registered in url-index.tsv."""
    url_index = cfg.url_index
    if not url_index.exists():
        return False

    normalized = _normalize_url(url).lower()
    for line in url_index.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts and _normalize_url(parts[0]).lower() == normalized:
            return True
    return False


def register_url(cfg: Config, url: str, entry_name: str) -> None:
    """Register a URL → entry mapping in url-index.tsv. Skips if already exists."""
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    url_index = cfg.url_index

    if url_exists(cfg, url):
        return

    normalized = _normalize_url(url)
    with url_index.open("a", encoding="utf-8") as f:
        f.write(f"{normalized}\t{entry_name}\n")


# ─── Reindex ──────────────────────────────────────────────────────────────────

_INDEX_HEADER = """\
# Wiki Index

Auto-maintained table of contents for the knowledge base.
Each entry and concept has a 1-sentence summary for retrieval.
This index is the primary retrieval layer — the LLM reads this
to find relevant notes instead of using RAG.

---

"""



def _extract_summary(content: str) -> str:
    """Extract first sentence from Summary / 摘要 section."""
    # Try English ## Summary
    match = re.search(r"^## Summary\s*\n(.*?)(?:\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    if match:
        text = match.group(1).strip().split("\n")[0].strip()
        if text:
            return text
    # Try Chinese ## 摘要
    match = re.search(r"^## 摘要\s*\n(.*?)(?:\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    if match:
        text = match.group(1).strip().split("\n")[0].strip()
        if text:
            return text
    # Try body after first heading
    match = re.search(r"^# .+\n+(.*?)(?:\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    if match:
        text = match.group(1).strip().split("\n")[0].strip()
        if text:
            return text
    return ""


def _extract_overview(content: str) -> str:
    """Extract first sentence from Overview / 概述 section."""
    match = re.search(r"^## Overview.*?\n(.*?)(?:\n##|\Z)", content, re.MULTILINE | re.DOTALL)
    if match:
        text = match.group(1).strip().split("\n")[0].strip()
        if text:
            return text
    return ""


def reindex(cfg: Config) -> str:
    """Rebuild wiki-index.md from all sources/entries/concepts/mocs.

    Returns the index content as a string.
    """
    lines = [_INDEX_HEADER]
    entries_added = 0
    concepts_added = 0
    mocs_added = 0

    # Scan Entries
    lines.append("## Entries\n")
    if cfg.entries_dir.exists():
        for entry_path in sorted(cfg.entries_dir.glob("*.md")):
            note_name = entry_path.stem
            try:
                content = entry_path.read_text(encoding="utf-8")
            except OSError:
                continue
            title = _extract_frontmatter_field(content, "title") or note_name
            summary = _extract_summary(content) or title
            lines.append(f"- [[{note_name}]]: {summary} (entry)")
            entries_added += 1
    lines.append("")

    # Scan Concepts
    lines.append("## Concepts\n")
    if cfg.concepts_dir.exists():
        for concept_path in sorted(cfg.concepts_dir.glob("*.md")):
            note_name = concept_path.stem
            try:
                content = concept_path.read_text(encoding="utf-8")
            except OSError:
                continue
            title = _extract_frontmatter_field(content, "title") or note_name
            body_text = _extract_summary(content) or title
            lines.append(f"- [[{note_name}]]: {body_text} (concept)")
            concepts_added += 1
    lines.append("")

    # Scan MoCs
    if cfg.mocs_dir.exists():
        moc_files = sorted(cfg.mocs_dir.glob("*.md"))
        if moc_files:
            lines.append("## Maps of Content\n")
            for moc_path in moc_files:
                note_name = moc_path.stem
                try:
                    content = moc_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                title = _extract_frontmatter_field(content, "title") or note_name
                overview = _extract_overview(content) or title
                lines.append(f"- [[{note_name}]]: {overview} (moc)")
                mocs_added += 1
            lines.append("")

    # Summary footer
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines.append("---\n")
    lines.append(f"*Reindexed on {today}: {entries_added} entries, {concepts_added} concepts, {mocs_added} MoCs*\n")

    index_content = "\n".join(lines)

    # Write to disk
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    from pipeline.utils import _atomic_write
    _atomic_write(cfg.wiki_index, index_content)

    return index_content


# ─── Inbox Archive ────────────────────────────────────────────────────────────

def archive_inbox(cfg: Config, hashes: set[str]) -> int:
    """Move processed .url files from inbox to archive.

    Files whose hash (derived from URL in the .url file) is in the given set
    are moved to cfg.archive_dir.

    Returns the count of files archived.
    """
    if not cfg.inbox_dir.exists():
        return 0

    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    import hashlib

    count = 0
    for url_file in cfg.inbox_dir.glob("*.url"):
        content = url_file.read_text(encoding="utf-8", errors="replace")
        url = parse_url_file_content(content)
        if not url:
            continue

        file_hash = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]

        if file_hash in hashes:
            target = cfg.archive_dir / url_file.name
            if target.exists():
                idx = 1
                while True:
                    candidate = cfg.archive_dir / f"{url_file.stem}-{idx}{url_file.suffix}"
                    if not candidate.exists():
                        target = candidate
                        break
                    idx += 1
            url_file.rename(target)
            count += 1

    return count


# ─── Clippings Archive ─────────────────────────────────────────────────────

def archive_clippings(
    cfg: Config,
    archived_hashes: set[str],
    extract_dir: Path | None = None,
) -> int:
    """Move processed clipping files from 02-Clippings to archive.

    Each clipping's URL-derived hash is checked against `archived_hashes`.
    Matching files are moved to cfg.clippings_archive_dir.

    Returns the count of files archived.
    """
    clippings_dir = cfg.clippings_dir
    if not clippings_dir.exists():
        return 0

    cfg.clippings_archive_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for md_file in clippings_dir.glob("*.md"):
        h = _clipping_hash(md_file)
        if not h:
            continue
        if h not in archived_hashes:
            continue

        target = cfg.clippings_archive_dir / md_file.name
        if target.exists():
            idx = 1
            max_attempts = 100
            while idx <= max_attempts:
                candidate = cfg.clippings_archive_dir / f"{md_file.stem}-{idx}{md_file.suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                idx += 1
            else:
                log.warning("Too many archive collisions for %s — skipping", md_file.name)
                continue

        try:
            md_file.rename(target)
            count += 1
        except OSError:
            log.exception("Failed to archive clipping %s", md_file.name)

    return count


def _clipping_hash(md_file: Path) -> str | None:
    """Compute the source URL hash for a clipping markdown file.

    Reads frontmatter & body from the file, resolves the URL the same way
    parse_clipping_file does, then returns its 12-char MD5 hash.
    """
    from pipeline.utils import extract_body, parse_frontmatter

    try:
        text = md_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    fm = parse_frontmatter(text)
    body = extract_body(text)

    url = ""
    for key in ("source_url", "url", "source"):
        if fm.get(key):
            url = str(fm[key]).strip()
            break
    if not url:
        m = re.search(r"https?://\S+", body)
        url = m.group(0) if m else ""
    if not url:
        return None

    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]
