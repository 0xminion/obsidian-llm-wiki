"""Deterministic compile operations — wiki index, edges, duplicate detection."""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from pipeline.config import Config
from pipeline.models import Edge, EdgeType
from pipeline.utils import frontmatter_list_items as _frontmatter_list_items

log = logging.getLogger(__name__)


def _edge_key(source: str, target: str, edge_type: str) -> tuple[str, str, str]:
    """Canonicalize edge keys so symmetric relationships are idempotent."""
    if edge_type == EdgeType.RELATES_TO.value:
        left, right = sorted([source, target])
        return (left, right, edge_type)
    return (source, target, edge_type)


def _rebuild_wiki_index(cfg: Config) -> bool:
    """Rebuild wiki-index.md from vault content. Returns True if successful."""
    from pipeline.vault import reindex as vault_reindex
    try:
        content = vault_reindex(cfg)
        log.info("Rebuilt wiki-index.md (%d lines)", content.count("\n"))
        return True
    except OSError:
        log.exception("Failed to rebuild wiki-index.md")
        return False


def _build_edges(cfg: Config, bidirectional: bool = False) -> int:
    """Rebuild graph edges by scanning all notes for wikilinks, concept-entry evidence,
    MoC membership, and shared-tag inference (>=2 tags), then optionally add weak-link
    reverses for asymmetric edges. Manual (non-generated) edges are preserved. Returns
    the number of generated edges written, or 0 if the file is unchanged.
    """
    edges_file = cfg.edges_file
    edges_file.parent.mkdir(parents=True, exist_ok=True)
    previous_content = edges_file.read_text(encoding="utf-8", errors="replace") if edges_file.exists() else ""

    manual_edges: list[tuple[str, str, str, str]] = []
    generated_descriptions = {
        "auto-detected wikilink",
        "entry provides evidence for concept",
        "note belongs to MoC",
    }
    if edges_file.exists():
        for line in edges_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#") or not line.strip() or line.startswith("source\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            source, target, edge_type = parts[:3]
            description = parts[3] if len(parts) > 3 else ""
            if description in generated_descriptions or description.startswith("shared tags:"):
                continue
            manual_edges.append((source.strip(), target.strip(), edge_type.strip(), description.strip()))

    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()
    notes: dict[str, dict] = {}

    for note_dir, note_type in [
        (cfg.sources_dir, "source"),
        (cfg.entries_dir, "entry"),
        (cfg.concepts_dir, "concept"),
        (cfg.mocs_dir, "moc"),
    ]:
        if not note_dir.exists():
            continue
        for md in note_dir.glob("*.md"):
            name = md.stem
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            tags = set()
            sources = []
            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match:
                fm = fm_match.group(1)
                tags = {tag.lower() for tag in _frontmatter_list_items(fm, "tags") if tag}
                sources = [source for source in _frontmatter_list_items(fm, "sources") if source]

            links = set(re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", content))
            notes[name] = {
                "type": note_type,
                "tags": tags,
                "links": links,
                "sources": sources,
            }

    def add_edge(source: str, target: str, edge_type: EdgeType, description: str) -> None:
        key = _edge_key(source, target, edge_type.value)
        if key in seen:
            return
        source_name, target_name, _ = key
        edges.append(Edge(source=source_name, target=target_name, type=edge_type, description=description))
        seen.add(key)

    for name, info in notes.items():
        for linked in info["links"]:
            if linked in notes and linked != name:
                add_edge(name, linked, EdgeType.RELATES_TO, "auto-detected wikilink")

        if info["type"] == "concept":
            for src in info["sources"]:
                src_clean = re.sub(r"^\[\[|\]\]$", "", src).split("|", 1)[0].split("#", 1)[0]
                if src_clean in notes and notes[src_clean]["type"] == "entry":
                    add_edge(src_clean, name, EdgeType.TESTED_BY, "entry provides evidence for concept")

        if info["type"] == "moc":
            for linked in info["links"]:
                if linked in notes:
                    add_edge(linked, name, EdgeType.PART_OF, "note belongs to MoC")

    concept_names = sorted(name for name, info in notes.items() if info["type"] == "concept")
    for i, name in enumerate(concept_names):
        for other_name in concept_names[i + 1:]:
            shared_tags = notes[name]["tags"] & notes[other_name]["tags"]
            if len(shared_tags) >= 2:
                add_edge(name, other_name, EdgeType.RELATES_TO, f"shared tags: {', '.join(sorted(shared_tags)[:3])}")

    if bidirectional:
        asymmetric_types = {
            EdgeType.EXTENDS.value, EdgeType.CONTRADICTS.value, EdgeType.SUPPORTS.value,
            EdgeType.SUPERSEDES.value, EdgeType.TESTED_BY.value, EdgeType.DEPENDS_ON.value,
            EdgeType.INSPIRED_BY.value, EdgeType.PART_OF.value,
        }
        directed = {
            (e.source, e.target, e.type.value) for e in edges if e.type.value in asymmetric_types
        }
        for source, target, etype in list(directed):
            if (target, source, etype) not in directed:
                add_edge(target, source, EdgeType.WEAK_LINK, f"inferred reverse ({etype})")

    lines = ["source\ttarget\ttype\tdescription"]
    valid_notes = set(notes)
    written_edges: set[tuple[str, str, str]] = set()
    for source, target, edge_type, description in manual_edges:
        if source in valid_notes and target in valid_notes:
            key = _edge_key(source, target, edge_type)
            if key not in written_edges:
                source_name, target_name, type_name = key
                lines.append(f"{source_name}\t{target_name}\t{type_name}\t{description}")
                written_edges.add(key)
    lines.extend(edge.to_tsv() for edge in edges if _edge_key(edge.source, edge.target, edge.type.value) not in written_edges)
    new_content = "\n".join(lines) + "\n"
    from pipeline.utils import _atomic_write
    _atomic_write(edges_file, new_content)
    log.info("Rebuilt edges.tsv with %d generated edges", len(edges))
    return 0 if previous_content == new_content else len(edges)


def _detect_duplicates(cfg: Config) -> int:
    """Flag potential duplicates using word-overlap similarity on titles. Two same-type
    notes are flagged when (shared words / min word count) > 0.7.
    """
    report_lines = []
    notes: list[tuple[str, str, str]] = []

    for note_dir, note_type in [(cfg.entries_dir, "entry"), (cfg.concepts_dir, "concept")]:
        if not note_dir.exists():
            continue
        for md in note_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                title = md.stem
                if fm_match:
                    t_match = re.search(r"title:\s*[\"']?(.*?)[\"']?\s*$", fm_match.group(1), re.MULTILINE)
                    if t_match:
                        title = t_match.group(1).strip()
                notes.append((md.stem, note_type, title))
            except OSError:
                continue

    dup_count = 0
    for i, (name_a, type_a, title_a) in enumerate(notes):
        for name_b, type_b, title_b in notes[i + 1:]:
            if type_a != type_b:
                continue
            words_a = set(re.sub(r"[^a-zA-Z0-9一-鿿]", " ", title_a.lower()).split())
            words_b = set(re.sub(r"[^a-zA-Z0-9一-鿿]", " ", title_b.lower()).split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > 0.7 and name_a != name_b:
                report_lines.append(f"- **{name_a}** ↔ **{name_b}** (overlap: {overlap:.0%}, type: {type_a})")
                dup_count += 1

    if report_lines:
        report_dir = cfg.vault_path / "Meta" / "Scripts"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "compile-duplicate-report.md"
        report_content = (
            f"# Duplicate Detection Report\n\n"
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Found {dup_count} potential duplicate pairs:\n\n"
            + "\n".join(report_lines) + "\n"
        )
        from pipeline.utils import _atomic_write
        _atomic_write(report_path, report_content)
        log.info("Duplicate report: %d pairs flagged → %s", dup_count, report_path)

    return dup_count
