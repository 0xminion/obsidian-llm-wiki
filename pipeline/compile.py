"""Compile pass module — incremental wiki improvement (Karpathy-style).

Runs the agent-based compile pass: concept convergence, MoC updates,
edge construction, schema evolution. Consolidates compile-pass.sh into Python.

Architecture:
  - Deterministic ops (reindex, edges, report) run in Python — always work.
  - Agent ops (cross-linking, concept merging) delegated to hermes.
  - Pre/post vault snapshot diffs actual changes.
  - Structured CompileResult returned with real metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline._common import VaultLock
from pipeline.config import Config
from pipeline.models import Edge, EdgeType
from pipeline.store import ContentStore
from pipeline.utils import count_md
from pipeline.utils import load_prompt as _load_prompt

log = logging.getLogger(__name__)

_compiling = threading.Event()  # sentinel to prevent watch callbacks during compile


def _archive_duplicate(path: Path, cfg: Config) -> None:
    """Move a duplicate concept file to the deleted-concepts archive instead of unlinking.

    Creates the archive directory if it does not exist.
    """
    archive_dir = cfg.vault_path / "Meta" / "Scripts" / ".deleted-concepts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived_name = f"{path.stem}_{timestamp}.md"
    path.rename(archive_dir / archived_name)


# ─── IncrementalCompiler ──────────────────────────────────────────────────────

class IncrementalCompiler:
    """Tracks file mtimes/hashes in store.db and filters changed files for compile.

    On compile, only files whose mtime > last_compile are analyzed.
    In watch mode, only changed files + downstream MoCs are re-compiled.
    """

    def __init__(self, store: ContentStore) -> None:
        self.store = store

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Return a quick hash of file content."""
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return hashlib.md5(data, usedforsecurity=False).hexdigest()[:12]

    def get_changed_files(
        self,
        cfg: Config,
        full: bool = False,
    ) -> tuple[set[str], dict[str, float]]:
        """Return (changed_filenames, current_mtimes) relative to vault root.

        If full=True, returns all files and marks all as changed.
        """
        if full:
            current: dict[str, float] = {}
            for d in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
                if not d.exists():
                    continue
                for md in d.glob("*.md"):
                    rel = str(md.relative_to(cfg.vault_path))
                    try:
                        current[rel] = md.stat().st_mtime
                    except OSError:
                        continue
            return set(current.keys()), current

        previous = self.store.compile_state_get_all()
        changed: set[str] = set()
        current = {}
        for d in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
            if not d.exists():
                continue
            for md in d.glob("*.md"):
                rel = str(md.relative_to(cfg.vault_path))
                try:
                    mtime = md.stat().st_mtime
                except OSError:
                    continue
                current[rel] = mtime
                state = previous.get(rel)
                if state is None:
                    changed.add(rel)
                elif mtime > state.get("last_compile", 0) + 0.01:
                    old_hash = state.get("last_hash", "")
                    new_hash = self._file_hash(md)
                    if old_hash != new_hash:
                        changed.add(rel)
        return changed, current

    def update_state(
        self,
        files: dict[str, float],
        cfg: Config,
    ) -> None:
        """Write current compile_state for the given files (by relative path)."""
        now = time.time()
        for rel, mtime in files.items():
            path = cfg.vault_path / rel
            h = self._file_hash(path)
            self.store.compile_state_set(rel, mtime, h, now)

    def collect_downstream_mocs(self, cfg: Config, changed: set[str]) -> set[str]:
        """Return additional MoC files that link to changed files."""
        result: set[str] = set()
        changed_stems = {Path(c).stem for c in changed}
        if not cfg.mocs_dir.exists():
            return result
        for md in cfg.mocs_dir.glob("*.md"):
            rel = str(md.relative_to(cfg.vault_path))
            if rel in changed:
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            links = set(re.findall(r"\[\[[^\]|#]+(?:[|#][^\]]*)?\]\]", text))
            link_stems = {link[2:].split("|", 1)[0].split("#", 1)[0].rstrip("]") for link in links}
            for stem in changed_stems:
                if stem in link_stems:
                    result.add(rel)
                    break
        return result

    def get_files_for_compile(
        self,
        cfg: Config,
        full: bool = False,
    ) -> tuple[set[str], dict[str, float]]:
        """Return (files_to_compile, current_mtimes). Includes downstream MoCs when incremental."""
        changed, current = self.get_changed_files(cfg, full=full)
        if full:
            return changed, current
        if changed:
            downstream = self.collect_downstream_mocs(cfg, changed)
            changed |= downstream
        return changed, current


# ─── Compile Result ─────────────────────────────────────────────────────────

@dataclass
class CompileResult:
    """Structured result from a compile pass."""

    success: bool = True
    entries_before: int = 0
    entries_after: int = 0
    concepts_before: int = 0
    concepts_after: int = 0
    mocs_before: int = 0
    mocs_after: int = 0
    crosslinks_added: int = 0
    concepts_merged: int = 0
    mocs_updated: int = 0
    edges_added: int = 0
    wiki_index_rebuilt: bool = False
    duplicates_flagged: int = 0
    agent_succeeded: bool = False
    agent_duration_s: float = 0.0
    error: str = ""
    report_path: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "entries": self.entries_after,
            "concepts": self.concepts_after,
            "mocs": self.mocs_after,
            "crosslinks_added": self.crosslinks_added,
            "concepts_merged": self.concepts_merged,
            "mocs_updated": self.mocs_updated,
            "edges_added": self.edges_added,
            "wiki_index_rebuilt": self.wiki_index_rebuilt,
            "duplicates_flagged": self.duplicates_flagged,
            "agent_succeeded": self.agent_succeeded,
            "agent_duration_s": round(self.agent_duration_s, 1),
            "error": self.error,
        }


# ─── Vault Snapshot ──────────────────────────────────────────────────────────

@dataclass
class VaultSnapshot:
    """Point-in-time counts and fingerprints of vault content."""

    entries: int = 0
    concepts: int = 0
    mocs: int = 0
    total_files: int = 0
    file_mtimes: dict[str, float] = field(default_factory=dict)
    file_wikilinks: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def capture(cls, cfg: Config) -> VaultSnapshot:
        """Capture current vault state."""
        snap = cls()
        snap.entries = count_md(cfg.entries_dir)
        snap.concepts = count_md(cfg.concepts_dir)
        snap.mocs = count_md(cfg.mocs_dir)

        for d in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
            if not d.exists():
                continue
            for md in d.glob("*.md"):
                rel = str(md.relative_to(cfg.vault_path))
                try:
                    stat = md.stat()
                    snap.file_mtimes[rel] = stat.st_mtime
                    content = md.read_text(encoding="utf-8", errors="replace")
                    links = set(re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", content))
                    snap.file_wikilinks[rel] = links
                except OSError:
                    continue
        snap.total_files = len(snap.file_mtimes)
        return snap


def _diff_snapshots(before: VaultSnapshot, after: VaultSnapshot) -> dict:
    """Diff two vault snapshots to find what changed."""
    new_files = set(after.file_mtimes.keys()) - set(before.file_mtimes.keys())
    modified_files = set()
    for rel, mtime in after.file_mtimes.items():
        if rel in before.file_mtimes and mtime > before.file_mtimes[rel] + 0.1:
            modified_files.add(rel)

    # Count new wikilinks added
    new_links = 0
    for rel in after.file_wikilinks:
        before_links = before.file_wikilinks.get(rel, set())
        after_links = after.file_wikilinks[rel]
        new_links += len(after_links - before_links)

    return {
        "new_files": new_files,
        "modified_files": modified_files,
        "files_changed": len(new_files) + len(modified_files),
        "new_wikilinks": new_links,
        "entries_delta": after.entries - before.entries,
        "concepts_delta": after.concepts - before.concepts,
        "mocs_delta": after.mocs - before.mocs,
    }


# ─── Retry Logic ─────────────────────────────────────────────────────────────

_RETRY_ADVICE = """
RETRY CONTEXT: Previous attempt failed. Try alternatives:
- If Defuddle failed, fall back to LiteParse (lit parse <file> --format text).
- If PDF parsing failed, try lit with --no-ocr or different page ranges.
- If TranscriptAPI failed, try bare video ID instead of full URL.
- If a file operation failed, verify the target directory exists (create if needed).
- If rate-limited, use a simpler/shorter prompt.
- If note write failed, write to a temp location first, then mv.
Be resourceful. Find a way."""




def _run_agent(cfg: Config, prompt: str, description: str, max_retries: int = 3) -> tuple[bool, str]:
    """Run the agent with retry logic. Returns (success, raw_output)."""
    from pipeline.metrics import record_agent_call

    agent_cmd = cfg.agent_cmd or "hermes"
    delay = 5
    last_output = ""

    for attempt in range(1, max_retries + 1):
        log.info("Attempt %d/%d: %s (prompt: %d chars)", attempt, max_retries, description, len(prompt))

        try:
            t0 = time.monotonic()
            result = subprocess.run(
                [agent_cmd, "chat", "-q", prompt, "-Q"],
                cwd=str(cfg.vault_path),
                capture_output=True,
                text=True,
                timeout=600,
            )
            duration = time.monotonic() - t0
            last_output = result.stdout or ""
            record_agent_call(prompt_chars=len(prompt), output_chars=len(last_output))

            if result.returncode == 0:
                log.info("SUCCESS: %s (%.1fs)", description, duration)
                return True, last_output

            log.warning("FAILED (exit %d): %s — attempt %d/%d",
                        result.returncode, description, attempt, max_retries)
        except subprocess.TimeoutExpired:
            log.warning("TIMEOUT: %s — attempt %d/%d", description, attempt, max_retries)
        except FileNotFoundError:
            log.error("Agent command not found: %s", agent_cmd)
            return False, ""

        if attempt < max_retries:
            log.info("Waiting %ds before retry...", delay)
            time.sleep(delay)
            delay *= 2
            if attempt == 1:
                prompt = prompt + _RETRY_ADVICE

    log.error("GIVING UP after %d attempts: %s", max_retries, description)
    return False, last_output


# ─── Deterministic: Wiki Index Rebuild ───────────────────────────────────────

def _rebuild_wiki_index(cfg: Config) -> bool:
    """Rebuild wiki-index.md from vault content. Returns True if successful."""
    from pipeline.vault import reindex as vault_reindex
    try:
        content = vault_reindex(cfg)
        log.info("Rebuilt wiki-index.md (%d lines)", content.count("\n"))
        return True
    except Exception:
        log.exception("Failed to rebuild wiki-index.md")
        return False


# ─── Deterministic: Typed Edges Construction ─────────────────────────────────


def _edge_key(source: str, target: str, edge_type: str) -> tuple[str, str, str]:
    """Canonicalize edge keys so symmetric relationships are idempotent.

    Only RELATES_TO is sorted — EXTENDS is directional (A extends B ≠ B extends A).
    """
    if edge_type == EdgeType.RELATES_TO.value:
        left, right = sorted([source, target])
        return (left, right, edge_type)
    return (source, target, edge_type)


def _frontmatter_list_items(frontmatter: str, field_name: str) -> list[str]:
    """Extract a YAML-style list field from raw frontmatter text."""
    match = re.search(rf"^{re.escape(field_name)}:[ \t]*\n((?:[ \t]*-[ \t]+.*\n?)*)", frontmatter, re.MULTILINE)
    if not match:
        return []
    items = []
    for item in re.finditer(r"^[ \t]*-[ \t]+(.+)$", match.group(1), re.MULTILINE):
        items.append(item.group(1).strip().strip('"'))
    return items


def _build_edges(cfg: Config, bidirectional: bool = False) -> int:
    """Rebuild deterministic graph edges from the current vault state.

    Generated edges are derived data, so this command rewrites them instead of
    appending. Manually-authored edges are preserved unless they point at notes
    that no longer exist.

    Args:
        bidirectional: If True, flag edges without reverse as WEAK_LINK.

    Returns number of generated edges written.
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
    notes: dict[str, dict] = {}  # name -> {path, type, tags, links, sources}

    # Index all notes
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

    # ─── Bidirectional Edge Inference (Rec 5) ────────────────────────────
    if bidirectional:
        # Only flag ASYMMETRIC edge types where a reverse edge makes semantic sense.
        # RELATES_TO is already canonicalised as symmetric by _edge_key.
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


# ─── Deterministic: Duplicate Detection ──────────────────────────────────────

def _detect_duplicates(cfg: Config) -> int:
    """Find potential duplicate entries/concepts by title similarity.

    Returns count of duplicate pairs flagged.
    """
    report_lines = []
    notes: list[tuple[str, str, str]] = []  # (name, type, title)

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

    # Compare title similarity
    dup_count = 0
    for i, (name_a, type_a, title_a) in enumerate(notes):
        for name_b, type_b, title_b in notes[i + 1:]:
            if type_a != type_b:
                continue
            # Simple similarity: shared words
            words_a = set(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", " ", title_a.lower()).split())
            words_b = set(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", " ", title_b.lower()).split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > 0.7 and name_a != name_b:
                report_lines.append(f"- **{name_a}** ↔ **{name_b}** (overlap: {overlap:.0%}, type: {type_a})")
                dup_count += 1

    # Write report
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


# ─── Semantic Compile (Direct LLM) ───────────────────────────────────────────

@dataclass
class NoteIndex:
    """In-memory index of vault notes for semantic operations."""
    notes: dict[str, dict] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    def load(self, cfg: Config) -> None:
        """Load all notes from the vault."""
        for note_dir, note_type in [
            (cfg.entries_dir, "entry"),
            (cfg.concepts_dir, "concept"),
            (cfg.mocs_dir, "moc"),
        ]:
            if not note_dir.exists():
                continue
            for md in note_dir.glob("*.md"):
                try:
                    content = md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                name = md.stem
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                tags: set[str] = set()
                title = name
                if fm_match:
                    fm = fm_match.group(1)
                    t_match = re.search(r"title:\s*[\"']?(.*?)[\"']?\s*$", fm, re.MULTILINE)
                    if t_match:
                        title = t_match.group(1).strip()
                    tags = {tag.lower() for tag in _frontmatter_list_items(fm, "tags") if tag}
                links = set(re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", content))
                preview = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)[:500]
                self.notes[name] = {
                    "type": note_type,
                    "title": title,
                    "tags": tags,
                    "links": links,
                    "preview": preview,
                    "path": str(md.relative_to(cfg.vault_path)),
                }

    def embed_all(self, client) -> None:
        """Generate embeddings for all note previews.

        If QMD MCP server is available, skips local Ollama batch embedding
        because semantic similarity will be computed via QMD query instead.
        """
        if not self.notes:
            return
        from pipeline.qmd import _get_client
        if _get_client() is not None:
            log.info("QMD enabled — skipping local embed batch; relying on QMD semantic search")
            return
        texts = [f"{n['title']}\n{n['preview']}" for n in self.notes.values()]
        names = list(self.notes.keys())
        batch = client.embed_batch(texts)
        if batch:
            for name, text in zip(names, texts):
                if text in batch:
                    self.embeddings[name] = batch[text]
            log.info("Embedded %d/%d notes", len(self.embeddings), len(self.notes))
        else:
            log.warning("Embedding batch failed; semantic operations will use heuristics only")

    def similarity(self, name_a: str, name_b: str) -> float:
        """Cosine similarity between two notes (0.0 if no embeddings)."""
        emb_a = self.embeddings.get(name_a)
        emb_b = self.embeddings.get(name_b)
        if not emb_a or not emb_b:
            return 0.0
        dot = sum(x * y for x, y in zip(emb_a, emb_b))
        norm_a = math.sqrt(sum(x * x for x in emb_a))
        norm_b = math.sqrt(sum(x * x for x in emb_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def _add_wikilink(cfg: Config, source_name: str, target_name: str, reason: str) -> bool:
    """Add a wikilink from source to target in the appropriate section."""
    source_dirs = [cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir]
    source_path = None
    for d in source_dirs:
        candidate = d / f"{source_name}.md"
        if candidate.exists():
            source_path = candidate
            break
    if not source_path:
        return False

    try:
        content = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    # Skip if link already exists
    if f"[[{target_name}]]" in content:
        return False

    # Find appropriate section to append link
    sections = {
        "entry": ["Linked concepts", "Links", "关联概念"],
        "concept": ["Links", "Context", "链接"],
        "moc": ["Related MoCs", "Cross-References", "关联图谱"],
    }
    note_type = "entry"  # default
    for d, nt in [(cfg.entries_dir, "entry"), (cfg.concepts_dir, "concept"), (cfg.mocs_dir, "moc")]:
        if (d / f"{source_name}.md").exists():
            note_type = nt
            break

    target_sections = sections.get(note_type, ["Links"])
    lines = content.splitlines()
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        for sec in target_sections:
            if line.strip().startswith(f"## {sec}"):
                insert_idx = i + 1
                # Skip empty lines after heading
                while insert_idx < len(lines) and lines[insert_idx].strip() == "":
                    insert_idx += 1
                break
        if insert_idx < len(lines):
            break

    link_line = f"- [[{target_name}]]"
    if reason:
        link_line += f" — {reason}"
    lines.insert(insert_idx, link_line)
    from pipeline.utils import _atomic_write
    _atomic_write(source_path, "\n".join(lines))
    log.debug("Added link: %s -> %s", source_name, target_name)
    return True


def _semantic_crosslink(cfg: Config, client, index: NoteIndex) -> int:
    """Add missing semantic cross-links between related notes.

    Uses embedding similarity + shared tags to find candidates,
    then validates with an LLM prompt.
    """
    if len(index.notes) < 2:
        return 0

    candidates: list[tuple[str, str, float, set[str]]] = []
    names = list(index.notes.keys())
    for i, name_a in enumerate(names):
        info_a = index.notes[name_a]
        for name_b in names[i + 1:]:
            info_b = index.notes[name_b]
            # Skip if already linked
            if name_b in info_a["links"] or name_a in info_b["links"]:
                continue
            sim = index.similarity(name_a, name_b)
            shared_tags = info_a["tags"] & info_b["tags"]
            score = sim + (len(shared_tags) * 0.1)
            if score > 0.5 or len(shared_tags) >= 2:
                candidates.append((name_a, name_b, score, shared_tags))

    if not candidates:
        return 0

    # Sort by score and take top 30 to keep prompt size reasonable
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:30]

    prompt_lines = [
        "You are a knowledge base editor. Review these candidate note pairs and decide which should link to each other.",
        "For each pair, respond with exactly one of these formats (use pipe separators | ):",
        "  LINK <note_a> | <note_b> | <brief reason>",
        "  SKIP <note_a> | <note_b>",
        "",
        "Candidates:",
    ]
    for a, b, score, tags in candidates:
        prompt_lines.append(f"\n--- {a} ↔ {b} (score: {score:.2f}) ---")
        prompt_lines.append(f"{a}: {index.notes[a]['title']} — {index.notes[a]['preview'][:200]}")
        prompt_lines.append(f"{b}: {index.notes[b]['title']} — {index.notes[b]['preview'][:200]}")
        if tags:
            prompt_lines.append(f"shared tags: {', '.join(tags)}")

    prompt = "\n".join(prompt_lines)
    response = client.generate(prompt, timeout=120)

    links_added = 0
    for line in response.splitlines():
        # Parse pipe-delimited format: LINK note_a | note_b | reason
        m = re.match(r"LINK\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)", line)
        if m:
            a, b, reason = m.groups()
            a = a.strip().strip('"').strip("'")
            b = b.strip().strip('"').strip("'")
            if _add_wikilink(cfg, a, b, reason.strip()):
                links_added += 1
            # Also add reverse link if appropriate
            if index.notes.get(b, {}).get("type") in ("concept", "moc"):
                if _add_wikilink(cfg, b, a, reason.strip()):
                    links_added += 1

    log.info("Semantic cross-linking: %d links added", links_added)
    return links_added


def _semantic_concept_merge(cfg: Config, client, index: NoteIndex) -> int:
    """Merge near-duplicate concepts using LLM validation.

    Finds candidates via title similarity + embedding similarity,
    asks LLM whether to merge, and performs the merge if approved.
    """
    concepts = {n: info for n, info in index.notes.items() if info["type"] == "concept"}
    if len(concepts) < 2:
        return 0

    candidates: list[tuple[str, str, float]] = []
    names = list(concepts.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            info_a = concepts[name_a]
            info_b = concepts[name_b]
            # Title word overlap
            words_a = set(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", " ", info_a["title"].lower()).split())
            words_b = set(re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", " ", info_b["title"].lower()).split())
            overlap = 0.0
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            sim = index.similarity(name_a, name_b)
            score = max(overlap, sim)
            if score > 0.75:
                candidates.append((name_a, name_b, score))

    if not candidates:
        return 0

    # Limit to top 10 pairs
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:10]

    prompt_lines = [
        "You are a knowledge base editor. Review these concept pairs and decide if they should be merged.",
        "For each pair, respond with exactly one of (use pipe separators | ):",
        "  MERGE <canonical_name> | <duplicate_name> | <reason>",
        "  KEEP_BOTH <name_a> | <name_b> | <reason>",
        "",
        "Rules:",
        "- If two concepts cover the SAME idea (even in different languages), merge them.",
        "- Choose the older/canonical concept as the first name.",
        "- If they overlap only partially, keep both.",
        "",
        "Candidates:",
    ]
    for a, b, score in candidates:
        prompt_lines.append(f"\n--- {a} ↔ {b} (similarity: {score:.2f}) ---")
        prompt_lines.append(f"{a}: {concepts[a]['title']}")
        prompt_lines.append(f"  {concepts[a]['preview'][:250]}")
        prompt_lines.append(f"{b}: {concepts[b]['title']}")
        prompt_lines.append(f"  {concepts[b]['preview'][:250]}")

    prompt = "\n".join(prompt_lines)
    response = client.generate(prompt, timeout=120)

    merged = 0
    for line in response.splitlines():
        # Parse pipe-delimited format: MERGE canonical | duplicate | reason
        m = re.match(r"MERGE\s+(.+?)\s*\|\s*(.+?)\s*\|\s*(.*)", line)
        if m:
            canonical, duplicate, reason = m.groups()
            canonical = canonical.strip()
            duplicate = duplicate.strip()
            if _merge_concepts(cfg, canonical, duplicate, index):
                merged += 1

    log.info("Semantic concept merge: %d concepts merged", merged)
    return merged


def _merge_concepts(cfg: Config, canonical_name: str, duplicate_name: str, index: NoteIndex) -> bool:
    """Merge duplicate concept into canonical. Returns True if merged.

    Updates all references to the duplicate across entries, concepts, MoCs, and edges.
    """
    canonical_path = cfg.concepts_dir / f"{canonical_name}.md"
    duplicate_path = cfg.concepts_dir / f"{duplicate_name}.md"
    if not canonical_path.exists() or not duplicate_path.exists():
        return False

    try:
        canonical_content = canonical_path.read_text(encoding="utf-8", errors="replace")
        duplicate_content = duplicate_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    # Extract body from both
    duplicate_body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", duplicate_content, flags=re.DOTALL)
    merged_content = canonical_content.rstrip() + f"\n\n## Merged from {duplicate_name}\n\n{duplicate_body.strip()}\n"
    from pipeline.utils import _atomic_write
    _atomic_write(canonical_path, merged_content)

    # Archive duplicate instead of deleting
    _archive_duplicate(duplicate_path, cfg)

    # Update ALL notes that link to the duplicate (entries, concepts, MoCs, and edges)
    all_dirs = [cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir]
    for directory in all_dirs:
        if not directory.exists():
            continue
        for note_md in directory.glob("*.md"):
            try:
                text = note_md.read_text(encoding="utf-8", errors="replace")
                original = text
                text = re.sub(
                    rf"\[\[{re.escape(duplicate_name)}(?P<suffix>[|#][^\]]*)?\]\]",
                    lambda m: f"[[{canonical_name}{m.group('suffix') or ''}]]",
                    text,
                )
                if text != original:
                    from pipeline.utils import _atomic_write
                    _atomic_write(note_md, text)
            except OSError:
                continue

    if cfg.edges_file.exists():
        rewritten: list[str] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for line in cfg.edges_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("source\t") or line.startswith("#"):
                rewritten.append(line)
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            parts[0] = canonical_name if parts[0] == duplicate_name else parts[0]
            parts[1] = canonical_name if parts[1] == duplicate_name else parts[1]
            if parts[0] == parts[1]:
                continue
            key = (parts[0], parts[1], parts[2])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            rewritten.append("\t".join(parts))
        from pipeline.utils import _atomic_write
        _atomic_write(cfg.edges_file, "\n".join(rewritten).rstrip() + "\n")

    # Invalidate entry in index so stale data isn't reused
    if duplicate_name in index.notes:
        del index.notes[duplicate_name]
    if duplicate_name in index.embeddings:
        del index.embeddings[duplicate_name]

    log.info("Merged concept %s into %s", duplicate_name, canonical_name)
    return True


def _replace_wikilink_in_dir(directory: Path, old_name: str, new_name: str) -> None:
    """Replace [[old_name]] with [[new_name]] in all .md files under directory."""
    if not directory.exists():
        return
    for md in directory.glob("*.md"):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
            if f"[[{old_name}]]" in text:
                text = text.replace(f"[[{old_name}]]", f"[[{new_name}]]")
                md.write_text(text, encoding="utf-8")
        except OSError:
            continue


def _semantic_moc_rebuild(cfg: Config, client, index: NoteIndex) -> int:
    """Rebuild MoCs with related notes using LLM synthesis.

    For each MoC, finds top related notes via embedding similarity,
    asks LLM for an updated structure, and writes the result.
    """
    mocs = {n: info for n, info in index.notes.items() if info["type"] == "moc"}
    if not mocs:
        return 0

    updated = 0
    for moc_name, moc_info in mocs.items():
        # Find top 10 related notes by embedding similarity
        related: list[tuple[str, float]] = []
        for name, info in index.notes.items():
            if info["type"] == "moc" or name == moc_name:
                continue
            sim = index.similarity(moc_name, name)
            # Boost if tag overlap
            shared_tags = moc_info["tags"] & info["tags"]
            sim += len(shared_tags) * 0.05
            if sim > 0.3 or moc_name.lower() in info["preview"].lower():
                related.append((name, sim))
        related.sort(key=lambda x: x[1], reverse=True)
        related = related[:10]

        if not related:
            continue

        prompt_lines = [
            f"You are updating a Map of Content (MoC) for the topic: {moc_info['title']}.",
            "",
            "Current MoC preview:",
            moc_info["preview"][:400],
            "",
            "Related notes to include:",
        ]
        for name, score in related:
            info = index.notes[name]
            prompt_lines.append(f"- [[{name}]] ({info['type']}): {info['title']} — {info['preview'][:150]}")

        prompt_lines.extend([
            "",
            "Write an updated MoC section (just the body, no frontmatter). Structure:",
            "## Overview / 概述",
            "<2-3 sentence synthesized summary>",
            "",
            "## <Topic Sections>",
            "- [[Note]] — <1-sentence summary>",
            "",
            "## Bridge Concepts",
            "- <concepts connecting subtopics>",
            "",
            "## Cross-References",
            "- <relevant links>",
            "",
            "Use [[wikilinks]] for all internal links. Keep it concise.",
        ])

        prompt = "\n".join(prompt_lines)
        response = client.generate(prompt, timeout=120)
        if not response:
            continue

        moc_path = cfg.mocs_dir / f"{moc_name}.md"
        try:
            current = moc_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Preserve frontmatter, replace body
        # Split on first \n---\n to avoid matching --- inside YAML values
        parts = current.split("\n---\n", 1)
        frontmatter = parts[0] + "\n---\n" if len(parts) > 1 else ""
        new_content = frontmatter + f"# {moc_info['title']}\n\n" + response + "\n"
        moc_path.write_text(new_content, encoding="utf-8")
        updated += 1

    log.info("Semantic MoC rebuild: %d MoCs updated", updated)
    return updated


def _run_semantic_compile(cfg: Config, result: CompileResult) -> tuple[bool, str]:
    """Run semantic compile operations via direct LLM calls (no subprocess).

    Uses embedding similarity + LLM validation for:
      - Cross-linking
      - Concept merging
      - MoC rebuilding

    Falls back to Hermes agent if llm_provider == 'hermes'.
    """
    from pipeline.llm_client import get_llm_client

    client = get_llm_client(cfg)

    # If user explicitly chose hermes provider, use the legacy agent path
    if cfg.llm_provider == "hermes":
        return _run_agent_compile(cfg, result)

    t0 = time.time()

    # Build note index with embeddings
    index = NoteIndex()
    index.load(cfg)
    if len(index.notes) > 0:
        index.embed_all(client)

    # Run semantic operations with exception tracking
    all_ok = True
    try:
        result.crosslinks_added = _semantic_crosslink(cfg, client, index)
    except Exception as e:
        log.error("Semantic cross-linking failed: %s", e)
        all_ok = False

    try:
        result.concepts_merged = _semantic_concept_merge(cfg, client, index)
    except Exception as e:
        log.error("Semantic concept merge failed: %s", e)
        all_ok = False

    try:
        result.mocs_updated = _semantic_moc_rebuild(cfg, client, index)
    except Exception as e:
        log.error("Semantic MoC rebuild failed: %s", e)
        all_ok = False

    result.agent_duration_s = time.time() - t0
    result.agent_succeeded = all_ok

    summary = (
        f"cross-links added: {result.crosslinks_added}\n"
        f"concepts merged: {result.concepts_merged}\n"
        f"mocs updated: {result.mocs_updated}"
    )
    return all_ok, summary


# ─── Agent: Concept Merge + Cross-Linking (Legacy) ───────────────────────────

def _run_agent_compile(cfg: Config, result: CompileResult) -> tuple[bool, str]:
    """Run the agent for cross-linking and concept merging.

    Returns (success, raw_output).
    The agent prompt focuses on operations that require semantic judgment:
    - Operation 1: Cross-link analysis
    - Operation 2: Concept convergence (merge near-duplicates)
    - Operation 3: MoC rebuild
    - Operation 6: Entry template assessment
    - Operation 9: Schema co-evolution

    Deterministic operations (reindex, edges, report, duplicate detection)
    are handled by Python code in this module.
    """
    prompt = ""
    if cfg.prompts_dir.exists():
        prompt = _load_prompt("compile-pass", cfg.prompts_dir)
    if not prompt:
        prompt = _load_prompt("compile-pass", Path(__file__).parent / "assets" / "prompts")
    if not prompt:
        return False, ""

    entry_count = count_md(cfg.entries_dir)
    concept_count = count_md(cfg.concepts_dir)
    moc_count = count_md(cfg.mocs_dir)

    prompt = prompt.replace("{VAULT_PATH}", str(cfg.vault_path))
    prompt = prompt.replace("{ENTRY_COUNT}", str(entry_count))
    prompt = prompt.replace("{CONCEPT_COUNT}", str(concept_count))
    prompt = prompt.replace("{MOC_COUNT}", str(moc_count))

    t0 = time.time()
    success, output = _run_agent(cfg, prompt, "Wiki compile pass")
    result.agent_duration_s = time.time() - t0
    result.agent_succeeded = success

    return success, output


def _parse_agent_metrics(output: str) -> dict:
    """Parse agent output for compile metrics.

    Looks for structured patterns in the agent's output:
    - Cross-links added: N
    - Concepts merged: N
    - MoCs updated: N
    """
    metrics = {
        "crosslinks_added": 0,
        "concepts_merged": 0,
        "mocs_updated": 0,
    }

    patterns = {
        "crosslinks_added": [
            r"cross[- ]?links?\s*(?:added|created|found):\s*(\d+)",
            r"added\s*(\d+)\s*(?:new\s+)?(?:cross[- ]?)?(?:wiki)?links?",
            r"(\d+)\s*(?:new\s+)?(?:cross[- ]?)?(?:wiki)?links?\s*(?:added|created)",
        ],
        "concepts_merged": [
            r"concepts?\s*(?:merged|combine[d]?):\s*(\d+)",
            r"merged\s*(\d+)\s*concepts?",
            r"(\d+)\s*(?:concept\s*)?(?:pairs?\s*)?merged",
        ],
        "mocs_updated": [
            r"mocs?\s*(?:updated|rebuilt|revised):\s*(\d+)",
            r"updated\s*(\d+)\s*mocs?",
            r"rebuilt\s*(\d+)\s*(?:maps?\s*of\s*content|mocs?)",
        ],
    }

    output_lower = output.lower()
    for metric, pats in patterns.items():
        for pat in pats:
            match = re.search(pat, output_lower)
            if match:
                try:
                    metrics[metric] = int(match.group(1))
                except (ValueError, IndexError):
                    pass
                break

    return metrics


# ─── Merge Queue Processing (Rec 8) ─────────────────────────────────────────

def _process_merge_queue(
    cfg: Config,
    store: ContentStore,
    confidence_threshold: float = 0.95,
) -> int:
    """Review pending merge queue and execute merges.

    - If similarity >= confidence_threshold, merge immediately (heuristic).
    - Otherwise, propose via a lightweight LLM call.
    - Returns count of merges executed.
    """
    pending = store.merge_queue_get_pending()
    if not pending:
        return 0

    from pipeline.llm_client import get_llm_client

    client = get_llm_client(cfg)
    merged = 0

    for item in pending:
        new_concept = item["new_concept"]
        existing_concept = item["existing_concept"]
        similarity = float(item["similarity"])

        # High-confidence heuristic: auto-merge
        if similarity >= confidence_threshold:
            log.info("Auto-merge (high confidence): %s -> %s", new_concept, existing_concept)
            _do_merge(cfg, new_concept, existing_concept)
            store.merge_queue_approve(item["id"])
            merged += 1
            continue

        # Lightweight LLM validation for borderline cases
        prompt = (
            f"You are a knowledge base editor.\n"
            f"Two wiki concepts may be duplicates:\n"
            f"  Existing concept: {existing_concept}\n"
            f"  New concept: {new_concept}\n"
            f"  Embedding similarity: {similarity:.3f}\n\n"
            f"Should they be merged? Answer with exactly one word: MERGE or KEEP.\n"
        )
        answer = ""
        try:
            resp = client.generate(prompt, timeout=30)
            if resp and resp.strip():
                answer = resp.strip().upper()
        except Exception as e:
            log.debug("LLM merge proposal failed: %s", e)

        if answer.startswith("MERGE"):
            log.info("LLM approved merge: %s -> %s", new_concept, existing_concept)
            _do_merge(cfg, new_concept, existing_concept)
            store.merge_queue_approve(item["id"])
            merged += 1
        else:
            log.info("LLM rejected merge: %s vs %s", new_concept, existing_concept)
            store.merge_queue_reject(item["id"])

    return merged


def _do_merge(cfg: Config, duplicate_name: str, canonical_name: str) -> bool:
    """Lightweight in-Python merge of duplicate concept into canonical.

    Updates references and deletes the duplicate concept file.
    """
    canonical_path = cfg.concepts_dir / f"{canonical_name}.md"
    duplicate_path = cfg.concepts_dir / f"{duplicate_name}.md"
    if not canonical_path.exists() or not duplicate_path.exists():
        return False

    try:
        canonical_content = canonical_path.read_text(encoding="utf-8", errors="replace")
        duplicate_content = duplicate_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    duplicate_body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", duplicate_content, flags=re.DOTALL)
    merged_content = canonical_content.rstrip() + f"\n\n## Merged from {duplicate_name}\n\n{duplicate_body.strip()}\n"
    from pipeline.utils import _atomic_write
    _atomic_write(canonical_path, merged_content)

    # Archive duplicate instead of deleting
    _archive_duplicate(duplicate_path, cfg)

    # Update references across all dirs
    all_dirs = [cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir]
    for directory in all_dirs:
        if not directory.exists():
            continue
        for note_md in directory.glob("*.md"):
            try:
                text = note_md.read_text(encoding="utf-8", errors="replace")
                original = text
                text = re.sub(
                    rf"\[\[{re.escape(duplicate_name)}(?P<suffix>[|#][^\]]*)?\]\]",
                    lambda m: f"[[{canonical_name}{m.group('suffix') or ''}]]",
                    text,
                )
                if text != original:
                    note_md.write_text(text, encoding="utf-8")
            except OSError:
                continue

    return True


# ─── Compile Report + Log ────────────────────────────────────────────────────

def _write_compile_report(cfg: Config, result: CompileResult) -> Path:
    """Write compile-report.md with actual metrics."""
    report_dir = cfg.vault_path / "Meta" / "Scripts"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "compile-report.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# Compile Report

Generated: {now}

## Metrics

| Metric | Value |
|---|---|
| Entries | {result.entries_before} → {result.entries_after} |
| Concepts | {result.concepts_before} → {result.concepts_after} |
| MoCs | {result.mocs_before} → {result.mocs_after} |
| Cross-links added | {result.crosslinks_added} |
| Concepts merged | {result.concepts_merged} |
| MoCs updated | {result.mocs_updated} |
| Edges added | {result.edges_added} |
| Duplicates flagged | {result.duplicates_flagged} |
| Wiki index rebuilt | {'yes' if result.wiki_index_rebuilt else 'no'} |

## Agent

- Succeeded: {'yes' if result.agent_succeeded else 'no'}
- Duration: {result.agent_duration_s:.1f}s

"""
    if result.error:
        report += f"## Errors\n\n{result.error}\n"

    report_path.write_text(report, encoding="utf-8")
    return report_path


def _append_log_entry(cfg: Config, result: CompileResult) -> None:
    """Append structured entry to 06-Config/log.md."""
    log_file = cfg.log_md
    log_file.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = (
        f"\n## [{now}] compile | Compile pass\n"
        f"- Entries: {result.entries_after}\n"
        f"- Concepts: {result.concepts_after}\n"
        f"- Cross-links added: {result.crosslinks_added}\n"
        f"- Concept merges: {result.concepts_merged}\n"
        f"- MoCs updated: {result.mocs_updated}\n"
        f"- Edges added: {result.edges_added}\n"
        f"- Duplicates flagged: {result.duplicates_flagged}\n"
        f"- Wiki index rebuilt: {'yes' if result.wiki_index_rebuilt else 'no'}\n"
        f"- Agent: {'ok' if result.agent_succeeded else 'failed'} ({result.agent_duration_s:.1f}s)\n"
    )

    try:
        if not log_file.exists():
            log_file.write_text(
                "# Wiki Activity Log\n\n"
                "Chronological record of all operations on the knowledge base.\n\n---\n",
                encoding="utf-8",
            )
        with log_file.open("a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        log.exception("Failed to write log entry")


# ─── Main Entry Point ────────────────────────────────────────────────────────

def run_compile(cfg: Config, process_merges: bool = False, incremental: bool = False, full: bool = False, bidirectional: bool = True, watch_mode: bool = False) -> dict:
    """Run the compile pass. Returns structured result dict.

    Args:
        process_merges: If True, process the merge queue after semantic compile.
        incremental: If True, only analyze files whose mtime > last_compile.
        full: If True, force full recompile regardless of incremental state.
        bidirectional: If True, flag edges without reverse as WEAK_LINK.
        watch_mode: If True, acquire VaultLock and set compiling sentinel.
    """
    if _compiling.is_set():
        log.info("Compile already in progress, skipping")
        return CompileResult(success=True).to_dict()

    _compiling.set()
    lock: Optional[VaultLock] = None
    if watch_mode:
        lock = VaultLock(cfg.vault_path, name="pipeline")
        if not lock.acquire():
            log.warning("Watch mode: could not acquire VaultLock, skipping compile")
            _compiling.clear()
            return CompileResult(success=False, error="VaultLock busy").to_dict()

    # Determine whether to use incremental mode
    use_incremental = incremental and not full

    # Default incremental on for >100 notes
    total_notes = (
        count_md(cfg.entries_dir)
        + count_md(cfg.concepts_dir)
        + count_md(cfg.mocs_dir)
        + count_md(cfg.sources_dir)
    )
    if not full and total_notes > 100:
        use_incremental = True

    store = ContentStore.open_vault_cache(cfg.vault_path)
    try:
        inc = IncrementalCompiler(store)
        changed_files: set[str] = set()
        current_mtimes: dict[str, float] = {}
        if use_incremental:
            changed_files, current_mtimes = inc.get_files_for_compile(cfg, full=full)
            if changed_files:
                log.info("Incremental compile: %d changed files + downstream MoCs", len(changed_files))
            else:
                log.info("Incremental compile: no changed files, skipping semantic analysis")

        result = CompileResult()

        # 1. Snapshot before
        before = VaultSnapshot.capture(cfg)
        result.entries_before = before.entries
        result.concepts_before = before.concepts
        result.mocs_before = before.mocs

        log.info("=== Compile pass: %d entries, %d concepts, %d MoCs ===",
                 before.entries, before.concepts, before.mocs)

        # 2. Semantic compile: cross-linking + concept merging + MoC rebuild
        agent_ok = True
        agent_output = ""
        if changed_files or not use_incremental:
            agent_ok, agent_output = _run_semantic_compile(cfg, result)
        result.agent_succeeded = agent_ok

        # Parse agent output for metrics (legacy compat)
        if agent_output and result.crosslinks_added == 0:
            agent_metrics = _parse_agent_metrics(agent_output)
            result.crosslinks_added = agent_metrics["crosslinks_added"]
            result.concepts_merged = agent_metrics["concepts_merged"]
            result.mocs_updated = agent_metrics["mocs_updated"]

        # 8. Merge queue processing (Rec 8)
        if process_merges:
            try:
                merge_count = _process_merge_queue(cfg, store)
                if merge_count:
                    result.concepts_merged += merge_count
            finally:
                pass

        # 3. Deterministic: wiki index rebuild
        result.wiki_index_rebuilt = _rebuild_wiki_index(cfg)

        # 4. Deterministic: typed edges construction
        result.edges_added = _build_edges(cfg, bidirectional=bidirectional)

        # 5. Deterministic: duplicate detection
        result.duplicates_flagged = _detect_duplicates(cfg)

        # Update compile state (M15: re-capture mtimes after compile)
        _, current_mtimes = inc.get_changed_files(cfg, full=False)
        inc.update_state(current_mtimes, cfg)

        # 6. Snapshot after
        after = VaultSnapshot.capture(cfg)
        result.entries_after = after.entries
        result.concepts_after = after.concepts
        result.mocs_after = after.mocs

        # 7. Diff for validation
        diff = _diff_snapshots(before, after)
        log.info("Compile diff: %d files changed, %d new wikilinks, "
                 "entries %+d, concepts %+d, MoCs %+d",
                 diff["files_changed"], diff["new_wikilinks"],
                 diff["entries_delta"], diff["concepts_delta"], diff["mocs_delta"])

        # If metrics claim 0 but vault actually changed, update from diff
        if result.crosslinks_added == 0 and diff["new_wikilinks"] > 0:
            result.crosslinks_added = diff["new_wikilinks"]
        if result.concepts_merged == 0 and diff["concepts_delta"] < 0:
            result.concepts_merged = abs(diff["concepts_delta"])

        # Determine overall success
        if not agent_ok:
            result.success = False
            result.error = "Semantic compile failed; deterministic maintenance still ran"
            if not result.wiki_index_rebuilt:
                result.error += "; wiki index rebuild also failed"

        # 8. Write report + log
        report_path = _write_compile_report(cfg, result)
        result.report_path = str(report_path)
        _append_log_entry(cfg, result)

        log.info("Compile complete: %s", json.dumps(result.to_dict(), indent=2))

        return result.to_dict()
    finally:
        store.close()
        if lock:
            lock.release()
        _compiling.clear()


def _watch_with_watchdog(cfg: Config, incremental: bool = True, bidirectional: bool = True) -> None:
    """Watch vault for changes using watchdog library (preferred)."""
    from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]
    from watchdog.observers import Observer  # type: ignore[import-untyped]

    class _CompileHandler(FileSystemEventHandler):
        def __init__(self, inc: IncrementalCompiler) -> None:
            self.inc = inc
            self._last_compile = 0.0

        def on_modified(self, event) -> None:
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix != ".md":
                return
            self._maybe_compile()

        def on_created(self, event) -> None:
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix != ".md":
                return
            self._maybe_compile()

        def _maybe_compile(self) -> None:
            if _compiling.is_set():
                return
            now = time.time()
            if now - self._last_compile < 5.0:
                return
            self._last_compile = now
            log.info("Watchdog: file change detected, triggering compile...")
            try:
                run_compile(cfg, incremental=incremental, full=False, bidirectional=bidirectional, watch_mode=True)
            except Exception:
                log.exception("Watch compile failed")

    store = ContentStore.open_vault_cache(cfg.vault_path)
    handler = _CompileHandler(IncrementalCompiler(store))
    observer = Observer()
    for d in (cfg.entries_dir, cfg.concepts_dir, cfg.mocs_dir, cfg.sources_dir):
        if d.exists():
            observer.schedule(handler, str(d), recursive=False)
    observer.start()
    log.info("Watchdog watching vault: %s", cfg.vault_path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        store.close()


def _watch_with_polling(cfg: Config, incremental: bool = True, bidirectional: bool = True, interval: float = 30.0) -> None:
    """Poll vault for changes every `interval` seconds and recompile when changed."""
    store = ContentStore.open_vault_cache(cfg.vault_path)
    inc = IncrementalCompiler(store)
    log.info("Polling watch started (every %.0fs): %s", interval, cfg.vault_path)
    try:
        while True:
            time.sleep(interval)
            changed, current = inc.get_files_for_compile(cfg, full=False)
            if changed:
                log.info("Polling: %d changed file(s), triggering compile...", len(changed))
                if _compiling.is_set():
                    log.debug("Polling: compile already in progress, skipping")
                else:
                    try:
                        run_compile(cfg, incremental=incremental, full=False, bidirectional=bidirectional, watch_mode=True)
                    except Exception:
                        log.exception("Watch compile failed")
            else:
                log.debug("Polling: no changes")
    except KeyboardInterrupt:
        pass
    finally:
        store.close()


def watch_compile(cfg: Config, incremental: bool = True, bidirectional: bool = True) -> None:
    """Auto-trigger compile on file changes.

    Uses watchdog if available, otherwise falls back to polling every 30s.
    In watch mode, only re-compiles changed files + downstream MoCs.
    """
    import importlib.util
    if importlib.util.find_spec("watchdog") is not None:
        _watch_with_watchdog(cfg, incremental=incremental, bidirectional=bidirectional)
    else:
        log.info("watchdog not installed; falling back to 30s polling")
        _watch_with_polling(cfg, incremental=incremental, bidirectional=bidirectional)
