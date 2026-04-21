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

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.config import Config
from pipeline.models import Edge, EdgeType
from pipeline.utils import count_md, extract_body

log = logging.getLogger(__name__)


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
                    links = set(re.findall(r"\[\[([^\]|#]+)\]", content))
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


def _load_prompt(name: str, prompts_dir: Path) -> str:
    """Load a prompt template from prompts/."""
    prompt_file = prompts_dir / f"{name}.prompt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    log.warning("Prompt file not found: %s", prompt_file)
    return ""


def _run_agent(cfg: Config, prompt: str, description: str, max_retries: int = 3) -> tuple[bool, str]:
    """Run the agent with retry logic. Returns (success, raw_output)."""
    agent_cmd = cfg.agent_cmd or "hermes"
    delay = 5
    last_output = ""

    for attempt in range(1, max_retries + 1):
        log.info("Attempt %d/%d: %s", attempt, max_retries, description)

        try:
            result = subprocess.run(
                [agent_cmd, "chat", "-q", prompt, "-Q"],
                cwd=str(cfg.vault_path),
                capture_output=True,
                text=True,
                timeout=600,
            )
            last_output = result.stdout
            if result.returncode == 0:
                log.info("SUCCESS: %s", description)
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

def _build_edges(cfg: Config) -> int:
    """Scan vault for relationships and build edges.tsv.

    Sources for edges:
    1. Existing wikilinks between notes → RELATES_TO
    2. Concept.source fields pointing to entries → tested_by
    3. MoC links to entries/concepts → part_of
    4. Cross-references between concepts with shared tags → extends/supports

    Returns number of edges added (not counting pre-existing).
    """
    edges_file = cfg.edges_file
    edges_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing edges to avoid duplicates
    existing_edges: set[tuple[str, str, str]] = set()
    if edges_file.exists():
        for line in edges_file.read_text(encoding="utf-8", errors="replace").strip().split("\n"):
            if line.startswith("#") or not line.strip() or line.startswith("source\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                existing_edges.add((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    edges: list[Edge] = []
    notes: dict[str, dict] = {}  # name -> {path, type, tags, links, sources}

    # Index all notes
    for note_dir, note_type in [
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

            # Parse frontmatter for tags and sources
            tags = set()
            sources = []
            fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if fm_match:
                fm = fm_match.group(1)
                for tag_match in re.finditer(r"^\s+-\s+(.+)$", fm, re.MULTILINE):
                    tags.add(tag_match.group(1).strip().strip('"').lower())
                src_match = re.search(r"sources:\s*\n((?:\s+-\s+.*\n?)*)", fm)
                if src_match:
                    for s in re.finditer(r"-\s+(.+)$", src_match.group(1), re.MULTILINE):
                        sources.append(s.group(1).strip().strip('"'))

            # Extract wikilinks
            links = set(re.findall(r"\[\[([^\]|#]+)\]", content))

            notes[name] = {
                "type": note_type,
                "tags": tags,
                "links": links,
                "sources": sources,
            }

    # Build edges from relationships
    for name, info in notes.items():
        # 1. Wikilinks → RELATES_TO (bidirectional, only emit one direction)
        for linked in info["links"]:
            if linked in notes and linked != name:
                edge_key = tuple(sorted([name, linked])) + ("relates_to",)
                if edge_key not in existing_edges:
                    edges.append(Edge(
                        source=name, target=linked,
                        type=EdgeType.RELATES_TO,
                        description="auto-detected wikilink",
                    ))
                    existing_edges.add(edge_key)

        # 2. Concept sources → tested_by (entry tests a concept)
        if info["type"] == "concept":
            for src in info["sources"]:
                src_clean = re.sub(r"^\[\[|\]\]$", "", src)
                if src_clean in notes and notes[src_clean]["type"] == "entry":
                    edge_key = (src_clean, name, "tested_by")
                    if edge_key not in existing_edges:
                        edges.append(Edge(
                            source=src_clean, target=name,
                            type=EdgeType.TESTED_BY,
                            description="entry provides evidence for concept",
                        ))
                        existing_edges.add(edge_key)

        # 3. MoC → entries/concepts → part_of
        if info["type"] == "moc":
            for linked in info["links"]:
                if linked in notes:
                    edge_key = (linked, name, "part_of")
                    if edge_key not in existing_edges:
                        edges.append(Edge(
                            source=linked, target=name,
                            type=EdgeType.PART_OF,
                            description="note belongs to MoC",
                        ))
                        existing_edges.add(edge_key)

        # 4. Concepts sharing tags → extends/supports
        if info["type"] == "concept":
            for other_name, other_info in notes.items():
                if other_name == name or other_info["type"] != "concept":
                    continue
                shared_tags = info["tags"] & other_info["tags"]
                if len(shared_tags) >= 2:  # 2+ shared tags suggests relationship
                    edge_key = tuple(sorted([name, other_name])) + ("extends",)
                    if edge_key not in existing_edges:
                        edges.append(Edge(
                            source=name, target=other_name,
                            type=EdgeType.EXTENDS,
                            description=f"shared tags: {', '.join(sorted(shared_tags)[:3])}",
                        ))
                        existing_edges.add(edge_key)

    # Write edges
    if edges:
        if not edges_file.exists():
            edges_file.write_text("source\ttarget\ttype\tdescription\n", encoding="utf-8")
        with edges_file.open("a", encoding="utf-8") as f:
            for edge in edges:
                f.write(edge.to_tsv() + "\n")
        log.info("Added %d edges to edges.tsv", len(edges))

    return len(edges)


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
        report_path.write_text(report_content, encoding="utf-8")
        log.info("Duplicate report: %d pairs flagged → %s", dup_count, report_path)

    return dup_count


# ─── Agent: Concept Merge + Cross-Linking ─────────────────────────────────────

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
    prompts_dir = cfg.prompts_dir if cfg.prompts_dir.exists() else Path(__file__).parent.parent / "prompts"

    entry_count = count_md(cfg.entries_dir)
    concept_count = count_md(cfg.concepts_dir)
    moc_count = count_md(cfg.mocs_dir)

    prompt = _load_prompt("compile-pass", prompts_dir)
    if not prompt:
        return False, ""

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

def run_compile(cfg: Config) -> dict:
    """Run the compile pass. Returns structured result dict.

    Flow:
      1. Capture vault snapshot (before)
      2. Run agent for cross-linking + concept merging (semantic ops)
      3. Deterministic: rebuild wiki-index
      4. Deterministic: construct typed edges
      5. Deterministic: detect duplicates
      6. Capture vault snapshot (after)
      7. Diff to get actual metrics
      8. Write compile report + log entry
    """
    result = CompileResult()

    # 1. Snapshot before
    before = VaultSnapshot.capture(cfg)
    result.entries_before = before.entries
    result.concepts_before = before.concepts
    result.mocs_before = before.mocs

    log.info("=== Compile pass: %d entries, %d concepts, %d MoCs ===",
             before.entries, before.concepts, before.mocs)

    # 2. Agent: cross-linking + concept merging
    agent_ok, agent_output = _run_agent_compile(cfg, result)

    # Parse agent output for metrics (even if agent "failed" — partial results may exist)
    if agent_output:
        agent_metrics = _parse_agent_metrics(agent_output)
        result.crosslinks_added = agent_metrics["crosslinks_added"]
        result.concepts_merged = agent_metrics["concepts_merged"]
        result.mocs_updated = agent_metrics["mocs_updated"]

    # 3. Deterministic: wiki index rebuild
    result.wiki_index_rebuilt = _rebuild_wiki_index(cfg)

    # 4. Deterministic: typed edges construction
    result.edges_added = _build_edges(cfg)

    # 5. Deterministic: duplicate detection
    result.duplicates_flagged = _detect_duplicates(cfg)

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

    # If agent claims 0 but vault actually changed, update from diff
    if result.crosslinks_added == 0 and diff["new_wikilinks"] > 0:
        result.crosslinks_added = diff["new_wikilinks"]
    if result.concepts_merged == 0 and diff["concepts_delta"] < 0:
        result.concepts_merged = abs(diff["concepts_delta"])

    # 8. Write report + log
    report_path = _write_compile_report(cfg, result)
    result.report_path = str(report_path)
    _append_log_entry(cfg, result)

    # Determine overall success
    if not agent_ok and not result.wiki_index_rebuilt:
        result.success = False
        result.error = "Agent failed and wiki index rebuild failed"

    log.info("Compile complete: %s", json.dumps(result.to_dict(), indent=2))

    return result.to_dict()
