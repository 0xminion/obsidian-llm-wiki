"""Core compile orchestration — IncrementalCompiler, CompileResult, run_compile."""

from __future__ import annotations

import hashlib
import json
import logging
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
from pipeline.utils import CircuitBreaker, count_md
from pipeline.utils import load_prompt as _load_prompt

log = logging.getLogger(__name__)

_compiling = threading.Event()
_agent_breaker = CircuitBreaker(threshold=5, reset_seconds=120)


def _archive_duplicate(path: Path, cfg: Config) -> None:
    archive_dir = cfg.vault_path / "Meta" / "Scripts" / ".deleted-concepts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived_name = f"{path.stem}_{timestamp}.md"
    path.rename(archive_dir / archived_name)


class IncrementalCompiler:
    """Track file modification times and content hashes in store.db to identify what
    changed since the last compile, avoiding redundant semantic analysis on large vaults.
    Also collects downstream MoCs that reference changed notes.
    """

    def __init__(self, store: ContentStore) -> None:
        self.store = store

    @staticmethod
    def _file_hash(path: Path) -> str:
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

    def update_state(self, files: dict[str, float], cfg: Config) -> None:
        now = time.time()
        for rel, mtime in files.items():
            path = cfg.vault_path / rel
            h = self._file_hash(path)
            self.store.compile_state_set(rel, mtime, h, now)

    def collect_downstream_mocs(self, cfg: Config, changed: set[str]) -> set[str]:
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
        changed, current = self.get_changed_files(cfg, full=full)
        if full:
            return changed, current
        if changed:
            downstream = self.collect_downstream_mocs(cfg, changed)
            changed |= downstream
        return changed, current


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
    new_files = set(after.file_mtimes.keys()) - set(before.file_mtimes.keys())
    modified_files = set()
    for rel, mtime in after.file_mtimes.items():
        if rel in before.file_mtimes and mtime > before.file_mtimes[rel] + 0.1:
            modified_files.add(rel)

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
    from pipeline.metrics import record_agent_call

    if _agent_breaker.is_open():
        log.warning("Circuit breaker OPEN (%d consecutive failures) — skipping: %s",
                     _agent_breaker.failure_count, description)
        return False, ""

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
                _agent_breaker.record_success()
                return True, last_output

            log.warning("FAILED (exit %d): %s — attempt %d/%d",
                        result.returncode, description, attempt, max_retries)
            _agent_breaker.record_failure()
        except subprocess.TimeoutExpired:
            log.warning("TIMEOUT: %s — attempt %d/%d", description, attempt, max_retries)
            _agent_breaker.record_failure()
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


def _run_agent_compile(cfg: Config, result: CompileResult) -> tuple[bool, str]:
    prompt = ""
    if cfg.prompts_dir.exists():
        prompt = _load_prompt("compile-pass", cfg.prompts_dir)
    if not prompt:
        prompt = _load_prompt("compile-pass", Path(__file__).parent.parent / "assets" / "prompts")
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


def _process_merge_queue(
    cfg: Config,
    store: ContentStore,
    confidence_threshold: float = 0.95,
) -> int:
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

        if similarity >= confidence_threshold:
            log.info("Auto-merge (high confidence): %s -> %s", new_concept, existing_concept)
            _do_merge(cfg, new_concept, existing_concept)
            store.merge_queue_approve(item["id"])
            merged += 1
            continue

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

    _archive_duplicate(duplicate_path, cfg)

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


def _write_compile_report(cfg: Config, result: CompileResult) -> Path:
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


def run_compile(
    cfg: Config,
    process_merges: bool = False,
    incremental: bool = False,
    full: bool = False,
    bidirectional: bool = True,
    watch_mode: bool = False,
    dry_run: bool = False,
) -> dict:
    """Orchestrate a full compile pass: incremental change detection -> vault snapshot ->
    semantic compile (cross-link, merge, MoC rebuild) -> structural passes (wiki index,
    edges, duplicate detection) -> merge queue processing -> report generation.
    Returns a structured result dict.
    """
    import pipeline.compile as _compile_pkg
    _run_semantic_compile = _compile_pkg._run_semantic_compile
    _build_edges = _compile_pkg._build_edges
    _detect_duplicates = _compile_pkg._detect_duplicates
    _rebuild_wiki_index = _compile_pkg._rebuild_wiki_index

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

    use_incremental = incremental and not full
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

        before = VaultSnapshot.capture(cfg)
        result.entries_before = before.entries
        result.concepts_before = before.concepts
        result.mocs_before = before.mocs

        log.info("=== Compile pass: %d entries, %d concepts, %d MoCs ===",
                 before.entries, before.concepts, before.mocs)

        if dry_run:
            log.info("[DRY RUN] Would run semantic compile on %d changed files", len(changed_files))
            log.info("[DRY RUN] Would rebuild wiki index, edges, and check duplicates")
            result.entries_after = before.entries
            result.concepts_after = before.concepts
            result.mocs_after = before.mocs
            return result.to_dict()

        agent_ok = True
        agent_output = ""
        if changed_files or not use_incremental:
            agent_ok, agent_output = _run_semantic_compile(cfg, result)
        result.agent_succeeded = agent_ok

        if agent_output and result.crosslinks_added == 0:
            agent_metrics = _parse_agent_metrics(agent_output)
            result.crosslinks_added = agent_metrics["crosslinks_added"]
            result.concepts_merged = agent_metrics["concepts_merged"]
            result.mocs_updated = agent_metrics["mocs_updated"]

        if process_merges:
            try:
                merge_count = _process_merge_queue(cfg, store)
                if merge_count:
                    result.concepts_merged += merge_count
            finally:
                pass

        result.wiki_index_rebuilt = _rebuild_wiki_index(cfg)
        result.edges_added = _build_edges(cfg, bidirectional=bidirectional)
        result.duplicates_flagged = _detect_duplicates(cfg)

        _, current_mtimes = inc.get_changed_files(cfg, full=False)
        inc.update_state(current_mtimes, cfg)

        after = VaultSnapshot.capture(cfg)
        result.entries_after = after.entries
        result.concepts_after = after.concepts
        result.mocs_after = after.mocs

        diff = _diff_snapshots(before, after)
        log.info("Compile diff: %d files changed, %d new wikilinks, "
                 "entries %+d, concepts %+d, MoCs %+d",
                 diff["files_changed"], diff["new_wikilinks"],
                 diff["entries_delta"], diff["concepts_delta"], diff["mocs_delta"])

        if result.crosslinks_added == 0 and diff["new_wikilinks"] > 0:
            result.crosslinks_added = diff["new_wikilinks"]
        if result.concepts_merged == 0 and diff["concepts_delta"] < 0:
            result.concepts_merged = abs(diff["concepts_delta"])

        if not agent_ok:
            result.success = False
            result.error = "Semantic compile failed; deterministic maintenance still ran"
            if not result.wiki_index_rebuilt:
                result.error += "; wiki index rebuild also failed"

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
