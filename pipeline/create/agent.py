"""Agent orchestration — subprocess execution, concept convergence, batch creation.

⚠️ DEPRECATED: This module uses Hermes subprocess for creation. It remains for
backward compatibility but is superseded by template-based creation with direct
LLM calls (pipeline/create/templates.py + pipeline/llm_client.py).

Do NOT add new features here. Use templates.py for all new creation work.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.create.prompts import build_batch_prompt

log = logging.getLogger(__name__)


def _validate_timeout_output(cfg: Config, batch: list[Plan], before_state: dict[str, dict[Path, int]]) -> list[str]:
    """After agent timeout, perform health check on files created so far.

    Validates each plan in the batch using the existing _plan_outputs_created
    helper (validates frontmatter + file existence).

    Returns list of plan hashes that FAILED health check.
    """
    failed_hashes = []
    for plan in batch:
        if not _plan_outputs_created(plan, cfg, before_state.get(plan.hash, {})):
            log.warning("Health check: no valid files for %s (hash %s)", plan.title, plan.hash)
            failed_hashes.append(plan.hash)
        else:
            log.info("Health check passed for %s", plan.title)
    return failed_hashes


class AgentResult:
    """Result of an agent subprocess run, including exit-code metadata."""

    def __init__(self, stdout: str, returncode: int, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr

    @property
    def timed_out(self) -> bool:
        """Hermes returns 124 on internal timeout."""
        return self.returncode == 124


def _run_agent_result(prompt: str, cfg: Config, timeout: int = 900) -> AgentResult:
    """Run the agent command and return full result including exit code.

    Public API _run_agent() returns stdout string for backward compatibility.
    create_batch() calls this to handle timeout recovery.
    """
    from pipeline.metrics import record_agent_call

    agent_cmd = os.environ.get("AGENT_CMD", cfg.agent_cmd)
    prompt_file = cfg.resolved_extract_dir / f"_agent_prompt_{os.getpid()}.md"
    try:
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")

        t0 = time.monotonic()
        result = subprocess.run(
            [agent_cmd, "chat", "-q", prompt, "-Q"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - t0

        output = result.stdout or ""
        record_agent_call(prompt_chars=len(prompt), output_chars=len(output))

        log.info("Agent call: %d chars in, %d chars out, %.1fs",
                 len(prompt), len(output), duration)
        if result.returncode == 124:
            log.warning("Agent timed out (exit 124) — files created before timeout are still valid")
        if result.returncode != 0:
            log.error("Agent exited with code %d: %s", result.returncode, result.stderr[:500])
        return AgentResult(output, result.returncode, result.stderr or "")
    except subprocess.TimeoutExpired as e:
        log.warning("Agent subprocess timed out after %ds", timeout)
        return AgentResult(e.stdout or "", 124, "")
    except FileNotFoundError:
        log.error("Agent command not found: %s", agent_cmd)
        return AgentResult("", 127, "")
    finally:
        try:
            if prompt_file.exists():
                prompt_file.unlink()
        except OSError:
            pass


def _run_agent(prompt: str, cfg: Config, timeout: int = 900) -> str:
    """Run the agent command with the given prompt.

    Uses: hermes chat -q "prompt" -Q
    Saves prompt to disk for debugging/replay).
    Handles hermes internal timeout (exit 124) gracefully — files created
    before timeout are still valid.
    """
    return _run_agent_result(prompt, cfg, timeout).stdout


def concept_convergence(plans: list[Plan], cfg: Config) -> dict[str, list[dict]]:
    """Search existing concepts via qmd for each plan.

    Returns hash -> list of {concept, score} mappings.
    Scores >0.5 = likely duplicate, 0.2-0.5 = tangential.
    Uses shared qmd module for the actual queries.
    """
    from pipeline.qmd import run_qmd_convergence
    return run_qmd_convergence(plans, cfg)


def _candidate_note_paths(directory: Path, base_filename: str) -> list[Path]:
    """Return exact and collision-resolved note paths for a staged base filename."""
    if not directory.exists():
        return []
    matches = []
    for path in directory.glob(f"{base_filename}*.md"):
        if path.stem == base_filename or path.stem.startswith(f"{base_filename}-"):
            matches.append(path)
    return sorted(matches)


def _snapshot_candidate_state(batch: list[Plan], cfg: Config) -> dict[str, dict[Path, int]]:
    """Capture mtimes for candidate output files before the agent runs."""
    from pipeline.vault import title_to_filename

    snapshot: dict[str, dict[Path, int]] = {}
    for plan in batch:
        base_filename = title_to_filename(plan.title)
        states: dict[Path, int] = {}
        for directory in (cfg.entries_dir, cfg.sources_dir):
            for path in _candidate_note_paths(directory, base_filename):
                try:
                    states[path] = path.stat().st_mtime_ns
                except OSError:
                    continue
        snapshot[plan.hash] = states
    return snapshot


def _is_valid_created_note(path: Path, require_frontmatter: bool = True) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if require_frontmatter:
        return content.startswith("---\n") or content.startswith("---\r\n")
    return True


def _plan_outputs_created(plan: Plan, cfg: Config, before: dict[Path, int]) -> bool:
    """Return True if this plan produced new or changed note files.

    Accepts collision-resolved filenames (e.g. foo-1.md) but rejects batches that
    merely observe pre-existing files without writing anything for this run.
    """
    from pipeline.vault import title_to_filename

    base_filename = title_to_filename(plan.title)
    candidate_paths = [
        * _candidate_note_paths(cfg.entries_dir, base_filename),
        * _candidate_note_paths(cfg.sources_dir, base_filename),
    ]
    for path in candidate_paths:
        previous_mtime = before.get(path)
        try:
            current_mtime = path.stat().st_mtime_ns
        except OSError:
            continue
        if previous_mtime is not None and current_mtime <= previous_mtime:
            continue
        if _is_valid_created_note(path, require_frontmatter=True):
            return True
    return False


def create_batch(batch: list[Plan], batch_idx: int, cfg: Config) -> dict:
    """Create vault files for a single batch of plans.

    1. Build batch prompt (with concept convergence data)
    2. Call hermes agent (with retry on failure)
    3. Validate output was created
    4. Return result dict with status and plan hashes
    """
    from pipeline.vault import title_to_filename

    # Run concept convergence for this batch
    convergence = concept_convergence(batch, cfg)

    # Build prompt
    prompt = build_batch_prompt(batch, cfg, convergence)

    # Save prompt for debugging
    prompt_file = cfg.resolved_extract_dir / f"batch_{batch_idx}_prompt.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt, encoding="utf-8")

    hashes = [plan.hash for plan in batch]
    max_retries = cfg.max_retries
    before_state = _snapshot_candidate_state(batch, cfg)

    for attempt in range(max_retries):
        log.info("Batch %d: spawning agent (attempt %d/%d, prompt: %d chars)",
                 batch_idx, attempt + 1, max_retries, len(prompt))

        # Run agent (internal call that returns exit code)
        agent_result = _run_agent_result(prompt, cfg, timeout=cfg.agent_timeout)
        output = agent_result.stdout

        # If agent timed out, run health check on created files
        if agent_result.timed_out and output:
            failed_hashes = _validate_timeout_output(cfg, batch, before_state)
            if failed_hashes:
                log.warning(
                    "Batch %d: timeout with %d/%d plans incomplete — retrying",
                    batch_idx, len(failed_hashes), len(batch),
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                # All retries exhausted — return partial failure
                return {
                    "batch_idx": batch_idx,
                    "status": "partial",
                    "plans": len(batch),
                    "hashes": [h for h in hashes if h not in failed_hashes],
                    "failed_hashes": failed_hashes,
                }

        # Save agent output for debugging
        output_file = cfg.resolved_extract_dir / f"batch_{batch_idx}_output.txt"
        output_file.write_text(output, encoding="utf-8")

        if not output:
            log.warning("Batch %d: agent returned empty output (attempt %d/%d)",
                        batch_idx, attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue

        # Check if files were actually created for this batch's plans
        created_all = True
        missing = []
        for plan in batch:
            if _plan_outputs_created(plan, cfg, before_state.get(plan.hash, {})):
                continue
            created_all = False
            missing.append(title_to_filename(plan.title))

        if created_all:
            return {
                "batch_idx": batch_idx,
                "status": "ok",
                "plans": len(batch),
                "hashes": hashes,
            }

        log.warning("Batch %d: agent ran but %d/%d files missing (%s) (attempt %d/%d)",
                    batch_idx, len(missing), len(batch), ", ".join(missing), attempt + 1, max_retries)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    # All retries exhausted
    log.error("Batch %d: all %d attempts failed", batch_idx, max_retries)
    return {
        "batch_idx": batch_idx,
        "status": "failed",
        "plans": len(batch),
        "hashes": hashes,
    }
