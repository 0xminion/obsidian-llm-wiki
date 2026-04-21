"""Agent orchestration — subprocess execution, concept convergence, batch creation."""

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


def _run_agent(prompt: str, cfg: Config, timeout: int = 900) -> str:
    """Run the agent command with the given prompt.

    Uses: hermes chat -q "prompt" -Q
    Saves prompt to disk for debugging/replay).
    Handles hermes internal timeout (exit 124) gracefully — files created
    before timeout are still valid.
    """
    from pipeline.metrics import record_agent_call

    agent_cmd = os.environ.get("AGENT_CMD", cfg.agent_cmd)
    try:
        # Save prompt to disk for debugging/replay
        prompt_file = cfg.resolved_extract_dir / "_agent_prompt.md"
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

        # Record metrics
        record_agent_call(prompt_chars=len(prompt), output_chars=len(output))

        log.info("Agent call: %d chars in, %d chars out, %.1fs",
                 len(prompt), len(output), duration)
        # hermes may return 124 on internal timeout (not subprocess.TimeoutExpired)
        if result.returncode == 124:
            log.warning("Agent timed out (exit 124) — files created before timeout are still valid")
            return result.stdout
        if result.returncode != 0:
            log.error("Agent exited with code %d: %s", result.returncode, result.stderr[:500])
            return result.stdout
        return result.stdout
    except subprocess.TimeoutExpired:
        log.warning("Agent subprocess timed out after %ds", timeout)
        return ""
    except FileNotFoundError:
        log.error("Agent command not found: %s", agent_cmd)
        return ""


def concept_convergence(plans: list[Plan], cfg: Config) -> dict[str, list[dict]]:
    """Search existing concepts via qmd for each plan.

    Returns hash -> list of {concept, score} mappings.
    Scores >0.5 = likely duplicate, 0.2-0.5 = tangential.
    Uses shared qmd module for the actual queries.
    """
    from pipeline.qmd import run_qmd_convergence
    return run_qmd_convergence(plans, cfg)


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

    for attempt in range(max_retries):
        log.info("Batch %d: spawning agent (attempt %d/%d, prompt: %d chars)",
                 batch_idx, attempt + 1, max_retries, len(prompt))

        # Run agent
        output = _run_agent(prompt, cfg, timeout=cfg.agent_timeout)

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
        created_any = False
        for plan in batch:
            filename = title_to_filename(plan.title)
            entry_file = cfg.entries_dir / f"{filename}.md"
            source_file = cfg.sources_dir / f"{filename}.md"
            if entry_file.exists() or source_file.exists():
                created_any = True
                break

        if created_any:
            return {
                "batch_idx": batch_idx,
                "status": "ok",
                "plans": len(batch),
                "hashes": hashes,
            }

        log.warning("Batch %d: agent ran but no files created (attempt %d/%d)",
                    batch_idx, attempt + 1, max_retries)
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
