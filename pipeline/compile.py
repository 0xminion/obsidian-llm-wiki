"""Compile pass module — incremental wiki improvement (Karpathy-style).

Runs the agent-based compile pass: concept convergence, MoC updates,
edge construction, schema evolution. Consolidates compile-pass.sh into Python.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from pipeline.config import Config
from pipeline.utils import count_md

log = logging.getLogger(__name__)

# Retry advice (same as common.sh RETRY_ADVICE)
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




def _run_agent(cfg: Config, prompt: str, description: str, max_retries: int = 3) -> bool:
    """Run the agent with retry logic. Returns True on success."""
    import os
    import time

    agent_cmd = cfg.agent_cmd or "hermes"
    delay = 5

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
            if result.returncode == 0:
                log.info("SUCCESS: %s", description)
                return True

            log.warning("FAILED (exit %d): %s — attempt %d/%d",
                        result.returncode, description, attempt, max_retries)
        except subprocess.TimeoutExpired:
            log.warning("TIMEOUT: %s — attempt %d/%d", description, attempt, max_retries)
        except FileNotFoundError:
            log.error("Agent command not found: %s", agent_cmd)
            return False

        if attempt < max_retries:
            log.info("Waiting %ds before retry...", delay)
            time.sleep(delay)
            delay *= 2
            # Append retry advice after first failure
            if attempt == 1:
                prompt = prompt + _RETRY_ADVICE

    log.error("GIVING UP after %d attempts: %s", max_retries, description)
    return False


def run_compile(cfg: Config) -> dict:
    """Run the compile pass. Returns result dict."""
    # Load and substitute prompt
    prompts_dir = cfg.prompts_dir if cfg.prompts_dir.exists() else Path(__file__).parent.parent / "prompts"

    entry_count = count_md(cfg.entries_dir)
    concept_count = count_md(cfg.concepts_dir)
    moc_count = count_md(cfg.mocs_dir)

    prompt = _load_prompt("compile-pass", prompts_dir)
    if not prompt:
        return {"success": False, "error": "compile-pass.prompt not found"}

    prompt = prompt.replace("{VAULT_PATH}", str(cfg.vault_path))
    prompt = prompt.replace("{ENTRY_COUNT}", str(entry_count))
    prompt = prompt.replace("{CONCEPT_COUNT}", str(concept_count))
    prompt = prompt.replace("{MOC_COUNT}", str(moc_count))

    success = _run_agent(cfg, prompt, "Wiki compile pass")

    return {
        "success": success,
        "entries": entry_count,
        "concepts": concept_count,
        "mocs": moc_count,
    }
