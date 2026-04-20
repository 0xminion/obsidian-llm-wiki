"""Main orchestrator for Stage 3 — create_all entry point."""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from pipeline.config import Config
from pipeline.models import Plans
from pipeline.vault import archive_inbox, reindex

from pipeline.create import agent as _create_agent
from pipeline.create.validate import validate_output, _repair_violations

log = logging.getLogger(__name__)


def create_all(plans: Plans, cfg: Config, parallel: int = 3) -> dict:
    """Main entry point for Stage 3 creation.

    1. Split plans into batches
    2. Run concept convergence search
    3. Spawn parallel agents
    4. Post-processing: validate → reindex → log → archive → sync

    Returns stats: {"created": N, "failed": N, "sources": N, "entries": N}
    """
    plan_list = plans.plans
    plan_count = len(plan_list)

    log.info("=== Stage 3: Create Batch (parallel=%d, plans=%d) ===", parallel, plan_count)

    if plan_count == 0:
        log.info("No plans to process")
        return {"created": 0, "failed": 0, "sources": 0, "entries": 0}

    # Validate parallel is a positive integer
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f"PARALLEL must be a positive integer, got: {parallel}")

    # Split into batches (content-size-aware when extract_dir available)
    batches = plans.split_batches(parallel, extract_dir=cfg.resolved_extract_dir)
    log.info("Split %d plans into %d batches (content-size-aware)", plan_count, len(batches))

    # ─── Spawn parallel agents ────────────────────────────────────────────
    results: list[dict] = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_idx = {
            executor.submit(_create_agent.create_batch, batch, idx, cfg): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] != "ok":
                    failed_count += 1
                    log.warning("Batch %d failed", idx)
                else:
                    log.info("Batch %d completed successfully (%d plans)", idx, result["plans"])
            except Exception:
                failed_count += 1
                log.exception("Batch %d raised exception", idx)

    # ─── Post-processing ──────────────────────────────────────────────────

    # 1. Validate
    log.info("Running output validation...")
    manifest_path = cfg.resolved_extract_dir / "manifest.json"
    violations = validate_output(cfg, manifest_path)
    if violations:
        log.warning("Output validation found %d violations:", len(violations))
        for v in violations:
            log.warning("  %s", v)

        # Auto-repair missing sections
        repaired = _repair_violations(cfg, violations)
        if repaired:
            log.info("Auto-repaired %d files", repaired)
            # Re-validate after repair
            remaining = validate_output(cfg, manifest_path)
            if remaining:
                log.warning("After repair, %d violations remain:", len(remaining))
                for v in remaining:
                    log.warning("  %s", v)
            else:
                log.info("All violations repaired")
    else:
        log.info("Output validation passed")

    # 2. Reindex
    log.info("Rebuilding wiki-index...")
    try:
        reindex(cfg)
    except Exception:
        log.exception("Reindex failed")

    # 3. Log to vault
    try:
        cfg.config_dir.mkdir(parents=True, exist_ok=True)
        log_entry = (
            f"## [{date.today().isoformat()}] ingest | batch ({plan_count} sources)\n"
            f"- Pipeline: v2 (3-stage) — Python\n"
            f"- Sources processed: {plan_count}\n"
            f"- Failed agents: {failed_count}\n"
        )
        log_file = cfg.log_md
        with log_file.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except OSError:
        log.exception("Failed to write log entry")

    # 4. Archive inbox (only for successfully processed hashes)
    successful_hashes: set[str] = set()
    for result in results:
        if result["status"] == "ok":
            successful_hashes.update(result["hashes"])

    log.info("Archiving inbox files (only successfully processed)...")
    try:
        archived = archive_inbox(cfg, successful_hashes)
        log.info("Archived %d inbox files", archived)
    except Exception:
        log.exception("Archive inbox failed")

    # 5. Sync vault (if ob CLI is available)
    _sync_vault(cfg)

    # ─── Compute stats ────────────────────────────────────────────────────
    created = sum(1 for r in results if r["status"] == "ok")
    # Count entries and concepts from successful results
    entries_count = sum(r["plans"] for r in results if r["status"] == "ok")

    log.info(
        "=== Stage 3 complete: %d sources, %d failed ===",
        plan_count, failed_count,
    )

    if failed_count > 0:
        log.warning("Some agents failed — check logs for details")

    return {
        "created": created,
        "failed": min(failed_count, plan_count),  # bounds check
        "sources": plan_count,
        "entries": entries_count,
    }


def _sync_vault(cfg: Config) -> None:
    """Sync vault via ob CLI if available."""
    try:
        result = subprocess.run(
            ["which", "ob"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.info("ob CLI not found, skipping vault sync")
            return

        log.info("Syncing vault...")
        subprocess.run(
            ["ob", "sync", "--path", str(cfg.vault_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        log.warning("Vault sync failed")
