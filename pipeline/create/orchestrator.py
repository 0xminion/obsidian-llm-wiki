"""Main orchestrator for Stage 3 — create_all entry point."""

from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

from pipeline.config import Config
from pipeline.models import Plans
from pipeline.vault import archive_clippings, archive_inbox, reindex

from pipeline.create import agent as _create_agent
from pipeline.create.validate import validate_batch, validate_output, _repair_violations

log = logging.getLogger(__name__)


def _update_tag_registry(cfg: Config) -> None:
    """Rebuild tag-registry.md from actual tag usage across all notes."""
    from collections import Counter
    from pipeline.utils import extract_tags

    registry_path = cfg.config_dir / "tag-registry.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    entry_tags = Counter()
    concept_tags = Counter()
    moc_tags = Counter()

    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                entry_tags.update(extract_tags(content))
            except OSError:
                continue

    if cfg.concepts_dir.exists():
        for md in cfg.concepts_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                concept_tags.update(extract_tags(content))
            except OSError:
                continue

    if cfg.mocs_dir.exists():
        for md in cfg.mocs_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                moc_tags.update(extract_tags(content))
            except OSError:
                continue

    lines = [
        "# Tag Registry", "",
        "Canonical list of tags used in this wiki. Before minting a new tag,",
        "check this registry and prefer reuse.", "",
        f"Auto-updated on {now}", "",
    ]

    if entry_tags:
        lines.append("## Entry Tags")
        lines.append("")
        for tag, count in entry_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    if concept_tags:
        lines.append("## Concept Tags")
        lines.append("")
        for tag, count in concept_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    if moc_tags:
        lines.append("## MoC Tags")
        lines.append("")
        for tag, count in moc_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    lines.extend([
        "---", "",
        f"*Updated on {now}: {len(entry_tags)} entry tags, {len(concept_tags)} concept tags, {len(moc_tags)} MoC tags*",
        "",
    ])

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "Tag registry updated: %d entry, %d concept, %d MoC tags",
        len(entry_tags), len(concept_tags), len(moc_tags),
    )


def _validate_batch_files(batch: list, cfg: Config) -> dict:
    """Validate files created by a batch. Returns {ok: bool, violations: list}.

    Checks that:
    1. Expected files actually exist
    2. Files have valid frontmatter
    3. Required sections are present
    4. No stub content
    5. Minimum body length met
    """
    from pipeline.vault import title_to_filename

    files_to_check: list[tuple[Path, str]] = []
    missing_files: list[str] = []

    for plan in batch:
        filename = title_to_filename(plan.title)

        # Check entry
        entry_path = cfg.entries_dir / f"{filename}.md"
        if entry_path.exists():
            files_to_check.append((entry_path, "entry"))
        else:
            # Also try source dir — agent might have only written source
            source_path = cfg.sources_dir / f"{filename}.md"
            if source_path.exists():
                files_to_check.append((source_path, "source"))
            else:
                missing_files.append(filename)

        # Check source (may or may not exist depending on agent behavior)
        source_path = cfg.sources_dir / f"{filename}.md"
        if source_path.exists() and (source_path, "source") not in files_to_check:
            files_to_check.append((source_path, "source"))

        # Check concept (if plan requested new concepts)
        for concept_name in plan.concept_new:
            concept_filename = title_to_filename(concept_name)
            # Account for collision-resolved filenames (-1, -2, etc.)
            found = False
            suffix_re = re.compile(rf"^{re.escape(concept_filename)}-\d+$")
            for candidate in cfg.concepts_dir.glob(f"{concept_filename}*.md"):
                if candidate.stem == concept_filename or suffix_re.fullmatch(candidate.stem):
                    files_to_check.append((candidate, "concept"))
                    found = True
            if not found:
                missing_files.append(concept_filename)

    if not files_to_check and missing_files:
        return {
            "ok": False,
            "violations": [f"No files created for: {', '.join(missing_files)}"],
            "files_checked": 0,
        }

    # Validate all created files
    batch_results = validate_batch(files_to_check)
    all_violations = []
    for file_path, violations in batch_results.items():
        for v in violations:
            all_violations.append(f"{Path(file_path).name}: {v}")

    # Missing files are violations too
    for mf in missing_files:
        all_violations.append(f"missing file: {mf}.md")

    # Critical violations: missing frontmatter, missing files, stubs
    critical = [v for v in all_violations if any(
        kw in v for kw in ["missing frontmatter", "missing file:", "stub content", "banned tag"]
    )]

    return {
        "ok": len(critical) == 0,
        "violations": all_violations,
        "critical": critical,
        "files_checked": len(files_to_check),
    }


def postprocess_creation(
    cfg: Config,
    results: list[dict],
    plan_count: int,
    failed_count: int,
    manifest_path: Path | None = None,
) -> list[str]:
    """Run shared post-processing for Stage 3 creation flows."""
    log.info("Running global output validation...")
    from pipeline.telemetry import TelemetrySink, record_stage

    telemetry = TelemetrySink(cfg.telemetry_file)
    manifest_path = manifest_path or (cfg.resolved_extract_dir / "manifest.json")
    with record_stage(telemetry, "postprocess.validate", plan_count=plan_count, failed_count=failed_count) as event:
        violations = validate_output(cfg, manifest_path)
        event["violations"] = len(violations)
    if violations:
        log.warning("Global validation found %d violations:", len(violations))
        for v in violations[:10]:
            log.warning("  %s", v)

        repaired = _repair_violations(cfg, violations)
        if repaired:
            log.info("Auto-repaired %d files", repaired)
            remaining = validate_output(cfg, manifest_path)
            if remaining:
                log.warning("After repair, %d violations remain:", len(remaining))
                for v in remaining[:5]:
                    log.warning("  %s", v)
                violations = remaining
            else:
                log.info("All violations repaired")
                violations = []
    else:
        log.info("Global validation passed")

    log.info("Rebuilding wiki-index...")
    try:
        reindex(cfg)
    except Exception:
        log.exception("Reindex failed")

    log.info("Updating tag registry...")
    try:
        _update_tag_registry(cfg)
    except Exception:
        log.exception("Tag registry update failed")

    try:
        cfg.config_dir.mkdir(parents=True, exist_ok=True)
        log_entry = (
            f"## [{date.today().isoformat()}] ingest | batch ({plan_count} sources)\n"
            f"- Pipeline: v2 (3-stage) — Python\n"
            f"- Sources processed: {plan_count}\n"
            f"- Failed agents: {failed_count}\n"
            f"- Validation violations: {len(violations)}\n"
        )
        log_file = cfg.log_md
        with log_file.open("a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except OSError:
        log.exception("Failed to write log entry")

    successful_hashes: set[str] = set()
    for result in results:
        if result["status"] == "ok":
            successful_hashes.update(result.get("hashes", []))

    if violations:
        log.warning(
            "Skipping archive because %d validation violations remain",
            len(violations),
        )
    else:
        log.info("Archiving inbox files (only successfully processed)...")
        try:
            archived = archive_inbox(cfg, successful_hashes)
            log.info("Archived %d inbox files", archived)
        except Exception:
            log.exception("Archive inbox failed")

        # Also archive processed clippings (02-Clippings -> 10-Archive-Clippings)
        try:
            archived_clips = archive_clippings(cfg, successful_hashes)
            log.info("Archived %d clipping files", archived_clips)
        except Exception:
            log.exception("Archive clippings failed")

    _sync_vault(cfg)
    return violations


def create_all(plans: Plans, cfg: Config, parallel: int = 3) -> dict:
    """Main entry point for Stage 3 creation.

    1. Split plans into batches
    2. Run concept convergence search
    3. Spawn parallel agents
    4. Validate each batch immediately after creation
    5. Post-processing: global validate → reindex → log → archive → sync

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
            executor.submit(_create_agent.create_batch, batch, idx, cfg): (idx, batch)
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(future_to_idx):
            idx, batch = future_to_idx[future]
            try:
                result = future.result()

                # ─── Per-batch validation ─────────────────────────────────
                if result["status"] == "ok":
                    validation = _validate_batch_files(batch, cfg)
                    if not validation["ok"]:
                        log.warning("Batch %d created files but validation failed (%d violations, %d critical)",
                                    idx, len(validation["violations"]), len(validation.get("critical", [])))
                        for v in validation["violations"][:5]:
                            log.warning("  Validation: %s", v)
                        # Don't mark as ok if critical violations exist
                        result["status"] = "validation_failed"
                        result["validation_violations"] = validation.get("critical", validation["violations"])[:5]
                        failed_count += 1
                    else:
                        log.info("Batch %d validated (%d files OK)", idx, validation["files_checked"])

                if result["status"] != "ok":
                    if result["status"] != "validation_failed":
                        failed_count += 1
                        log.warning("Batch %d failed", idx)
                else:
                    log.info("Batch %d completed successfully (%d plans)", idx, result["plans"])

                results.append(result)
            except Exception:
                failed_count += 1
                log.exception("Batch %d raised exception", idx)
                results.append({"batch_idx": idx, "status": "exception", "plans": len(batch), "hashes": []})

    # ─── Post-processing ──────────────────────────────────────────────────

    postprocess_creation(cfg, results, plan_count, failed_count)

    # ─── Compute stats ────────────────────────────────────────────────────
    created = sum(1 for r in results if r["status"] == "ok")
    entries_count = sum(r.get("plans", 0) for r in results if r["status"] == "ok")

    log.info(
        "=== Stage 3 complete: %d/%d batches ok, %d failed ===",
        created, len(batches), failed_count,
    )

    if failed_count > 0:
        failed_batches = [r for r in results if r["status"] != "ok"]
        for fb in failed_batches:
            log.warning("Failed batch %d: %s", fb.get("batch_idx", "?"), fb.get("status", "unknown"))
            for v in fb.get("validation_violations", []):
                log.warning("  Violation: %s", v)

    return {
        "created": created,
        "failed": min(failed_count, plan_count),
        "sources": plan_count,
        "entries": entries_count,
    }


def _sync_vault(cfg: Config) -> None:
    """Sync vault via ob CLI if available."""
    import shutil
    if not shutil.which("ob"):
        log.info("ob CLI not found, skipping vault sync")
        return

    log.info("Syncing vault...")
    try:
        subprocess.run(
            ["ob", "sync", "--path", str(cfg.vault_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, subprocess.CalledProcessError) as e:
        log.warning("Vault sync failed: %s", e)
