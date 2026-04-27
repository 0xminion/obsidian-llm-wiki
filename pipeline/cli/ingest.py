"""CLI ingest command — the main 3-stage pipeline (extract → plan → create)."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from pipeline.models import ExtractedSource, Manifest, Plans, SourceType

from pipeline.cli._helpers import (
    PipelineLock,
    _auto_setup,
    _collect_clipping_files,
    _collect_url_files,
    _load_cfg,
    _setup_logging,
    app,
    check_dependencies,
)


@app.command()
def ingest(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    parallel: int = typer.Option(3, "--parallel", "-p", help="Parallel workers per stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline without writing files"),
    review: bool = typer.Option(False, "--review", help="Stage files for review, skip Stage 3"),
    resume: bool = typer.Option(False, "--resume", help="Resume from saved plans (skip Stages 1+2)"),
    agent: bool = typer.Option(False, "--agent", "-a", help="Use full agent mode for creation (slower, may timeout — not recommended)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Process inbox: extract → plan → create."""
    cfg = _load_cfg(vault)
    cfg.parallel = parallel
    t0 = time.time()
    if dry_run:
        from pipeline.vault_setup import detect_vault

        if verbose:
            _setup_logging(True, None)

        state = detect_vault(cfg.vault_path)
        typer.echo(f"Pipeline ingest — vault: {cfg.vault_path}")

        if state.state == "new":
            typer.echo("[DRY RUN] Vault does not exist; would initialize it during a real ingest.")
            typer.echo("[DRY RUN] No files were created.")
            raise typer.Exit(code=0)

        if state.state == "incomplete":
            typer.echo(state.summary)
            typer.echo("[DRY RUN] Vault is incomplete; would migrate missing directories/files during a real ingest.")
            typer.echo("[DRY RUN] No files were created.")
            raise typer.Exit(code=0)

        typer.echo(f"Extract dir: {cfg.resolved_extract_dir}")
        url_entries = _collect_url_files(cfg.inbox_dir)
        urls = [u for _, u in url_entries]
        if not urls and not resume:
            typer.echo("No .url files found in inbox.")
            raise typer.Exit(code=0)
        typer.echo(f"Found {len(urls)} URL(s) in inbox.")
        typer.echo("  [DRY RUN] Would extract the following URLs:")
        for url in urls:
            typer.echo(f"    - {url}")
        typer.echo("  [DRY RUN] Would generate plans for extracted sources.")
        if review and not resume:
            typer.echo("  [DRY RUN] Would stage generated files for review.")
        else:
            typer.echo("  [DRY RUN] Would create vault files for generated plans.")
        typer.echo("[DRY RUN] No files were created.")
        raise typer.Exit(code=0)

    _setup_logging(verbose, cfg.log_file)

    vault_state = _auto_setup(cfg.vault_path)
    if vault_state == "new":
        typer.echo(f"New vault initialized at {cfg.vault_path}")
    elif vault_state == "migrated":
        typer.echo(f"Vault structure migrated at {cfg.vault_path}")

    typer.echo(f"Pipeline ingest — vault: {cfg.vault_path}")
    typer.echo(f"Extract dir: {cfg.resolved_extract_dir}")

    errors = cfg.validate()
    if errors:
        for e in errors:
            typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)

    missing = check_dependencies(cfg.agent_cmd)
    if missing:
        typer.echo(f"ERROR: Missing required commands: {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)

    lock = PipelineLock(cfg.vault_path)
    if not lock.acquire():
        typer.echo("ERROR: Another pipeline run is in progress. If stale, delete: "
                    f"{lock.lock_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

        from pipeline.log import set_correlation
        from pipeline.metrics import end_stage, get_metrics, reset_metrics, start_stage
        reset_metrics()
        import uuid
        set_correlation(batch_id=uuid.uuid4().hex[:8])

        import pipeline.cli as _cli_pkg
        extract_all = _cli_pkg.extract_all
        plan_sources = _cli_pkg.plan_sources
        create_all = _cli_pkg.create_all
        create_file_templates = _cli_pkg.create_file_templates

        url_entries = _collect_url_files(cfg.inbox_dir)
        urls = [u for _, u in url_entries]
        clipping_entries = _collect_clipping_files(cfg.clippings_dir)

        has_work = bool(urls or clipping_entries)
        if not has_work and not resume:
            typer.echo("No .url files in inbox and no .md clippings found.")
            raise typer.Exit(code=0)

        inbox_msg_parts: list[str] = []
        if urls:
            inbox_msg_parts.append(f"{len(urls)} URL(s)")
        if clipping_entries:
            inbox_msg_parts.append(f"{len(clipping_entries)} clipping(s)")
        typer.echo(f"Found {' + '.join(inbox_msg_parts)} in inbox.")

        # ─── Stage 1: Extract ─────────────────────────────────────────────────
        t1 = time.time()
        if resume:
            typer.echo("Stage 1: SKIPPED (--resume)")
            manifest = Manifest.load(extract_dir)
            if not manifest.entries:
                typer.echo("ERROR: No manifest found for --resume. Run without --resume first.", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"  Loaded {len(manifest.entries)} sources from saved manifest.")
            t1 = t0
        else:
            typer.echo("Stage 1: Extracting...")
            set_correlation(stage="extract")
            start_stage("extract")
            if dry_run:
                if urls:
                    typer.echo("  [DRY RUN] Would extract the following URLs:")
                    for url in urls:
                        typer.echo(f"    - {url}")
                if clipping_entries:
                    typer.echo("  [DRY RUN] Would ingest clippings (skip Stage 1):")
                    for _, clipped in clipping_entries:
                        typer.echo(f"    - {clipped.get('title', 'untitled')}")
                manifest = Manifest(entries=[])
            else:
                manifest = extract_all(urls, cfg, parallel=parallel)
                for _fp, clipped in clipping_entries:
                    source = ExtractedSource(
                        url=clipped["url"],
                        title=clipped["title"],
                        content=clipped["content"],
                        type=SourceType(clipped.get("type", "web")),
                        author=clipped.get("author", ""),
                        source_file=clipped.get("source_file", ""),
                    )
                    manifest.entries.append(source)
                manifest.save(extract_dir)
            elapsed_1 = time.time() - t1
            end_stage("extract")
            typer.echo(f"  Extracted {len(manifest.entries)} sources in {elapsed_1:.1f}s")

        # ─── Stage 2: Plan ─────────────────────────────────────────────────
        t2 = time.time()
        if resume:
            typer.echo("Stage 2: SKIPPED (--resume)")
            plans = Plans.load(extract_dir)
            if not plans.plans:
                typer.echo("ERROR: No plans found for --resume. Run without --resume first.", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"  Loaded {len(plans.plans)} plans from saved file.")
            t2 = t1
        else:
            typer.echo("Stage 2: Planning...")
            set_correlation(stage="plan")
            start_stage("plan")
            if dry_run:
                typer.echo("  [DRY RUN] Would generate plans for extracted sources.")
                plans = Plans(plans=[])
            else:
                plans = plan_sources(manifest, cfg)
            elapsed_2 = time.time() - t2
            end_stage("plan")
            typer.echo(f"  Generated {len(plans.plans)} plans in {elapsed_2:.1f}s")

        if review and resume:
            typer.echo("Review mode with --resume: staging saved plans for approval...")
            from pipeline.review import stage_for_review
            t_review = time.time()
            review_stats = stage_for_review(plans, cfg)
            elapsed_review = time.time() - t_review
            typer.echo(f"  Staged: {review_stats['staged']}, Failed: {review_stats['failed']} in {elapsed_review:.1f}s")
            typer.echo("  Run 'pipeline approve' to write to vault or 'pipeline reject' to discard.")
            elapsed_total = time.time() - t0
            typer.echo(f"Done (review mode, resumed) in {elapsed_total:.1f}s")
            raise typer.Exit(code=0)

        if review and not resume:
            typer.echo("Review mode: staging files for approval...")
            from pipeline.review import stage_for_review
            t_review = time.time()
            review_stats = stage_for_review(plans, cfg)
            elapsed_review = time.time() - t_review
            typer.echo(f"  Staged: {review_stats['staged']}, Failed: {review_stats['failed']} in {elapsed_review:.1f}s")
            typer.echo("  Run 'pipeline approve' to write to vault or 'pipeline reject' to discard.")
            elapsed_total = time.time() - t0
            typer.echo(f"Done (review mode) in {elapsed_total:.1f}s")
            raise typer.Exit(code=0)

        # ─── Stage 3: Create ───────────────────────────────────────────────
        typer.echo("Stage 3: Creating vault files...")
        set_correlation(stage="create")
        start_stage("create")
        t3 = time.time()
        if dry_run:
            typer.echo("  [DRY RUN] Would create vault files for plans.")
            stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
        elif agent:
            typer.echo("  Using full agent mode (heavy — may timeout)")
            stats = create_all(plans, cfg, parallel=parallel)
        else:
            typer.echo("  Using template-based creation (deterministic + insight agent)")
            stats = create_file_templates(plans.plans, cfg, use_agent_insights=True)
        elapsed_3 = time.time() - t3
        end_stage("create")
        typer.echo(f"  Created: {stats['created']}, Failed: {stats['failed']} in {elapsed_3:.1f}s")

        elapsed_total = time.time() - t0
        typer.echo("")
        typer.echo("─── Timing Summary ───")
        if not resume:
            typer.echo(f"  Stage 1 (Extract):  {t2 - t1:.1f}s")
            typer.echo(f"  Stage 2 (Plan):     {t3 - t2:.1f}s")
        typer.echo(f"  Stage 3 (Create):   {elapsed_3:.1f}s")
        typer.echo(f"  Total:              {elapsed_total:.1f}s")
        typer.echo("")

        metrics = get_metrics()
        if metrics.total_agent_calls > 0:
            typer.echo(metrics.summary())
            typer.echo("")

        typer.echo(f"Done in {elapsed_total:.1f}s")

    finally:
        lock.release()
