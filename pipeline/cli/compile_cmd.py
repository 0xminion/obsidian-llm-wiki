"""CLI compile command."""

from __future__ import annotations

from pathlib import Path

import typer

from pipeline.vault import reindex as vault_reindex

from pipeline.cli._helpers import _load_cfg, app


@app.command(name="compile")
def compile_pass(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without writing"),
):
    """Run the compile pass — concept convergence, MoC updates, edge construction."""
    from pipeline._common import VaultLock
    from pipeline.compile import run_compile
    from pipeline.log import set_correlation
    from pipeline.metrics import end_stage, get_metrics, reset_metrics, start_stage

    cfg = _load_cfg(vault)
    set_correlation(stage="compile")
    if dry_run:
        typer.echo(f"[DRY RUN] Compile pass — vault: {cfg.vault_path}")
    else:
        typer.echo(f"Compile pass — vault: {cfg.vault_path}")

    lock = VaultLock(cfg.vault_path, name="pipeline")
    if not lock.acquire():
        typer.echo("ERROR: Another pipeline run is in progress. If stale, delete: "
                    f"{lock.lock_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        reset_metrics()
        start_stage("compile")
        result = run_compile(cfg, dry_run=dry_run)
        end_stage("compile")

        if dry_run:
            typer.echo(f"[DRY RUN] Vault snapshot: {result['entries']} entries, "
                        f"{result['concepts']} concepts, {result['mocs']} MoCs")
            typer.echo("[DRY RUN] No files were modified.")
        elif result["success"]:
            typer.echo(f"Compile pass complete. ({result['entries']} entries, "
                        f"{result['concepts']} concepts, {result['mocs']} MoCs)")
            content = vault_reindex(cfg)
            typer.echo(f"Reindexed wiki-index.md ({content.count(chr(10))} lines)")

            metrics = get_metrics()
            if metrics.total_agent_calls > 0:
                typer.echo("")
                typer.echo(metrics.summary())
        else:
            error = result.get("error", "Unknown error")
            typer.echo(f"Compile pass failed: {error}", err=True)
            raise typer.Exit(code=1)
    finally:
        lock.release()
