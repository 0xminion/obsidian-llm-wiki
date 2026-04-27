"""CLI review commands — approve, reject, review-status."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pipeline.cli._helpers import _load_cfg, app


@app.command()
def approve(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be written"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Approve pending reviews and write files to the vault."""
    from pipeline.review import approve_reviews, show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        if json_output:
            typer.echo(json.dumps({"pending": 0, "approved": 0, "written": 0, "failed": 0}, ensure_ascii=False, sort_keys=True))
        else:
            typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    if json_output and dry_run:
        typer.echo(json.dumps({"pending": len(pending), "items": pending}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    elif not json_output:
        typer.echo(f"Pending reviews: {len(pending)}")
        for r in pending:
            typer.echo(f"  [{r['file_type']}] {Path(r['file_path']).name}")

    if dry_run:
        if not json_output:
            typer.echo("\nDry run — no files written.")
        raise typer.Exit(code=0)

    stats = approve_reviews(cfg)
    if json_output:
        typer.echo(json.dumps({"pending": len(pending), **stats}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"\nApproved: {stats['approved']}, Written: {stats['written']}, Failed: {stats['failed']}")
        if stats.get("written_paths"):
            typer.echo("Written paths:")
            for path in stats["written_paths"]:
                typer.echo(f"  - {path}")


@app.command()
def reject(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Reject and discard all pending reviews."""
    from pipeline.review import reject_reviews, show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        if json_output:
            typer.echo(json.dumps({"pending": 0, "rejected": 0}, ensure_ascii=False, sort_keys=True))
        else:
            typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    count = reject_reviews(cfg)
    if json_output:
        typer.echo(json.dumps({"pending": len(pending), "rejected": count}, ensure_ascii=False, sort_keys=True))
    else:
        typer.echo(f"Rejected {count} pending reviews.")


@app.command(name="review-status")
def review_status(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show pending review queue status without approving/rejecting anything."""
    from pipeline.review import show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)
    by_type: dict[str, int] = {}
    for item in pending:
        by_type[item["file_type"]] = by_type.get(item["file_type"], 0) + 1
    report = {"pending": len(pending), "by_type": dict(sorted(by_type.items())), "items": pending}
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(f"Pending reviews: {len(pending)}")
        for file_type, count in report["by_type"].items():
            typer.echo(f"  {file_type}: {count}")
