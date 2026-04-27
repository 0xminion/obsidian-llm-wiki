"""CLI quality commands — lint, validate, doctor, config-doctor, release-check."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pipeline.cli._helpers import _load_cfg, app


@app.command()
def lint(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix safe issues (frontmatter, format, banned tags)"),
):
    """Run comprehensive vault health checks (15 checks)."""
    from pipeline.lint import run_lint

    cfg = _load_cfg(vault)
    result = run_lint(cfg.vault_path, fix=fix)

    typer.echo(f"Files checked: {result.files_checked}")
    typer.echo(f"Total issues:  {result.total_issues}")
    if result.fixes_applied:
        typer.echo(f"Fixes applied: {result.fixes_applied}")

    if result.total_issues == 0:
        typer.echo("Vault health check passed ✓")
        raise typer.Exit(code=0)

    for check_name, count in sorted(result.issues_by_check.items()):
        if count:
            typer.echo(f"  ⚠ {check_name}: {count}")

    report_path = cfg.vault_path / "Meta" / "Scripts" / "lint-report.md"
    typer.echo(f"\nFull report: {report_path}")
    raise typer.Exit(code=1)


@app.command()
def validate(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix safe issues"),
):
    """Validate pipeline output (frontmatter, sections, stubs, tags, format)."""
    from pipeline.lint import run_validate

    cfg = _load_cfg(vault)
    result = run_validate(cfg.vault_path, fix=fix)

    typer.echo(f"Files checked: {result.files_checked}")
    if result.fixes_applied:
        typer.echo(f"Fixes applied:  {result.fixes_applied}")

    if result.total_issues == 0:
        typer.echo("Output validation passed ✓")
        raise typer.Exit(code=0)

    typer.echo(f"Violations:    {result.total_issues}")
    for issue in result.issues:
        prefix = {"error": "✗", "warning": "⚠", "info": "ℹ"}[issue.severity.value]
        typer.echo(f"  {prefix} [{issue.check}] {issue.note}: {issue.detail}")

    raise typer.Exit(code=1)


@app.command()
def doctor(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Run first-run and configuration diagnostics."""
    from pipeline.doctor import run_doctor

    cfg = _load_cfg(vault)
    report = run_doctor(cfg)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Doctor: {'ok' if report['ok'] else 'issues found'}")
        for check in report["checks"]:
            mark = "✓" if check["ok"] else "✗"
            typer.echo(f"  {mark} {check['name']}: {check['detail']}")
    raise typer.Exit(code=0 if report["ok"] else 1)


@app.command(name="config-doctor")
def config_doctor(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Alias for doctor focused on redacted configuration diagnostics."""
    doctor(vault=vault, json_output=json_output)


@app.command(name="release-check")
def release_check(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Check release metadata, docs, and changelog alignment."""
    from pipeline.release import check_release_hygiene

    report = check_release_hygiene(Path(__file__).parent.parent.parent)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Release hygiene: {'ok' if report['ok'] else 'issues found'} ({report['version']})")
        for check in report["checks"]:
            mark = "✓" if check["ok"] else "✗"
            typer.echo(f"  {mark} {check['name']}: {check['detail']}")
    raise typer.Exit(code=0 if report["ok"] else 1)
