"""``olw identity-migration --dry-run`` — read-only legacy identity inventory.

There is deliberately no ``--apply``: applying a migration is the job of the
future identity-plan build, which is not implemented here. The command only
reports, and it exits non-zero whenever it could not establish a consistent,
write-free view of the vault.

Exit codes:
  0  preview completed and found no blockers;
  2  refused (busy/uncheckpointed WAL, malformed registry, unsafe path,
     missing vault, usage error) — nothing was inventoried;
  3  preview completed but reported blockers that need human review
     (malformed cache, unknown schema version, missing registry, ...).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from obsidian_llm_wiki.cli import app

_REPORT_NAME = "identity-migration-preview.json"


def _emit_refusal(reasons: list[str], action: str, *, as_json: bool, vault: str,
                  registry: dict | None = None) -> None:
    payload = {
        "mode": "dry-run", "status": "refused", "vault": vault, "bindings": [],
        "apply": {"implemented": False, "handled_by": "future plan build"},
        "refusal": {"reasons": reasons, "action": action},
        "registry": registry or {},
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    typer.echo("Identity migration preview refused; no files written.", err=True)
    for reason in reasons:
        typer.echo(f"  - {reason}", err=True)
    typer.echo(f"  Action: {action}", err=True)


def _scratch_target(scratch: str, vault: Path) -> Path:
    target = Path(scratch).expanduser().absolute()
    for candidate in (target, target.resolve()):
        if candidate == vault or vault in candidate.parents:
            raise ValueError(f"scratch directory {target} is inside the vault {vault}")
    return target


@app.command("identity-migration")
def identity_migration(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Required. Inventory legacy identity bindings read-only; never applies anything.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON"),
    scratch: str | None = typer.Option(
        None, "--scratch",
        help="Optional directory OUTSIDE the vault where the JSON report is also written",
    ),
) -> None:
    """Preview what a future identity-plan migration would have to decide.

    Reads sources, the synthesis cache, generated pages and a byte copy of
    ``registry.db`` without opening the live registry, acquiring the compile
    lock, writing metrics or contacting any provider. Refuses when a writer is
    active or the WAL holds uncheckpointed frames.

    Examples:
        olw identity-migration ~/MyVault --dry-run
        olw identity-migration ~/MyVault --dry-run --json
        olw identity-migration ~/MyVault --dry-run --json --scratch /tmp/olw-preview
    """
    vault_path = Path(vault).expanduser().absolute()
    if not vault_path.is_symlink():
        vault_path = vault_path.resolve()
    if not dry_run:
        typer.echo(
            "identity-migration only supports --dry-run: applying a migration is handled by "
            "the future plan build and is not implemented here.",
            err=True,
        )
        raise typer.Exit(code=2)

    from obsidian_llm_wiki.core.identity_migration import (
        PreviewRefusedError,
        preview_identity_migration,
    )

    scratch_dir: Path | None = None
    if scratch is not None:
        try:
            scratch_dir = _scratch_target(scratch, vault_path)
        except ValueError as exc:
            _emit_refusal([str(exc)], "choose a scratch directory outside the vault",
                          as_json=as_json, vault=vault_path.as_posix())
            raise typer.Exit(code=2) from exc

    try:
        report = preview_identity_migration(vault_path)
    except PreviewRefusedError as exc:
        _emit_refusal(
            exc.reasons, exc.action, as_json=as_json, vault=vault_path.as_posix(),
            registry=getattr(exc, "registry", None),
        )
        raise typer.Exit(code=2) from exc

    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / _REPORT_NAME).write_text(rendered + "\n", encoding="utf-8")

    if as_json:
        print(rendered)
    else:
        _print_summary(report)

    if report["status"] != "preview":
        typer.echo(
            "Preview inconsistent: the vault changed while it was read; re-run with "
            "writers stopped.", err=True,
        )
        raise typer.Exit(code=2)
    if report["blockers"]:
        raise typer.Exit(code=3)


def _print_summary(report: dict) -> None:
    counts = report["summary"]["bindings"]
    print("🔍 Identity migration preview (dry-run; read-only)")
    print(f"   Vault: {report['vault']}")
    print(f"   Registry: {report['registry']['strategy']} "
          f"(schema {report['registry']['schema_version']}, "
          f"binary {report['registry']['binary_schema_version']})")
    publication = report["publication"]
    print(f"   Publication: {publication['status']}; verified: {publication['verified']}; "
          f"recovery required: {publication.get('recovery_required', False)}")
    print("   Identity bindings (inventory, not rendered-page certification or apply authority):")
    print(f"   Rendered pages certified: {report['rendered_pages']['certified']}")
    for key in ("resolved", "unresolved", "needs_assessment"):
        print(f"     {key}: {counts[key]}")
    print(f"     total: {counts['total']}")
    refs = report["summary"]["references"]
    print(f"   Page references: resolved {refs['resolved']}, unresolved {refs['unresolved']}")
    print(f"   Human-edited pages preserved as-is: {report['summary']['human_edited_pages']}")
    estimate = report["assessment_estimate"]
    print(f"   Assessment estimate: {estimate['status']} "
          f"(bindings needing assessment: {estimate['needs_assessment_bindings']}; "
          f"model calls made: {estimate['model_calls_made']})")
    if report["blockers"]:
        print(f"   Blockers ({len(report['blockers'])}):")
        for blocker in report["blockers"][:20]:
            print(f"     - {blocker['kind']}: {blocker['path']} — {blocker['detail']}")
    for binding in report["bindings"]:
        if binding["status"] != "resolved":
            print(f"     · {binding['source_file']} → {binding['slug']} "
                  f"[{binding['status']}]: {binding['reason']}")
    print("   apply: not implemented here; handled by the future plan build.")
    print(f"   Vault unchanged: {report['vault_unchanged']}")
