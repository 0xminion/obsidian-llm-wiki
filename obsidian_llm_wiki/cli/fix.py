"""``olw fix`` — explicit, backed-up deterministic vault maintenance."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer
import yaml

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.cli.health import (
    _normalize_wikilink_target,
    _parse_relation_target,
    _scan_maintenance_findings,
)
from obsidian_llm_wiki.core.backups import backup_file, list_backups
from obsidian_llm_wiki.core.contradictions import ContradictionStore
from obsidian_llm_wiki.core.maintenance import FixKind, PlannedFix, plan_fixes
from obsidian_llm_wiki.render.frontmatter import (
    atomic_write,
    parse_frontmatter,
    safe_read_file,
    sanitize_tag,
)

__all__ = ["fix"]

_BACKUP_RETENTION = 10


@app.command()
def fix(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print a plan without writing files"),
    apply: bool = typer.Option(
        False, "--apply", help="Publish a new projection; never patch legacy or published files"
    ),
    restore: Path | None = typer.Option(
        None, "--restore",
        help="Retained compatibility option; always refused under immutable publication",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the plan or result as JSON"),
) -> None:
    """Plan or explicitly apply conservative maintenance repairs.

    Default mode is a dry run. ``--apply`` rebuilds a complete new immutable
    projection through normal review gates (changed inputs can require model
    calls). It never edits existing pages or resolves human review findings.
    """
    if dry_run and apply:
        raise typer.BadParameter("choose either --dry-run or --apply, not both")
    _, config = resolve_vault(vault)
    from obsidian_llm_wiki.core.application import read_publication

    if restore is not None:
        _emit_error(
            "Refusing --restore: immutable publication never replaces human or published pages. "
            "Inspect/copy the backup to a separate annotation manually, or run "
            "build --refresh-render to publish a new generation.", json_output,
        )
    try:
        bundle_dir, manifest = read_publication(config)
    except (OSError, ValueError, RuntimeError) as exc:
        _emit_error(str(exc), json_output)
    if not bundle_dir.is_dir():
        _emit_error(f"Bundle directory not found: {bundle_dir}", json_output)


    findings_count, findings = _scan_maintenance_findings(bundle_dir)
    plan = plan_fixes(findings)
    contradiction_review_actions = _contradiction_review_actions(bundle_dir)
    skipped_reviewed = sum(1 for finding in findings if finding.reviewed)
    requires_review = [item for item in plan if item.requires_review]
    applicable = [item for item in plan if not item.requires_review]

    if not apply:
        result = {
            "mode": "dry-run",
            "files_scanned": findings_count,
            "plan": [_planned_fix_payload(item) for item in plan],
            "contradiction_review_actions": contradiction_review_actions,
            "summary": {
                "applicable": len(applicable),
                "requires_review": len(requires_review) + len(contradiction_review_actions),
                "skipped_reviewed": skipped_reviewed,
            },
        }
        _emit(result, json_output)
        return

    # Repair the projection from accepted evidence; never patch a published or
    # legacy human page. The existing semantic pipeline preserves review gates.
    import asyncio

    from obsidian_llm_wiki.core.application import source_inventory
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    try:
        sources = source_inventory(config).sources
        if not sources:
            raise ValueError(
                "Refusing maintenance without source inputs; inspect annotations/backups manually."
            )
        compiled = asyncio.run(run_pipeline(config.vault, sources, config, refresh_render=True))
        if compiled.errors:
            raise ValueError("; ".join(compiled.errors))
        _, published = read_publication(config)
    except (OSError, ValueError, RuntimeError) as exc:
        _emit_error("Maintenance publication failed: " + str(exc), json_output)
    result = {
        "mode": "publish",
        "generation": published["generation"],
        "action": "new immutable projection; legacy/human files retained; no in-place fixes",
        "files_scanned": findings_count,
        "plan": [_planned_fix_payload(item) for item in plan],
        "contradiction_review_actions": contradiction_review_actions,
        "summary": {
            "pages_published": len(compiled.pages),
            "requires_review": len(requires_review) + len(contradiction_review_actions),
            "skipped_reviewed": skipped_reviewed,
        },
    }
    _emit(result, json_output)


def _emit_error(message: str, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({"error": message}, sort_keys=True))
    else:
        typer.echo(f"❌ {message}")
    raise typer.Exit(code=1)


def _planned_fix_payload(item: PlannedFix) -> dict[str, Any]:
    return {
        "kind": item.kind.value,
        "path": item.path,
        "payload": item.payload,
        "requires_review": item.requires_review,
    }


def _contradiction_review_actions(bundle_dir: Path) -> list[dict[str, Any]]:
    """Expose active contradiction records as non-mutating human-review work."""
    store_path = bundle_dir / ".llmwiki" / "contradictions.json"
    if not store_path.is_file():
        return []
    try:
        records = ContradictionStore(store_path).records()
    except ValueError:
        return []
    return [
        {
            "action": "review_contradiction",
            "path": ".llmwiki/contradictions.json",
            "payload": {"record_id": record.id, "status": record.status.value},
            "requires_review": True,
        }
        for record in records
        if record.status.value not in {"resolved", "suppressed"}
    ]


def _emit(payload: dict[str, Any], json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    typer.echo(f"Maintenance {payload['mode']} plan")
    for item in payload.get("plan", []):
        safety = "requires human review" if item["requires_review"] else "safe when --apply is used"
        typer.echo(f"- {item['kind']}: {item['path']} ({safety})")
    for action in payload.get("contradiction_review_actions", []):
        typer.echo(f"- {action['action']}: {action['path']} (requires human review)")
    for key, value in payload["summary"].items():
        typer.echo(f"{key.replace('_', ' ')}: {value}")


def _apply_fixes(bundle_dir: Path, plan: list[PlannedFix]) -> int:
    """Apply deterministic planned repairs, making one backup per changed page."""
    by_path: dict[str, list[PlannedFix]] = defaultdict(list)
    for item in plan:
        by_path[item.path].append(item)

    applied = 0
    backup_root = bundle_dir / ".llmwiki" / "backups"
    for relative_path, page_fixes in sorted(by_path.items()):
        page = bundle_dir / relative_path
        if not page.is_file():
            continue
        if any(item.kind is FixKind.REMOVE_EMPTY_GENERATED_STUB for item in page_fixes):
            backup_file(page, backup_root, max_backups=_BACKUP_RETENTION)
            page.unlink()
            applied += sum(
                item.kind is FixKind.REMOVE_EMPTY_GENERATED_STUB for item in page_fixes
            )
            continue

        raw = safe_read_file(page)
        replacement, fixed_count = _fixed_page_content(raw, page_fixes)
        if fixed_count and replacement != raw:
            backup_file(page, backup_root, max_backups=_BACKUP_RETENTION)
            atomic_write(page, replacement)
            applied += fixed_count
    return applied


def _fixed_page_content(raw: str, page_fixes: list[PlannedFix]) -> tuple[str, int]:
    metadata, body = parse_frontmatter(raw)
    if not raw.startswith("---\n") or not metadata:
        return raw, 0
    broken_targets = {
        str(item.payload.get("target", ""))
        for item in page_fixes
        if item.kind is FixKind.REMOVE_BROKEN_RELATION
    }
    normalized_tags = {
        str(item.payload.get("tag", "")): str(item.payload.get("normalized_tag", ""))
        for item in page_fixes
        if item.kind is FixKind.NORMALIZE_TAG
    }
    fixed_count = 0

    if broken_targets and isinstance(metadata.get("relations"), list):
        original_relations = metadata["relations"]
        retained_relations = [
            relation
            for relation in original_relations
            if _normalize_wikilink_target(
                _parse_relation_target(relation) or ""
            )
            not in broken_targets
        ]
        removed = len(original_relations) - len(retained_relations)
        if removed:
            metadata["relations"] = retained_relations
            fixed_count += removed

    if normalized_tags and isinstance(metadata.get("tags"), list):
        tags = metadata["tags"]
        replacement_tags = [
            normalized_tags.get(tag, sanitize_tag(tag))
            if isinstance(tag, str)
            else tag
            for tag in tags
        ]
        changed_tags = sum(old != new for old, new in zip(tags, replacement_tags, strict=True))
        if changed_tags:
            metadata["tags"] = replacement_tags
            fixed_count += changed_tags

    if not fixed_count:
        return raw, 0
    yaml_block = yaml.safe_dump(
        metadata, allow_unicode=True, default_flow_style=False, sort_keys=False
    ).rstrip()
    return f"---\n{yaml_block}\n---\n{body}", fixed_count


def _restore_backup(bundle_dir: Path, backup_path: Path, json_output: bool) -> dict[str, Any]:
    backup_root = (bundle_dir / ".llmwiki" / "backups").resolve()
    candidate = backup_path.expanduser().resolve()
    try:
        candidate.relative_to(backup_root)
    except ValueError:
        _emit_error("Restore path is outside the vault backup root", json_output)
    if not candidate.is_file() or candidate.suffix != ".bak":
        _emit_error("Restore path is not a trusted backup file", json_output)

    destination = _source_for_backup(bundle_dir, backup_root, candidate)
    if destination is None:
        _emit_error("Backup does not match a current vault page", json_output)
    assert destination is not None

    backup_file(destination, backup_root, max_backups=_BACKUP_RETENTION)
    atomic_write(destination, candidate.read_text(encoding="utf-8"))
    return {"mode": "restore", "restored": str(destination.relative_to(bundle_dir))}


def _source_for_backup(bundle_dir: Path, backup_root: Path, candidate: Path) -> Path | None:
    for page in sorted(bundle_dir.rglob("*.md")):
        if candidate in {backup.resolve() for backup in list_backups(page, backup_root)}:
            return page
    return None
