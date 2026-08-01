"""``olw health`` — vault health reports and machine-readable maintenance findings."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.contradictions import ContradictionStore
from obsidian_llm_wiki.core.maintenance import (
    FindingKind,
    FindingSeverity,
    MaintenanceFinding,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.review import is_reviewed_page
from obsidian_llm_wiki.render.frontmatter import sanitize_tag
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

__all__ = ["health"]

# Reserved files that don't need type frontmatter.
_RESERVED = frozenset({"index.md", "log.md"})

# Wikilink regex: [[slug]] or [[slug|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_PARENTHETICAL_ALIAS_RE = re.compile(r"\(([^()]+)\)")
_MAX_ALIAS_CANDIDATE_CHARS = 160


def _content_markdown_files(bundle_dir: Path) -> list[Path]:
    """Return live vault notes, never pipeline state or quarantine evidence."""
    return sorted(
        path
        for path in bundle_dir.rglob("*.md")
        if not ({".llmwiki", "views"} & set(path.relative_to(bundle_dir).parts))
    )


def _normalize_wikilink_target(target: str) -> str:
    """Reduce a raw wikilink target to the bare note stem Obsidian resolves."""
    stem = target.split("#", 1)[0].strip()
    stem = stem.rsplit("/", maxsplit=1)[-1]
    return stem.removesuffix(".md").strip()


@app.command()
def health(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    write: bool = typer.Option(
        False, "--write", "-w", help="Write report to vault/04-Wiki/.llmwiki/health-report.md"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit stable typed maintenance findings as JSON"
    ),
    full: bool = typer.Option(
        False, "--full", help="Scan every generated note instead of one bounded batch"
    ),
    batch_size: int = typer.Option(
        1_000, "--batch-size", min=1, help="Maximum notes in an incremental scan"
    ),
) -> None:
    """Generate a vault health report or JSON maintenance findings.

    The default markdown report is retained for human review.  ``--json`` is
    intended for automation and contains only deterministic typed findings.
    """
    _, config = resolve_vault(vault)
    bundle_dir = config.wiki_dir

    if not bundle_dir.is_dir():
        if json_output:
            typer.echo(
                json.dumps({"error": f"Bundle directory not found: {bundle_dir}"}, sort_keys=True)
            )
        else:
            typer.echo(f"❌ Bundle directory not found: {bundle_dir}")
        raise typer.Exit(code=1)

    incremental_scan: tuple[int, list[MaintenanceFinding]] | None = None
    indexed_full_scan: tuple[int, list[MaintenanceFinding]] | None = None
    cycle_complete = True
    snapshot_complete = True
    if not full:
        with VaultRegistry(config.llmwiki_dir) as registry:
            relative_paths, cycle_complete = registry.next_maintenance_batch(
                limit=batch_size
            )
            (
                all_stems,
                concept_stems,
                source_stems,
                moced_slugs,
                moc_counts,
            ) = registry.maintenance_page_sets()
            selected = [
                bundle_dir / path
                for path in relative_paths
                if (bundle_dir / path).is_file()
            ]
            if relative_paths or not _content_markdown_files(bundle_dir):
                batch_scan = _scan_selected_maintenance_findings(
                    bundle_dir,
                    selected,
                    all_stems,
                    concept_stems,
                    source_stems,
                    moced_slugs,
                    moc_counts,
                )
                registry.commit_maintenance_batch(
                    relative_paths,
                    (_finding_payload(finding) for finding in batch_scan[1]),
                    cycle_complete=cycle_complete,
                )
                snapshot_complete = registry.maintenance_snapshot_complete()
                incremental_scan = (
                    batch_scan[0],
                    [
                        _finding_from_registry_payload(finding)
                        for finding in registry.health_findings()
                    ],
                )
        if incremental_scan is None:
            # Legacy vaults have no render manifest yet. Perform one explicit
            # full scan and seed the durable snapshot instead of returning a
            # batch-shaped result that silently omits unindexed pages.
            with VaultRegistry(config.llmwiki_dir) as registry:
                indexed_full_scan = _append_source_hash_findings(
                    _scan_maintenance_findings(bundle_dir),
                    bundle_dir,
                    registry.source_content_hashes(),
                )
                registry.replace_all_health_findings(
                    _finding_payload(finding) for finding in indexed_full_scan[1]
                )
    else:
        with VaultRegistry(config.llmwiki_dir) as registry:
            indexed_full_scan = _append_source_hash_findings(
                _scan_maintenance_findings(bundle_dir),
                bundle_dir,
                registry.source_content_hashes(),
            )
            indexed_full_scan = _append_unmanaged_generated_findings(
                indexed_full_scan,
                bundle_dir,
                registry.rendered_paths(),
            )
            registry.replace_all_health_findings(
                _finding_payload(finding) for finding in indexed_full_scan[1]
            )

    if json_output:
        payload = _health_json_payload(
            bundle_dir,
            scan=incremental_scan or indexed_full_scan,
            incremental=incremental_scan is not None,
            cycle_complete=cycle_complete,
            snapshot_complete=snapshot_complete,
        )
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    report = (
        _generate_incremental_health_report(
            incremental_scan[0], incremental_scan[1], cycle_complete
        )
        if incremental_scan is not None
        else _generate_health_report(bundle_dir)
    )
    typer.echo(report)

    if write:
        llmwiki_dir = bundle_dir / ".llmwiki"
        llmwiki_dir.mkdir(parents=True, exist_ok=True)
        report_path = llmwiki_dir / "health-report.md"
        report_path.write_text(report, encoding="utf-8")
        typer.echo(f"\n📄 Report written to: {report_path}")


def _health_json_payload(
    bundle_dir: Path,
    *,
    scan: tuple[int, list[MaintenanceFinding]] | None = None,
    incremental: bool = False,
    cycle_complete: bool = True,
    snapshot_complete: bool = True,
) -> dict[str, Any]:
    """Build a stable JSON-safe health result without making any changes."""
    files_scanned, findings = scan or _scan_maintenance_findings(bundle_dir)
    payload: dict[str, Any] = {
        "files_scanned": files_scanned,
        "findings": [_finding_payload(finding) for finding in findings],
        "summary": dict(Counter(finding.kind.value for finding in findings)),
    }
    if incremental:
        payload.update(
            incremental=True,
            cycle_complete=cycle_complete,
            snapshot_complete=snapshot_complete,
        )
    store_path = bundle_dir / ".llmwiki" / "contradictions.json"
    if store_path.is_file():
        try:
            records = ContradictionStore(store_path).records()
        except ValueError:
            records = []
        payload["contradictions"] = [_contradiction_payload(record) for record in records]
        payload["contradiction_summary"] = dict(
            sorted(Counter(record.status.value for record in records).items())
        )
    return payload


def _contradiction_payload(record: Any) -> dict[str, Any]:
    """Serialize durable contradiction evidence without mutating its status."""
    return {
        "evidence": list(record.evidence),
        "id": record.id,
        "sources": [
            {
                "content_hash": source.content_hash,
                "revision": source.revision,
                "source_path": source.source_path,
            }
            for source in record.sources
        ],
        "status": record.status.value,
        "summary": record.summary,
    }


def _finding_payload(finding: MaintenanceFinding) -> dict[str, Any]:
    return {
        "details": finding.details,
        "kind": finding.kind.value,
        "message": finding.message,
        "path": finding.path,
        "reviewed": finding.reviewed,
        "severity": finding.severity.value,
    }


def _finding_from_registry_payload(payload: dict[str, object]) -> MaintenanceFinding:
    """Restore a typed finding from one validated registry row."""
    details = payload.get("details", {})
    return MaintenanceFinding(
        kind=FindingKind(str(payload["kind"])),
        path=str(payload["path"]),
        message=str(payload["message"]),
        severity=FindingSeverity(str(payload.get("severity", "warning"))),
        details=details if isinstance(details, dict) else {},
        reviewed=bool(payload.get("reviewed", False)),
    )


def _scan_maintenance_findings(bundle_dir: Path) -> tuple[int, list[MaintenanceFinding]]:
    """Scan deterministic repair candidates without writing vault content."""
    all_md_files = _content_markdown_files(bundle_dir)
    all_stems = {file.stem for file in all_md_files}
    concept_stems = {file.stem for file in all_md_files if file.parent.name == "concepts"}
    source_stems = {file.stem for file in all_md_files if file.parent.name == "sources"}
    moced_slugs, moc_counts = _moc_membership_data(bundle_dir, concept_stems)
    findings: list[MaintenanceFinding] = []

    for md_file in all_md_files:
        if md_file.name in _RESERVED:
            continue
        raw = safe_read_file(md_file)
        if not raw.strip():
            continue
        metadata, body = parse_frontmatter(raw)
        relative_path = str(md_file.relative_to(bundle_dir))
        reviewed = is_reviewed_page(raw)

        findings.extend(
            _relation_findings(metadata, concept_stems, relative_path, reviewed)
        )
        findings.extend(_tag_findings(metadata, relative_path, reviewed))
        findings.extend(_alias_findings(metadata, relative_path, reviewed))

        findings.extend(
            _content_findings(
                metadata,
                body,
                relative_path,
                md_file.stem,
                reviewed,
                all_stems=all_stems,
                source_stems=source_stems,
                moc_counts=moc_counts,
            )
        )

        if metadata.get("generated") is True and not body.strip():
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.EMPTY_GENERATED_STUB,
                    path=relative_path,
                    message="Generated page has an empty body",
                    reviewed=reviewed,
                )
            )

        if str(metadata.get("type", "")).strip() == "Concept" and md_file.stem not in moced_slugs:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.ORPHAN_CONCEPT,
                    path=relative_path,
                    message="Concept is not in any MoC",
                    reviewed=reviewed,
                )
            )

    return len(all_md_files), sorted(
        findings, key=lambda item: (item.path, item.kind.value, item.message)
    )


def _append_source_hash_findings(
    scan: tuple[int, list[MaintenanceFinding]],
    bundle_dir: Path,
    expected_hashes: dict[str, str],
) -> tuple[int, list[MaintenanceFinding]]:
    """Verify source bytes during explicit full scans, never normal builds."""
    from obsidian_llm_wiki.core.state import hash_content
    from obsidian_llm_wiki.ingest.sources import load_source_file

    files_scanned, findings = scan
    for filename, expected_hash in sorted(expected_hashes.items()):
        source_path = bundle_dir / "sources" / filename
        source = load_source_file(source_path) if source_path.is_file() else None
        actual_hash = hash_content(source.content) if source is not None else ""
        if actual_hash != expected_hash:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.SOURCE_HASH_MISMATCH,
                    path=f"sources/{filename}",
                    message="Source content differs from the authoritative registry hash",
                    severity=FindingSeverity.ERROR,
                    details={"actual": actual_hash, "expected": expected_hash},
                )
            )
    return files_scanned, sorted(
        findings, key=lambda item: (item.path, item.kind.value, item.message)
    )


def _append_unmanaged_generated_findings(
    scan: tuple[int, list[MaintenanceFinding]],
    bundle_dir: Path,
    managed_paths: set[str],
) -> tuple[int, list[MaintenanceFinding]]:
    """Flag legacy generated pages no longer represented by a render manifest."""
    files_scanned, findings = scan
    if not managed_paths:
        return scan
    generated_types = {"Concept", "Entry", "Map of Content"}
    for directory in ("concepts", "entries", "mocs"):
        for path in sorted((bundle_dir / directory).glob("*.md")):
            relative = path.relative_to(bundle_dir).as_posix()
            if path.name in _RESERVED or relative in managed_paths:
                continue
            raw = safe_read_file(path)
            metadata, _ = parse_frontmatter(raw)
            if (
                str(metadata.get("type", "")).strip() not in generated_types
                or metadata.get("orphaned") is True
                or is_reviewed_page(raw)
            ):
                continue
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.UNMANAGED_GENERATED_PAGE,
                    path=relative,
                    message="Generated page is not owned by the current render manifest",
                    details={"path": relative},
                )
            )
    return files_scanned, sorted(
        findings, key=lambda item: (item.path, item.kind.value, item.message)
    )


def _scan_selected_maintenance_findings(
    bundle_dir: Path,
    markdown_files: list[Path],
    all_stems: set[str],
    concept_stems: set[str],
    source_stems: set[str],
    moced_slugs: set[str],
    moc_counts: dict[str, int],
) -> tuple[int, list[MaintenanceFinding]]:
    """Apply deterministic checks to one registry-selected maintenance batch."""
    findings: list[MaintenanceFinding] = []
    for md_file in markdown_files:
        if md_file.name in _RESERVED:
            continue
        raw = safe_read_file(md_file)
        if not raw.strip():
            continue
        metadata, body = parse_frontmatter(raw)
        relative_path = str(md_file.relative_to(bundle_dir))
        reviewed = is_reviewed_page(raw)
        findings.extend(
            _relation_findings(metadata, concept_stems, relative_path, reviewed)
        )
        findings.extend(_tag_findings(metadata, relative_path, reviewed))
        findings.extend(_alias_findings(metadata, relative_path, reviewed))
        findings.extend(
            _content_findings(
                metadata,
                body,
                relative_path,
                md_file.stem,
                reviewed,
                all_stems=all_stems,
                source_stems=source_stems,
                moc_counts=moc_counts,
            )
        )
        if metadata.get("generated") is True and not body.strip():
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.EMPTY_GENERATED_STUB,
                    path=relative_path,
                    message="Generated page has an empty body",
                    reviewed=reviewed,
                )
            )
        if (
            str(metadata.get("type", "")).strip() == "Concept"
            and md_file.stem not in moced_slugs
        ):
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.ORPHAN_CONCEPT,
                    path=relative_path,
                    message="Concept is not in any MoC",
                    reviewed=reviewed,
                )
            )
    return len(markdown_files), sorted(
        findings, key=lambda item: (item.path, item.kind.value, item.message)
    )


def _moc_membership_data(
    bundle_dir: Path,
    concept_stems: set[str],
) -> tuple[set[str], dict[str, int]]:
    mocs_dir = bundle_dir / "mocs"
    if not mocs_dir.is_dir():
        return set(), {}
    moced_slugs: set[str] = set()
    moc_counts: dict[str, int] = {}
    for moc_file in sorted(mocs_dir.glob("*.md")):
        if moc_file.name in _RESERVED:
            continue
        _, body = parse_frontmatter(safe_read_file(moc_file))
        slugs = {
            _normalize_wikilink_target(target)
            for target, _ in _WIKILINK_RE.findall(body)
            if _normalize_wikilink_target(target) in concept_stems
        }
        moced_slugs.update(slugs)
        moc_counts[moc_file.stem] = len(slugs)
    return moced_slugs, moc_counts


def _moced_concept_slugs(bundle_dir: Path, concept_stems: set[str]) -> set[str]:
    return _moc_membership_data(bundle_dir, concept_stems)[0]


def _content_findings(
    metadata: dict[str, Any],
    body: str,
    path: str,
    stem: str,
    reviewed: bool,
    *,
    all_stems: set[str],
    source_stems: set[str],
    moc_counts: dict[str, int],
) -> list[MaintenanceFinding]:
    """Return typed content-quality findings shared by full and batch scans."""
    findings: list[MaintenanceFinding] = []
    for raw_target, _ in _WIKILINK_RE.findall(body):
        target = _normalize_wikilink_target(raw_target)
        if target and target not in all_stems and "." not in target:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.BROKEN_WIKILINK,
                    path=path,
                    message=f"Wikilink target does not exist: {target}",
                    severity=FindingSeverity.ERROR,
                    details={"target": target},
                    reviewed=reviewed,
                )
            )

    page_type = str(metadata.get("type", "")).strip()
    if page_type == "Concept":
        confidence = metadata.get("confidence")
        try:
            if confidence is not None and float(confidence) < 0.5:
                findings.append(
                    MaintenanceFinding(
                        kind=FindingKind.LOW_CONFIDENCE,
                        path=path,
                        message=f"Concept confidence is below 0.5: {confidence}",
                        reviewed=reviewed,
                    )
                )
        except (TypeError, ValueError):
            pass
        body_length = len(body.strip())
        if body_length < 500:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.SHORT_CONCEPT_STUB,
                    path=path,
                    message=f"Concept body is shorter than 500 characters: {body_length}",
                    reviewed=reviewed,
                )
            )
    elif page_type == "Entry":
        has_source = any(
            target.strip().lower().startswith("source")
            or _normalize_wikilink_target(target) in source_stems
            for target, _ in _WIKILINK_RE.findall(body)
        )
        if not has_source and "## Source" not in body:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.MISSING_SOURCE_LINK,
                    path=path,
                    message="Entry has no source wikilink",
                    reviewed=reviewed,
                )
            )
    elif page_type == "Map of Content" and moc_counts.get(stem, 0) < 2:
        findings.append(
            MaintenanceFinding(
                kind=FindingKind.UNDERSIZED_MOC,
                path=path,
                message=f"MoC contains fewer than two concepts: {moc_counts.get(stem, 0)}",
                reviewed=reviewed,
            )
        )
    return findings


def _parse_relation_target(relation: object) -> str | None:
    """Extract the target slug from a relation entry.

    Relations are serialized as pipe-separated strings
    ``"slug|relation_type|display_label"`` so Obsidian's Properties panel
    treats them as a simple list.  Legacy dict entries with a ``target``
    key are still accepted for pages rendered before the format change.
    """
    if isinstance(relation, dict):
        target = relation.get("target")
        return str(target) if isinstance(target, str) and target.strip() else None
    if isinstance(relation, str):
        # Format: "slug|relation_type|display_label" — use the first part.
        slug = relation.split("|", 1)[0].strip()
        return slug or None
    return None


def _relation_findings(
    metadata: dict[str, Any], concept_stems: set[str], path: str, reviewed: bool
) -> list[MaintenanceFinding]:
    relations = metadata.get("relations")
    if not isinstance(relations, list):
        return []
    findings: list[MaintenanceFinding] = []
    for relation in relations:
        raw_target = _parse_relation_target(relation)
        if raw_target is None:
            continue
        target = _normalize_wikilink_target(raw_target)
        if target and target not in concept_stems:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.BROKEN_RELATION,
                    path=path,
                    message=f"Relation target does not exist: {target}",
                    severity=FindingSeverity.ERROR,
                    details={"target": target},
                    reviewed=reviewed,
                )
            )
    return findings


def _tag_findings(metadata: dict[str, Any], path: str, reviewed: bool) -> list[MaintenanceFinding]:
    tags = metadata.get("tags")
    if not isinstance(tags, list):
        return []
    findings: list[MaintenanceFinding] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        normalized_tag = sanitize_tag(tag)
        if normalized_tag != tag:
            findings.append(
                MaintenanceFinding(
                    kind=FindingKind.TAG_NORMALIZATION,
                    path=path,
                    message=f"Tag needs Obsidian normalization: {tag}",
                    details={"tag": tag, "normalized_tag": normalized_tag},
                    reviewed=reviewed,
                )
            )
    return findings


def _alias_findings(
    metadata: dict[str, Any], path: str, reviewed: bool
) -> list[MaintenanceFinding]:
    """Propose absent parenthetical title aliases without modifying any page."""
    title = metadata.get("title")
    if not isinstance(title, str):
        return []
    title = " ".join(title.split())
    if not title:
        return []
    aliases = metadata.get("aliases")
    existing: set[str] = set()
    if isinstance(aliases, list):
        existing = {
            " ".join(alias.split()).casefold()
            for alias in aliases
            if isinstance(alias, str)
        }
    findings: list[MaintenanceFinding] = []
    seen: set[str] = set()
    for match in _PARENTHETICAL_ALIAS_RE.finditer(title):
        alias = " ".join(match.group(1).split())
        key = alias.casefold()
        if (
            not alias
            or len(alias) > _MAX_ALIAS_CANDIDATE_CHARS
            or not any(char.isalnum() for char in alias)
            or key == title.casefold()
            or key in existing
            or key in seen
        ):
            continue
        seen.add(key)
        findings.append(
            MaintenanceFinding(
                kind=FindingKind.ALIAS_CANDIDATE,
                path=path,
                message=f"Alias candidate from title: {alias}",
                details={"alias": alias, "title": title},
                reviewed=reviewed,
            )
        )
    return findings


def _generate_incremental_health_report(
    files_scanned: int,
    findings: list[MaintenanceFinding],
    cycle_complete: bool,
) -> str:
    lines = [
        "# Vault Health Report (Incremental)",
        "",
        f"- Files scanned this batch: **{files_scanned}**",
        f"- Maintenance cycle complete: **{'yes' if cycle_complete else 'no'}**",
        f"- Findings: **{len(findings)}**",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(
            f"- **{finding.severity.value} / {finding.kind.value}** "
            f"`{finding.path}` — {finding.message}"
            for finding in findings
        )
        lines.append("")
    else:
        lines.extend(["✅ No issues in this batch.", ""])
    return "\n".join(lines)


def _generate_health_report(bundle_dir: Path) -> str:
    """Generate the original human-oriented health report markdown string."""
    sections: list[str] = []
    all_md_files = _content_markdown_files(bundle_dir)
    all_stems = {file.stem for file in all_md_files}
    concept_stems = {file.stem for file in all_md_files if file.parent.name == "concepts"}
    source_stems = {file.stem for file in all_md_files if file.parent.name == "sources"}

    broken_wikilinks: list[str] = []
    orphan_concepts: list[str] = []
    stub_entries: list[str] = []
    low_confidence: list[str] = []
    missing_source_links: list[str] = []
    tag_violations: list[str] = []
    small_mocs: list[str] = []

    moced_slugs: set[str] = set()
    moc_concept_counts: dict[str, int] = {}
    mocs_dir = bundle_dir / "mocs"
    if mocs_dir.is_dir():
        for moc_file in mocs_dir.glob("*.md"):
            if moc_file.name in _RESERVED:
                continue
            raw = safe_read_file(moc_file)
            _, body = parse_frontmatter(raw)
            moc_slugs = [_normalize_wikilink_target(link[0]) for link in _WIKILINK_RE.findall(body)]
            moc_concept_slugs = [slug for slug in moc_slugs if slug in concept_stems]
            moc_concept_counts[moc_file.stem] = len(moc_concept_slugs)
            moced_slugs.update(moc_concept_slugs)
            if len(moc_concept_slugs) < 2:
                small_mocs.append(
                    f"{moc_file.relative_to(bundle_dir)}: {len(moc_concept_slugs)} concept(s)"
                )

    for md_file in all_md_files:
        if md_file.name in _RESERVED:
            continue
        rel = md_file.relative_to(bundle_dir)
        raw = safe_read_file(md_file)
        if not raw.strip():
            continue
        meta, body = parse_frontmatter(raw)
        type_val = str(meta.get("type", "")).strip()

        for match in _WIKILINK_RE.finditer(body):
            target_slug = match.group(1).strip()
            stem = _normalize_wikilink_target(target_slug)
            if not stem or stem in all_stems or "." in stem:
                continue
            broken_wikilinks.append(f"{rel}: [[{target_slug}]]")

        tags = meta.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and " " in tag:
                    tag_violations.append(f"{rel}: tag '{tag}' contains spaces")

        if type_val == "Concept":
            concept_slug = md_file.stem
            if concept_slug not in moced_slugs:
                orphan_concepts.append(f"{rel}: not in any MoC")
            confidence = meta.get("confidence")
            if confidence is not None:
                try:
                    confidence_value = float(confidence)
                    if confidence_value < 0.5:
                        low_confidence.append(f"{rel}: confidence={confidence_value}")
                except (ValueError, TypeError):
                    pass
            if len(body.strip()) < 500:
                stub_entries.append(f"{rel}: {len(body.strip())} chars")

        if type_val == "Entry":
            has_source = any(
                target.strip().lower().startswith("source")
                or _normalize_wikilink_target(target) in source_stems
                for target in (target.strip() for target, _ in _WIKILINK_RE.findall(body))
            )
            if not has_source and "## Source" not in body:
                missing_source_links.append(f"{rel}: no source wikilink found")

    sections.extend(
        [
            "# Vault Health Report",
            "",
            f"**Vault**: `{bundle_dir}`",
            f"**Files scanned**: {len(all_md_files)}",
            "",
            "## Summary",
            "",
            "| Check | Count |",
            "|-------|-------|",
            f"| Broken wikilinks | {len(broken_wikilinks)} |",
            f"| Orphan concepts | {len(orphan_concepts)} |",
            f"| Stub entries (<500 chars) | {len(stub_entries)} |",
            f"| Low-confidence concepts (<0.5) | {len(low_confidence)} |",
            f"| Missing source links | {len(missing_source_links)} |",
            f"| Tag violations | {len(tag_violations)} |",
            f"| MoCs with <2 concepts | {len(small_mocs)} |",
            "",
        ]
    )

    def detail_section(title: str, items: list[str]) -> None:
        if not items:
            sections.extend([f"## {title}", "", "✅ No issues found.", ""])
            return
        sections.extend([f"## {title} ({len(items)})", ""])
        sections.extend(f"- {item}" for item in items[:50])
        if len(items) > 50:
            sections.append(f"- ... and {len(items) - 50} more")
        sections.append("")

    detail_section("Broken Wikilinks", broken_wikilinks)
    detail_section("Orphan Concepts", orphan_concepts)
    detail_section("Stub Entries", stub_entries)
    detail_section("Low-Confidence Concepts", low_confidence)
    detail_section("Missing Source Links", missing_source_links)
    detail_section("Tag Violations", tag_violations)
    detail_section("MoCs with <2 Concepts", small_mocs)
    return "\n".join(sections)
