"""``olw health`` — vault health report.

Reports on vault quality issues:
  - Broken wikilinks (target file not found)
  - Orphan concepts (not in any MoC)
  - Stub entries (<500 chars body)
  - Low-confidence concepts (<0.5)
  - Missing source links (entries without a source wikilink)
  - Tag violations (spaces in tags)
  - MoCs with <2 concepts

Output is markdown to stdout. Optionally writes to
``vault/04-Wiki/.llmwiki/health-report.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

__all__ = ["health"]

# Reserved files that don't need type frontmatter.
_RESERVED = frozenset({"index.md", "log.md"})

# Wikilink regex: [[slug]] or [[slug|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


@app.command()
def health(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    write: bool = typer.Option(
        False, "--write", "-w",
        help="Write report to vault/04-Wiki/.llmwiki/health-report.md",
    ),
):
    """Generate a vault health report.

    Checks for broken wikilinks, orphan concepts, stub entries,
    low-confidence concepts, missing source links, tag violations,
    and MoCs with fewer than 2 concepts.

    Examples:
        olw health ~/MyVault
        olw health ~/MyVault --write
    """
    vault_path, config = resolve_vault(vault)
    bundle_dir = config.wiki_dir

    if not bundle_dir.is_dir():
        print(f"❌ Bundle directory not found: {bundle_dir}")
        raise typer.Exit(code=1)

    report = _generate_health_report(bundle_dir)
    print(report)

    if write:
        llmwiki_dir = bundle_dir / ".llmwiki"
        llmwiki_dir.mkdir(parents=True, exist_ok=True)
        report_path = llmwiki_dir / "health-report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n📄 Report written to: {report_path}")


def _generate_health_report(bundle_dir: Path) -> str:
    """Generate the health report markdown string."""
    sections: list[str] = []
    all_md_files = sorted(bundle_dir.rglob("*.md"))
    all_stems = {f.stem for f in all_md_files}

    broken_wikilinks: list[str] = []
    orphan_concepts: list[str] = []
    stub_entries: list[str] = []
    low_confidence: list[str] = []
    missing_source_links: list[str] = []
    tag_violations: list[str] = []
    small_mocs: list[str] = []

    # Track which concepts are in MoCs.
    moced_slugs: set[str] = set()
    moc_concept_counts: dict[str, int] = {}

    # First pass: parse MoCs to know which concepts are assigned.
    mocs_dir = bundle_dir / "mocs"
    if mocs_dir.is_dir():
        for moc_file in mocs_dir.glob("*.md"):
            if moc_file.name in _RESERVED:
                continue
            raw = safe_read_file(moc_file)
            _, body = parse_frontmatter(raw)
            # Count wikilinks in the MoC body as concept references.
            moc_links = _WIKILINK_RE.findall(body)
            moc_slugs = [link[0].strip() for link in moc_links]
            # Filter to only concept-directory files.
            moc_concept_slugs = [
                s for s in moc_slugs
                if (bundle_dir / "concepts" / f"{s}.md").exists()
            ]
            moc_concept_counts[moc_file.stem] = len(moc_concept_slugs)
            moced_slugs.update(moc_concept_slugs)

            if len(moc_concept_slugs) < 2:
                small_mocs.append(
                    f"{moc_file.relative_to(bundle_dir)}: "
                    f"{len(moc_concept_slugs)} concept(s)"
                )

    # Second pass: check all files for issues.
    for md_file in all_md_files:
        if md_file.name in _RESERVED:
            continue
        rel = md_file.relative_to(bundle_dir)
        raw = safe_read_file(md_file)
        if not raw.strip():
            continue

        meta, body = parse_frontmatter(raw)
        type_val = str(meta.get("type", "")).strip()

        # Check for broken wikilinks.
        for match in _WIKILINK_RE.finditer(body):
            target_slug = match.group(1).strip()
            target_slug_no_ext = target_slug.replace(".md", "")
            if target_slug_no_ext not in all_stems and target_slug not in all_stems:
                broken_wikilinks.append(f"{rel}: [[{target_slug}]]")

        # Check tag violations (spaces in tags).
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and " " in tag:
                    tag_violations.append(f"{rel}: tag '{tag}' contains spaces")

        # Concept-specific checks.
        if type_val == "Concept":
            concept_slug = md_file.stem

            # Orphan: not in any MoC.
            if concept_slug not in moced_slugs:
                orphan_concepts.append(f"{rel}: not in any MoC")

            # Low confidence.
            confidence = meta.get("confidence")
            if confidence is not None:
                try:
                    conf_val = float(confidence)
                    if conf_val < 0.5:
                        low_confidence.append(
                            f"{rel}: confidence={conf_val}"
                        )
                except (ValueError, TypeError):
                    pass

            # Stub: body < 500 chars.
            if len(body.strip()) < 500:
                stub_entries.append(
                    f"{rel}: {len(body.strip())} chars"
                )

        # Entry-specific checks: must have a source wikilink.
        if type_val == "Entry":
            has_source = any(
                target.strip().lower().startswith("source")
                or (bundle_dir / "sources" / f"{target.strip()}.md").exists()
                for target in [
                    match.group(1).strip()
                    for match in _WIKILINK_RE.finditer(body)
                ]
            )
            # Also check the ## Source section pattern.
            if not has_source and "## Source" not in body:
                missing_source_links.append(f"{rel}: no source wikilink found")

    # Build report.
    sections.append("# Vault Health Report")
    sections.append("")
    sections.append(f"**Vault**: `{bundle_dir}`")
    sections.append(f"**Files scanned**: {len(all_md_files)}")
    sections.append("")

    # Summary table.
    sections.append("## Summary")
    sections.append("")
    sections.append("| Check | Count |")
    sections.append("|-------|-------|")
    sections.append(f"| Broken wikilinks | {len(broken_wikilinks)} |")
    sections.append(f"| Orphan concepts | {len(orphan_concepts)} |")
    sections.append(f"| Stub entries (<500 chars) | {len(stub_entries)} |")
    sections.append(f"| Low-confidence concepts (<0.5) | {len(low_confidence)} |")
    sections.append(f"| Missing source links | {len(missing_source_links)} |")
    sections.append(f"| Tag violations | {len(tag_violations)} |")
    sections.append(f"| MoCs with <2 concepts | {len(small_mocs)} |")
    sections.append("")

    # Details.
    def _detail_section(title: str, items: list[str]) -> None:
        if not items:
            sections.append(f"## {title}")
            sections.append("")
            sections.append("✅ No issues found.")
            sections.append("")
            return
        sections.append(f"## {title} ({len(items)})")
        sections.append("")
        for item in items[:50]:
            sections.append(f"- {item}")
        if len(items) > 50:
            sections.append(f"- ... and {len(items) - 50} more")
        sections.append("")

    _detail_section("Broken Wikilinks", broken_wikilinks)
    _detail_section("Orphan Concepts", orphan_concepts)
    _detail_section("Stub Entries", stub_entries)
    _detail_section("Low-Confidence Concepts", low_confidence)
    _detail_section("Missing Source Links", missing_source_links)
    _detail_section("Tag Violations", tag_violations)
    _detail_section("MoCs with <2 Concepts", small_mocs)

    return "\n".join(sections)
