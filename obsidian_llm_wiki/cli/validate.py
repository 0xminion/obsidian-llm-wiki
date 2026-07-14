"""``olw validate`` — check vault for conformance issues."""

from __future__ import annotations

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.render.frontmatter import extract_wikilinks
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

# Reserved files that don't need type frontmatter.
_RESERVED = frozenset({"index.md", "log.md"})


@app.command()
def validate(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    strict: bool = typer.Option(
        False, "--strict", "-s", help="Treat warnings as errors"
    ),
):
    """Validate the vault for conformance issues.

    Checks:
      - Missing YAML frontmatter (error)
      - Missing/empty 'type' field (error)
      - Broken wikilinks (warning)
      - Missing index.md files (info)

    Examples:
        olw validate ~/MyVault
        olw validate ~/MyVault --strict
    """
    vault_path, config = resolve_vault(vault)
    bundle_dir = config.wiki_dir

    print(f"🔍 Validating: {bundle_dir}")

    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    files_checked = 0

    if not bundle_dir.is_dir():
        print(f"❌ Bundle directory not found: {bundle_dir}")
        raise typer.Exit(code=1)

    all_md_files = sorted(bundle_dir.rglob("*.md"))
    note_stems = {f.stem for f in all_md_files}

    for md_file in all_md_files:
        rel = md_file.relative_to(bundle_dir)
        files_checked += 1
        raw = safe_read_file(md_file)
        if not raw.strip():
            continue

        if md_file.name in _RESERVED:
            continue

        has_frontmatter = raw.startswith("---\n")
        meta, body = parse_frontmatter(raw)

        if not has_frontmatter:
            errors.append(f"{rel}: missing YAML frontmatter")
            continue

        type_val = meta.get("type")
        if not type_val or (isinstance(type_val, str) and not type_val.strip()):
            errors.append(f"{rel}: missing or empty 'type' field")

        # Check Obsidian wikilinks, including aliases, headings, block refs,
        # relative directory prefixes, and explicit .md suffixes.
        for target, _alias in extract_wikilinks(body):
            normalized = target.split("#", 1)[0].strip().rsplit("/", 1)[-1]
            normalized = normalized.removesuffix(".md").strip()
            if normalized and normalized not in note_stems:
                message = f"{rel}: broken link → {target}"
                (errors if strict else warnings).append(message)

    # Check for index.md files.
    for subdir in ["sources", "entries", "concepts", "mocs"]:
        idx = bundle_dir / subdir / "index.md"
        if idx.parent.is_dir() and not idx.exists():
            infos.append(f"{subdir}/: missing index.md")

    # Report.
    if errors:
        print(f"\n❌ {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {w}")
    if infos:
        print(f"\nℹ {len(infos)} info(s):")
        for i in infos:
            print(f"  {i}")

    if not errors and not warnings and not infos:
        print("✅ No issues found.")

    if errors or (strict and warnings):
        raise typer.Exit(code=1)
