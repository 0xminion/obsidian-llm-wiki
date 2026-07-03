"""``olw ingest`` — extract URLs + collect clippings → write source files."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.clippings import collect_clippings
from obsidian_llm_wiki.ingest.web import extract_web
from obsidian_llm_wiki.render.obsidian import atomic_write, slugify


@app.command()
def ingest(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    urls: list[str] | None = typer.Option(
        None, "--url", "-u", help="URLs to ingest (can be repeated)"
    ),
    parallel: int = typer.Option(
        3, "--parallel", "-p", help="Concurrent LLM calls during synthesis"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview extraction without writing files"
    ),
    skip_synthesis: bool = typer.Option(
        False,
        "--skip-synthesis",
        help="Only extract sources; skip LLM synthesis and rendering",
    ),
):
    """Ingest URLs and clippings, then synthesise + render the vault.

    Extracts full content from URLs, collects clippings from 02-Clippings/,
    then runs the LLM synthesis pipeline to generate concepts, MOCs, and
    entries.

    Examples:
        olw ingest ~/MyVault --url https://example.com/article
        olw ingest ~/MyVault -u URL1 -u URL2 --parallel 5
        olw ingest ~/MyVault -u URL1 --dry-run
    """
    vault_path, config = resolve_vault(vault)

    if parallel:
        import os
        os.environ["COMPILE_CONCURRENCY"] = str(parallel)
        config = load_config_with_concurrency(vault_path, parallel)

    print(f"📂 Vault: {vault_path}")
    print(f"🤖 Model: {config.llm.model}")

    # ── Collect clippings ──────────────────────────────────────────────
    sources: dict[str, SourceDoc] = {}

    passed_clippings = collect_clippings(config)
    if passed_clippings:
        print(f"\n📋 Clippings passing quality gate: {len(passed_clippings)}")
        for clip_path, source in passed_clippings:
            key = clip_path.name
            sources[key] = source
            print(f"   ✅ {source.title[:60]} ({len(source.content)} chars)")

    # ── Extract URLs ───────────────────────────────────────────────────
    if urls:
        print(f"\n🌐 Extracting {len(urls)} URL(s)...")
        if dry_run:
            print("   🔍 Dry run — would extract:")
            for url in urls:
                print(f"      {url}")
        else:
            for url in urls:
                try:
                    source = extract_web(url)
                    filename = f"{slugify(source.title)}.md"
                    filepath = config.sources_dir / filename
                    if not dry_run:
                        config.sources_dir.mkdir(parents=True, exist_ok=True)
                        # Write source page with frontmatter.
                        from obsidian_llm_wiki.render.obsidian import render_source_page
                        page = render_source_page(source)
                        atomic_write(filepath, page)
                    sources[filename] = source
                    print(f"   ✅ {source.title[:60]} ({len(source.content)} chars)")
                except Exception as exc:
                    print(f"   ❌ {url}: {exc}")

    if not sources:
        print("\n⚠ No sources to process.")
        if not urls and not passed_clippings:
            print("   Tip: Use --url to add URLs or add .md files to 02-Clippings/")
        return

    print(f"\n📦 Total sources: {len(sources)}")

    if skip_synthesis:
        print("   ⏭ Skipping synthesis (--skip-synthesis)")
        return

    if dry_run:
        print("   🔍 Dry run — would synthesise these sources:")
        for key, source in sorted(sources.items()):
            print(f"      {source.title[:60]} ← {key}")
        return

    # ── Run synthesis + render pipeline ────────────────────────────────
    print("\n🤖 Running LLM synthesis pipeline...")
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    result = asyncio.run(run_pipeline(vault_path, sources, config, force=True))

    print_result_summary(result)


def load_config_with_concurrency(vault_path: Path, parallel: int):
    """Reload config with concurrency override."""
    from obsidian_llm_wiki.config import load_config

    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    return load_config(
        env_file=env_file,
        VAULT_PATH=str(vault_path),
        COMPILE_CONCURRENCY=str(parallel),
    )
