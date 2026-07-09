"""``olw ingest`` — extract URLs + collect clippings → write source files."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
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
    writes them to sources/, then runs the LLM synthesis pipeline on the
    FULL corpus (existing + new).  Unchanged sources reuse their cached
    synthesis; only new/changed sources trigger LLM calls.

    Examples:
        olw ingest ~/MyVault --url https://example.com/article
        olw ingest ~/MyVault -u URL1 -u URL2 --parallel 5
        olw ingest ~/MyVault -u URL1 --dry-run
    """
    vault_path, config = resolve_vault(vault)

    if parallel:
        import os
        os.environ["COMPILE_CONCURRENCY"] = str(parallel)
        config = _reload_config_with_concurrency(vault_path, parallel)

    print(f"📂 Vault: {vault_path}")
    print(f"🤖 Model: {config.llm.model}")

    # ── Collect clippings ──────────────────────────────────────────────
    from obsidian_llm_wiki.ingest.clippings import collect_clippings

    new_count = 0

    passed_clippings = collect_clippings(config)
    if passed_clippings:
        print(f"\n📋 Clippings passing quality gate: {len(passed_clippings)}")
        for clip_path, source in passed_clippings:
            if not dry_run:
                config.sources_dir.mkdir(parents=True, exist_ok=True)
                from obsidian_llm_wiki.render.obsidian import render_source_page
                page = render_source_page(source)
                atomic_write(config.sources_dir / clip_path.name, page)
            new_count += 1
            print(f"   ✅ {source.title[:60]} ({len(source.content)} chars)")

    # ── Extract URLs and write to sources/ ──────────────────────────────
    if urls:
        print(f"\n🌐 Extracting {len(urls)} URL(s)...")
        from obsidian_llm_wiki.ingest.extractors import extract
        from obsidian_llm_wiki.render.obsidian import render_source_page

        for url in urls:
            if dry_run:
                print(f"   🔍 Would extract: {url}")
                continue
            try:
                source = extract(url)
                filename = f"{slugify(source.title)}.md"
                filepath = config.sources_dir / filename
                config.sources_dir.mkdir(parents=True, exist_ok=True)
                page = render_source_page(source)
                atomic_write(filepath, page)
                new_count += 1
                print(f"   ✅ {source.title[:60]} ({len(source.content)} chars)")
            except Exception as exc:
                print(f"   ❌ {url}: {exc}")

    if new_count == 0 and not dry_run:
        print("\n⚠ No new sources to ingest.")
        print("   Tip: Use --url to add URLs or add .md files to 02-Clippings/")

    if skip_synthesis:
        print("\n   ⏭ Skipping synthesis (--skip-synthesis)")
        return

    if dry_run:
        print("\n   🔍 Dry run — no files written.")
        return

    # ── Load the FULL corpus from sources/ ────────────────────────────
    # This is critical: run_pipeline treats any source not in the dict
    # as deleted.  We must pass the complete set of sources, not just
    # the newly extracted ones.
    full_corpus = load_sources_from_dir(config.sources_dir)

    if not full_corpus:
        print("\n⚠ No source files found in sources/.")
        return

    print(f"\n📦 Total corpus: {len(full_corpus)} source(s)")
    if new_count:
        print(f"   ({new_count} new/changed this run)")

    # ── Run synthesis + render pipeline on the full corpus ─────────────
    print("\n🤖 Running LLM synthesis pipeline...")
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    result = asyncio.run(run_pipeline(vault_path, full_corpus, config))

    print_result_summary(result)


def _reload_config_with_concurrency(vault_path: Path, parallel: int):
    """Reload config with concurrency override."""
    from obsidian_llm_wiki.config import load_config

    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    return load_config(
        env_file=env_file,
        VAULT_PATH=str(vault_path),
        COMPILE_CONCURRENCY=str(parallel),
    )
