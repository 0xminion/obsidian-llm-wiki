"""``olw build`` — re-synthesise changed sources and re-render the vault."""

from __future__ import annotations

import asyncio

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


@app.command()
def build(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-synthesis of all sources"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Detect changes but skip synthesis"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the LLM model"
    ),
):
    """Re-synthesise changed sources and re-render the vault.

    Detects changes in sources/, runs the LLM synthesis pipeline on changed
    sources, and re-renders the vault.  Unchanged sources reuse their cached
    synthesis so the full corpus is always rendered.

    Examples:
        olw build ~/MyVault
        olw build ~/MyVault --force
        olw build ~/MyVault --dry-run
    """
    vault_path, config = resolve_vault(vault)

    if model:
        import os
        os.environ["LLM_MODEL"] = model
        config.llm.model = model

    print(f"📂 Vault: {vault_path}")
    print(f"🤖 Model: {config.llm.model}")

    # Enumerate the full corpus but read bodies only for new/changed sources.
    # Registry metadata stubs preserve source indexes for unchanged pages.
    with VaultRegistry(config.llmwiki_dir) as registry:
        source_load = load_sources_incremental(
            config.sources_dir,
            registry,
            force=force,
        )
    sources = source_load.sources

    print(f"\n📦 Found {len(sources)} source file(s)")

    if dry_run:
        print("   🔍 Dry run — would synthesise changed sources")
        return

    # ── Run pipeline with the full corpus ───────────────────────────────
    print("\n🤖 Running synthesis pipeline...")
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    result = asyncio.run(
        run_pipeline(
            vault_path,
            sources,
            config,
            force=force,
            unchanged_filenames=source_load.unchanged_filenames,
            source_file_signatures=source_load.signatures,
        )
    )

    if not force and result.compiled == 0 and not result.errors:
        print("✅ Already up-to-date.")
        return

    print_result_summary(result)
