"""``olw metrics`` and ``olw recompile`` CLI commands.

Metrics: reads the latest metrics.json and prints a summary.
Recompile: manually retries a single failed source without truncating input.
"""

from __future__ import annotations

import asyncio

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import resolve_vault


@app.command()
def metrics(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
) -> None:
    """Print a summary of the latest pipeline run metrics.

    Reads ``.llmwiki/metrics.json`` from the vault and displays per-phase
    statistics: extractions, syntheses, rendering, and embedding.

    Examples:
        olw metrics ~/MyVault
    """
    vault_path, _ = resolve_vault(vault)

    from obsidian_llm_wiki.core.metrics import print_metrics_summary

    print_metrics_summary(vault_path)


@app.command()
def recompile(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    source_file: str = typer.Argument(..., help="Source filename to retry"),
) -> None:
    """Manually retry a single failed source file.

    Loads the source from sources/, retries synthesis of the complete input,
    and caches the result. Run build afterwards to render the updated cache.

    Examples:
        olw recompile ~/MyVault my-article.md
    """
    vault_path, config = resolve_vault(vault)

    print(f"📂 Vault: {vault_path}")
    print(f"📄 Source: {source_file}")
    print(f"🤖 Model: {config.llm.ingest_model or config.llm.model}")
    print("\n🔄 Recompiling with complete-source retry...")

    from obsidian_llm_wiki.core.pipeline import recompile_single_source

    try:
        result = asyncio.run(recompile_single_source(vault_path, source_file, config))
    except KeyboardInterrupt:
        typer.echo("Recompile cancelled.", err=True)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        typer.echo(f"Recompile failed: {str(exc) or type(exc).__name__}", err=True)
        raise typer.Exit(code=1) from exc

    if result.compiled > 0 and not result.errors:
        print(f"\n✅ Recompiled: {result.compiled} source, "
              f"{len(result.concepts)} concepts")
    else:
        print("\n❌ Recompile failed.")
    if result.errors:
        print(f"   Errors: {len(result.errors)}")
        for err in result.errors[:10]:
            print(f"     - {err}")
    if result.errors or result.compiled == 0:
        raise typer.Exit(code=1)
