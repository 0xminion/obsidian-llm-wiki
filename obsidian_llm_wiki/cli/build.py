"""``olw build`` — re-synthesise changed sources and re-render the vault."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import typer

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.cli._helpers import print_result_summary, resolve_vault
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.state import hash_content
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


class _ReadOnlyPlanState:
    """The loader's two read methods, backed only by the legacy JSON snapshot.

    Do not open SQLite for planning: even a read-only WAL connection can create
    sidecars. No signature stubs are trusted here; every source body is checked.
    """

    def __init__(self, path: Path) -> None:
        self.hashes: dict[str, str] = {}
        self.missing = False
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.missing = True
            return
        try:
            data = json.loads(raw)
            entries = data.get("sources", {}) if isinstance(data, dict) else None
            if not isinstance(entries, dict):
                raise ValueError("sources must be an object")
            for filename, entry in entries.items():
                if not isinstance(entry, dict) or not isinstance(entry.get("hash"), str):
                    raise ValueError(f"invalid source state for {filename!r}")
                self.hashes[filename] = entry["hash"]
        except (ValueError, TypeError) as exc:
            raise OSError(f"Cannot plan from invalid state.json: {exc}") from exc

    def source_filenames(self) -> set[str]:
        return set(self.hashes)

    def source_file_states(self) -> dict[str, tuple[int, int, SourceDoc]]:
        return {}


def _print_build_plan(config: Config, *, force: bool, refresh_render: bool) -> None:
    """Report a content-only plan using the same fail-closed source inventory."""
    state = _ReadOnlyPlanState(config.state_file)
    # Structural adapter: the loader only uses these two read-only methods.
    loaded = load_sources_incremental(config.sources_dir, cast(VaultRegistry, state))
    groups: dict[str, list[str]] = {key: [] for key in ("new", "changed", "unchanged", "deleted")}
    for filename, source in loaded.sources.items():
        if filename not in state.hashes:
            status = "new"
        elif hash_content(source.content) != state.hashes[filename]:
            status = "changed"
        else:
            status = "unchanged"
        groups[status].append(filename)
    groups["deleted"] = sorted(state.source_filenames() - loaded.sources.keys())
    print("\n🔍 Dry run — content-only plan from state.json; no network or files written.")
    if state.missing:
        print("   state.json absent — all present sources are classified as new.")
    for status, filenames in groups.items():
        print(f"   {status}: {len(filenames)}")
        for filename in filenames:
            print(f"     - {filename}")
    candidates = len(loaded.sources) if force else len(groups["new"]) + len(groups["changed"])
    print(f"   Would synthesise (content-only): {candidates}")
    print("   --force: all present sources, including unchanged" if force else "   --force: off")
    if refresh_render:
        print("   --refresh-render: reconcile generated pages without forcing synthesis.")
    print(
        "   Limits: reads all source bodies; state.json may lag the registry. "
        "Model/schema compatibility, cache validity, queue eligibility and render changes "
        "are not checked."
    )


@app.command()
def build(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-synthesis of all sources"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Read-only content plan from state.json; no synthesis or writes"
    ),
    refresh_render: bool = typer.Option(
        False, "--refresh-render",
        help="Reconcile generated pages without forcing LLM re-synthesis",
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
        config = replace(config, llm=replace(config.llm, model=model, ingest_model=model))

    print(f"📂 Vault: {vault_path}")
    print(f"🤖 Model: {config.llm.ingest_model or config.llm.model}")

    if dry_run:
        try:
            _print_build_plan(config, force=force, refresh_render=refresh_render)
        except (OSError, ValueError) as exc:
            typer.echo(f"Build plan failed; no files written: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        return

    # Enumerate the full corpus but read bodies only for new/changed sources.
    # Registry metadata stubs preserve source indexes for unchanged pages.
    try:
        with VaultRegistry(config.llmwiki_dir) as registry:
            source_load = load_sources_incremental(
                config.sources_dir,
                registry,
                force=force,
            )
    except (OSError, ValueError) as exc:
        typer.echo(f"Source inventory failed; no build performed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    sources = source_load.sources

    print(f"\n📦 Found {len(sources)} source file(s)")

    # ── Run pipeline with the full corpus ───────────────────────────────
    print("\n🤖 Running synthesis pipeline...")
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    try:
        result = asyncio.run(
            run_pipeline(
                vault_path,
                sources,
                config,
                force=force,
                unchanged_filenames=source_load.unchanged_filenames,
                source_file_signatures=source_load.signatures,
                refresh_render=refresh_render,
            )
        )
    except KeyboardInterrupt:
        typer.echo("Build cancelled.", err=True)
        raise typer.Exit(code=130) from None
    except Exception as exc:
        typer.echo(f"Build failed: {str(exc) or type(exc).__name__}", err=True)
        raise typer.Exit(code=1) from exc

    if not force and result.compiled == 0 and not result.errors:
        print("✅ Already up-to-date.")
        return

    print_result_summary(result)
    if result.errors:
        raise typer.Exit(code=1)
