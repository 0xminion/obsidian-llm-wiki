"""CLI management commands — init, stats, reindex, tags, query, dlq, store,
telemetry, fixture, enrich, setup-qmd, setup-hooks."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import typer

from pipeline.vault import reindex as vault_reindex

from pipeline.cli._helpers import (
    _build_query_prompt,
    _collision_safe_path,
    _load_cfg,
    _resolve_vault,
    app,
    query_vault_fast,
)


@app.command()
def init(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    force: bool = typer.Option(False, "--force", "-f", help="Auto-migrate without prompting"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Initialize or migrate vault structure."""
    from pipeline.vault_setup import detect_vault, migrate_vault, setup_vault

    vault_path = vault or Path.home() / "MyVault"
    repo_root = Path(__file__).parent.parent.parent

    state = detect_vault(vault_path)

    if state.state == "existing":
        typer.echo(f"Vault ready: {vault_path}")
        raise typer.Exit(code=0)

    if state.state == "new":
        typer.echo(f"Setting up new vault at {vault_path}")
        actions = setup_vault(vault_path, repo_root=repo_root, quiet=quiet)
        if not quiet:
            for a in actions:
                typer.echo(f"  + {a}")
        typer.echo(f"\nSetup complete: {len(actions)} actions.")
        raise typer.Exit(code=0)

    typer.echo(f"Incomplete vault at {vault_path}:")
    if state.missing_dirs:
        typer.echo(f"  Missing dirs: {', '.join(state.missing_dirs)}")
    if state.missing_files:
        typer.echo(f"  Missing files: {', '.join(state.missing_files)}")

    if not force:
        response = typer.confirm("Migrate vault structure?", default=True)
        if not response:
            typer.echo("Migration skipped.")
            raise typer.Exit(code=1)

    actions = migrate_vault(vault_path, state, repo_root=repo_root)
    if not quiet:
        for a in actions:
            typer.echo(f"  + {a}")
    typer.echo(f"\nMigration complete: {len(actions)} actions.")
    raise typer.Exit(code=0)


@app.command()
def reindex(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Rebuild wiki-index.md."""
    cfg = _load_cfg(vault)
    content = vault_reindex(cfg)
    lines = content.count("\n")
    typer.echo(f"Rebuilt wiki-index.md ({lines} lines)")
    typer.echo(f"  Location: {cfg.wiki_index}")


@app.command()
def stats(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Generate vault dashboard with growth, review status, and health metrics."""
    from pipeline.stats import run_stats

    cfg = _load_cfg(vault)
    summary = run_stats(cfg)

    if json_output:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        raise typer.Exit(code=0)

    typer.echo(f"Vault: {cfg.vault_path}")
    typer.echo(f"  Entries:  {summary['entries']}")
    typer.echo(f"  Concepts: {summary['concepts']}")
    typer.echo(f"  Sources:  {summary['sources']}")
    typer.echo(f"  MoCs:     {summary['mocs']}")
    typer.echo(f"  Total:    {summary['total']}")
    typer.echo(f"\nDashboard written to: {summary['dashboard_path']}")


@app.command()
def dlq(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    clear: bool = typer.Option(False, "--clear", help="Clear all pending DLQ items"),
    reason: str = typer.Option("", "--reason", help="Filter/clear by reason"),
):
    """Show or manage the dead letter queue (failed extractions)."""
    from pipeline.store import ContentStore

    cfg = _load_cfg(vault)
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        if clear:
            cleared = store.dlq_clear(reason=reason or None)
            typer.echo(f"Cleared {cleared} DLQ items.")
            raise typer.Exit(code=0)

        pending = store.dlq_get_pending()

        if not pending:
            typer.echo("Dead letter queue is empty.")
            raise typer.Exit(code=0)

        typer.echo(f"Dead letter queue: {len(pending)} items")
        for item in pending:
            typer.echo(f"\n  URL: {item['url']}")
            typer.echo(f"  Reason: {item['reason']}")
            typer.echo(f"  Attempts: {item['attempts']}")
            if item.get("last_error"):
                typer.echo(f"  Error: {item['last_error'][:100]}")
    finally:
        store.close()


@app.command()
def store_stats(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Show content store statistics."""
    from pipeline.store import ContentStore

    cfg = _load_cfg(vault)
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        s = store.get_stats()
    finally:
        store.close()

    typer.echo(f"Content Store: {cfg.resolved_extract_dir / 'store.db'}")
    typer.echo(f"  URLs:    {s['urls_total']} total ({s['urls_ok']} ok, {s['urls_failed']} failed)")
    typer.echo(f"  Content: {s['content_total']} entries")
    typer.echo(f"  DLQ:     {s['dlq_pending']} pending")
    typer.echo(f"  Reviews: {s['reviews_pending']} pending")


@app.command(name="tags")
def update_tags(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Rebuild tag-registry.md from actual tag usage across all notes."""
    from collections import Counter

    from pipeline.utils import extract_tags

    cfg = _load_cfg(vault)
    registry_path = cfg.config_dir / "tag-registry.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    entry_tags = Counter()
    concept_tags = Counter()
    moc_tags = Counter()
    source_tags = Counter()

    for note_dir, counter in [
        (cfg.entries_dir, entry_tags),
        (cfg.concepts_dir, concept_tags),
        (cfg.mocs_dir, moc_tags),
        (cfg.sources_dir, source_tags),
    ]:
        if note_dir.exists():
            for md in note_dir.glob("*.md"):
                try:
                    content = md.read_text(encoding="utf-8", errors="replace")
                    counter.update(extract_tags(content))
                except OSError:
                    continue

    lines = [
        "# Tag Registry", "",
        "Canonical list of tags used in this wiki. Before minting a new tag,",
        "check this registry and prefer reuse.", "",
        f"Auto-updated on {now}", "",
    ]

    for section_name, counter in [
        ("Entry Tags", entry_tags),
        ("Concept Tags", concept_tags),
        ("MoC Tags", moc_tags),
        ("Source Tags", source_tags),
    ]:
        if counter:
            lines.append(f"## {section_name}")
            lines.append("")
            for tag, count in counter.most_common():
                lines.append(f"- `{tag}` ({count} uses)")
            lines.append("")

    lines.extend([
        "---", "",
        f"*Updated on {now}: {len(entry_tags)} entry tags, {len(concept_tags)} concept tags, "
        f"{len(moc_tags)} MoC tags, {len(source_tags)} source tags*",
        "",
    ])

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"Tag registry: {len(entry_tags)} entry, {len(concept_tags)} concept, {len(moc_tags)} MoC tags")
    typer.echo(f"  Written to: {registry_path}")


@app.command()
def query(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    question: str = typer.Option("", "--ask", "-q", help="Question to ask the vault"),
    all_queries: bool = typer.Option(False, "--all", "-a", help="Process all pending queries"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode: direct LLM call instead of Hermes agent"),
):
    """Query the vault wiki with a question (compound-back Q&A).

    Drop query .md files in 03-Queries/. Answers are written to 05-Outputs/
    and queries are archived to 09-Archive-Queries/.

    Use --fast for simple lookups (sub-5s via direct LLM).
    Omit --fast for complex research questions (Hermes agent with tool use).
    """
    cfg = _load_cfg(vault)
    queries_dir = cfg.vault_path / "03-Queries"
    outputs_dir = cfg.vault_path / "05-Outputs"
    archive_dir = cfg.vault_path / "09-Archive-Queries"
    archive_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    had_failures = False

    query_files: list[Path] = []
    if question:
        query_files = [Path("__command_line__")]
    else:
        if not queries_dir.exists():
            typer.echo("No query files found. Use --ask 'your question'")
            raise typer.Exit(code=0)
        query_files = sorted(queries_dir.glob("*.md"))
        if not query_files:
            typer.echo("No query files found. Use --ask 'your question'")
            raise typer.Exit(code=0)
        if not all_queries:
            query_files = query_files[:1]

    for qf in query_files:
        if str(qf) == "__command_line__":
            qtext = question
            qname = "cli-query"
        else:
            qtext = qf.read_text(encoding="utf-8").strip()
            qname = qf.stem

        prompt = _build_query_prompt(cfg, qtext)

        if fast:
            answer = query_vault_fast(cfg, qtext)
            if answer:
                typer.echo(f"\n--- Answer ({qname}) ---\n")
                typer.echo(answer)
                out_file = _collision_safe_path(outputs_dir / f"{qname}.md")
                out_file.write_text(
                    f"# Query: {qname}\n\n"
                    f"**Question:**\n{qtext}\n\n"
                    f"**Answer:**\n{answer}\n",
                    encoding="utf-8",
                )
                typer.echo(f"\nWritten to: {out_file}")
                if str(qf) != "__command_line__":
                    archive_path = _collision_safe_path(archive_dir / qf.name)
                    qf.rename(archive_path)
                    typer.echo(f"Archived to: {archive_path}")
            else:
                had_failures = True
                typer.echo(f"LLM failed for {qname} (empty response)", err=True)
        else:
            try:
                result = subprocess.run(
                    [cfg.agent_cmd, "chat", "-q", prompt, "-Q"],
                    cwd=str(cfg.vault_path), capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    answer = result.stdout.strip()
                    typer.echo(f"\n--- Answer ({qname}) ---\n")
                    typer.echo(answer)
                    out_file = _collision_safe_path(outputs_dir / f"{qname}.md")
                    out_file.write_text(
                        f"# Query: {qname}\n\n"
                        f"**Question:**\n{qtext}\n\n"
                        f"**Answer:**\n{answer}\n",
                        encoding="utf-8",
                    )
                    typer.echo(f"\nWritten to: {out_file}")
                    if str(qf) != "__command_line__":
                        archive_path = _collision_safe_path(archive_dir / qf.name)
                        qf.rename(archive_path)
                        typer.echo(f"Archived to: {archive_path}")
                else:
                    had_failures = True
                    typer.echo(f"Agent failed for {qname}: {result.stderr[:200]}", err=True)
            except subprocess.TimeoutExpired:
                had_failures = True
                typer.echo(f"Agent timed out for {qname}", err=True)
            except FileNotFoundError:
                typer.echo(f"Agent command not found: {cfg.agent_cmd}", err=True)
                raise typer.Exit(code=127)

    if had_failures:
        raise typer.Exit(code=1)


@app.command(name="fixture")
def fixture_cmd(
    vault: Path = typer.Argument(..., help="Vault path to populate"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing fixture files"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Create a deterministic example vault for demos and snapshot tests."""
    from pipeline.fixtures import create_example_vault

    summary = create_example_vault(vault, overwrite=overwrite)
    if json_output:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Fixture vault ready: {vault}")
        typer.echo(f"  Files written: {summary['files_written']}")


@app.command(name="telemetry")
def telemetry_cmd(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    limit: int = typer.Option(20, "--limit", min=1, help="Number of recent events"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show recent structured pipeline telemetry events."""
    from pipeline.telemetry import read_recent_events

    cfg = _load_cfg(vault)
    events = read_recent_events(cfg.telemetry_file, limit=limit)
    if json_output:
        typer.echo(json.dumps({"events": events}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if not events:
            typer.echo("No telemetry events found.")
        for event in events:
            typer.echo(f"{event.get('timestamp')} {event.get('stage')} {event.get('status')} {event.get('duration_s')}s")


@app.command(name="setup-qmd")
def setup_qmd_cmd(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Install and configure qmd for semantic concept search."""
    from pipeline.setup import setup_qmd

    vault_path = _resolve_vault(vault)
    setup_qmd(vault_path)


@app.command()
def enrich(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    note: str = typer.Argument(..., help="Relative path to the note inside the vault"),
):
    """Re-extract a source note, diff against existing, and propose updates."""
    from pipeline.enrich import enrich_note

    cfg = _load_cfg(vault)
    note_path = cfg.vault_path / note
    resolved = note_path.resolve()
    try:
        if not resolved.is_relative_to(cfg.vault_path):
            typer.echo("ERROR: note path must be inside the vault.", err=True)
            raise typer.Exit(code=1)
    except ValueError:
        typer.echo("ERROR: note path must be inside the vault.", err=True)
        raise typer.Exit(code=1)
    if not resolved.exists():
        typer.echo(f"ERROR: note not found: {resolved}", err=True)
        raise typer.Exit(code=1)

    result = enrich_note(resolved, cfg)
    status = result.get("status", "unknown")
    if status == "error":
        typer.echo(f"ERROR: {result.get('error', 'unknown error')}", err=True)
        raise typer.Exit(code=1)
    if status == "no_changes":
        typer.echo("No changes detected.")
        raise typer.Exit(code=0)
    typer.echo(f"Plan generated ({result.get('diff_length', 0)} chars diff)")
    if result.get("plan_path"):
        typer.echo(f"Plan written to: {result['plan_path']}")
    raise typer.Exit(code=0)


@app.command(name="setup-hooks")
def setup_hooks_cmd(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Install git hooks (pre-commit, commit-msg) in the vault repo."""
    from pipeline.setup import setup_git_hooks

    vault_path = _resolve_vault(vault)
    setup_git_hooks(vault_path)
