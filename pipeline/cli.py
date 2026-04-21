"""CLI entry point for the obsidian-llm-wiki pipeline.

Provides commands for the full 3-stage pipeline and vault maintenance:
  ingest  — extract → plan → create (full pipeline)
  lint    — vault health checks
  reindex — rebuild wiki-index.md
  stats   — show vault statistics
  validate — validate pipeline output
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from pipeline.config import Config, load_config
from pipeline.extract import extract_all
from pipeline.plan import plan_sources
from pipeline.create import create_all, create_file_templates
from pipeline.models import Manifest, Plans
from pipeline.vault import archive_inbox, reindex as vault_reindex

app = typer.Typer(
    name="pipeline",
    help="Obsidian wiki pipeline — extract, plan, create.",
    no_args_is_help=True,
)

log = logging.getLogger(__name__)


def check_dependencies() -> list[str]:
    """Check for required CLI tools. Returns list of missing commands."""
    missing = []
    for cmd in ["curl", "jq", "python3", "hermes"]:
        if not shutil.which(cmd):
            missing.append(cmd)
    return missing


def _pid_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class PipelineLock:
    """Directory-based lock file for pipeline runs."""

    def __init__(self, vault_path: Path):
        self.lock_dir = vault_path / "06-Config" / ".pipeline.lock"
        self.acquired = False

    def acquire(self) -> bool:
        try:
            self.lock_dir.mkdir(exist_ok=False)
            self.acquired = True
            (self.lock_dir / "pid").write_text(str(os.getpid()))
            import atexit
            atexit.register(self.release)
            return True
        except FileExistsError:
            pid_file = self.lock_dir / "pid"
            if pid_file.exists():
                # Check lock age — stale after 30 minutes
                try:
                    lock_age = time.time() - self.lock_dir.stat().st_mtime
                    if lock_age > 1800:
                        log.warning("Stale lock detected (age: %.0fs), forcing release", lock_age)
                        self._force_release()
                        return self.acquire()
                except OSError:
                    pass
                try:
                    old_pid = int(pid_file.read_text().strip())
                    if not _pid_running(old_pid):
                        self._force_release()
                        return self.acquire()
                except ValueError:
                    self._force_release()
                    return self.acquire()
            return False

    def _force_release(self) -> None:
        shutil.rmtree(self.lock_dir, ignore_errors=True)

    def release(self) -> None:
        if self.acquired:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
            self.acquired = False


def _setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)


def _resolve_vault(vault: Optional[Path]) -> Path:
    """Resolve vault path from argument or default."""
    if vault is not None:
        return vault
    return Path.home() / "MyVault"


def _load_cfg(vault: Optional[Path]) -> Config:
    """Load config with resolved vault path."""
    vault_path = _resolve_vault(vault)
    return load_config(vault_path=vault_path)


def _collect_url_files(inbox_dir: Path) -> list[tuple[Path, str]]:
    """Scan inbox for .url files, return list of (filepath, url) tuples."""
    import re
    results = []
    if not inbox_dir.exists():
        return results
    for url_file in sorted(inbox_dir.glob("*.url")):
        content = url_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^URL=(.+)$", content, re.MULTILINE)
        if match:
            results.append((url_file, match.group(1).strip()))
    return results


@app.command()
def init(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    force: bool = typer.Option(False, "--force", "-f", help="Auto-migrate without prompting"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Initialize or migrate vault structure."""
    from pipeline.vault_setup import detect_vault, setup_vault, migrate_vault

    vault_path = vault or Path.home() / "MyVault"
    repo_root = Path(__file__).parent.parent

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

    # Incomplete
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


def _auto_setup(vault_path: Path) -> str:
    """Auto-detect and setup/migrate vault. Returns state string."""
    from pipeline.vault_setup import ensure_vault_ready
    repo_root = Path(__file__).parent.parent
    return ensure_vault_ready(vault_path, repo_root=repo_root, force=True)


# ─── Main: ingest ─────────────────────────────────────────────────────────────

@app.command()
def ingest(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    parallel: int = typer.Option(3, "--parallel", "-p", help="Parallel workers per stage"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline without writing files"),
    review: bool = typer.Option(False, "--review", help="Stage files for review, skip Stage 3"),
    resume: bool = typer.Option(False, "--resume", help="Resume from saved plans (skip Stages 1+2)"),
    template: bool = typer.Option(False, "--template", "-t", help="Use template-based creation (deterministic + insight agent)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Process inbox: extract → plan → create."""
    cfg = _load_cfg(vault)
    _setup_logging(verbose, cfg.log_file)
    t0 = time.time()

    # Auto-setup vault if new or incomplete
    vault_state = _auto_setup(cfg.vault_path)
    if vault_state == "new":
        typer.echo(f"New vault initialized at {cfg.vault_path}")
    elif vault_state == "migrated":
        typer.echo(f"Vault structure migrated at {cfg.vault_path}")

    typer.echo(f"Pipeline ingest — vault: {cfg.vault_path}")
    typer.echo(f"Extract dir: {cfg.resolved_extract_dir}")

    # Validate vault structure
    errors = cfg.validate()
    if errors:
        for e in errors:
            typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=1)

    # Check dependencies
    missing = check_dependencies()
    if missing:
        typer.echo(f"ERROR: Missing required commands: {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)

    # Acquire lock
    lock = PipelineLock(cfg.vault_path)
    if not lock.acquire():
        typer.echo("ERROR: Another pipeline run is in progress. If stale, delete: "
                    f"{cfg.vault_path / '06-Config' / '.pipeline.lock'}", err=True)
        raise typer.Exit(code=1)

    try:
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

        # ─── Collect URLs ──────────────────────────────────────────────────
        url_entries = _collect_url_files(cfg.inbox_dir)
        urls = [u for _, u in url_entries]

        if not urls and not resume:
            typer.echo("No .url files found in inbox.")
            raise typer.Exit(code=0)

        typer.echo(f"Found {len(urls)} URL(s) in inbox.")

        # ─── Stage 1: Extract ──────────────────────────────────────────────
        t1 = time.time()
        if resume:
            typer.echo("Stage 1: SKIPPED (--resume)")
            manifest = Manifest.load(extract_dir)
            if not manifest.entries:
                typer.echo("ERROR: No manifest found for --resume. Run without --resume first.", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"  Loaded {len(manifest.entries)} sources from saved manifest.")
            t1 = t0  # stage was skipped, elapsed is 0
        else:
            typer.echo("Stage 1: Extracting...")
            if dry_run:
                typer.echo("  [DRY RUN] Would extract the following URLs:")
                for url in urls:
                    typer.echo(f"    - {url}")
                manifest = Manifest(entries=[])
            else:
                manifest = extract_all(urls, cfg, parallel=parallel)
            elapsed_1 = time.time() - t1
            typer.echo(f"  Extracted {len(manifest.entries)} sources in {elapsed_1:.1f}s")

        # ─── Stage 2: Plan ─────────────────────────────────────────────────
        t2 = time.time()
        if resume:
            typer.echo("Stage 2: SKIPPED (--resume)")
            plans = Plans.load(extract_dir)
            if not plans.plans:
                typer.echo("ERROR: No plans found for --resume. Run without --resume first.", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"  Loaded {len(plans.plans)} plans from saved file.")
            t2 = t1  # stage was skipped, elapsed is 0
        else:
            typer.echo("Stage 2: Planning...")
            if dry_run:
                typer.echo("  [DRY RUN] Would generate plans for extracted sources.")
                plans = Plans(plans=[])
            else:
                plans = plan_sources(manifest, cfg)
            elapsed_2 = time.time() - t2
            typer.echo(f"  Generated {len(plans.plans)} plans in {elapsed_2:.1f}s")

        if review and not resume:
            typer.echo("Review mode: staging files for approval...")
            from pipeline.review import stage_for_review
            t_review = time.time()
            review_stats = stage_for_review(plans, cfg)
            elapsed_review = time.time() - t_review
            typer.echo(f"  Staged: {review_stats['staged']}, Failed: {review_stats['failed']} in {elapsed_review:.1f}s")
            typer.echo(f"  Run 'pipeline approve' to write to vault or 'pipeline reject' to discard.")
            elapsed_total = time.time() - t0
            typer.echo(f"Done (review mode) in {elapsed_total:.1f}s")
            raise typer.Exit(code=0)

        # ─── Stage 3: Create ───────────────────────────────────────────────
        typer.echo("Stage 3: Creating vault files...")
        t3 = time.time()
        if dry_run:
            typer.echo("  [DRY RUN] Would create vault files for plans.")
            stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
        elif template:
            typer.echo("  Using template-based creation (deterministic + insight agent)")
            stats = create_file_templates(plans.plans, cfg, use_agent_insights=True)
        else:
            stats = create_all(plans, cfg, parallel=parallel)
        elapsed_3 = time.time() - t3
        typer.echo(f"  Created: {stats['created']}, Failed: {stats['failed']} in {elapsed_3:.1f}s")

        # ─── Summary ───────────────────────────────────────────────────────
        elapsed_total = time.time() - t0
        typer.echo("")
        typer.echo("─── Timing Summary ───")
        if not resume:
            typer.echo(f"  Stage 1 (Extract):  {t2 - t1:.1f}s")
            typer.echo(f"  Stage 2 (Plan):     {t3 - t2:.1f}s")
        typer.echo(f"  Stage 3 (Create):   {elapsed_3:.1f}s")
        typer.echo(f"  Total:              {elapsed_total:.1f}s")
        typer.echo("")
        typer.echo(f"Done in {elapsed_total:.1f}s")

    finally:
        lock.release()


# ─── lint ──────────────────────────────────────────────────────────────────────

@app.command()
def lint(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix safe issues (frontmatter, format, banned tags)"),
):
    """Run comprehensive vault health checks (12 checks)."""
    from pipeline.lint import run_lint

    cfg = _load_cfg(vault)
    result = run_lint(cfg.vault_path, fix=fix)

    typer.echo(f"Files checked: {result.files_checked}")
    typer.echo(f"Total issues:  {result.total_issues}")
    if result.fixes_applied:
        typer.echo(f"Fixes applied: {result.fixes_applied}")

    if result.total_issues == 0:
        typer.echo("Vault health check passed ✓")
        raise typer.Exit(code=0)

    for check_name, count in sorted(result.issues_by_check.items()):
        if count:
            typer.echo(f"  ⚠ {check_name}: {count}")

    report_path = cfg.vault_path / "Meta" / "Scripts" / "lint-report.md"
    typer.echo(f"\nFull report: {report_path}")
    raise typer.Exit(code=1)


# ─── reindex ───────────────────────────────────────────────────────────────────

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


# ─── stats ─────────────────────────────────────────────────────────────────────

@app.command()
def stats(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Generate vault dashboard with growth, review status, and health metrics."""
    from pipeline.stats import run_stats

    cfg = _load_cfg(vault)
    summary = run_stats(cfg)

    typer.echo(f"Vault: {cfg.vault_path}")
    typer.echo(f"  Entries:  {summary['entries']}")
    typer.echo(f"  Concepts: {summary['concepts']}")
    typer.echo(f"  Sources:  {summary['sources']}")
    typer.echo(f"  MoCs:     {summary['mocs']}")
    typer.echo(f"  Total:    {summary['total']}")
    typer.echo(f"\nDashboard written to: {summary['dashboard_path']}")


# ─── validate ──────────────────────────────────────────────────────────────────

@app.command()
def validate(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix safe issues"),
):
    """Validate pipeline output (frontmatter, sections, stubs, tags, format)."""
    from pipeline.lint import run_validate

    cfg = _load_cfg(vault)
    result = run_validate(cfg.vault_path, fix=fix)

    typer.echo(f"Files checked: {result.files_checked}")
    if result.fixes_applied:
        typer.echo(f"Fixes applied:  {result.fixes_applied}")

    if result.total_issues == 0:
        typer.echo("Output validation passed ✓")
        raise typer.Exit(code=0)

    typer.echo(f"Violations:    {result.total_issues}")
    for issue in result.issues:
        prefix = {"error": "✗", "warning": "⚠", "info": "ℹ"}[issue.severity.value]
        typer.echo(f"  {prefix} [{issue.check}] {issue.note}: {issue.detail}")

    raise typer.Exit(code=1)


# ─── compile ──────────────────────────────────────────────────────────────────

@app.command(name="compile")
def compile_pass(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Run the compile pass — concept convergence, MoC updates, edge construction."""
    from pipeline.compile import run_compile

    cfg = _load_cfg(vault)
    typer.echo(f"Compile pass — vault: {cfg.vault_path}")

    result = run_compile(cfg)

    if result["success"]:
        typer.echo(f"Compile pass complete. ({result['entries']} entries, "
                    f"{result['concepts']} concepts, {result['mocs']} MoCs)")
        # Reindex after compile
        content = vault_reindex(cfg)
        typer.echo(f"Reindexed wiki-index.md ({content.count(chr(10))} lines)")
    else:
        error = result.get("error", "Unknown error")
        typer.echo(f"Compile pass failed: {error}", err=True)
        raise typer.Exit(code=1)


# ─── approve ──────────────────────────────────────────────────────────────────

@app.command()
def approve(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be written"),
):
    """Approve pending reviews and write files to the vault."""
    from pipeline.review import show_pending, approve_reviews

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    typer.echo(f"Pending reviews: {len(pending)}")
    for r in pending:
        typer.echo(f"  [{r['file_type']}] {Path(r['file_path']).name}")

    if dry_run:
        typer.echo("\nDry run — no files written.")
        raise typer.Exit(code=0)

    stats = approve_reviews(cfg)
    typer.echo(f"\nApproved: {stats['approved']}, Written: {stats['written']}, Failed: {stats['failed']}")


# ─── reject ───────────────────────────────────────────────────────────────────

@app.command()
def reject(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Reject and discard all pending reviews."""
    from pipeline.review import reject_reviews, show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    count = reject_reviews(cfg)
    typer.echo(f"Rejected {count} pending reviews.")


# ─── dlq ──────────────────────────────────────────────────────────────────────

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

    if clear:
        cleared = store.dlq_clear(reason=reason or None)
        typer.echo(f"Cleared {cleared} DLQ items.")
        store.close()
        raise typer.Exit(code=0)

    pending = store.dlq_get_pending()
    store.close()

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


# ─── store ────────────────────────────────────────────────────────────────────

@app.command()
def store_stats(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Show content store statistics."""
    from pipeline.store import ContentStore

    cfg = _load_cfg(vault)
    store = ContentStore.open(cfg.resolved_extract_dir)
    stats = store.get_stats()
    store.close()

    typer.echo(f"Content Store: {cfg.resolved_extract_dir / 'store.db'}")
    typer.echo(f"  URLs:    {stats['urls_total']} total ({stats['urls_ok']} ok, {stats['urls_failed']} failed)")
    typer.echo(f"  Content: {stats['content_total']} entries")
    typer.echo(f"  DLQ:     {stats['dlq_pending']} pending")
    typer.echo(f"  Reviews: {stats['reviews_pending']} pending")


# ─── tags ────────────────────────────────────────────────────────────────────

@app.command(name="tags")
def update_tags(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Rebuild tag-registry.md from actual tag usage across all notes."""
    from collections import Counter
    from pipeline.lint import _parse_frontmatter

    cfg = _load_cfg(vault)
    registry_path = cfg.config_dir / "tag-registry.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _extract_tags(content: str) -> list[str]:
        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        if isinstance(tags, list):
            return [str(t).strip().strip('"').lower() for t in tags if str(t).strip()]
        return []

    entry_tags = Counter()
    concept_tags = Counter()
    moc_tags = Counter()

    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                entry_tags.update(_extract_tags(content))
            except OSError:
                continue

    if cfg.concepts_dir.exists():
        for md in cfg.concepts_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                concept_tags.update(_extract_tags(content))
            except OSError:
                continue

    if cfg.mocs_dir.exists():
        for md in cfg.mocs_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                moc_tags.update(_extract_tags(content))
            except OSError:
                continue

    lines = [
        "# Tag Registry", "",
        "Canonical list of tags used in this wiki. Before minting a new tag,",
        "check this registry and prefer reuse.", "",
        f"Auto-updated on {now}", "",
    ]

    if entry_tags:
        lines.append("## Entry Tags")
        lines.append("")
        for tag, count in entry_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    if concept_tags:
        lines.append("## Concept Tags")
        lines.append("")
        for tag, count in concept_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    if moc_tags:
        lines.append("## MoC Tags")
        lines.append("")
        for tag, count in moc_tags.most_common():
            lines.append(f"- `{tag}` ({count} uses)")
        lines.append("")

    lines.extend([
        "---", "",
        f"*Updated on {now}: {len(entry_tags)} entry tags, {len(concept_tags)} concept tags*",
        "",
    ])

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"Tag registry: {len(entry_tags)} entry, {len(concept_tags)} concept tags")
    typer.echo(f"  Written to: {registry_path}")


# ─── query ───────────────────────────────────────────────────────────────────

@app.command()
def query(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    question: str = typer.Option("", "--ask", "-q", help="Question to ask the vault"),
):
    """Query the vault wiki with a question (compound-back Q&A)."""
    import subprocess

    cfg = _load_cfg(vault)

    if not question:
        queries_dir = cfg.vault_path / "03-Queries"
        if not queries_dir.exists():
            typer.echo("No query files found. Use --ask 'your question'")
            raise typer.Exit(code=0)
        query_files = sorted(queries_dir.glob("*.md"))
        if not query_files:
            typer.echo("No query files found. Use --ask 'your question'")
            raise typer.Exit(code=0)
        question = query_files[0].read_text(encoding="utf-8").strip()

    wiki_index = cfg.wiki_index
    vault_summary = ""
    if wiki_index.exists():
        vault_summary = wiki_index.read_text(encoding="utf-8", errors="replace")[:3000]

    prompt = (
        "You are querying an Obsidian wiki knowledge base.\n\n"
        f"VAULT INDEX:\n{vault_summary}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer based on the vault content. Cite notes using [[wikilinks]]."
    )

    try:
        result = subprocess.run(
            [cfg.agent_cmd, "chat", "-q", prompt, "-Q"],
            cwd=str(cfg.vault_path), capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            typer.echo("\n--- Answer ---\n")
            typer.echo(result.stdout)
        else:
            typer.echo(f"Agent failed: {result.stderr[:200]}", err=True)
            raise typer.Exit(code=1)
    except subprocess.TimeoutExpired:
        typer.echo("Agent timed out", err=True)
        raise typer.Exit(code=1)


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
