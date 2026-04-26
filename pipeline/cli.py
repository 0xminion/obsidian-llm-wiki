"""CLI entry point for the obsidian-llm-wiki pipeline.

Provides commands for the full 3-stage pipeline and vault maintenance:
  ingest  — extract → plan → create (full pipeline)
  lint    — vault health checks
  reindex — rebuild wiki-index.md
  stats   — show vault statistics
  validate — validate pipeline output
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from pipeline._common import VaultLock
from pipeline.config import Config, load_config
from pipeline.extract import extract_all
from pipeline.plan import plan_sources
from pipeline.create import create_all, create_file_templates
from pipeline.models import ExtractedSource, Manifest, Plans, SourceType
from pipeline.utils import extract_body, parse_url_file_content
from pipeline.vault import reindex as vault_reindex

app = typer.Typer(
    name="pipeline",
    help="Obsidian wiki pipeline — extract, plan, create.",
    no_args_is_help=True,
)

log = logging.getLogger(__name__)


def _collision_safe_path(path: Path) -> Path:
    """Return path, or a numbered sibling if path already exists."""
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def query_vault_fast(cfg: Config, question: str) -> str:
    """Fast direct-LLM query path. Kept separate for testability."""
    from pipeline.llm_client import get_llm_client

    return get_llm_client(cfg).generate(_build_query_prompt(cfg, question), timeout=120)


def check_dependencies(agent_cmd: str = "hermes") -> list[str]:
    """Check for baseline CLI tools needed before ingest starts.

    Do not preflight the agent binary here.
    Agent-backed stages handle missing binaries at the actual call path so
    dry runs, empty inboxes, mocked tests, and resume flows without real agent
    execution do not fail early for the wrong reason.
    """
    missing = []
    required_cmds = ["curl", "python3"]
    for cmd in required_cmds:
        if not shutil.which(cmd):
            missing.append(cmd)
    return missing


class PipelineLock(VaultLock):
    """Directory-based lock file for pipeline runs (delegates to VaultLock)."""

    def __init__(self, vault_path: Path):
        super().__init__(vault_path, name="pipeline")


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
    results = []
    if not inbox_dir.exists():
        return results
    for url_file in sorted(inbox_dir.glob("*.url")):
        content = url_file.read_text(encoding="utf-8", errors="replace")
        url = parse_url_file_content(content)
        if url:
            results.append((url_file, url))
    return results


def _collect_clipping_files(clippings_dir: Path) -> list[tuple[Path, dict]]:
    """Scan 02-Clippings for markdown files, return list of (filepath, data_dict)."""
    from pipeline.utils import collect_clipping_files

    if not clippings_dir.exists():
        return []
    return collect_clipping_files(clippings_dir)


def _query_keywords(question: str) -> set[str]:
    """Extract meaningful keywords from a question for note retrieval."""
    stopwords = {
        "about", "this", "that", "what", "which", "when", "where", "who",
        "does", "with", "from", "into", "your", "their", "there", "have",
        "vault",
    }
    return {
        w.lower() for w in re.split(r"[^\w]+", question)
        if len(w) > 3 and w.lower() not in stopwords
    }


def _gather_query_note_context(cfg: Config, question: str, limit: int = 6) -> str:
    """Gather relevant note snippets from entries, sources, concepts, and MoCs."""
    keywords = _query_keywords(question)

    def _display_name(raw: str, fallback: str) -> str:
        body = extract_body(raw)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and len(stripped) > 2:
                return stripped[2:].strip()
        return fallback

    candidates: list[tuple[int, str, str]] = []
    note_dirs = [
        (cfg.entries_dir, "entry"),
        (cfg.sources_dir, "source"),
        (cfg.concepts_dir, "concept"),
        (cfg.mocs_dir, "moc"),
    ]

    for directory, label in note_dirs:
        if not directory.is_dir():
            continue
        for md in directory.glob("*.md"):
            try:
                raw = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = extract_body(raw).strip()
            display_name = _display_name(raw, md.stem)
            haystack = f"{display_name}\n{body}".lower()
            # Simple keyword scoring normalized by document length to avoid long-note bias
            raw_score = sum(1 for kw in keywords if kw in haystack)
            if raw_score <= 0:
                continue
            score = raw_score / (len(haystack) / 2000 + 1)
            snippet = re.sub(r"\s+", " ", body)[:600]
            candidates.append((score, md.stem, f"- [[{md.stem}]] ({display_name}; {label}): {snippet}"))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not candidates:
        # Fallback: most recently modified notes (more relevant than alphabetical)
        for directory, label in note_dirs:
            if not directory.is_dir():
                continue
            recent = sorted(
                directory.glob("*.md"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )[:2]
            for md in recent:
                try:
                    raw = md.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                body = extract_body(raw).strip()
                display_name = _display_name(raw, md.stem)
                snippet = re.sub(r"\s+", " ", body)[:600]
                candidates.append((0, md.stem, f"- [[{md.stem}]] ({display_name}; {label}): {snippet}"))

    if not candidates:
        return ""

    lines = ["Relevant note excerpts:"]
    seen: set[str] = set()
    for _, stem, line in candidates:
        if stem in seen:
            continue
        seen.add(stem)
        lines.append(line)
        if len(seen) >= limit:
            break
    return "\n".join(lines)


def _build_query_prompt(cfg: Config, question: str) -> str:
    """Build a retrieval-augmented prompt for vault Q&A."""
    vault_summary = ""
    if cfg.wiki_index.exists():
        vault_summary = cfg.wiki_index.read_text(encoding="utf-8", errors="replace")[:2500]

    note_context = _gather_query_note_context(cfg, question)
    sections = [
        "You are querying an Obsidian wiki knowledge base.",
        "",
        "VAULT INDEX:",
        vault_summary,
    ]
    if note_context:
        sections.extend(["", note_context])
    sections.extend([
        "",
        f"QUESTION: {question}",
        "",
        "Answer based on the vault content. Cite notes using [[wikilinks]]. If the vault is incomplete, say so.",
    ])
    return "\n".join(sections)


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
    agent: bool = typer.Option(False, "--agent", "-a", help="Use full agent mode for creation (slower, may timeout — not recommended)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Process inbox: extract → plan → create."""
    cfg = _load_cfg(vault)
    cfg.parallel = parallel
    t0 = time.time()
    if dry_run:
        from pipeline.vault_setup import detect_vault

        if verbose:
            _setup_logging(True, None)

        state = detect_vault(cfg.vault_path)
        typer.echo(f"Pipeline ingest — vault: {cfg.vault_path}")

        if state.state == "new":
            typer.echo("[DRY RUN] Vault does not exist; would initialize it during a real ingest.")
            typer.echo("[DRY RUN] No files were created.")
            raise typer.Exit(code=0)

        if state.state == "incomplete":
            typer.echo(state.summary)
            typer.echo("[DRY RUN] Vault is incomplete; would migrate missing directories/files during a real ingest.")
            typer.echo("[DRY RUN] No files were created.")
            raise typer.Exit(code=0)

        typer.echo(f"Extract dir: {cfg.resolved_extract_dir}")
        url_entries = _collect_url_files(cfg.inbox_dir)
        urls = [u for _, u in url_entries]
        if not urls and not resume:
            typer.echo("No .url files found in inbox.")
            raise typer.Exit(code=0)
        typer.echo(f"Found {len(urls)} URL(s) in inbox.")
        typer.echo("  [DRY RUN] Would extract the following URLs:")
        for url in urls:
            typer.echo(f"    - {url}")
        typer.echo("  [DRY RUN] Would generate plans for extracted sources.")
        if review and not resume:
            typer.echo("  [DRY RUN] Would stage generated files for review.")
        else:
            typer.echo("  [DRY RUN] Would create vault files for generated plans.")
        typer.echo("[DRY RUN] No files were created.")
        raise typer.Exit(code=0)

    _setup_logging(verbose, cfg.log_file)

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
    missing = check_dependencies(cfg.agent_cmd)
    if missing:
        typer.echo(f"ERROR: Missing required commands: {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)

    # Acquire lock
    lock = PipelineLock(cfg.vault_path)
    if not lock.acquire():
        typer.echo("ERROR: Another pipeline run is in progress. If stale, delete: "
                    f"{lock.lock_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

        # ─── Metrics ────────────────────────────────────────────────────────
        from pipeline.metrics import reset_metrics, start_stage, end_stage, get_metrics
        reset_metrics()

        # ─── Collect URLs + Clippings ────────────────────────────────────────
        url_entries = _collect_url_files(cfg.inbox_dir)
        urls = [u for _, u in url_entries]
        clipping_entries = _collect_clipping_files(cfg.clippings_dir)

        has_work = bool(urls or clipping_entries)
        if not has_work and not resume:
            typer.echo("No .url files in inbox and no .md clippings found.")
            raise typer.Exit(code=0)

        inbox_msg_parts: list[str] = []
        if urls:
            inbox_msg_parts.append(f"{len(urls)} URL(s)")
        if clipping_entries:
            inbox_msg_parts.append(f"{len(clipping_entries)} clipping(s)")
        typer.echo(f"Found {' + '.join(inbox_msg_parts)} in inbox.")

        # ─── Stage 1: Extract ─────────────────────────────────────────────────
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
            start_stage("extract")
            if dry_run:
                if urls:
                    typer.echo("  [DRY RUN] Would extract the following URLs:")
                    for url in urls:
                        typer.echo(f"    - {url}")
                if clipping_entries:
                    typer.echo("  [DRY RUN] Would ingest clippings (skip Stage 1):")
                    for _, clipped in clipping_entries:
                        typer.echo(f"    - {clipped.get('title', 'untitled')}")
                manifest = Manifest(entries=[])
            else:
                manifest = extract_all(urls, cfg, parallel=parallel)
                # ── Clippings bypass Stage 1 (already processed / defuddled)
                for _fp, clipped in clipping_entries:
                    source = ExtractedSource(
                        url=clipped["url"],
                        title=clipped["title"],
                        content=clipped["content"],
                        type=SourceType(clipped.get("type", "web")),
                        author=clipped.get("author", ""),
                        source_file=clipped.get("source_file", ""),
                    )
                    manifest.entries.append(source)
                    source.save(extract_dir)
                manifest.save(extract_dir)
            elapsed_1 = time.time() - t1
            end_stage("extract")
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
            start_stage("plan")
            if dry_run:
                typer.echo("  [DRY RUN] Would generate plans for extracted sources.")
                plans = Plans(plans=[])
            else:
                plans = plan_sources(manifest, cfg)
            elapsed_2 = time.time() - t2
            end_stage("plan")
            typer.echo(f"  Generated {len(plans.plans)} plans in {elapsed_2:.1f}s")

        if review and resume:
            typer.echo("Review mode with --resume: staging saved plans for approval...")
            from pipeline.review import stage_for_review
            t_review = time.time()
            review_stats = stage_for_review(plans, cfg)
            elapsed_review = time.time() - t_review
            typer.echo(f"  Staged: {review_stats['staged']}, Failed: {review_stats['failed']} in {elapsed_review:.1f}s")
            typer.echo("  Run 'pipeline approve' to write to vault or 'pipeline reject' to discard.")
            elapsed_total = time.time() - t0
            typer.echo(f"Done (review mode, resumed) in {elapsed_total:.1f}s")
            raise typer.Exit(code=0)

        if review and not resume:
            typer.echo("Review mode: staging files for approval...")
            from pipeline.review import stage_for_review
            t_review = time.time()
            review_stats = stage_for_review(plans, cfg)
            elapsed_review = time.time() - t_review
            typer.echo(f"  Staged: {review_stats['staged']}, Failed: {review_stats['failed']} in {elapsed_review:.1f}s")
            typer.echo("  Run 'pipeline approve' to write to vault or 'pipeline reject' to discard.")
            elapsed_total = time.time() - t0
            typer.echo(f"Done (review mode) in {elapsed_total:.1f}s")
            raise typer.Exit(code=0)

        # ─── Stage 3: Create ───────────────────────────────────────────────
        typer.echo("Stage 3: Creating vault files...")
        start_stage("create")
        t3 = time.time()
        if dry_run:
            typer.echo("  [DRY RUN] Would create vault files for plans.")
            stats = {"created": 0, "failed": 0, "sources": 0, "entries": 0}
        elif agent:
            typer.echo("  Using full agent mode (heavy — may timeout)")
            stats = create_all(plans, cfg, parallel=parallel)
        else:
            typer.echo("  Using template-based creation (deterministic + insight agent)")
            stats = create_file_templates(plans.plans, cfg, use_agent_insights=True)
        elapsed_3 = time.time() - t3
        end_stage("create")
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

        # Agent metrics
        metrics = get_metrics()
        if metrics.total_agent_calls > 0:
            typer.echo(metrics.summary())
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
    """Run comprehensive vault health checks (15 checks)."""
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
    from pipeline.metrics import reset_metrics, start_stage, end_stage, get_metrics
    from pipeline._common import VaultLock

    cfg = _load_cfg(vault)
    typer.echo(f"Compile pass — vault: {cfg.vault_path}")

    # Acquire lock (same pattern as ingest)
    lock = VaultLock(cfg.vault_path, name="pipeline")
    if not lock.acquire():
        typer.echo("ERROR: Another pipeline run is in progress. If stale, delete: "
                    f"{lock.lock_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        reset_metrics()
        start_stage("compile")
        result = run_compile(cfg)
        end_stage("compile")

        if result["success"]:
            typer.echo(f"Compile pass complete. ({result['entries']} entries, "
                        f"{result['concepts']} concepts, {result['mocs']} MoCs)")
            # Reindex after compile
            content = vault_reindex(cfg)
            typer.echo(f"Reindexed wiki-index.md ({content.count(chr(10))} lines)")

            metrics = get_metrics()
            if metrics.total_agent_calls > 0:
                typer.echo("")
                typer.echo(metrics.summary())
        else:
            error = result.get("error", "Unknown error")
            typer.echo(f"Compile pass failed: {error}", err=True)
            raise typer.Exit(code=1)
    finally:
        lock.release()


# ─── approve ──────────────────────────────────────────────────────────────────

@app.command()
def approve(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be written"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Approve pending reviews and write files to the vault."""
    from pipeline.review import show_pending, approve_reviews

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        if json_output:
            typer.echo(json.dumps({"pending": 0, "approved": 0, "written": 0, "failed": 0}, ensure_ascii=False, sort_keys=True))
        else:
            typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    if json_output and dry_run:
        typer.echo(json.dumps({"pending": len(pending), "items": pending}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    elif not json_output:
        typer.echo(f"Pending reviews: {len(pending)}")
        for r in pending:
            typer.echo(f"  [{r['file_type']}] {Path(r['file_path']).name}")

    if dry_run:
        if not json_output:
            typer.echo("\nDry run — no files written.")
        raise typer.Exit(code=0)

    stats = approve_reviews(cfg)
    if json_output:
        typer.echo(json.dumps({"pending": len(pending), **stats}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"\nApproved: {stats['approved']}, Written: {stats['written']}, Failed: {stats['failed']}")
        if stats.get("written_paths"):
            typer.echo("Written paths:")
            for path in stats["written_paths"]:
                typer.echo(f"  - {path}")


# ─── reject ───────────────────────────────────────────────────────────────────

@app.command()
def reject(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Reject and discard all pending reviews."""
    from pipeline.review import reject_reviews, show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)

    if not pending:
        if json_output:
            typer.echo(json.dumps({"pending": 0, "rejected": 0}, ensure_ascii=False, sort_keys=True))
        else:
            typer.echo("No pending reviews.")
        raise typer.Exit(code=0)

    count = reject_reviews(cfg)
    if json_output:
        typer.echo(json.dumps({"pending": len(pending), "rejected": count}, ensure_ascii=False, sort_keys=True))
    else:
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
    from pipeline.utils import extract_tags

    cfg = _load_cfg(vault)
    registry_path = cfg.config_dir / "tag-registry.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    entry_tags = Counter()
    concept_tags = Counter()
    moc_tags = Counter()
    source_tags = Counter()

    if cfg.entries_dir.exists():
        for md in cfg.entries_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                entry_tags.update(extract_tags(content))
            except OSError:
                continue

    if cfg.concepts_dir.exists():
        for md in cfg.concepts_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                concept_tags.update(extract_tags(content))
            except OSError:
                continue

    if cfg.mocs_dir.exists():
        for md in cfg.mocs_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                moc_tags.update(extract_tags(content))
            except OSError:
                continue

    if cfg.sources_dir.exists():
        for md in cfg.sources_dir.glob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                source_tags.update(extract_tags(content))
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

    if source_tags:
        lines.append("## Source Tags")
        lines.append("")
        for tag, count in source_tags.most_common():
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


# ─── query ───────────────────────────────────────────────────────────────────

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
    import subprocess

    cfg = _load_cfg(vault)
    queries_dir = cfg.vault_path / "03-Queries"
    outputs_dir = cfg.vault_path / "05-Outputs"
    archive_dir = cfg.vault_path / "09-Archive-Queries"
    archive_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    had_failures = False

    # Build file-based query list
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
            # Fast path: direct LLM call (no Hermes subprocess overhead)
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
            # Full agent path: Hermes subprocess with tool access
            try:
                result = subprocess.run(
                    [cfg.agent_cmd, "chat", "-q", prompt, "-Q"],
                    cwd=str(cfg.vault_path), capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    answer = result.stdout.strip()
                    typer.echo(f"\n--- Answer ({qname}) ---\n")
                    typer.echo(answer)
                    # Write to 05-Outputs/
                    out_file = _collision_safe_path(outputs_dir / f"{qname}.md")
                    out_file.write_text(
                        f"# Query: {qname}\n\n"
                        f"**Question:**\n{qtext}\n\n"
                        f"**Answer:**\n{answer}\n",
                        encoding="utf-8",
                    )
                    typer.echo(f"\nWritten to: {out_file}")
                    # Archive original query
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


# ─── diagnostics / fixture / release ─────────────────────────────────────────

@app.command()
def doctor(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Run first-run and configuration diagnostics."""
    from pipeline.doctor import run_doctor

    cfg = _load_cfg(vault)
    report = run_doctor(cfg)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Doctor: {'ok' if report['ok'] else 'issues found'}")
        for check in report["checks"]:
            mark = "✓" if check["ok"] else "✗"
            typer.echo(f"  {mark} {check['name']}: {check['detail']}")
    raise typer.Exit(code=0 if report["ok"] else 1)


@app.command(name="config-doctor")
def config_doctor(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Alias for doctor focused on redacted configuration diagnostics."""
    doctor(vault=vault, json_output=json_output)


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


@app.command(name="review-status")
def review_status(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show pending review queue status without approving/rejecting anything."""
    from pipeline.review import show_pending

    cfg = _load_cfg(vault)
    pending = show_pending(cfg)
    by_type: dict[str, int] = {}
    for item in pending:
        by_type[item["file_type"]] = by_type.get(item["file_type"], 0) + 1
    report = {"pending": len(pending), "by_type": dict(sorted(by_type.items())), "items": pending}
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(f"Pending reviews: {len(pending)}")
        for file_type, count in report["by_type"].items():
            typer.echo(f"  {file_type}: {count}")


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


@app.command(name="release-check")
def release_check(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Check release metadata, docs, and changelog alignment."""
    from pipeline.release import check_release_hygiene

    report = check_release_hygiene(Path(__file__).parent.parent)
    if json_output:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(f"Release hygiene: {'ok' if report['ok'] else 'issues found'} ({report['version']})")
        for check in report["checks"]:
            mark = "✓" if check["ok"] else "✗"
            typer.echo(f"  {mark} {check['name']}: {check['detail']}")
    raise typer.Exit(code=0 if report["ok"] else 1)


# ─── setup-qmd ────────────────────────────────────────────────────────────────

@app.command(name="setup-qmd")
def setup_qmd_cmd(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Install and configure qmd for semantic concept search."""
    from pipeline.setup import setup_qmd

    vault_path = _resolve_vault(vault)
    setup_qmd(vault_path)


# ─── setup-hooks ──────────────────────────────────────────────────────────────

@app.command(name="setup-hooks")
def setup_hooks_cmd(
    vault: Path = typer.Argument(None, help="Vault path (default: ~/MyVault)"),
):
    """Install git hooks (pre-commit, commit-msg) in the vault repo."""
    from pipeline.setup import setup_git_hooks

    vault_path = _resolve_vault(vault)
    setup_git_hooks(vault_path)


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
