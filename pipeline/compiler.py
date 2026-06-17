"""Compile orchestrator — ties all pipeline stages together.

Central orchestration modelled on llm-wiki-compiler/src/compiler/index.ts
runCompilePipeline(). Handles the full pipeline: change detection, concept
extraction, page generation, wikilink resolution, index/MOC generation,
and state persistence.

OKF migration (Task 12): imports now prefer the OKF-native modules
(``pipeline.okf_models``, ``pipeline.okf_markdown``, ``pipeline.okf_resolver``,
``pipeline.okf_indexgen``) while retaining backward-compat fallbacks to the
legacy modules for symbols that have no OKF equivalent yet (``SchemaConfig``,
``PageKind``).

Usage:
    import asyncio
    from pipeline.compiler import compile
    result = asyncio.run(compile("/path/to/vault"))
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import Config, load_config
from pipeline.hasher import detect_changes
from pipeline.lock import acquire_lock, release_lock

# SchemaConfig / PageKind are schema-layer types that have not yet been
# migrated to okf_models — keep importing from the legacy models module.
from pipeline.models import SchemaConfig  # noqa: F401  (re-exported)

# ── OKF-preferred imports (with legacy fallback) ────────────────────────
# OKF models carry the same names for the core pipeline dataclasses.
from pipeline.okf_markdown import parse_frontmatter, safe_read_file, slugify
from pipeline.okf_models import (
    CompileResult,
    IngestedSource,
    SourceChange,
    SourceStatus,
    WikiState,
)
from pipeline.state import read_state, remove_source_state, write_state

# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────


async def compile(
    root_dir: str,
    options: dict | None = None,
) -> CompileResult:
    """Run the full compilation pipeline against a wiki vault.

    Flow:
      1. Load config from root_dir/.env
      2. Acquire PID lock
      3. Read persisted state
      4. Detect source file changes
      5. Bucket changes: to_compile, deleted, unchanged
      6. Early-exit if nothing to do
      7. Mark deleted sources as orphaned
      8. Find frozen slugs (shared concepts across deleted+live sources)
      9. Find indirectly-affected sources → add to to_compile
      10. Read source files + extract concepts via LLM
      11. Generate entry + concept pages (run_create)
      12. Resolve wikilinks
      13. Generate index
      14. Generate MOC
      15. Generate per-directory index.md files (OKF)
      16. Generate/update log.md (OKF)
      17. Persist state
      18. Release lock
      19. Return CompileResult

    Args:
        root_dir: Path to the Obsidian vault root.
        options: Optional overrides:
            - force: bool — recompile all sources regardless of changes.
            - files: list[str] — compile only these specific source files.
            - dry_run: bool — detect changes but skip actual compilation.
            - model: str — override the LLM model.
            - concurrency: int — override compile concurrency.

    Returns:
        CompileResult with counts and errors.
    """
    options = options or {}
    root = Path(root_dir).resolve()

    # ── Step 1: Load config ────────────────────────────────────────────────
    env_file = str(root / ".env") if (root / ".env").exists() else None
    overrides = {}
    if options.get("model"):
        overrides["OLLAMA_MODEL"] = options["model"]
    if options.get("concurrency"):
        overrides["COMPILE_CONCURRENCY"] = str(options["concurrency"])
    config = load_config(env_file=env_file, **overrides)

    print(f"[compiler] Vault: {config.vault}")
    print(f"[compiler] Model: {config.ollama_model}")

    # ── Step 2: Acquire lock ────────────────────────────────────────────────
    if not acquire_lock(config.lock_file):
        print("[compiler] ❌ Could not acquire lock — another compile running.")
        return CompileResult(errors=["lock: another compilation is running"])

    try:
        # ── Step 3: Read state ─────────────────────────────────────────────
        state = read_state(config.state_file)
        print(f"[compiler] State: {len(state.sources)} tracked sources")

        # ── Step 4: Detect changes ─────────────────────────────────────────
        force = options.get("force", False)
        specific_files = options.get("files")

        if force:
            changes = _force_all_changed(config.sources_dir, state)
            print("[compiler] 🔄 Force mode — all sources marked for recompile")
        elif specific_files:
            changes = _filter_changes(config.sources_dir, state, specific_files)
            print(
                f"[compiler] 🎯 Targeting {len(specific_files)} specific file(s)"
            )
        else:
            changes = detect_changes(config.sources_dir, state)

        # ── Step 5: Bucket changes ─────────────────────────────────────────
        to_compile: set[str] = set()
        deleted: set[str] = set()

        for ch in changes:
            if ch.status == SourceStatus.NEW or ch.status == SourceStatus.CHANGED:
                to_compile.add(ch.file)
            elif ch.status == SourceStatus.DELETED:
                deleted.add(ch.file)
            # UNCHANGED files are ignored

        new_count = sum(
            1 for ch in changes if ch.status == SourceStatus.NEW
        )
        changed_count = sum(
            1 for ch in changes if ch.status == SourceStatus.CHANGED
        )

        print(
            f"[compiler] Changes: {new_count} new, {changed_count} changed, "
            f"{len(deleted)} deleted"
        )

        # ── Step 6: Early-exit ──────────────────────────────────────────────
        if not to_compile and not deleted:
            print("[compiler] ✅ Nothing to compile — already up-to-date.")
            return CompileResult(skipped=len(changes))

        dry_run = options.get("dry_run", False)
        if dry_run:
            print("[compiler] 🔍 Dry run — skipping compilation.")
            return CompileResult(
                compiled=0,
                skipped=len(
                    [c for c in changes if c.status == SourceStatus.UNCHANGED]
                ),
                deleted=len(deleted),
            )

        # ── Step 7: Mark deleted sources as orphaned ────────────────────────
        log_entries = []  # collected for log.md generation (Step 16)
        for del_file in sorted(deleted):
            print(f"[compiler] 🗑 Orphaning concepts from deleted source: {del_file}")
            from pipeline.orphan import mark_orphaned

            mark_orphaned(str(config.wiki_dir), del_file, state)
            remove_source_state(state, del_file)
            log_entries.append(
                _make_log_entry("deleted", f"sources/{del_file}", del_file)
            )

        # ── Step 8: Find frozen slugs ───────────────────────────────────────
        from pipeline.deps import find_frozen_slugs

        frozen_slugs = find_frozen_slugs(state, changes)
        if frozen_slugs:
            print(
                f"[compiler] 🧊 {len(frozen_slugs)} frozen slug(s) "
                f"(shared with live sources)"
            )

        # ── Step 9: Find affected sources ───────────────────────────────────
        from pipeline.deps import find_affected_sources

        affected_files = find_affected_sources(state, changes)
        if affected_files:
            print(
                f"[compiler] 🔗 {len(affected_files)} indirectly affected source(s)"
            )
            for af in affected_files:
                if af not in to_compile:
                    to_compile.add(af)
                    print(f"[compiler]   + {af}")

        if not to_compile:
            print("[compiler] ✅ No sources to compile.")
            return CompileResult()

        print(f"[compiler] 📦 Compiling {len(to_compile)} source(s)")

        # ── Step 10+11: Read sources and generate pages ─────────────────────
        sources_dict: dict[str, IngestedSource] = {}
        for filename in sorted(to_compile):
            filepath = config.sources_dir / filename
            raw = safe_read_file(filepath)
            if not raw.strip():
                print(f"[compiler] ⚠ Skipping empty source: {filename}")
                continue

            meta, body = parse_frontmatter(raw)
            title = meta.get("title", filename.replace(".md", ""))
            sources_dict[filename] = IngestedSource(title=title, content=body)

        from pipeline.create.orchestrator import run_create

        print("[compiler] 🤖 Running LLM creation phase...")
        result = await run_create(config, sources_dict, state)

        # Collect log entries for compiled sources
        for filename in sorted(to_compile):
            log_entries.append(
                _make_log_entry("compiled", f"sources/{filename}", filename)
            )
        for concept in result.concepts:
            slug = slugify(concept.concept)
            log_entries.append(
                _make_log_entry(
                    "created" if getattr(concept, "is_new", False) else "updated",
                    f"concepts/{slug}",
                    concept.concept,
                )
            )

        # ── Step 12: Resolve wikilinks ──────────────────────────────────────
        # Collect all concept slugs from this run
        all_slugs: list[str] = []
        new_slugs: list[str] = []
        for concept in result.concepts:
            slug = slugify(concept.concept)
            all_slugs.append(slug)
            if concept.is_new:
                new_slugs.append(slug)

        if all_slugs:
            # Prefer OKF resolver; fall back to legacy if unavailable.
            try:
                from pipeline.okf_resolver import resolve_links
            except ImportError:  # pragma: no cover
                from pipeline.resolver import resolve_links

            print("[compiler] 🔗 Resolving wikilinks...")
            modified = resolve_links(str(config.wiki_dir), all_slugs, new_slugs)
            if modified:
                print(f"[compiler]   Updated {modified} page(s) with wikilinks")

        # ── Step 13: Generate index ─────────────────────────────────────────
        # OKF: generate per-directory index.md + bundle-root index.md.
        from pipeline.okf_indexgen import (
            generate_bundle_index,
        )
        from pipeline.okf_markdown import atomic_write

        print("[compiler] 📇 Generating OKF directory indexes...")
        _generate_all_directory_indexes(config, atomic_write)

        print("[compiler] 📇 Generating OKF bundle index...")
        bundle_idx = generate_bundle_index(config.bundle_dir, config.okf_version)
        bundle_index_path = config.bundle_dir / "index.md"
        atomic_write(bundle_index_path, bundle_idx)
        print(f"[compiler]   → {bundle_index_path}")

        # ── Step 14: Generate MOC ───────────────────────────────────────────
        # The OKF indexgen module does not produce a separate MOC.md —
        # MoC pages are generated as concept files during the create phase
        # (see orchestrator).  We keep the legacy MOC generation as a
        # backward-compat courtesy so existing consumers that read
        # wiki/MOC.md still work.
        try:
            from pipeline.indexgen import generate_moc

            print("[compiler] 🗺 Generating MOC (legacy)...")
            moc_path = generate_moc(config.wiki_dir, config.concepts_dir)
            print(f"[compiler]   → {moc_path}")
        except Exception:
            print("[compiler] ⚠ MOC generation skipped (legacy indexgen unavailable)")

        # ── Step 15: Persist state ──────────────────────────────────────────
        write_state(config.state_file, state)
        print(f"[compiler] 💾 State persisted ({len(state.sources)} sources)")

        # ── Step 16: Generate/update log.md ─────────────────────────────────
        from pipeline.okf_indexgen import generate_log

        print("[compiler] 📝 Generating log.md...")
        log_md = generate_log(log_entries)
        log_path = config.bundle_dir / "log.md"
        atomic_write(log_path, log_md)
        print(f"[compiler]   → {log_path}")

        # ── Result summary ──────────────────────────────────────────────────
        print(
            f"[compiler] ✅ Done! "
            f"{result.compiled} compiled, "
            f"{len(result.concepts)} concepts, "
            f"{len(deleted)} deleted, "
            f"{len(result.errors)} errors"
        )
        result.deleted = len(deleted)
        return result

    finally:
        # ── Step 18: Release lock ───────────────────────────────────────────
        release_lock(config.lock_file)


# ──────────────────────────────────────────────────────────────────────────────
# OKF index/log helpers
# ──────────────────────────────────────────────────────────────────────────────


def _generate_all_directory_indexes(config: Config, atomic_write) -> None:
    """Generate ``index.md`` for each OKF bundle sub-directory.

    Covers: sources/, entries/, concepts/, mocs/, references/.
    Directories that do not exist are silently skipped.
    """
    from pipeline.okf_indexgen import generate_directory_index

    dir_map = {
        "sources": config.sources_dir,
        "entries": config.entries_dir,
        "concepts": config.concepts_dir,
        "mocs": config.mocs_dir,
        "references": config.references_dir,
    }

    for dir_name, dir_path in dir_map.items():
        if not dir_path.is_dir():
            continue
        md_files = sorted(dir_path.glob("*.md"))
        # Skip index.md / log.md themselves
        md_files = [
            f for f in md_files if f.name not in {"index.md", "log.md"}
        ]
        if not md_files:
            continue
        content = generate_directory_index(dir_path, md_files)
        index_path = dir_path / "index.md"
        atomic_write(index_path, content)


def _make_log_entry(action: str, concept_id: str, description: str):
    """Build a :class:`LogEntry` for the compilation change log."""
    from pipeline.okf_models import LogEntry

    now = datetime.now(UTC)
    return LogEntry(
        date=now.strftime("%Y-%m-%d"),
        action=action,
        concept_id=concept_id,
        description=description,
        timestamp=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _force_all_changed(
    sources_dir: Path,
    state: WikiState,
) -> list[SourceChange]:
    """Mark all source files as CHANGED for a full recompilation."""
    changes: list[SourceChange] = []
    current_files: set[str] = set()

    if sources_dir.exists():
        for f in sources_dir.iterdir():
            if f.suffix == ".md" and f.is_file():
                current_files.add(f.name)
                changes.append(SourceChange(file=f.name, status=SourceStatus.CHANGED))

    for filename in state.sources:
        if filename not in current_files:
            changes.append(SourceChange(file=filename, status=SourceStatus.DELETED))

    return changes


def _filter_changes(
    sources_dir: Path,
    state: WikiState,
    target_files: list[str],
) -> list[SourceChange]:
    """Detect changes only for a specific set of files."""
    from pipeline.hasher import hash_file

    changes: list[SourceChange] = []
    target_set = set(target_files)

    for filename in target_set:
        filepath = sources_dir / filename
        if filepath.exists():
            # File exists — classify as new or changed
            file_hash = hash_file(filepath)
            prev = state.sources.get(filename)
            if prev is None:
                status = SourceStatus.NEW
            elif prev.hash != file_hash:
                status = SourceStatus.CHANGED
            else:
                status = SourceStatus.UNCHANGED
            changes.append(SourceChange(file=filename, status=status))
        else:
            # File doesn't exist on disk but is in target
            changes.append(SourceChange(file=filename, status=SourceStatus.DELETED))

    # Also detect deletions for files in state but not on disk
    if sources_dir.exists():
        current_files = {
            f.name
            for f in sources_dir.iterdir()
            if f.suffix == ".md" and f.is_file()
        }
        for filename in state.sources:
            if filename not in current_files and filename not in target_set:
                changes.append(
                    SourceChange(file=filename, status=SourceStatus.DELETED)
                )

    return changes


# ──────────────────────────────────────────────────────────────────────────────
# Seed page generation
# ──────────────────────────────────────────────────────────────────────────────


async def _generate_seed_pages(
    root_dir: str,
    schema: SchemaConfig,
    config: Config | None = None,
) -> list[str]:
    """Generate schema-declared seed pages (overview, comparison, etc.).

    Seed pages are non-concept wiki pages declared in schema.toml/json.
    They provide high-level structure like topic overviews and cross-concept
    comparisons.

    Args:
        root_dir: Wiki vault root directory.
        schema: Resolved SchemaConfig with seed_pages populated.
        config: Optional pipeline config (loaded if not provided).

    Returns:
        List of file paths written (empty if no seed pages configured).
    """
    if not schema.seed_pages:
        return []

    if config is None:
        config = load_config(
            env_file=str(Path(root_dir) / ".env")
            if (Path(root_dir) / ".env").exists()
            else None
        )

    written: list[str] = []

    for seed in schema.seed_pages:
        slug = slugify(seed.title)
        kind_dir = _kind_to_dir(seed.kind, config)
        kind_dir.mkdir(parents=True, exist_ok=True)

        page_path = kind_dir / f"{slug}.md"

        # Build frontmatter + body — prefer OKF markdown helpers.
        from pipeline.okf_markdown import atomic_write, build_frontmatter

        meta: dict = {
            "title": seed.title,
            "kind": seed.kind.value,
            "slug": slug,
        }
        if seed.summary:
            meta["summary"] = seed.summary
        if seed.related_slugs:
            meta["related"] = seed.related_slugs

        fm = build_frontmatter(meta)

        body_lines: list[str] = []
        body_lines.append(f"# {seed.title}")
        body_lines.append("")

        if seed.summary:
            body_lines.append(seed.summary)
            body_lines.append("")

        if seed.related_slugs:
            body_lines.append("## Related Concepts")
            body_lines.append("")
            for rel_slug in seed.related_slugs:
                body_lines.append(f"- [[{rel_slug}]]")
            body_lines.append("")

        body_lines.append(
            "*This is a seed page generated from the wiki schema. "
            "Edit freely — it will not be overwritten unless deleted.*"
        )

        atomic_write(page_path, fm + "\n" + "\n".join(body_lines) + "\n")
        written.append(str(page_path))
        print(f"[compiler] 🌱 Seed page: {page_path}")

    return written


def _kind_to_dir(kind, config: Config) -> Path:
    """Map a PageKind to the appropriate wiki subdirectory."""
    # PageKind is still a legacy-model enum; import from legacy models.
    from pipeline.models import PageKind

    if kind == PageKind.CONCEPT:
        return config.concepts_dir
    elif kind == PageKind.ENTITY:
        return config.wiki_dir / "entities"
    elif kind == PageKind.OVERVIEW:
        return config.wiki_dir / "overviews"
    elif kind == PageKind.COMPARISON:
        return config.wiki_dir / "comparisons"
    else:
        return config.wiki_dir


# ──────────────────────────────────────────────────────────────────────────────
# Synchronous convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────


def compile_sync(
    root_dir: str,
    options: dict | None = None,
) -> CompileResult:
    """Synchronous wrapper around async compile() — usable from CLI entry points.

    Args:
        root_dir: Wiki vault root directory.
        options: Same as compile().

    Returns:
        CompileResult.
    """
    return asyncio.run(compile(root_dir, options))
