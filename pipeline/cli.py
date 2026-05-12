"""Typer CLI for the llmwiki knowledge compiler pipeline.

Sources in, interlinked wiki out. All commands are real implementations
connected to the pipeline modules.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

app = typer.Typer(
    name="llmwiki",
    help="Knowledge compiler CLI. Sources in, interlinked wiki out.",
    no_args_is_help=True,
)


# ── Shared helpers ────────────────────────────────────────────────────────


def _resolve_vault(vault: str) -> tuple[Path, Config]:  # noqa: F821
    """Resolve vault path and load config. Returns (vault_path, config)."""
    from pipeline.config import load_config

    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file)
    # Override vault_path if env didn't have one or user specified a different one
    if config.vault_path and Path(config.vault_path).expanduser().resolve() != vault_path or not config.vault_path:
        import os
        os.environ["VAULT_PATH"] = str(vault_path)
        config = load_config(env_file=env_file, VAULT_PATH=str(vault_path))
    return vault_path, config


def _print_result_summary(result: CompileResult) -> None:  # noqa: F821
    """Pretty-print a CompileResult."""
    typer.echo(
        f"\n✅ Compilation complete: "
        f"{result.compiled} compiled, "
        f"{len(result.concepts)} concepts, "
        f"{result.deleted} deleted"
    )
    if result.skipped:
        typer.echo(f"   Skipped: {result.skipped} (unchanged)")
    if result.errors:
        typer.echo(f"   Errors:  {len(result.errors)}")
        for err in result.errors[:10]:
            typer.echo(f"     - {err}")
        if len(result.errors) > 10:
            typer.echo(f"     ... and {len(result.errors) - 10} more")
    if result.candidates:
        typer.echo(f"   Candidates (pending review): {len(result.candidates)}")


# ── Commands ──────────────────────────────────────────────────────────────


@app.command()
def ingest(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    urls: list[str] | None = typer.Option(
        None, "--url", "-u", help="URLs to ingest (can be repeated)"
    ),
    parallel: int = typer.Option(
        3, "--parallel", "-p", help="Concurrent LLM calls during create phase"
    ),
    review: bool = typer.Option(
        False, "--review", help="Stage generated pages as candidates for manual review"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview extraction without writing any files"
    ),
    skip_compile: bool = typer.Option(
        False,
        "--skip-compile",
        help="Only extract sources; skip concept extraction and page generation",
    ),
):
    """Ingest URLs and generate wiki pages from them.

    Extracts full content from URLs, writes source files, then optionally
    runs the LLM compilation pipeline to generate entries, concepts, and MoCs.

    Clippings in 02-Clippings/ that pass the quality gate are included
    automatically (no Stage 1 extraction needed for them).

    Examples:
        llmwiki ingest ~/MyVault --url https://example.com/article
        llmwiki ingest ~/MyVault -u URL1 -u URL2 --parallel 5 --review
        llmwiki ingest ~/MyVault -u URL1 --dry-run
    """
    from pipeline.clippings import collect_clippings
    from pipeline.config import load_config
    from pipeline.extract import run_extraction

    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file)
    # Ensure vault_path is set
    import os
    if not config.vault_path or Path(config.vault_path).expanduser().resolve() != vault_path:
        os.environ["VAULT_PATH"] = str(vault_path)
        config = load_config(env_file=env_file, VAULT_PATH=str(vault_path))

    if parallel:
        os.environ["COMPILE_CONCURRENCY"] = str(parallel)
        config = load_config(env_file=env_file, VAULT_PATH=str(vault_path), COMPILE_CONCURRENCY=str(parallel))

    typer.echo(f"📂 Vault: {config.vault}")
    typer.echo(f"🤖 Model: {config.ollama_model}")

    # ── Collect clippings that pass the quality gate ──────────────────
    clipping_sources: dict[str, IngestedSource] = {}  # noqa: F821
    passed_clippings = collect_clippings(config)
    if passed_clippings:
        typer.echo(f"\n📋 Clippings passing quality gate: {len(passed_clippings)}")
        for clip_path, source in passed_clippings:
            key = clip_path.name
            clipping_sources[key] = source
            typer.echo(f"   ✅ {source.title[:60]} ({len(source.content)} chars)")

    # ── Extract URLs ─────────────────────────────────────────────────
    extracted_sources: dict[str, IngestedSource] = {}  # noqa: F821
    if urls:
        typer.echo(f"\n🌐 Extracting {len(urls)} URL(s)...")
        if dry_run:
            typer.echo("   🔍 Dry run — would extract:")
            for url in urls:
                typer.echo(f"      {url}")
        else:
            extracted_sources = run_extraction(list(urls), config)
            if not extracted_sources:
                typer.echo("   ⚠ No URLs were extracted (all skipped or failed)")

    # ── Combine sources ──────────────────────────────────────────────
    all_sources = {**clipping_sources, **extracted_sources}
    if not all_sources:
        typer.echo("\n⚠ No sources to process (no URLs extracted, no clippings passed).")
        if not urls and not passed_clippings:
            typer.echo("   Tip: Use --url to add URLs or add .md files to 02-Clippings/")
        return

    typer.echo(f"\n📦 Total sources to compile: {len(all_sources)}")

    if skip_compile:
        typer.echo("   ⏭ Skipping compilation (--skip-compile)")
        return

    if dry_run:
        typer.echo("   🔍 Dry run — would compile these sources:")
        for key, source in sorted(all_sources.items()):
            typer.echo(f"      {source.title[:60]} ← {key}")
        return

    # ── Run create pipeline ──────────────────────────────────────────
    typer.echo("\n🤖 Running LLM creation phase...")
    from pipeline.create.orchestrator import run_create
    from pipeline.state import read_state

    state = read_state(config.state_file)

    if review:
        # In review mode, render pages but stage as candidates
        result = asyncio.run(
            _run_create_with_review(config, all_sources, state)
        )
    else:
        result = asyncio.run(run_create(config, all_sources, state))

    # ── Post-compile: resolve links, indexes ─────────────────────────
    from pipeline.hasher import slugify as _slugify
    from pipeline.indexgen import generate_index, generate_moc
    from pipeline.resolver import resolve_links
    from pipeline.state import write_state

    # Collect concept slugs
    all_slugs = [_slugify(c.concept) for c in result.concepts]
    new_slugs = [_slugify(c.concept) for c in result.concepts if c.is_new]

    if all_slugs:
        typer.echo("🔗 Resolving wikilinks...")
        modified = resolve_links(str(config.wiki_dir), all_slugs, new_slugs)
        if modified:
            typer.echo(f"   Updated {modified} page(s)")

    typer.echo("📇 Generating index...")
    idx_path = generate_index(config.wiki_dir, config.concepts_dir)
    typer.echo(f"   → {idx_path}")

    typer.echo("🗺 Generating MOC...")
    moc_path = generate_moc(config.wiki_dir, config.concepts_dir)
    typer.echo(f"   → {moc_path}")

    write_state(config.state_file, state)
    typer.echo(f"💾 State persisted ({len(state.sources)} sources)")

    _print_result_summary(result)


async def _run_create_with_review(
    config: Config, sources: dict, state: WikiState  # noqa: F821
) -> CompileResult:  # noqa: F821
    """Run create and stage outputs as review candidates instead of writing."""
    from pipeline.candidates import write_candidate
    from pipeline.create.orchestrator import run_create

    # First run normally to get the result
    result = await run_create(config, sources, state)

    # For each concept, stage as a candidate
    for concept in result.concepts:
        from pipeline.hasher import slugify as _slugify
        slug = _slugify(concept.concept)
        candidate_data = {
            "title": concept.concept,
            "slug": slug,
            "summary": concept.summary,
            "sources": list(sources.keys()),
            "body": f"# {concept.concept}\n\n{concept.summary}\n\n"
                    f"*Tags: {', '.join(concept.tags)}*",
        }
        try:
            candidate = write_candidate(str(config.vault), candidate_data)
            result.candidates.append(candidate.id)
        except Exception as exc:
            result.errors.append(f"candidate:{concept.concept}:{exc}")

    return result


@app.command()
def compile_cmd(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    review: bool = typer.Option(
        False, "--review", help="Write generated pages to candidates/ instead of wiki/"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force recompilation of all sources"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Detect changes but skip compilation"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the LLM model"
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", "-c", help="Override compile concurrency"
    ),
    files: list[str] | None = typer.Option(
        None, "--file", help="Compile only these specific source files"
    ),
):
    """Run the full compilation pipeline.

    Detects changes in sources/, generates entry and concept pages via LLM,
    resolves wikilinks, and rebuilds the index and MOC.

    Examples:
        llmwiki compile ~/MyVault
        llmwiki compile ~/MyVault --force
        llmwiki compile ~/MyVault --dry-run
        llmwiki compile ~/MyVault --file article1.md --file article2.md
    """
    from pipeline.compiler import compile as compile_pipeline

    options: dict = {
        "force": force,
        "dry_run": dry_run,
    }
    if model:
        options["model"] = model
    if concurrency:
        options["concurrency"] = concurrency
    if files:
        options["files"] = list(files)

    if review:
        # In review mode, compile first, then stage outputs as candidates
        result = asyncio.run(
            _compile_with_review(vault, options)
        )
    else:
        result = asyncio.run(compile_pipeline(vault, options))

    if not dry_run and not force and result.compiled == 0 and not result.errors:
        typer.echo("✅ Already up-to-date.")
        return

    _print_result_summary(result)


async def _compile_with_review(vault: str, options: dict) -> CompileResult:  # noqa: F821
    """Compile and stage all generated pages as review candidates."""
    from pipeline.candidates import write_candidate
    from pipeline.compiler import compile as compile_pipeline
    from pipeline.config import load_config

    # Run the normal pipeline
    result = await compile_pipeline(vault, options)

    # Stage each generated concept as a review candidate
    if result.concepts:
        vault_path = Path(vault).expanduser().resolve()
        env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
        config = load_config(env_file=env_file)

        for concept in result.concepts:
            from pipeline.hasher import slugify as _slugify
            slug = _slugify(concept.concept)
            candidate_data = {
                "title": concept.concept,
                "slug": slug,
                "summary": concept.summary,
                "sources": [],  # Source names are captured during compilation
                "body": f"# {concept.concept}\n\n{concept.summary}\n\n"
                        f"*Tags: {', '.join(concept.tags)}*",
            }
            try:
                candidate = write_candidate(str(config.vault), candidate_data)
                result.candidates.append(candidate.id)
            except Exception as exc:
                result.errors.append(f"candidate:{concept.concept}:{exc}")

    return result


@app.command()
def lint(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    strict: bool = typer.Option(
        False, "--strict", "-s", help="Treat warnings as errors"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON"
    ),
):
    """Run lint checks on the vault.

    Checks for:
      - Malformed frontmatter
      - Broken wikilinks (targets that don't exist)
      - Orphaned pages with no incoming links
      - Missing or malformed citations (^[...])
      - Pages below minimum content thresholds
      - Empty source files

    Examples:
        llmwiki lint ~/MyVault
        llmwiki lint ~/MyVault --strict
        llmwiki lint ~/MyVault --json
    """
    import json

    from pipeline.config import load_config
    from pipeline.markdown import (
        is_malformed_citation_entry,
        parse_frontmatter,
        safe_read_file,
        slugify,
    )

    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file)

    issues: list[dict] = []
    warnings: list[dict] = []
    errors: list[dict] = []

    sources_dir = config.sources_dir
    concepts_dir = config.concepts_dir
    entries_dir = config.entries_dir

    # ── Check sources ─────────────────────────────────────────────────
    if sources_dir.exists():
        for f in sources_dir.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "empty-source",
                    "message": f"Source file is empty: {f.name}",
                })
                continue
            meta, body = parse_frontmatter(raw)
            if not meta.get("title"):
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "missing-title",
                    "message": f"Source file missing title in frontmatter: {f.name}",
                })
            if len(body.strip()) < config.min_source_chars:
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "short-source",
                    "message": (
                        f"Source body too short: {len(body.strip())} chars "
                        f"(min: {config.min_source_chars})"
                    ),
                })

    # ── Check concepts ───────────────────────────────────────────────
    all_concept_slugs: set[str] = set()
    if concepts_dir.exists():
        for f in concepts_dir.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "empty-concept",
                    "message": f"Concept page is empty: {f.name}",
                })
                continue
            meta, body = parse_frontmatter(raw)
            slug = meta.get("slug", f.stem)
            all_concept_slugs.add(slug)

            if not meta.get("title"):
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "error",
                    "rule": "missing-title",
                    "message": f"Concept page missing title: {f.name}",
                })

            if len(body.strip()) < config.concept_min_body_chars:
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "short-concept",
                    "message": (
                        f"Concept body too short: {len(body.strip())} chars "
                        f"(min: {config.concept_min_body_chars})"
                    ),
                })

            # Check for malformed citations
            import re
            for match in re.finditer(r"\^\[([^\]]+)\]", body):
                raw_cite = match.group(1)
                for part in raw_cite.split(","):
                    part = part.strip()
                    if part and is_malformed_citation_entry(part):
                        issues.append({
                            "file": str(f.relative_to(config.vault)),
                            "severity": "warning",
                            "rule": "malformed-citation",
                            "message": f"Malformed citation entry: {part}",
                        })

    # ── Check entries ────────────────────────────────────────────────
    if entries_dir.exists():
        for f in entries_dir.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "empty-entry",
                    "message": f"Entry page is empty: {f.name}",
                })
                continue
            meta, body = parse_frontmatter(raw)
            if len(body.strip()) < config.entry_min_body_chars:
                issues.append({
                    "file": str(f.relative_to(config.vault)),
                    "severity": "warning",
                    "rule": "short-entry",
                    "message": (
                        f"Entry body too short: {len(body.strip())} chars "
                        f"(min: {config.entry_min_body_chars})"
                    ),
                })

    # ── Check for broken wikilinks ───────────────────────────────────
    import re as _re
    _WIKILINK_RE = _re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")
    if concepts_dir.exists():
        for f in concepts_dir.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                continue
            for match in _WIKILINK_RE.finditer(raw):
                target = match.group(1).strip()
                target_slug = slugify(target)
                if target_slug not in all_concept_slugs:
                    # Check if it might exist as an entry or source
                    entry_exists = (entries_dir / f"{target_slug}.md").exists() if entries_dir.exists() else False
                    source_exists = (sources_dir / f"{target_slug}.md").exists() if sources_dir.exists() else False
                    if not entry_exists and not source_exists:
                        issues.append({
                            "file": str(f.relative_to(config.vault)),
                            "severity": "warning",
                            "rule": "broken-wikilink",
                            "message": f"Broken wikilink: [[{target}]] — target not found",
                        })

    # ── Separate errors and warnings ─────────────────────────────────
    for issue in issues:
        if issue["severity"] == "error":
            errors.append(issue)
        else:
            warnings.append(issue)

    # ── Output ───────────────────────────────────────────────────────
    if json_output:
        result = {
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": issues,
        }
        typer.echo(json.dumps(result, indent=2))
    else:
        if errors:
            typer.echo(f"\n❌ {len(errors)} error(s):")
            for e in errors:
                typer.echo(f"  {e['file']}: {e['message']}")

        if warnings:
            typer.echo(f"\n⚠ {len(warnings)} warning(s):")
            for w in warnings:
                typer.echo(f"  {w['file']}: {w['message']}")

        if not errors and not warnings:
            typer.echo("✅ No issues found.")

        if strict and warnings:
            typer.echo("\n❌ Strict mode: warnings treated as errors. Exiting with failure.")
            raise typer.Exit(code=1)

    if errors:
        raise typer.Exit(code=1)


@app.command()
def query(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    ask: str = typer.Option(..., "--ask", "-a", help="Question to ask the knowledge wiki"),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override the LLM model"
    ),
    max_results: int = typer.Option(
        10, "--max-results", "-n", help="Maximum number of relevant pages to retrieve"
    ),
):
    """Query the knowledge wiki using LLM with retrieval-augmented generation.

    Searches concept pages, entries, and sources for relevant content, then
    uses the LLM to answer your question grounded in the retrieved context.

    Examples:
        llmwiki query ~/MyVault --ask "What is a transformer model?"
        llmwiki query ~/MyVault -a "Compare RAG vs fine-tuning" -n 5
    """
    from pipeline.config import load_config
    from pipeline.llm_client import call_llm
    from pipeline.markdown import parse_frontmatter, safe_read_file

    vault_path = Path(vault).expanduser().resolve()
    env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
    config = load_config(env_file=env_file)

    if model:
        import os
        os.environ["OLLAMA_MODEL"] = model
        config = load_config(env_file=env_file, OLLAMA_MODEL=model)

    typer.echo(f"🔍 Searching for: \"{ask}\"")
    typer.echo(f"🤖 Model: {config.ollama_model}")

    # ── Gather context from wiki pages ───────────────────────────────
    context_parts: list[str] = []
    pages_scanned = 0

    # Scan concepts for keyword matches
    query_lower = ask.lower()
    query_words = set(query_lower.split())

    scored_pages: list[tuple[int, str, str]] = []  # (score, filepath, content_snippet)

    for dir_path, _label in [
        (config.concepts_dir, "concept"),
        (config.entries_dir, "entry"),
        (config.sources_dir, "source"),
    ]:
        if not dir_path.exists():
            continue
        for f in dir_path.glob("*.md"):
            raw = safe_read_file(f)
            if not raw.strip():
                continue
            meta, body = parse_frontmatter(raw)
            title = meta.get("title", f.stem)

            # Simple TF scoring: count query word occurrences
            body_lower = body.lower()
            score = sum(body_lower.count(w) for w in query_words if len(w) > 2)
            # Title matches get a bonus
            title_lower = title.lower()
            score += sum(10 for w in query_words if len(w) > 2 and w in title_lower)

            if score > 0:
                snippet = body[:800] if len(body) > 800 else body
                scored_pages.append((score, str(f.relative_to(config.vault)), snippet))

    # Sort by score descending, take top N
    scored_pages.sort(key=lambda x: x[0], reverse=True)
    top_pages = scored_pages[:max_results]

    if not top_pages:
        typer.echo("⚠ No relevant pages found in the wiki.")
        typer.echo("   Try ingesting some sources first with: llmwiki ingest")
        return

    # Build context
    for score, relpath, snippet in top_pages:
        context_parts.append(f"--- PAGE: {relpath} (relevance: {score}) ---\n{snippet}")

    context = "\n\n".join(context_parts)
    pages_scanned = len(top_pages)

    typer.echo(f"📄 Retrieved {pages_scanned} relevant page(s)")

    # ── Build prompt ─────────────────────────────────────────────────
    system = (
        "You are a knowledgeable assistant answering questions based on a "
        "personal knowledge wiki. Ground your answer in the provided context. "
        "If the context doesn't contain enough information, say so honestly. "
        "Cite the specific pages you used.\n\n"
        f"--- KNOWLEDGE WIKI CONTEXT ---\n{context}\n--- END CONTEXT ---"
    )

    messages: list[dict] = [
        {
            "role": "user",
            "content": ask,
        }
    ]

    typer.echo("\n💭 Thinking...\n")

    try:
        answer = asyncio.run(call_llm(system, messages, config, max_tokens=2048))
        typer.echo(answer)
        typer.echo(
            f"\n---\n*Answer based on {pages_scanned} wiki page(s). "
            f"Model: {config.ollama_model}*"
        )
    except Exception as exc:
        typer.echo(f"❌ Query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def setup():
    """Interactive setup wizard — configure LLM provider, vault path, API keys.

    Runs through a series of prompts to configure your llmwiki environment.
    Validates connectivity and writes a .env file to your vault.

    Example:
        llmwiki setup
    """
    from pipeline.setup import run_setup
    run_setup()


@app.command()
def candidates(
    vault: str = typer.Argument(..., help="Path to Obsidian vault"),
    action: str = typer.Argument(
        "list", help="Action: list, approve, reject, show"
    ),
    candidate_id: str | None = typer.Argument(
        None, help="Candidate ID (required for approve/reject/show)"
    ),
):
    """Manage review candidates — draft pages awaiting human approval.

    Actions:
      list    — Show all pending candidates
      show    — Display a candidate's full content
      approve — Approve and publish a candidate to the wiki
      reject  — Reject and archive a candidate

    Examples:
        llmwiki candidates ~/MyVault list
        llmwiki candidates ~/MyVault show my-concept-a1b2c3d4
        llmwiki candidates ~/MyVault approve my-concept-a1b2c3d4
        llmwiki candidates ~/MyVault reject my-concept-a1b2c3d4
    """
    from pipeline.candidates import (
        approve_candidate,
        list_candidates,
        read_candidate,
        reject_candidate,
    )

    vault_path = Path(vault).expanduser().resolve()

    if action == "list":
        cands = list_candidates(str(vault_path))
        if not cands:
            typer.echo("No pending candidates.")
            return
        typer.echo(f"\n📋 {len(cands)} pending candidate(s):\n")
        for c in cands:
            typer.echo(f"  [{c.id}] {c.title}")
            typer.echo(f"       Summary: {c.summary[:100]}{'...' if len(c.summary) > 100 else ''}")
            typer.echo(f"       Sources: {', '.join(c.sources) if c.sources else '(none)'}")
            typer.echo(f"       Generated: {c.generated_at}")
            if c.schema_violations:
                typer.echo(f"       ⚠ Schema violations: {len(c.schema_violations)}")
            typer.echo()

    elif action == "show":
        if not candidate_id:
            typer.echo("❌ Candidate ID required for 'show'.", err=True)
            raise typer.Exit(code=1)
        cand = read_candidate(str(vault_path), candidate_id)
        if cand is None:
            typer.echo(f"❌ Candidate not found: {candidate_id}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"\n# {cand.title}")
        typer.echo(f"ID: {cand.id}")
        typer.echo(f"Slug: {cand.slug}")
        typer.echo(f"Generated: {cand.generated_at}")
        if cand.sources:
            typer.echo(f"Sources: {', '.join(cand.sources)}")
        typer.echo(f"\n{cand.body}")
        if cand.schema_violations:
            typer.echo("\n--- Schema Violations ---")
            for v in cand.schema_violations:
                typer.echo(f"  - {v}")
        if cand.provenance_violations:
            typer.echo("\n--- Provenance Violations ---")
            for v in cand.provenance_violations:
                typer.echo(f"  - {v}")

    elif action == "approve":
        if not candidate_id:
            typer.echo("❌ Candidate ID required for 'approve'.", err=True)
            raise typer.Exit(code=1)
        from pipeline.config import load_config
        env_file = str(vault_path / ".env") if (vault_path / ".env").exists() else None
        config = load_config(env_file=env_file)
        ok = approve_candidate(str(vault_path), candidate_id, str(config.wiki_dir))
        if ok:
            typer.echo(f"✅ Approved: {candidate_id}")
        else:
            typer.echo(f"❌ Candidate not found: {candidate_id}", err=True)
            raise typer.Exit(code=1)

    elif action == "reject":
        if not candidate_id:
            typer.echo("❌ Candidate ID required for 'reject'.", err=True)
            raise typer.Exit(code=1)
        ok = reject_candidate(str(vault_path), candidate_id)
        if ok:
            typer.echo(f"🗑 Rejected and archived: {candidate_id}")
        else:
            typer.echo(f"❌ Candidate not found: {candidate_id}", err=True)
            raise typer.Exit(code=1)

    else:
        typer.echo(f"❌ Unknown action: {action}. Valid: list, show, approve, reject", err=True)
        raise typer.Exit(code=1)


# ── Entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    app()
