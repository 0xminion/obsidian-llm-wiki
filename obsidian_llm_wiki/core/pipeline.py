"""Single pipeline orchestrator — the heart of obsidian-llm-wiki.

Replaces the legacy dual-orchestrator design (compiler.py + create/orchestrator.py)
with a single coherent flow:

  1. Load config + acquire lock
  2. Read state + detect changes
  3. Detect deleted sources → orphan exclusively-owned concepts
  4. Load cached syntheses for unchanged sources
  5. Synthesise changed/new sources (one LLM call each) → cache results
  6. Merge ALL syntheses (cached + fresh) → SynthesisBundle
  7. Render: deterministic markdown from full SynthesisBundle → Obsidian vault
  8. Persist state + release lock

The synthesis cache (``.llmwiki/cache/<filename>.json``) is what makes
incremental builds correct: unchanged sources reuse their cached
synthesis, so the rendered corpus is always complete — not just the
subset that changed in this run.

The LLM only produces the synthesis intermediate.  All markdown generation
is pure functions in ``render.obsidian``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.cache import (
    delete_cached_synthesis,
    load_all_cached_syntheses,
    save_synthesis,
)
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.metrics import MetricsCollector
from obsidian_llm_wiki.core.models import (
    CompileResult,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
    WikiState,
)
from obsidian_llm_wiki.core.orphan import mark_orphaned_concepts
from obsidian_llm_wiki.core.state import (
    hash_content,
    read_state,
    remove_source_state,
    update_source_state,
    write_state,
)
from obsidian_llm_wiki.render.obsidian import render_vault
from obsidian_llm_wiki.synth.dedupe import merge_bundle
from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt

logger = logging.getLogger("obswiki.core.pipeline")

# ── Retry truncation levels ────────────────────────────────────────────────
# When a source fails synthesis, progressively truncate content and retry.
_TRUNCATION_LEVELS = [None, 50_000, 20_000]  # full → 50K → 20K


async def run_pipeline(
    vault_path: str | Path,
    sources: dict[str, SourceDoc],
    config: Config | None = None,
    *,
    force: bool = False,
) -> CompileResult:
    """Run the full synthesis + render pipeline.

    Args:
        vault_path: Path to the Obsidian vault root.
        sources: Dict mapping source filename → SourceDoc (the FULL corpus).
        config: Pipeline config (loaded from vault_path/.env if None).
        force: Force re-synthesis of all sources.

    Returns:
        CompileResult with counts and any errors.
    """
    vault = Path(vault_path).resolve()
    result = CompileResult()

    # ── Load config ────────────────────────────────────────────────────
    if config is None:
        env_file = str(vault / ".env") if (vault / ".env").exists() else None
        config = load_config(env_file=env_file, VAULT_PATH=str(vault))

    bundle_dir = config.wiki_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # ── Acquire lock ───────────────────────────────────────────────────
    if not acquire_lock(config.lock_file):
        result.errors.append("lock: another compilation is running")
        return result

    # ── Initialise metrics collector ──────────────────────────────────
    metrics = MetricsCollector(vault)
    metrics.start_run()

    try:
        # ── Read state ─────────────────────────────────────────────────
        state = read_state(config.state_file)

        # ── Detect deleted sources and orphan their concepts ───────────
        live_filenames = set(sources.keys())
        state_filenames = set(state.sources.keys())
        deleted_filenames = state_filenames - live_filenames

        for filename in deleted_filenames:
            orphaned = mark_orphaned_concepts(
                config.concepts_dir, filename, state
            )
            delete_cached_synthesis(config.llmwiki_dir, filename)
            remove_source_state(state, filename)
            result.deleted += 1
            if orphaned:
                logger.info(
                    "Deleted source '%s' orphaned %d concept(s): %s",
                    filename, len(orphaned), ", ".join(orphaned),
                )

        # ── Load cached syntheses for unchanged sources ────────────────
        cached_syntheses = load_all_cached_syntheses(config.llmwiki_dir)

        # ── Classify sources: compile vs reuse cache ───────────────────
        to_compile: dict[str, SourceDoc] = {}
        all_syntheses_by_file: dict[str, SourceSynthesis] = {}

        for filename, source in sources.items():
            # Source length gate (B3 fix).
            content_len = len(source.content)
            if content_len < config.min_source_chars:
                result.errors.append(
                    f"source:{filename}: too short ({content_len} < "
                    f"{config.min_source_chars} chars)"
                )
                continue
            if content_len > config.max_source_chars:
                logger.warning(
                    "Source '%s' is %d chars — exceeds max_source_chars (%d), "
                    "truncating to safety cap. Sources above chunk_size (%d) "
                    "will be chunked during two-pass synthesis.",
                    filename, content_len, config.max_source_chars,
                    config.chunk_size,
                )
                source = SourceDoc(
                    title=source.title,
                    content=source.content[:config.max_source_chars],
                    url=source.url,
                    source_file=filename,
                )

            current_hash = hash_content(source.content)
            prev = state.sources.get(filename)
            cache = cached_syntheses.get(filename)

            unchanged = (
                not force
                and prev is not None
                and prev.hash == current_hash
                and cache is not None
            )

            if unchanged:
                all_syntheses_by_file[filename] = cache
                result.skipped += 1
            else:
                to_compile[filename] = source

        if not to_compile and not deleted_filenames:
            # Everything unchanged — re-render full corpus from cache.
            logger.info("All sources unchanged — re-rendering from cache.")

        if to_compile:
            logger.info("Synthesising %d source(s)...", len(to_compile))

            # ── Build existing concept index for dedup context ─────────
            existing_concepts = _existing_concept_slugs(state, all_syntheses_by_file)

            # ── Synthesise with truncation-based retry ──────────────────
            # Each source is tried at full content, then 50K, then 20K.
            sem = asyncio.Semaphore(config.compile_concurrency)

            async def _synth_one(filename: str, source: SourceDoc):
                async with sem:
                    return await _synthesize_with_retry(
                        config, filename, source, existing_concepts,
                        metrics,
                    )

            synth_results = await asyncio.gather(
                *[_synth_one(f, s) for f, s in to_compile.items()],
                return_exceptions=True,
            )

            # ── Collect successful syntheses + cache them ──────────────
            filenames_done: list[str] = []
            for i, res in enumerate(synth_results):
                filename = list(to_compile.keys())[i]
                if isinstance(res, BaseException):
                    logger.error("Synthesis failed for '%s': %s", filename, res)
                    result.errors.append(f"synth:{filename}:{res}")
                    if metrics:
                        metrics.record_synthesis(
                            source_file=filename,
                            success=False,
                            error_type=type(res).__name__,
                        )
                    continue
                if res is None:
                    logger.warning(
                        "Synthesis produced no output for '%s' "
                        "(tried all truncation levels)", filename,
                    )
                    result.errors.append(f"synth:{filename}: no output (permanent failure)")
                    if metrics:
                        metrics.record_synthesis(
                            source_file=filename,
                            success=False,
                            error_type="no_output_all_truncation_levels",
                        )
                    continue
                all_syntheses_by_file[filename] = res
                save_synthesis(res, config.llmwiki_dir, filename)
                filenames_done.append(filename)
                result.compiled += 1
        else:
            filenames_done = []

        if not all_syntheses_by_file:
            if result.errors:
                logger.error("No syntheses available (%d errors).", len(result.errors))
            # Persist state even when no syntheses (e.g. all sources deleted).
            write_state(config.state_file, state)
            return result

        # ── Merge: full corpus dedup ───────────────────────────────────
        all_syntheses = list(all_syntheses_by_file.values())
        bundle = merge_bundle(all_syntheses)
        result.errors.extend(bundle.errors)

        # ── Backlink propagation: ensure bidirectional edges ──────────
        from obsidian_llm_wiki.synth.dedupe import propagate_backlinks
        propagate_backlinks(bundle)

        # ── Incremental concept re-synthesis ──────────────────────────
        # When new sources reference existing concepts (from cached syntheses),
        # re-synthesize those concepts to integrate new information coherently
        # rather than just appending sections.
        if filenames_done and config.synthesis_mode == "two_pass":
            try:
                await _resynthesize_referenced_concepts(
                    config, bundle, filenames_done,
                    all_syntheses_by_file, sources, metrics,
                )
            except Exception as exc:
                logger.warning("Concept re-synthesis skipped: %s", exc)

        # ── Render: full corpus deterministic markdown ─────────────────
        logger.info(
            "Rendering %d concepts, %d MOCs from %d sources...",
            len(bundle.concepts), len(bundle.maps), len(all_syntheses),
        )
        # Render with the FULL corpus, not just changed sources.
        all_sources_for_render = {
            f: sources.get(f) or _source_from_synthesis(all_syntheses_by_file[f])
            for f in all_syntheses_by_file
        }
        render_start = time.monotonic()
        written = render_vault(bundle_dir, bundle, all_sources_for_render, config=config)
        result.pages.extend(written)
        result.concepts = bundle.concepts

        # ── Record rendering metrics ───────────────────────────────────
        render_time = time.monotonic() - render_start
        metrics.record_rendering(
            concepts_rendered=len(bundle.concepts),
            mocs_rendered=len(bundle.maps),
            cross_lingual_links=0,  # populated by embedding pass if enabled
            backlinks_added=0,  # approximate: not separately tracked yet
            time_seconds=render_time,
        )

        # ── Update state for compiled sources ─────────────────────────
        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for filename in filenames_done:
            source = to_compile[filename]
            synth = all_syntheses_by_file.get(filename)
            concept_slugs = [c.slug for c in synth.concepts] if synth else []
            update_source_state(
                state, filename,
                hash_content(source.content),
                concept_slugs,
                compiled_at,
            )

        write_state(config.state_file, state)

        logger.info(
            "Done: %d compiled, %d skipped, %d deleted, %d concepts, %d MOCs, %d errors",
            result.compiled, result.skipped, result.deleted,
            len(bundle.concepts), len(bundle.maps), len(result.errors),
        )
        return result

    finally:
        # ── Persist metrics ─────────────────────────────────────────────
        metrics.finish_run()
        metrics.save()
        release_lock(config.lock_file)


async def _synthesize_source(
    config: Config,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
) -> SourceSynthesis | None:
    """Call the LLM to synthesise one source into a SourceSynthesis.

    Dispatches to the two-pass quality synthesis when
    ``config.synthesis_mode == "two_pass"``; otherwise uses the default
    single-pass synthesis.

    Language is always detected from source content (not config.output_language)
    so that Chinese sources stay Chinese, English stays English, etc.
    """
    # Detect language once — used by both single and two-pass paths
    source_lang = _detect_source_language(source.content, filename)

    if config.synthesis_mode == "two_pass":
        from obsidian_llm_wiki.synth.quality import quality_synthesize_source
        synth = await quality_synthesize_source(
            config, filename, source, existing_concepts,
        )
        if synth is not None and source_lang and not synth.language:
            synth.language = source_lang
        return synth

    from obsidian_llm_wiki.providers.llm import acall_llm

    prompt = build_synthesis_prompt(
        source.title,
        source.content,
        existing_concepts=existing_concepts,
        language=source_lang,
    )

    messages = [{"role": "user", "content": "Synthesise the source document above."}]

    try:
        response = await acall_llm(prompt, messages, config)
    except Exception as exc:
        logger.error("LLM call failed for '%s': %s", filename, exc)
        raise

    synthesis = parse_single_source_synthesis(response)
    if synthesis is None:
        logger.warning("Could not parse synthesis JSON for '%s'", filename)
        return None

    # Ensure source title and file are set.
    if not synthesis.source_title:
        synthesis.source_title = source.title
    synthesis.source_file = filename

    if source_lang and not synthesis.language:
        synthesis.language = source_lang

    return synthesis


async def _synthesize_with_retry(
    config: Config,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
    metrics: MetricsCollector | None = None,
) -> SourceSynthesis | None:
    """Synthesize a source with progressive content truncation on failure.

    Tries (in order):
      1. Full content
      2. Content truncated to 50K chars
      3. Content truncated to 20K chars

    If all attempts fail, logs as permanently failed and returns None.
    Records each attempt via the metrics collector if provided.
    """
    for level_idx, truncation_chars in enumerate(_TRUNCATION_LEVELS):
        if truncation_chars is not None and len(source.content) > truncation_chars:
            truncated_source = SourceDoc(
                title=source.title,
                content=source.content[:truncation_chars],
                url=source.url,
                source_file=filename,
            )
            level_label = f"{truncation_chars // 1000}K"
        else:
            truncated_source = source
            level_label = "full"

        if level_idx > 0:
            logger.info(
                "Retrying '%s' with truncated content (%s chars, level %d)",
                filename, level_label, level_idx,
            )

        synth_start = time.monotonic()
        try:
            synth = await _synthesize_source(
                config, filename, truncated_source, existing_concepts,
            )
        except Exception as exc:
            synth_time = time.monotonic() - synth_start
            logger.error(
                "Synthesis attempt %d failed for '%s' (%s): %s",
                level_idx + 1, filename, level_label, exc,
            )
            if metrics:
                metrics.record_synthesis(
                    source_file=filename,
                    pass1_time=synth_time,
                    success=False,
                    error_type=type(exc).__name__,
                )
            if level_idx < len(_TRUNCATION_LEVELS) - 1:
                continue
            # Last level — re-raise so the caller handles it as a failure
            raise

        synth_time = time.monotonic() - synth_start

        if synth is not None:
            if metrics:
                metrics.record_synthesis(
                    source_file=filename,
                    pass1_time=synth_time,
                    concepts_extracted=len(synth.concepts),
                    success=True,
                )
            return synth

        # synth is None — no output, try next truncation level
        if metrics:
            metrics.record_synthesis(
                source_file=filename,
                pass1_time=synth_time,
                success=False,
                error_type="no_output",
            )
        logger.warning(
            "Synthesis produced no output for '%s' at level %d (%s)",
            filename, level_idx + 1, level_label,
        )

    # All truncation levels exhausted — permanent failure
    logger.error(
        "Permanent synthesis failure for '%s': all %d truncation levels exhausted. "
        "Content length: %d chars.",
        filename, len(_TRUNCATION_LEVELS), len(source.content),
    )
    return None


async def recompile_single_source(
    vault_path: str | Path,
    source_file: str,
    config: Config | None = None,
) -> CompileResult:
    """Manually retry a single failed source file.

    Loads the source from the vault's sources/ directory, runs synthesis
    with truncation-based retry, and caches the result.

    Args:
        vault_path: Path to the Obsidian vault root.
        source_file: The source filename (e.g. "my-article.md").
        config: Pipeline config (loaded from vault_path/.env if None).

    Returns:
        CompileResult with the compilation outcome.
    """
    vault = Path(vault_path).resolve()
    result = CompileResult()

    if config is None:
        env_file = str(vault / ".env") if (vault / ".env").exists() else None
        config = load_config(env_file=env_file, VAULT_PATH=str(vault))

    bundle_dir = config.wiki_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Load the single source file
    source_path = config.sources_dir / source_file
    if not source_path.exists():
        result.errors.append(f"Source file not found: {source_path}")
        return result

    content = source_path.read_text(encoding="utf-8")
    title = source_path.stem
    source = SourceDoc(title=title, content=content, url=str(source_path), source_file=source_file)

    if len(content) < config.min_source_chars:
        result.errors.append(
            f"source:{source_file}: too short ({len(content)} < "
            f"{config.min_source_chars} chars)"
        )
        return result

    # Acquire lock
    if not acquire_lock(config.lock_file):
        result.errors.append("lock: another compilation is running")
        return result

    metrics = MetricsCollector(vault)
    metrics.start_run()

    try:
        state = read_state(config.state_file)
        existing_concepts = _existing_concept_slugs(state)

        synth = await _synthesize_with_retry(
            config, source_file, source, existing_concepts, metrics,
        )

        if synth is None:
            result.errors.append(f"synth:{source_file}: permanent failure (all truncation levels)")
            return result

        save_synthesis(synth, config.llmwiki_dir, source_file)
        result.compiled += 1
        result.concepts = synth.concepts

        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        update_source_state(
            state, source_file,
            hash_content(source.content),
            [c.slug for c in synth.concepts],
            compiled_at,
        )
        write_state(config.state_file, state)

        logger.info(
            "Recompiled '%s': %d concepts extracted",
            source_file, len(synth.concepts),
        )
        return result

    finally:
        metrics.finish_run()
        metrics.save()
        release_lock(config.lock_file)


def _detect_source_language(content: str, filename: str) -> str:
    """Detect the primary language of source content."""
    try:
        from obsidian_llm_wiki.synth.language import detect_language
        lang = detect_language(content)
        logger.debug("Detected language '%s' for '%s'", lang, filename)
        return lang
    except Exception:
        return ""


def _existing_concept_slugs(
    state: WikiState,
    cached_syntheses: dict[str, SourceSynthesis] | None = None,
) -> list[str]:
    """Extract all known concept slugs from state + cached syntheses.

    Includes concepts from cached syntheses (unchanged sources) so the
    LLM knows the full existing corpus for dedup.
    """
    slugs: set[str] = set()
    for src_state in state.sources.values():
        slugs.update(src_state.concepts)
    if cached_syntheses:
        for synth in cached_syntheses.values():
            slugs.update(c.slug for c in synth.concepts)
    return sorted(slugs)


def _source_from_synthesis(synth: SourceSynthesis) -> SourceDoc:
    """Build a minimal SourceDoc from a cached synthesis (for rendering)."""
    return SourceDoc(
        title=synth.source_title,
        content=synth.source_summary or "",
        source_file=synth.source_file,
    )


async def _resynthesize_referenced_concepts(
    config: Config,
    bundle: SynthesisBundle,
    new_filenames: list[str],
    all_syntheses_by_file: dict[str, SourceSynthesis],
    sources: dict[str, SourceDoc],
    metrics: MetricsCollector | None = None,
) -> None:
    """Re-synthesize existing concepts that are referenced by new sources.

    For each new source, find concepts in the new synthesis whose slug
    already existed in the corpus (from cached syntheses). For each such
    concept, call resynthesize_concept() to produce a coherent updated
    concept body that integrates the new source's information.

    Only runs in two_pass mode. Skipped if no new sources or no overlaps.
    """
    if not new_filenames:
        return

    from obsidian_llm_wiki.synth.quality import resynthesize_concept

    # Build set of concept slugs from cached (unchanged) sources
    cached_slugs: set[str] = set()
    for filename, synth in all_syntheses_by_file.items():
        if filename in new_filenames:
            continue  # This is a new source
        cached_slugs.update(c.slug for c in synth.concepts)

    if not cached_slugs:
        return  # No existing concepts to re-synthesize

    # Find concepts in new sources that reference existing concepts
    concepts_to_resynth: list[tuple[str, str, str]] = []  # (slug, source_content, source_title)
    for filename in new_filenames:
        synth = all_syntheses_by_file.get(filename)
        if not synth:
            continue
        source = sources.get(filename)
        source_content = source.content if source else ""
        source_title = source.title if source else synth.source_title

        for concept in synth.concepts:
            if concept.slug in cached_slugs:
                concepts_to_resynth.append((concept.slug, source_content, source_title))

    if not concepts_to_resynth:
        return

    logger.info(
        "Re-synthesizing %d concepts referenced by new sources...",
        len(concepts_to_resynth),
    )

    # Build concept map from bundle
    concept_map = {c.slug: c for c in bundle.concepts}

    # Re-synthesize each concept
    sem = asyncio.Semaphore(config.compile_concurrency)

    async def _resynth_one(slug: str, content: str, title: str):
        async with sem:
            existing = concept_map.get(slug)
            if not existing:
                return None
            return await resynthesize_concept(config, existing, content, title)

    results = await asyncio.gather(
        *[_resynth_one(s, c, t) for s, c, t in concepts_to_resynth],
        return_exceptions=True,
    )

    # Replace updated concepts in the bundle
    updated = 0
    for i, res in enumerate(results):
        if isinstance(res, BaseException) or res is None:
            continue
        slug = concepts_to_resynth[i][0]
        # Replace in bundle.concepts
        for j, c in enumerate(bundle.concepts):
            if c.slug == slug:
                bundle.concepts[j] = res
                updated += 1
                break

    if updated:
        logger.info("Re-synthesized %d/%d concepts successfully", updated, len(concepts_to_resynth))
