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
import hashlib
import logging
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.cache import (
    delete_cached_synthesis,
    load_all_cached_syntheses,
    load_resynthesis_overlay,
    save_resynthesis_overlay,
    save_synthesis,
)
from obsidian_llm_wiki.core.contradictions import (
    ContradictionRecord,
    ContradictionStore,
    SourceRevision,
)
from obsidian_llm_wiki.core.evidence import resolve_synthesis_evidence
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.metrics import MetricsCollector
from obsidian_llm_wiki.core.models import (
    CompileResult,
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
    WikiState,
)
from obsidian_llm_wiki.core.orphan import mark_orphaned_concepts
from obsidian_llm_wiki.core.schema import (
    DEFAULT_SCHEMA_POLICY,
    SchemaPolicy,
    load_schema_policy,
    select_synthesis_granularity,
)
from obsidian_llm_wiki.core.source_content import bound_source_content
from obsidian_llm_wiki.core.source_files import source_file_path, validate_source_filename
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

# System prompt for LLM calls — kept short to avoid proxy system-prompt
# truncation. The full synthesis instructions go in the user message.
_SYSTEM_PROMPT = (
    "You are a knowledge synthesis engine. "
    "Return ONLY a JSON object, no prose, no code fences."
)

# ── Retry truncation levels ────────────────────────────────────────────────
# When a source fails synthesis for a SIZE-RELATED reason, progressively
# truncate content and retry.
_TRUNCATION_LEVELS = [None, 50_000, 20_000]  # full → 50K → 20K


def _is_size_related_failure(exc: BaseException) -> bool:
    """Whether truncating the source content could plausibly fix *exc*.

    Only timeouts qualify: on a local LLM they usually mean the prompt is too
    large for the model/host to finish in time. Everything else (connection
    refused, HTTP errors, auth, bugs) is NOT fixed by discarding content —
    the provider layer already retried transient errors with backoff, and
    "retrying" such a failure at 20K of a 200K source would just build the
    wiki from a fraction of the document while reporting success.
    """
    import httpx

    return isinstance(exc, TimeoutError | httpx.TimeoutException)


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
    # Keep the bounded representation used for hashing, synthesis, cache, and
    # rendering local to this run without mutating the caller's source mapping.
    sources = dict(sources)
    result = CompileResult()

    # ── Load config ────────────────────────────────────────────────────
    if config is None:
        env_file = str(vault / ".env") if (vault / ".env").exists() else None
        config = load_config(env_file=env_file, VAULT_PATH=str(vault))

    # Policy is vault-level configuration, so read it once per compilation and
    # pass the same sanitized object to every source synthesis task.
    schema_policy = load_schema_policy(vault)

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
        # Render-time bilingual normalization used to leave cache/state with
        # the old identity while pages used the new slug. Migrate cached items
        # before they participate in dedup, prompts, or state updates.
        for filename, synthesis in cached_syntheses.items():
            if _normalize_synthesis_identity(synthesis):
                save_synthesis(synthesis, config.llmwiki_dir, filename)

        # ── Classify sources: compile vs reuse cache ───────────────────
        to_compile: dict[str, SourceDoc] = {}
        all_syntheses_by_file: dict[str, SourceSynthesis] = {}
        changed_sources: dict[str, tuple[str, SourceSynthesis | None]] = {}

        for filename, source in sources.items():
            # Source length gate (B3 fix).
            content_len = len(source.content.encode("utf-8"))
            if content_len < config.min_source_chars:
                result.errors.append(
                    f"source:{filename}: too short ({content_len} < "
                    f"{config.min_source_chars} bytes)"
                )
                continue
            source, truncated = bound_source_content(source, config.max_source_chars)
            if truncated:
                logger.warning(
                    "Source '%s' is %d UTF-8 bytes — exceeds max_source_chars (%d), "
                    "truncating to safety cap. Sources above chunk_size (%d) "
                    "will be chunked during two-pass synthesis.",
                    filename, content_len, config.max_source_chars,
                    config.chunk_size,
                )
                sources[filename] = source

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
                if prev is not None and prev.hash != current_hash:
                    changed_sources[filename] = (prev.hash, cache)

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
                        config,
                        filename,
                        source,
                        existing_concepts,
                        metrics=metrics,
                        schema_policy=schema_policy,
                    )

            synth_results = await asyncio.gather(
                *[_synth_one(f, s) for f, s in to_compile.items()],
                return_exceptions=True,
            )

            # ── Collect successful syntheses + cache them ──────────────
            filenames_done: list[str] = []
            for filename, res in zip(to_compile.keys(), synth_results, strict=True):
                # Failure metrics were already recorded per attempt inside
                # _synthesize_with_retry — recording again here would count
                # one failed source as 4 failures in metrics.json.
                if isinstance(res, BaseException):
                    logger.error("Synthesis failed for '%s': %s", filename, res)
                    result.errors.append(f"synth:{filename}:{res}")
                    # A transient failure must not erase a changed source's
                    # last successful knowledge from this render. Its state
                    # remains unchanged, so the source retries next run.
                    previous = changed_sources.get(filename, ("", None))[1]
                    if previous is not None:
                        all_syntheses_by_file[filename] = previous
                    continue
                if res is None:
                    logger.warning(
                        "Synthesis produced no output for '%s' "
                        "(tried all truncation levels)", filename,
                    )
                    result.errors.append(f"synth:{filename}: no output (permanent failure)")
                    previous = changed_sources.get(filename, ("", None))[1]
                    if previous is not None:
                        all_syntheses_by_file[filename] = previous
                    continue
                assert isinstance(res, SourceSynthesis)
                _normalize_synthesis_identity(res)
                all_syntheses_by_file[filename] = res
                save_synthesis(res, config.llmwiki_dir, filename)
                filenames_done.append(filename)
                result.compiled += 1
        else:
            filenames_done = []

        if not all_syntheses_by_file:
            if result.errors:
                logger.error("No syntheses available (%d errors).", len(result.errors))
            # Run the renderer with an empty bundle so obsolete generated pages
            # are pruned after the final source is removed.
            written = render_vault(bundle_dir, SynthesisBundle(), {}, config=config)
            result.pages.extend(written)
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

        # ── Re-apply persisted resynthesized concepts ──────────────────
        # merge_bundle rebuilt every concept from the per-source caches, which
        # still hold pre-resynthesis bodies. Without re-applying the overlay,
        # every coherently rewritten concept would revert to its mechanical
        # merge on the next run. Entries whose slug was freshly extracted this
        # run are dropped: the fresh synthesis supersedes them (and overlapping
        # slugs get re-resynthesized below).
        overlay = load_resynthesis_overlay(config.llmwiki_dir)
        overlay_dirty = False
        if overlay:
            fresh_slugs = {
                c.slug
                for f in filenames_done
                for c in all_syntheses_by_file[f].concepts
            }
            for slug in fresh_slugs & overlay.keys():
                del overlay[slug]
                overlay_dirty = True
            concept_index = {c.slug: i for i, c in enumerate(bundle.concepts)}
            for slug, concept in overlay.items():
                idx = concept_index.get(slug)
                if idx is not None:
                    bundle.concepts[idx] = concept

        # ── Incremental concept re-synthesis ──────────────────────────
        # When new sources reference existing concepts (from cached syntheses),
        # re-synthesize those concepts to integrate new information coherently
        # rather than just appending sections.
        resynthesized: dict[str, ConceptNote] = {}
        if filenames_done and config.synthesis_mode == "two_pass":
            try:
                resynthesized = await _resynthesize_referenced_concepts(
                    config, bundle, filenames_done,
                    all_syntheses_by_file, sources, metrics,
                )
                if resynthesized:
                    overlay.update(resynthesized)
                    overlay_dirty = True
            except Exception as exc:
                logger.warning("Concept re-synthesis skipped: %s", exc)

        if overlay_dirty:
            save_resynthesis_overlay(config.llmwiki_dir, overlay)

        # ── Corpus normalization: semantic dedup + MoC assignment ─────
        # These mutate the bundle (merge concepts, rewrite slugs), so they
        # belong in the synthesis stage — before rendering AND before the
        # state write — not buried inside render_vault where their failures
        # were previously swallowed at debug level. Both no-op when
        # embeddings are unavailable; an exception here is a real bug and
        # must be visible, but should not kill an otherwise good build.
        from obsidian_llm_wiki.synth.dedupe import (
            assign_orphans_to_mocs,
            semantic_dedupe_concepts,
        )
        embeddings_cache: dict[str, list[float]] = {}
        embedding_started = time.monotonic()
        embedding_options: dict[str, object] = {
            "enabled": config.embeddings_enabled,
            "model": config.embedding_model,
            "host": config.embedding_host,
        }
        # Load persisted embeddings to avoid recomputing for unchanged concepts.
        # The cache is model-keyed — switching embedding models invalidates it.
        embeddings_cache_path = config.llmwiki_dir / "embeddings.json"
        if config.embeddings_enabled:
            from obsidian_llm_wiki.synth.embedding import (
                load_embeddings_cache,
                save_embeddings_cache,
            )
            embeddings_cache = load_embeddings_cache(
                embeddings_cache_path, model=config.embedding_model
            )
        try:
            # Build the set of new/changed concept slugs for incremental dedup.
            new_slugs: set[str] | None = None
            if filenames_done:
                new_slugs = set()
                for f in filenames_done:
                    synth = all_syntheses_by_file.get(f)
                    if synth:
                        new_slugs.update(c.slug for c in synth.concepts)
                # Re-synthesis changes a concept's semantic payload without
                # changing its source filename. It must re-enter incremental
                # comparison or an updated vector can bypass dedup entirely.
                new_slugs.update(resynthesized)

            semantic_dedupe_concepts(
                bundle,
                threshold=config.similarity_dedup_threshold,
                embeddings_cache=embeddings_cache,
                embedding_options=embedding_options,
                new_slugs=new_slugs,
            )
        except Exception as exc:
            logger.warning("Semantic dedup failed (continuing without): %s", exc)
        try:
            assign_orphans_to_mocs(
                bundle,
                threshold=config.moc_assignment_threshold,
                embeddings_cache=embeddings_cache,
                embedding_options=embedding_options,
            )
        except Exception as exc:
            logger.warning("MoC orphan assignment failed (continuing without): %s", exc)

        # Semantic dedup remaps source-local concept slugs in memory. Persist
        # those canonical source syntheses before a later no-embedding run can
        # resurrect a dedup victim from a pre-dedup cache.
        for synthesis in bundle.sources:
            save_synthesis(synthesis, config.llmwiki_dir, synthesis.source_file)

        cross_lingual_links: dict[str, list[tuple[str, float, str]]] = {}
        try:
            from obsidian_llm_wiki.synth.embedding import find_cross_lingual_links

            cross_lingual_links = find_cross_lingual_links(
                bundle.concepts,
                enabled=config.embeddings_enabled,
                model=config.embedding_model,
                host=config.embedding_host,
                embeddings_cache=embeddings_cache,
            )
        except Exception as exc:
            logger.warning("Cross-lingual linking failed (continuing without): %s", exc)

        # Persist embeddings for reuse on the next run.
        persisted_embedding_count = 0
        if config.embeddings_enabled and embeddings_cache:
            # Prune stale entries: only keep embeddings for concepts that
            # survived dedup (are in the final bundle).
            live_slugs = {c.slug for c in bundle.concepts}
            pruned_cache = {
                slug: vec for slug, vec in embeddings_cache.items()
                if slug in live_slugs
            }
            save_embeddings_cache(
                embeddings_cache_path, pruned_cache, model=config.embedding_model
            )
            persisted_embedding_count = len(pruned_cache)

        metrics.record_embedding(
            model=config.embedding_model if config.embeddings_enabled else "",
            concepts_embedded=persisted_embedding_count,
            cross_lingual_matches=len({
                tuple(sorted((source_slug, target_slug)))
                for source_slug, targets in cross_lingual_links.items()
                for target_slug, _score, _display in targets
                if source_slug != target_slug
            }),
            time_seconds=time.monotonic() - embedding_started,
        )

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
        written = render_vault(
            bundle_dir,
            bundle,
            all_sources_for_render,
            config=config,
            cross_lingual_links=cross_lingual_links,
        )
        result.pages.extend(written)
        result.concepts = bundle.concepts

        # ── Record rendering metrics ───────────────────────────────────
        render_time = time.monotonic() - render_start
        metrics.record_rendering(
            concepts_rendered=len(bundle.concepts),
            mocs_rendered=len(bundle.maps),
            cross_lingual_links=len({
                tuple(sorted((source_slug, target_slug)))
                for source_slug, targets in cross_lingual_links.items()
                for target_slug, _score, _display in targets
                if source_slug != target_slug
            }),
            backlinks_added=0,  # approximate: not separately tracked yet
            time_seconds=render_time,
        )

        # ── Update state for compiled sources ─────────────────────────
        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _persist_changed_source_revisions(
            config,
            changed_sources,
            filenames_done,
            to_compile,
            all_syntheses_by_file,
        )
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
    *,
    schema_policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
) -> SourceSynthesis | None:
    """Call the LLM to synthesise one source into a SourceSynthesis.

    Dispatches to the two-pass quality synthesis when
    ``config.synthesis_mode == "two_pass"``; otherwise uses the default
    single-pass synthesis.

    Language is always detected from source content (not config.output_language)
    so that Chinese sources stay Chinese, English stays English, etc.
    """
    # Detect language and choose detail once per source — used by both single
    # and two-pass paths, while the vault-level policy is shared across sources.
    source_lang = _detect_source_language(source.content, filename)
    granularity = select_synthesis_granularity(
        source.content,
        source.source_type,
        schema_policy.granularity_override,
    )

    if config.synthesis_mode == "two_pass":
        from obsidian_llm_wiki.synth.quality import multi_model_entry_synthesize_source
        synth = await multi_model_entry_synthesize_source(
            config, filename, source, existing_concepts,
            schema_policy=schema_policy,
            granularity=granularity,
        )
        if synth is not None:
            if not synth.source_title:
                synth.source_title = source.title
            synth.source_file = filename
            if source_lang and not synth.language:
                synth.language = source_lang
            resolve_synthesis_evidence(synth, source, filename)
        return synth

    from obsidian_llm_wiki.providers.llm import acall_llm

    prompt = build_synthesis_prompt(
        source.title,
        source.content,
        existing_concepts=existing_concepts,
        language=source_lang,
        schema_policy=schema_policy,
        granularity=granularity,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await acall_llm(prompt, messages, config, task="ingest")
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

    resolve_synthesis_evidence(synthesis, source, filename)

    return synthesis


async def _synthesize_with_retry(
    config: Config,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
    metrics: MetricsCollector | None = None,
    *,
    schema_policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
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
            truncated_source = replace(source, content=source.content[:truncation_chars])
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
                config,
                filename,
                truncated_source,
                existing_concepts,
                schema_policy=schema_policy,
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
            # Truncation only helps size-related failures. Anything else
            # (connection errors, HTTP failures, bugs) re-raises immediately:
            # the provider layer already retried transient errors, and
            # silently discarding content would disguise data loss as
            # recovery.
            if (
                _is_size_related_failure(exc)
                and level_idx < len(_TRUNCATION_LEVELS) - 1
            ):
                continue
            raise

        synth_time = time.monotonic() - synth_start

        if synth is not None:
            if level_idx > 0:
                # Evidence must describe the source revision persisted and
                # rendered, never the temporary retry prefix.
                resolve_synthesis_evidence(synth, source, filename)
                # Success on truncated content is a degraded result, not a
                # clean one — the wiki was built from a fraction of the source.
                logger.warning(
                    "Synthesized '%s' from TRUNCATED content (%s of %d chars) "
                    "after %d failed attempt(s) — concepts beyond the cut "
                    "are missing.",
                    filename, level_label, len(source.content), level_idx,
                )
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

    # Policy is vault-level configuration, so read it once per compilation and
    # pass the same sanitized object to every source synthesis task.
    schema_policy = load_schema_policy(vault)

    bundle_dir = config.wiki_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Load the single source file through the same loader as `olw build`:
    # it strips YAML frontmatter and takes the title from it. Reading the raw
    # file here would synthesize frontmatter noise under the stem title AND
    # store a raw-content hash that never matches the build's body hash, so
    # the next build would immediately re-synthesize and discard this result.
    from obsidian_llm_wiki.ingest.sources import load_source_file

    try:
        source_file = validate_source_filename(source_file)
        source_path = source_file_path(config.sources_dir, source_file)
    except ValueError as exc:
        result.errors.append(f"Invalid source filename: {exc}")
        return result
    if not source_path.exists():
        result.errors.append(f"Source file not found: {source_path}")
        return result

    source = load_source_file(source_path)
    if source is None:
        result.errors.append(f"source:{source_file}: empty or unreadable")
        return result

    source, truncated = bound_source_content(source, config.max_source_chars)
    if truncated:
        logger.warning(
            "Source '%s' exceeds max_source_chars (%d); truncating before recompile.",
            source_file,
            config.max_source_chars,
        )

    if len(source.content) < config.min_source_chars:
        result.errors.append(
            f"source:{source_file}: too short ({len(source.content)} < "
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
            config,
            source_file,
            source,
            existing_concepts,
            metrics=metrics,
            schema_policy=schema_policy,
        )

        if synth is None:
            result.errors.append(f"synth:{source_file}: permanent failure (all truncation levels)")
            return result

        _normalize_synthesis_identity(synth)
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


def _normalize_synthesis_identity(synthesis: SourceSynthesis) -> bool:
    """Persist English-first bilingual slugs before cache/state/render diverge."""
    from obsidian_llm_wiki.render.bilingual import normalize_bilingual_titles_and_slugs

    before = (
        synthesis.source_title,
        tuple((concept.title, concept.slug) for concept in synthesis.concepts),
        tuple((moc.title, moc.slug, tuple(moc.concept_slugs)) for moc in synthesis.maps),
    )
    bundle = SynthesisBundle(
        sources=[synthesis], concepts=synthesis.concepts, maps=synthesis.maps,
    )
    normalize_bilingual_titles_and_slugs(bundle)
    after = (
        synthesis.source_title,
        tuple((concept.title, concept.slug) for concept in synthesis.concepts),
        tuple((moc.title, moc.slug, tuple(moc.concept_slugs)) for moc in synthesis.maps),
    )
    return before != after


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


def _persist_changed_source_revisions(
    config: Config,
    changed_sources: dict[str, tuple[str, SourceSynthesis | None]],
    filenames_done: list[str],
    compiled_sources: dict[str, SourceDoc],
    syntheses: dict[str, SourceSynthesis],
) -> None:
    """Persist successful changed-source revisions and conservative conflicts.

    A record is only detected when both revisions contain exactly one claim for
    the same ``(concept slug, source reference)`` key and the normalized claim
    text changed. This avoids guessing that changed summaries, unkeyed claims,
    or ambiguous repeated claims are contradictions.
    """
    successful_changes = sorted(set(filenames_done) & set(changed_sources))
    if not successful_changes:
        return

    store = ContradictionStore(config.llmwiki_dir / "contradictions.json")
    known_record_ids = {record.id for record in store.records()}
    for filename in successful_changes:
        previous_hash, previous_synthesis = changed_sources[filename]
        current_synthesis = syntheses.get(filename)
        source = compiled_sources[filename]
        current_hash = hash_content(source.content)
        previous_revision = SourceRevision(filename, previous_hash, previous_hash)
        current_revision = SourceRevision(filename, current_hash, current_hash)
        store.add_source_revision(previous_revision)
        store.add_source_revision(current_revision)

        if previous_synthesis is None or current_synthesis is None:
            continue
        for record in _detect_changed_claims(
            filename,
            previous_revision,
            current_revision,
            previous_synthesis,
            current_synthesis,
        ):
            if record.id not in known_record_ids:
                store.add(record)
                known_record_ids.add(record.id)


def _detect_changed_claims(
    filename: str,
    previous_revision: SourceRevision,
    current_revision: SourceRevision,
    previous_synthesis: SourceSynthesis,
    current_synthesis: SourceSynthesis,
) -> list[ContradictionRecord]:
    """Return deterministic records for materially changed, stable claim keys."""
    previous_claims = _keyed_claims(previous_synthesis)
    current_claims = _keyed_claims(current_synthesis)
    records: list[ContradictionRecord] = []
    for key in sorted(previous_claims.keys() & current_claims.keys()):
        old_claim = previous_claims[key]
        new_claim = current_claims[key]
        if old_claim is None or new_claim is None:
            continue
        concept_slug, source_ref, old_text = old_claim
        _, _, new_text = new_claim
        if _normalized_claim_text(old_text) == _normalized_claim_text(new_text):
            continue
        record_id_material = "\x1f".join(
            (filename, previous_revision.content_hash, current_revision.content_hash, key)
        )
        record_id = "source-revision-" + hashlib.sha256(
            record_id_material.encode("utf-8")
        ).hexdigest()[:20]
        records.append(
            ContradictionRecord(
                id=record_id,
                summary=(
                    f"{filename}: claim for concept '{concept_slug}' at "
                    f"'{source_ref}' changed between source revisions."
                ),
                sources=(previous_revision, current_revision),
                evidence=(f"previous: {old_text}", f"current: {new_text}"),
            )
        )
    return records


def _keyed_claims(
    synthesis: SourceSynthesis,
) -> dict[str, tuple[str, str, str] | None]:
    """Index only unique, explicitly sourced claims for safe comparison."""
    indexed: dict[str, tuple[str, str, str] | None] = {}
    for concept in synthesis.concepts:
        for claim in concept.claims:
            concept_slug = (claim.concept_slug or concept.slug).strip()
            source_ref = " ".join(claim.source_ref.split())
            text = claim.text.strip()
            if not concept_slug or not source_ref or not text:
                continue
            key = f"{concept_slug.casefold()}\x1f{source_ref.casefold()}"
            candidate = (concept_slug, source_ref, text)
            if key in indexed:
                indexed[key] = None
            else:
                indexed[key] = candidate
    return indexed


def _normalized_claim_text(text: str) -> str:
    """Normalize harmless case and whitespace variation before comparison."""
    return " ".join(text.casefold().split())


async def _resynthesize_referenced_concepts(
    config: Config,
    bundle: SynthesisBundle,
    new_filenames: list[str],
    all_syntheses_by_file: dict[str, SourceSynthesis],
    sources: dict[str, SourceDoc],
    metrics: MetricsCollector | None = None,
) -> dict[str, ConceptNote]:
    """Re-synthesize existing concepts that are referenced by new sources.

    For each concept whose slug appears in both a new synthesis and the cached
    corpus, call resynthesize_concept() once with ALL referencing new sources'
    content combined. One call per slug is essential: concurrent per-source
    calls would each rewrite the same original concept and the last write
    would silently discard the other sources' integrations.

    Returns the successfully resynthesized concepts by slug so the caller can
    persist them — the per-source caches were saved before this ran, so
    without persistence the rewrites would revert on the next build.

    Only runs in two_pass mode. Skipped if no new sources or no overlaps.
    """
    if not new_filenames:
        return {}

    from obsidian_llm_wiki.synth.quality import resynthesize_concept

    # Build set of concept slugs from cached (unchanged) sources
    cached_slugs: set[str] = set()
    for filename, synth in all_syntheses_by_file.items():
        if filename in new_filenames:
            continue  # This is a new source
        cached_slugs.update(c.slug for c in synth.concepts)

    if not cached_slugs:
        return {}  # No existing concepts to re-synthesize

    # Group the referencing new sources by concept slug.
    sources_by_slug: dict[str, list[tuple[str, str]]] = {}  # slug → [(content, title)]
    for filename in new_filenames:
        synth = all_syntheses_by_file.get(filename)
        if not synth:
            continue
        source = sources.get(filename)
        source_content = source.content if source else ""
        source_title = source.title if source else synth.source_title

        for concept in synth.concepts:
            if concept.slug in cached_slugs:
                sources_by_slug.setdefault(concept.slug, []).append(
                    (source_content, source_title),
                )

    if not sources_by_slug:
        return {}

    logger.info(
        "Re-synthesizing %d concepts referenced by new sources...",
        len(sources_by_slug),
    )

    # Build concept map from bundle
    concept_map = {c.slug: c for c in bundle.concepts}

    # Re-synthesize each concept once, with all referencing sources combined.
    sem = asyncio.Semaphore(config.compile_concurrency)

    async def _resynth_one(slug: str, refs: list[tuple[str, str]]):
        async with sem:
            existing = concept_map.get(slug)
            if not existing:
                return None
            combined_content = "\n\n---\n\n".join(
                f"### Source: {title}\n\n{content}" for content, title in refs
            )
            combined_title = ", ".join(dict.fromkeys(title for _, title in refs))
            return await resynthesize_concept(
                config, existing, combined_content, combined_title,
            )

    slugs = list(sources_by_slug.keys())
    results = await asyncio.gather(
        *[_resynth_one(slug, sources_by_slug[slug]) for slug in slugs],
        return_exceptions=True,
    )

    # Replace updated concepts in the bundle and collect them for persistence.
    concept_index = {c.slug: i for i, c in enumerate(bundle.concepts)}
    resynthesized: dict[str, ConceptNote] = {}
    for slug, res in zip(slugs, results, strict=True):
        if isinstance(res, BaseException) or res is None:
            continue
        idx = concept_index.get(slug)
        if idx is not None:
            bundle.concepts[idx] = res
            resynthesized[slug] = res

    if resynthesized:
        logger.info(
            "Re-synthesized %d/%d concepts successfully",
            len(resynthesized), len(sources_by_slug),
        )
    return resynthesized
