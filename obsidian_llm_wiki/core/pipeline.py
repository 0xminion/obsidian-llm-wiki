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
import json
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.cache import (
    delete_cached_synthesis,
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
from obsidian_llm_wiki.core.identity_policy import apply_equivalences, load_equivalences
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.metrics import MetricsCollector
from obsidian_llm_wiki.core.models import (
    CompileResult,
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.orphan import orphan_page
from obsidian_llm_wiki.core.registry import (
    AnnSearchResult,
    ArtifactVersions,
    VaultRegistry,
)
from obsidian_llm_wiki.core.request_limiter import (
    AsyncRequestLimiter,
    ThreadsafeRequestGate,
)
from obsidian_llm_wiki.core.schema import (
    DEFAULT_SCHEMA_POLICY,
    SchemaPolicy,
    load_schema_policy,
    select_synthesis_granularity,
)
from obsidian_llm_wiki.core.source_files import source_file_path, validate_source_filename
from obsidian_llm_wiki.core.state import (
    hash_content,
    read_state,
    remove_source_state,
    update_source_state,
    write_state,
)
from obsidian_llm_wiki.core.text_quality import dedupe_statements, dedupe_tags
from obsidian_llm_wiki.render.obsidian import render_vault
from obsidian_llm_wiki.synth.dedupe import merge_bundle
from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from obsidian_llm_wiki.synth.response_schema import (
    SYNTHESIS_RESPONSE_SCHEMA as _SYNTHESIS_RESPONSE_SCHEMA,
)
from obsidian_llm_wiki.synth.retrieval import ConceptRetriever
from obsidian_llm_wiki.synth.validation import (
    SynthesisValidationError,
    validate_local_identities,
    validate_synthesis,
)

logger = logging.getLogger("obswiki.core.pipeline")

# System prompt for LLM calls — kept short to avoid proxy system-prompt
# truncation. The full synthesis instructions go in the user message.
_SYSTEM_PROMPT = (
    "You are a knowledge synthesis engine. "
    "Return ONLY a JSON object, no prose, no code fences."
)

# ── Structured synthesis request policy ───────────────────────────────────
#
# The model supports a 256K context, but context and output share that budget.
# Asking for 128K output for every source both wastes capacity and makes large
# prompts exceed the window.  16K is ample for a concise 3–8-concept synthesis;
# larger documents are losslessly split into chunks below.
_SYNTHESIS_NUM_PREDICT = 16_384
_SYNTHESIS_TEMPERATURE = 0
_SYNTHESIS_PARSE_ATTEMPTS = 2
_EXTRACTOR_VERSION = "extract-v2"
_PROMPT_VERSION = "synthesis-v4"
_SYNTHESIS_SCHEMA_VERSION = "synthesis-schema-v3"


def _artifact_versions_for_config(
    config: Config,
    schema_policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
) -> ArtifactVersions:
    effective_synthesis = {
        "prompt": _PROMPT_VERSION,
        "provider": config.llm.provider,
        "host": config.llm.host,
        "ingest_model": config.llm.ingest_model or config.llm.model,
        "expand_model": config.llm.expand_model or config.llm.model,
        "synthesis_mode": config.synthesis_mode,
        "output_language": config.output_language,
        "chunk_size": config.chunk_size,
        "candidate_limit": config.candidate_limit,
        "candidate_char_budget": config.candidate_char_budget,
        "ann_candidate_pool": config.ann_candidate_pool,
        "evidence_window_chars": config.evidence_window_chars,
        "evidence_context_paragraphs": config.evidence_context_paragraphs,
        "similarity_dedup_threshold": config.similarity_dedup_threshold,
        "cross_lingual_threshold": config.cross_lingual_threshold,
        "moc_assignment_threshold": config.moc_assignment_threshold,
        "schema_policy": asdict(schema_policy),
    }
    effective_hash = hashlib.sha256(
        json.dumps(effective_synthesis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return ArtifactVersions(
        extractor=_EXTRACTOR_VERSION,
        prompt=f"{_PROMPT_VERSION}:{effective_hash}",
        schema=_SYNTHESIS_SCHEMA_VERSION,
        synthesis_model=config.llm.ingest_model or config.llm.model,
    )


async def _call_structured_synthesis(
    prompt: str,
    messages: list[dict[str, str]],
    config: Config,
    *,
    request_limiter: AsyncRequestLimiter | None = None,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> str:
    """Run one bounded, JSON-mode synthesis request.

    The semaphore belongs around the actual remote request rather than a
    whole source.  This keeps every in-flight LLM call within the server's
    parallelism limit while allowing chunks from a large source to share the
    same capacity with ordinary sources.
    """
    from obsidian_llm_wiki.providers.llm import acall_llm

    async def _request() -> str:
        return await acall_llm(
            prompt,
            messages,
            config,
            task="ingest",
            format=_SYNTHESIS_RESPONSE_SCHEMA,
            options={
                "num_predict": _SYNTHESIS_NUM_PREDICT,
                "temperature": _SYNTHESIS_TEMPERATURE,
            },
        )

    if request_limiter is None:
        if llm_semaphore is None:
            return await _request()
        async with llm_semaphore:
            return await _request()
    async with request_limiter.slot():
        return await _request()


async def _retry_chunk_synthesis(synthesize, filename: str, chunk_index: int, chunk_count: int):
    """Retry one failed chunk without accepting an incomplete source."""
    for attempt in range(_SYNTHESIS_PARSE_ATTEMPTS):
        try:
            result = await synthesize()
        except Exception as exc:
            logger.warning(
                "Chunk %d/%d for '%s' failed on attempt %d/%d: %s",
                chunk_index, chunk_count, filename, attempt + 1,
                _SYNTHESIS_PARSE_ATTEMPTS, exc,
            )
            continue
        if result is not None:
            return result
        logger.warning(
            "Chunk %d/%d for '%s' produced no parseable output on attempt %d/%d",
            chunk_index, chunk_count, filename, attempt + 1,
            _SYNTHESIS_PARSE_ATTEMPTS,
        )
    return None


async def _maintain_pipeline_leases(
    registry: VaultRegistry, worker_id: str, errors: list[str], *, interval: float = 30,
) -> None:
    """Keep claims alive while slow asynchronous inference is running."""
    while True:
        await asyncio.sleep(interval)
        try:
            registry.renew_dirty_worker(worker_id)
        except Exception as exc:
            errors.append(f"queue: lease heartbeat failed: {exc}")
            logger.error("Lease heartbeat failed: %s", exc)
            return


def _preflight_pipeline_paths(vault: Path, config: Config) -> None:
    """Check metadata before policy reads, locks, SQLite or error metrics."""
    from obsidian_llm_wiki.core.generated_paths import (
        checked_generated_path,
        checked_metadata_directory,
    )

    if config.vault != vault:
        raise ValueError("Pipeline vault does not match Config.vault")
    checked_metadata_directory(config.llmwiki_dir, config.vault)
    # The policy loader also accepts this historical fallback location.
    checked_generated_path(vault / ".llmwiki" / "schema.yaml", config.vault)


async def run_pipeline(
    vault_path: str | Path,
    sources: dict[str, SourceDoc],
    config: Config | None = None,
    *,
    force: bool = False,
    unchanged_filenames: set[str] | None = None,
    source_file_signatures: dict[str, tuple[int, int]] | None = None,
    refresh_render: bool = False,
) -> CompileResult:
    """Run the full synthesis + render pipeline.

    Args:
        vault_path: Path to the Obsidian vault root.
        sources: Dict mapping source filename → SourceDoc (the FULL corpus).
        config: Pipeline config (loaded from vault_path/.env if None).
        force: Force re-synthesis of all sources.
        unchanged_filenames: Files represented by registry metadata because
            their committed ``mtime_ns``/size signature is unchanged.
        source_file_signatures: Discovery signatures committed only after a
            successful render.
        refresh_render: Reconcile generated output even with an unchanged
            input checkpoint, without forcing new LLM requests.

    Returns:
        CompileResult with counts and any errors.
    """
    vault = Path(vault_path).resolve()
    # Keep the bounded representation used for hashing, synthesis, cache, and
    # rendering local to this run without mutating the caller's source mapping.
    sources = dict(sources)
    trusted_unchanged = set(unchanged_filenames or ())
    observed_signatures = dict(source_file_signatures or {})
    result = CompileResult()

    # ── Load config ────────────────────────────────────────────────────
    if config is None:
        env_file = str(vault / ".env") if (vault / ".env").exists() else None
        config = load_config(env_file=env_file, VAULT_PATH=str(vault))

    _preflight_pipeline_paths(vault, config)

    # Policy is vault-level configuration, so read it once per compilation and
    # pass the same sanitized object to every source synthesis task.
    schema_policy = load_schema_policy(vault)

    bundle_dir = config.wiki_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Construct metrics before acquiring a descriptor: validation may raise.
    metrics = MetricsCollector(vault)
    metrics.start_run()
    # Keep the lexical path for cleanup even if a child is later replaced.
    lock_file = config.lock_file
    # ── Acquire lock ───────────────────────────────────────────────────
    if not acquire_lock(lock_file):
        result.errors.append("lock: another compilation is running")
        return result

    registry: VaultRegistry | None = None
    pipeline_worker_id = f"pipeline:{uuid4().hex}"
    heartbeat: asyncio.Task | None = None

    try:
        registry = VaultRegistry(config.llmwiki_dir)
        # The exclusive compile lock proves every previous pipeline owner stopped.
        registry.release_abandoned_pipeline_workers()
        heartbeat = asyncio.create_task(_maintain_pipeline_leases(
            registry, pipeline_worker_id, result.errors,
        ))
        versions = _artifact_versions_for_config(config, schema_policy)

        registry.migrate_legacy(
            ArtifactVersions("legacy", "legacy", "legacy", "legacy")
        )
        live_filenames = set(sources.keys())
        indexed_filenames = registry.source_filenames()
        equivalences = load_equivalences(config.llmwiki_dir / "equivalences.json")
        checkpoint = _pipeline_checkpoint(config, versions, equivalences)
        if (
            not force
            and not refresh_render
            and indexed_filenames == live_filenames
            and trusted_unchanged == live_filenames
            and registry.get_pipeline_checkpoint() == checkpoint
            and all(registry.synthesis_versions_current(f, versions) for f in live_filenames)
        ):
            result.skipped = len(live_filenames)
            return result
        # Invalidate before any work: interruption or partial failure must never
        # leave an apparently complete checkpoint for the next invocation.
        registry.set_pipeline_checkpoint("")
        state = read_state(config.state_file)
        state_filenames = set(state.sources) | indexed_filenames
        deleted_filenames = state_filenames - live_filenames
        # Compare the entire deletion set against surviving owners once. Per-
        # source checks both cost O(deleted * corpus) and miss shared concepts
        # whose owners are all deleted in the same transaction.
        surviving_concepts = {
            slug for filename, entry in state.sources.items()
            if filename not in deleted_filenames for slug in entry.concepts
        }
        orphaned_slugs = {
            slug for filename, entry in state.sources.items()
            if filename in deleted_filenames for slug in entry.concepts
        } - surviving_concepts
        protected_stale_paths = {f"concepts/{slug}.md" for slug in orphaned_slugs}

        # ── Load cached syntheses for unchanged sources ────────────────
        cached_syntheses = registry.load_syntheses(live_filenames)
        # Render-time bilingual normalization used to leave cache/state with
        # the old identity while pages used the new slug. Migrate cached items
        # before they participate in dedup, prompts, or state updates.
        for filename, synthesis in cached_syntheses.items():
            identity_changed = _normalize_and_assign_synthesis_identity(synthesis, registry)
            # Identity-only repairs cannot certify output from an older prompt,
            # model, or schema. Keep its old version until synthesis succeeds.
            if identity_changed and registry.synthesis_versions_current(filename, versions):
                save_synthesis(
                    synthesis,
                    config.llmwiki_dir,
                    filename,
                    artifact_versions=versions.as_dict(),
                )
                registry.update_synthesis(filename, synthesis, versions)

        # ── Classify sources: compile vs reuse cache ───────────────────
        to_compile: dict[str, SourceDoc] = {}
        # Retain every live source's last committed synthesis until a replacement
        # succeeds. Forced/version-triggered retries and validation failures are
        # not source deletions and must never erase previously published notes.
        all_syntheses_by_file: dict[str, SourceSynthesis] = dict(cached_syntheses)
        changed_sources: dict[str, tuple[str, SourceSynthesis | None]] = {}
        skipped_filenames: set[str] = set()
        metadata_updated_filenames: set[str] = set()
        # Source Markdown is pipeline input, not a derived artifact. Never
        # rewrite an existing source page during rendering: doing so can erase
        # a concurrent human edit and makes the observed pre-read signature
        # diverge from the file solely because the renderer touched it.
        preserved_source_filenames: set[str] = {
            filename
            for filename in live_filenames
            if source_file_path(config.sources_dir, filename).is_file()
        }

        for filename, source in sources.items():
            cache = cached_syntheses.get(filename)
            if filename in trusted_unchanged:
                if (
                    not force
                    and cache is not None
                    and registry.synthesis_versions_current(filename, versions)
                ):
                    all_syntheses_by_file[filename] = cache
                    result.skipped += 1
                    skipped_filenames.add(filename)
                    preserved_source_filenames.add(filename)
                    continue
                from obsidian_llm_wiki.ingest.sources import load_source_file

                loaded_source = load_source_file(
                    source_file_path(config.sources_dir, filename)
                )
                trusted_unchanged.discard(filename)
                if loaded_source is None:
                    result.errors.append(f"source:{filename}: empty or unreadable")
                    continue
                source = loaded_source
                sources[filename] = loaded_source

            input_error = _source_input_error(source, config)
            if input_error:
                result.errors.append(f"source:{filename}: {input_error}")
                continue

            current_hash = hash_content(source.content)
            prev = state.sources.get(filename)
            cache = cached_syntheses.get(filename)

            unchanged = (
                not force
                and cache is not None
                and registry.synthesis_is_current(filename, source.content, versions)
            )
            if unchanged and cache is not None:
                if cache.source_title != source.title:
                    cache.source_title = source.title
                    metadata_updated_filenames.add(filename)
                all_syntheses_by_file[filename] = cache
                result.skipped += 1
                skipped_filenames.add(filename)
                registry.refresh_source_metadata(filename, source, versions)
                if _normalize_and_assign_synthesis_identity(cache, registry):
                    save_synthesis(
                        cache,
                        config.llmwiki_dir,
                        filename,
                        artifact_versions=versions.as_dict(),
                    )
                    registry.update_synthesis(filename, cache, versions)
            else:
                to_compile[filename] = source
                if prev is not None and prev.hash != current_hash:
                    changed_sources[filename] = (prev.hash, cache)

        if not to_compile and not deleted_filenames:
            logger.info("All sources unchanged — no synthesis work required.")

        request_limiter = AsyncRequestLimiter(
            config.compile_concurrency,
            min_interval_seconds=max(0, config.request_min_interval_ms) / 1_000,
        )
        embedding_request_gate = ThreadsafeRequestGate(
            request_limiter, asyncio.get_running_loop()
        )
        candidate_embedder_fn: Callable[[str], list[float] | None] | None = None
        if config.embeddings_enabled:
            from obsidian_llm_wiki.synth.embedding import embed_text

            def _candidate_embedder(text: str) -> list[float] | None:
                return embed_text(
                    text,
                    enabled=True,
                    model=config.embedding_model,
                    host=config.embedding_host,
                    request_gate=embedding_request_gate,
                )

            candidate_embedder_fn = _candidate_embedder

        retriever = ConceptRetriever(
            registry,
            candidate_limit=config.candidate_limit,
            candidate_char_budget=config.candidate_char_budget,
            ann_candidate_pool=config.ann_candidate_pool,
            embedding_model=config.embedding_model if config.embeddings_enabled else "",
        )
        synthesis_queue_ids: dict[str, int] = {}
        for filename, source in to_compile.items():
            synthesis_queue_ids[filename] = registry.enqueue_dirty(
                "source", filename, "synthesis_retry",
                {"versions": versions.as_dict(), "input_hash": hash_content(source.content)},
                reset_retry=force,
            )

        if to_compile:
            logger.info("Synthesising %d source(s)...", len(to_compile))

            # A semaphore alone does not provide queue backpressure: creating
            # one task per source still lets hundreds of prompt-building
            # coroutines pile up behind it.  Keep only server-capacity-sized
            # source jobs alive at once; each job shares the request limiter
            # above for its normal call or lossless chunk calls.
            pending_sources = iter(to_compile.items())
            results_by_filename: dict[str, SourceSynthesis | None | Exception] = {}
            completed_sources = 0
            completion_lock = asyncio.Lock()

            async def _worker() -> None:
                nonlocal completed_sources
                while True:
                    try:
                        filename, source = next(pending_sources)
                    except StopIteration:
                        return
                    try:
                        if not registry.start_dirty(
                            synthesis_queue_ids[filename], worker_id=pipeline_worker_id
                        ):
                            raise RuntimeError(
                                "queue: retry is not due or a live worker owns the claim; "
                                "retry after backoff or use --force to reset pending retries"
                            )
                        query_vector = None
                        if candidate_embedder_fn is not None:
                            query_vector = await asyncio.to_thread(
                                candidate_embedder_fn,
                                ConceptRetriever.query_text(source.title, source.content),
                            )
                        retrieval = retriever.retrieve(
                            source.title,
                            source.content,
                            query_vector=query_vector,
                        )
                        results_by_filename[filename] = await _synthesize_with_retry(
                            config,
                            filename,
                            source,
                            retrieval.prompt_lines,
                            metrics=metrics,
                            schema_policy=schema_policy,
                            request_limiter=request_limiter,
                        )
                    except Exception as exc:
                        results_by_filename[filename] = exc
                    async with completion_lock:
                        completed_sources += 1
                        completed = results_by_filename[filename]
                        if isinstance(completed, SourceSynthesis):
                            _normalize_and_assign_synthesis_identity(completed, registry)
                            save_synthesis(
                                completed,
                                config.llmwiki_dir,
                                filename,
                                artifact_versions=versions.as_dict(),
                            )
                            logger.info(
                                "Synthesis %d/%d ready: '%s' (%d concepts)",
                                completed_sources,
                                len(to_compile),
                                filename,
                                len(completed.concepts),
                            )
                        elif completed is None:
                            logger.warning(
                                "Synthesis %d/%d produced no output: '%s'",
                                completed_sources,
                                len(to_compile),
                                filename,
                            )
                        else:
                            logger.error(
                                "Synthesis %d/%d failed: '%s': %s",
                                completed_sources,
                                len(to_compile),
                                filename,
                                completed,
                            )

            worker_count = min(config.compile_concurrency, len(to_compile))
            await asyncio.gather(*(_worker() for _ in range(worker_count)))
            synth_results = [results_by_filename[f] for f in to_compile]

            # ── Collect successful syntheses + cache them ──────────────
            filenames_done: list[str] = []
            for filename, res in zip(to_compile.keys(), synth_results, strict=True):
                # Failure metrics were already recorded per attempt inside
                # _synthesize_with_retry — recording again here would count
                # one failed source as 4 failures in metrics.json.
                if isinstance(res, BaseException):
                    logger.error("Synthesis failed for '%s': %s", filename, res)
                    result.errors.append(f"synth:{filename}:{res}")
                    registry.fail_dirty(
                        synthesis_queue_ids[filename],
                        str(res),
                        worker_id=pipeline_worker_id,
                    )
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
                    registry.fail_dirty(
                        synthesis_queue_ids[filename],
                        "no output (permanent failure)",
                        worker_id=pipeline_worker_id,
                    )
                    previous = changed_sources.get(filename, ("", None))[1]
                    if previous is not None:
                        all_syntheses_by_file[filename] = previous
                    continue
                assert isinstance(res, SourceSynthesis)
                _normalize_and_assign_synthesis_identity(res, registry)
                all_syntheses_by_file[filename] = res
                save_synthesis(
                    res,
                    config.llmwiki_dir,
                    filename,
                    artifact_versions=versions.as_dict(),
                )
                filenames_done.append(filename)
                result.compiled += 1
        else:
            filenames_done = []

        # Empty corpora use the same render/commit path as nonempty ones.
        # Returning early here used to bypass deletion bookkeeping and orphan
        # protection when the last source disappeared.

        # ── Merge: full corpus dedup ───────────────────────────────────
        # Canonical aggregates use filename order, not cached/new arrival order.
        all_syntheses = [
            all_syntheses_by_file[filename] for filename in sorted(all_syntheses_by_file)
        ]
        for synthesis in all_syntheses:
            _normalize_and_assign_synthesis_identity(synthesis, registry)
        bundle = merge_bundle(all_syntheses)
        result.errors.extend(bundle.errors)
        source_synthesis_fingerprints = {
            synthesis.source_file: hash_content(
                json.dumps(
                    source_synthesis_to_dict(synthesis),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            for synthesis in bundle.sources
            if synthesis.source_file
        }

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
                    request_limiter=request_limiter,
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
            "request_gate": embedding_request_gate,
        }
        # Load persisted embeddings to avoid recomputing for unchanged concepts.
        # The cache is model-keyed — switching embedding models invalidates it.
        embeddings_cache_path = config.llmwiki_dir / "embeddings.json"
        embedding_artifact_versions = {
            **versions.as_dict(),
            "embedding_model": config.embedding_model,
            "embedding_host": config.embedding_host,
        }
        embedding_version_hash = hash_content(
            json.dumps(embedding_artifact_versions, sort_keys=True)
        )[:16]
        concept_embedding_schema = f"concept-embedding-v2:{embedding_version_hash}"
        moc_embedding_schema = f"moc-centroid-v2:{embedding_version_hash}"
        cross_lingual_schema = (
            f"cross-lingual-v1:{embedding_version_hash}:"
            f"{config.cross_lingual_threshold:.6f}"
        )
        registry_has_concept_embeddings = False
        if config.embeddings_enabled:
            from obsidian_llm_wiki.synth.embedding import (
                load_embeddings_cache,
                save_embeddings_cache,
            )
            registry_has_concept_embeddings = registry.has_embeddings(
                "concept",
                model_id=config.embedding_model,
                schema_version=concept_embedding_schema,
            )
            if not registry_has_concept_embeddings:
                embeddings_cache = load_embeddings_cache(
                    embeddings_cache_path,
                    model=config.embedding_model,
                    artifact_versions=embedding_artifact_versions,
                )
        # Build the set of new/changed concept slugs for incremental semantic
        # work. An empty set is intentionally a no-op, never a full-pair scan.
        new_slugs: set[str] = set()
        for f in filenames_done:
            synth = all_syntheses_by_file.get(f)
            if synth:
                new_slugs.update(c.slug for c in synth.concepts)
        new_slugs.update(resynthesized)
        semantic_work_slugs = set(new_slugs)
        if config.embeddings_enabled:
            concepts_by_id = {
                concept.canonical_id: concept
                for concept in bundle.concepts
                if concept.canonical_id
            }
            expected_hashes = {
                canonical_id: hash_content(f"{concept.title}\n{concept.summary}")
                for canonical_id, concept in concepts_by_id.items()
            }
            current_count = registry.embedding_count(
                "concept",
                model_id=config.embedding_model,
                schema_version=concept_embedding_schema,
            )
            ids_to_validate = (
                set(expected_hashes)
                if current_count != len(expected_hashes)
                else {
                    canonical_id
                    for canonical_id, concept in concepts_by_id.items()
                    if concept.slug in new_slugs
                }
            )
            current_hashes = registry.embedding_input_hashes(
                "concept",
                model_id=config.embedding_model,
                schema_version=concept_embedding_schema,
                entity_ids=ids_to_validate,
            )
            semantic_work_slugs.update(
                concept.slug
                for canonical_id, concept in concepts_by_id.items()
                if canonical_id in ids_to_validate
                and current_hashes.get(canonical_id) != expected_hashes[canonical_id]
            )
        concept_candidates: dict[str, list[str]] = {
            slug: [] for slug in semantic_work_slugs
        }
        moc_candidates: dict[str, list[str]] = {
            slug: [] for slug in semantic_work_slugs
        }
        try:
            if config.embeddings_enabled:
                from obsidian_llm_wiki.synth.dedupe import ensure_concept_embeddings

                cached_input_hashes = getattr(embeddings_cache, "input_hashes", None)
                if isinstance(cached_input_hashes, dict):
                    versioned_inputs = bool(
                        getattr(embeddings_cache, "versioned_inputs", False)
                    )
                    concepts_by_slug = {
                        concept.slug: concept for concept in bundle.concepts
                    }
                    validation_slugs = semantic_work_slugs
                    for slug in validation_slugs:
                        concept = concepts_by_slug.get(slug)
                        expected_hash = (
                            hash_content(f"{concept.title}\n{concept.summary}")
                            if concept is not None
                            else ""
                        )
                        if (
                            not versioned_inputs
                            or cached_input_hashes.get(slug) != expected_hash
                        ):
                            embeddings_cache.pop(slug, None)
                concepts_to_embed = [
                    concept
                    for concept in bundle.concepts
                    if concept.slug in semantic_work_slugs
                ]
                await asyncio.to_thread(
                    ensure_concept_embeddings,
                    concepts_to_embed,
                    embeddings_cache,
                    embedding_options,
                )
                concept_candidates = _ann_concept_candidates(
                    registry,
                    bundle,
                    embeddings_cache,
                    semantic_work_slugs,
                    config,
                    index_slugs=semantic_work_slugs,
                    schema_version=concept_embedding_schema,
                )
            await asyncio.to_thread(
                semantic_dedupe_concepts,
                bundle,
                threshold=config.similarity_dedup_threshold,
                embeddings_cache=embeddings_cache,
                embedding_options=embedding_options,
                new_slugs=semantic_work_slugs,
                candidate_slugs_by_slug=concept_candidates,
            )
        except Exception as exc:
            logger.warning("Semantic dedup failed (continuing without): %s", exc)
        try:
            if config.embeddings_enabled:
                moc_candidates = _ann_moc_candidates(
                    registry,
                    bundle,
                    embeddings_cache,
                    semantic_work_slugs,
                    config,
                    schema_version=moc_embedding_schema,
                )
            await asyncio.to_thread(
                assign_orphans_to_mocs,
                bundle,
                threshold=config.moc_assignment_threshold,
                embeddings_cache=embeddings_cache,
                embedding_options=embedding_options,
                candidate_mocs_by_slug=moc_candidates,
            )
        except Exception as exc:
            logger.warning("MoC orphan assignment failed (continuing without): %s", exc)

        # Semantic dedup remaps source-local concept slugs in memory. Track and
        # persist only source syntheses that actually changed; writing every
        # portable cache on every incremental run recreates the v1 full-corpus
        # I/O problem, while failing to update SQLite resurrects dedup victims.
        canonicalized_source_files: set[str] = set()
        for synthesis in bundle.sources:
            if not synthesis.source_file:
                continue
            current_fingerprint = hash_content(
                json.dumps(
                    source_synthesis_to_dict(synthesis),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if source_synthesis_fingerprints.get(synthesis.source_file) == current_fingerprint:
                continue
            canonicalized_source_files.add(synthesis.source_file)
            save_synthesis(
                synthesis,
                config.llmwiki_dir,
                synthesis.source_file,
                artifact_versions=versions.as_dict(),
            )

        cross_lingual_links: dict[str, list[tuple[str, float, str]]] = {}
        live_concept_slugs = {concept.slug for concept in bundle.concepts}
        cross_lingual_work_slugs = set(semantic_work_slugs)
        cross_lingual_full_backfill = False
        if config.embeddings_enabled and not registry.cross_lingual_namespace_complete(
            model_id=config.embedding_model,
            schema_version=cross_lingual_schema,
        ):
            cross_lingual_work_slugs = set(live_concept_slugs)
            cross_lingual_full_backfill = True
        try:
            from obsidian_llm_wiki.synth.embedding import find_cross_lingual_links

            cross_lingual_candidates = concept_candidates
            if config.embeddings_enabled and cross_lingual_work_slugs != semantic_work_slugs:
                concept_by_slug = {
                    concept.slug: concept for concept in bundle.concepts
                }
                _load_registry_concept_vectors(
                    registry,
                    concept_by_slug,
                    cross_lingual_work_slugs,
                    embeddings_cache,
                    config.embedding_model,
                )
                cross_lingual_candidates = _ann_concept_candidates(
                    registry,
                    bundle,
                    embeddings_cache,
                    cross_lingual_work_slugs,
                    config,
                    index_slugs=set(),
                    schema_version=concept_embedding_schema,
                )
            computed_cross_lingual_links = await asyncio.to_thread(
                find_cross_lingual_links,
                bundle.concepts,
                enabled=config.embeddings_enabled,
                model=config.embedding_model,
                host=config.embedding_host,
                request_gate=embedding_request_gate,
                threshold=config.cross_lingual_threshold,
                embeddings_cache=embeddings_cache,
                comparison_slugs=cross_lingual_work_slugs,
                candidate_slugs_by_slug=cross_lingual_candidates,
            )
            if config.embeddings_enabled:
                registry.replace_cross_lingual_links(
                    cross_lingual_work_slugs,
                    computed_cross_lingual_links,
                    model_id=config.embedding_model,
                    schema_version=cross_lingual_schema,
                    namespace_complete=cross_lingual_full_backfill,
                )
                cross_lingual_links = registry.load_cross_lingual_links(
                    model_id=config.embedding_model,
                    schema_version=cross_lingual_schema,
                    live_slugs=live_concept_slugs,
                )
            else:
                cross_lingual_links = computed_cross_lingual_links
        except Exception as exc:
            logger.warning("Cross-lingual linking failed (continuing without): %s", exc)
            if config.embeddings_enabled:
                cross_lingual_links = registry.load_cross_lingual_links(
                    model_id=config.embedding_model,
                    schema_version=cross_lingual_schema,
                    live_slugs=live_concept_slugs,
                )

        # Persist embeddings for reuse on the next run.
        persisted_embedding_count = (
            len(embeddings_cache) if config.embeddings_enabled else 0
        )
        if (
            config.embeddings_enabled
            and embeddings_cache
            and not registry_has_concept_embeddings
            and len(bundle.concepts) <= 5_000
        ):
            # Prune stale entries: only keep embeddings for concepts that
            # survived dedup (are in the final bundle).
            live_slugs = {c.slug for c in bundle.concepts}
            pruned_cache = {
                slug: vec for slug, vec in embeddings_cache.items()
                if slug in live_slugs
            }
            save_embeddings_cache(
                embeddings_cache_path,
                pruned_cache,
                model=config.embedding_model,
                input_hashes={
                    concept.slug: hash_content(
                        f"{concept.title}\n{concept.summary}"
                    )
                    for concept in bundle.concepts
                },
                artifact_versions=embedding_artifact_versions,
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
        confirmed_pairs, equivalence_errors = apply_equivalences(bundle, equivalences)
        result.errors.extend(equivalence_errors)
        by_slug = {concept.slug: concept for concept in bundle.concepts}
        for left, right in confirmed_pairs:
            # 1.0 here marks a human assertion, not a measured cosine score.
            for source_slug, target_slug in ((left, right), (right, left)):
                targets = cross_lingual_links.setdefault(source_slug, [])
                if not any(target == target_slug for target, _, _ in targets):
                    targets.append((target_slug, 1.0, by_slug[target_slug].title))
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
            render_registry=registry,
            force_render=force or refresh_render,
            preserve_source_filenames=preserved_source_filenames,
            protected_stale_paths=protected_stale_paths,
            confirmed_cross_lingual_pairs=confirmed_pairs,
        )
        result.pages.extend(written)
        result.concepts = bundle.concepts

        # Deletions are destructive registry/cache mutations. Apply them only
        # after rendering has committed so a late render failure leaves the
        # complete pre-run source state available for a clean retry.
        orphan_reason = "Source owners deleted: " + ", ".join(sorted(deleted_filenames))[:512]
        for slug in sorted(orphaned_slugs):
            orphan_page(config.concepts_dir, slug, orphan_reason)
        for filename in deleted_filenames:
            delete_cached_synthesis(config.llmwiki_dir, filename)
            registry.delete_source(filename)
            remove_source_state(state, filename)
            result.deleted += 1

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

        # Rendering is the commit boundary: only now may a source/version be
        # considered reusable by the next incremental run.
        source_syntheses_after_normalization = {
            synthesis.source_file: synthesis
            for synthesis in bundle.sources
            if synthesis.source_file
        }
        persisted_source_files = (
            set(filenames_done) | canonicalized_source_files | metadata_updated_filenames
        )
        for filename in sorted(persisted_source_files):
            synthesis = source_syntheses_after_normalization.get(filename)
            if synthesis is not None:
                if filename in filenames_done:
                    source = sources.get(filename) or _source_from_synthesis(synthesis)
                    if not registry.complete_dirty(
                        synthesis_queue_ids[filename], worker_id=pipeline_worker_id
                    ):
                        raise RuntimeError(f"Lost synthesis lease before commit: {filename}")
                    registry.upsert_synthesis(
                        filename,
                        source,
                        synthesis,
                        versions,
                    )

                else:
                    registry.update_synthesis(filename, synthesis, versions)
                if filename in metadata_updated_filenames:
                    save_synthesis(
                        synthesis, config.llmwiki_dir, filename,
                        artifact_versions=versions.as_dict(),
                    )
        registry.prune_cross_lingual_links()

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
        for filename in sorted(canonicalized_source_files - set(filenames_done)):
            synthesis = source_syntheses_after_normalization.get(filename)
            source = sources.get(filename)
            previous_state = state.sources.get(filename)
            if synthesis is None or (source is None and previous_state is None):
                continue
            source_hash = (
                hash_content(source.content)
                if source is not None
                else previous_state.hash if previous_state is not None else ""
            )
            update_source_state(
                state,
                filename,
                source_hash,
                [concept.slug for concept in synthesis.concepts],
                compiled_at,
            )

        write_state(config.state_file, state)

        verified_source_files = (
            skipped_filenames | set(filenames_done) | canonicalized_source_files
        )
        for filename in sorted(verified_source_files & set(observed_signatures)):
            mtime_ns, size_bytes = observed_signatures[filename]
            registry.record_source_file_state(
                filename,
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
            )

        logger.info(
            "Done: %d compiled, %d skipped, %d deleted, %d concepts, %d MOCs, %d errors",
            result.compiled, result.skipped, result.deleted,
            len(bundle.concepts), len(bundle.maps), len(result.errors),
        )
        if not result.errors:
            registry.set_pipeline_checkpoint(checkpoint)
        return result

    finally:
        # Cleanup is unconditional even if metrics persistence fails. Release
        # unfinished claims so a normal exception/cancellation can retry now,
        # rather than waiting for a lease that no worker still owns.
        try:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            metrics.finish_run()
            metrics.save()
        finally:
            try:
                if registry is not None:
                    try:
                        registry.release_dirty_worker(pipeline_worker_id)
                    finally:
                        registry.close()
            finally:
                release_lock(lock_file)


async def _synthesize_source(
    config: Config,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
    *,
    schema_policy: SchemaPolicy = DEFAULT_SCHEMA_POLICY,
    request_limiter: AsyncRequestLimiter | None = None,
) -> SourceSynthesis | None:
    """Call the LLM to synthesise one source into a SourceSynthesis.

    Dispatches to the two-pass quality synthesis when
    ``config.synthesis_mode == "two_pass"``; otherwise uses the default
    single-pass synthesis.

    Preserve source language unless the user explicitly requests translation
    through config.output_language.
    """
    # Detect language and choose detail once per source — used by both single
    # and two-pass paths, while the vault-level policy is shared across sources.
    source_lang = config.output_language or _detect_source_language(source.content, filename)
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
            request_limiter=request_limiter,
        )
        if synth is not None:
            # Source-frontmatter title is canonical. Letting a model-provided
            # title win creates duplicated/hallucinated entry filenames on a
            # retry even when extraction already established the source title.
            synth.source_title = source.title
            synth.source_file = filename
            if source_lang:
                synth.language = source_lang
            resolve_synthesis_evidence(synth, source, filename)
        return synth

    # ── Chunking for large sources ─────────────────────────────────────
    # Sources exceeding chunk_size are split and every chunk is synthesized.
    # This preserves the full document instead of treating a context-window
    # error as permission to silently drop its tail.
    chunk_size = getattr(config, "chunk_size", 30_000)
    content_len = len(source.content)

    if content_len > chunk_size:
        from obsidian_llm_wiki.synth.quality import _bounded_map, chunk_content, merge_skeletons

        chunks = chunk_content(source.content, chunk_size)
        logger.info(
            "Chunking '%s': %d chars → %d chunks (%d chars each)",
            filename, content_len, len(chunks), chunk_size,
        )

        async def _synth_chunk_once(chunk_content_str: str) -> SourceSynthesis | None:
            chunk_source = replace(source, content=chunk_content_str)
            chunk_prompt = build_synthesis_prompt(
                chunk_source.title,
                chunk_source.content,
                existing_concepts=existing_concepts,
                language=source_lang,
                schema_policy=schema_policy,
                granularity=granularity,
            )
            chunk_messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": chunk_prompt},
            ]
            chunk_response = await _call_structured_synthesis(
                chunk_prompt,
                chunk_messages,
                config,
                request_limiter=request_limiter,
            )
            parsed = parse_single_source_synthesis(chunk_response)
            if parsed is not None:
                validate_synthesis(parsed, chunk_source, filename)
            return parsed

        async def _synth_chunk(item: tuple[int, str]):
            index, chunk = item
            return await _retry_chunk_synthesis(
                lambda: _synth_chunk_once(chunk), filename, index, len(chunks),
            )

        chunk_synths = await _bounded_map(
            list(enumerate(chunks, start=1)), _synth_chunk,
            max_workers=config.compile_concurrency,
        )
        failures = [s for s in chunk_synths if s is None or isinstance(s, BaseException)]
        if failures:
            raise RuntimeError(
                f"chunked synthesis incomplete for '{filename}': "
                f"{len(failures)} of {len(chunks)} chunks failed"
            )
        valid_synths = [s for s in chunk_synths if isinstance(s, SourceSynthesis)]

        if len(valid_synths) == 1:
            synthesis = valid_synths[0]
        else:
            # Merge skeletons: union concepts by slug, union MoCs by slug,
            # dedup key_points/open_questions
            merged = merge_skeletons(valid_synths)
            synthesis = merged

        if synthesis is not None:
            synthesis.source_tags = dedupe_tags(synthesis.source_tags)
            synthesis.key_points = dedupe_statements(synthesis.key_points)
            synthesis.open_questions = dedupe_statements(synthesis.open_questions)
            synthesis.source_title = source.title
            synthesis.source_file = filename
            if source_lang:
                synthesis.language = source_lang
            resolve_synthesis_evidence(synthesis, source, filename)
        return synthesis

    # ── Single-call path for normal-sized sources ──────────────────────
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
        response = await _call_structured_synthesis(
            prompt,
            messages,
            config,
            request_limiter=request_limiter,
        )
    except Exception as exc:
        logger.error("LLM call failed for '%s': %s", filename, exc)
        raise

    synthesis = parse_single_source_synthesis(response)
    if synthesis is None:
        logger.warning("Could not parse synthesis JSON for '%s'", filename)
        return None

    # Source-frontmatter title is canonical; synthesis may summarize it but
    # must not rename the source identity used for entries and filenames.
    synthesis.source_title = source.title
    synthesis.source_file = filename

    if source_lang:
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
    request_limiter: AsyncRequestLimiter | None = None,
) -> SourceSynthesis | None:
    """Synthesize one source without dropping source content on failure.

    Context safety is handled by lossless chunking in ``_synthesize_source``.
    A parser failure retries the *same* complete source once because model
    sampling can still produce an invalid semantic payload despite JSON mode.
    """
    validation_failure = ""
    for attempt in range(_SYNTHESIS_PARSE_ATTEMPTS):

        synth_start = time.monotonic()
        try:
            synth = await _synthesize_source(
                config,
                filename,
                source,
                existing_concepts,
                schema_policy=schema_policy,
                request_limiter=request_limiter,
            )
        except Exception as exc:
            synth_time = time.monotonic() - synth_start
            logger.error(
                "Synthesis attempt %d failed for '%s': %s",
                attempt + 1, filename, exc,
            )
            if metrics:
                metrics.record_synthesis(
                    source_file=filename,
                    pass1_time=synth_time,
                    success=False,
                    error_type=type(exc).__name__,
                )
            raise

        synth_time = time.monotonic() - synth_start

        if synth is not None:
            try:
                validate_synthesis(synth, source, filename)
            except SynthesisValidationError as exc:
                validation_failure = str(exc)
                logger.warning("Rejected synthesis for '%s': %s", filename, exc)
                synth = None

        if synth is not None:
            if metrics:
                metrics.record_synthesis(
                    source_file=filename,
                    pass1_time=synth_time,
                    concepts_extracted=len(synth.concepts),
                    success=True,
                )
            return synth

        # synth is None — retry the same full source once, never a prefix.
        if metrics:
            metrics.record_synthesis(
                source_file=filename,
                pass1_time=synth_time,
                success=False,
                error_type="invalid_synthesis" if validation_failure else "no_output",
            )
        logger.warning(
            "Synthesis produced no output for '%s' on attempt %d of %d",
            filename, attempt + 1, _SYNTHESIS_PARSE_ATTEMPTS,
        )

    # All complete-source attempts exhausted — permanent failure.
    logger.error(
        "Permanent synthesis failure for '%s': all %d complete-source attempts exhausted. "
        "Content length: %d chars.",
        filename, _SYNTHESIS_PARSE_ATTEMPTS, len(source.content),
    )
    if validation_failure:
        raise SynthesisValidationError(validation_failure)
    return None


async def recompile_single_source(
    vault_path: str | Path,
    source_file: str,
    config: Config | None = None,
) -> CompileResult:
    """Manually retry a single failed source file.

    Loads one source, retries the complete input, and caches the result.
    A subsequent build renders it; the registry checkpoint is invalidated.

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

    _preflight_pipeline_paths(vault, config)

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

    input_error = _source_input_error(source, config)
    if input_error:
        result.errors.append(f"source:{source_file}: {input_error}")
        return result

    metrics = MetricsCollector(vault)
    metrics.start_run()
    lock_file = config.lock_file
    # Acquire lock
    if not acquire_lock(lock_file):
        result.errors.append("lock: another compilation is running")
        return result

    try:
        state = read_state(config.state_file)
        versions = _artifact_versions_for_config(config, schema_policy)
        with VaultRegistry(config.llmwiki_dir) as registry:
            registry.migrate_legacy(
                ArtifactVersions("legacy", "legacy", "legacy", "legacy")
            )
            request_limiter = AsyncRequestLimiter(
                1,
                min_interval_seconds=max(0, config.request_min_interval_ms) / 1_000,
            )
            embedding_request_gate = ThreadsafeRequestGate(
                request_limiter, asyncio.get_running_loop()
            )
            candidate_embedder_fn: Callable[[str], list[float] | None] | None = None
            if config.embeddings_enabled:
                from obsidian_llm_wiki.synth.embedding import embed_text

                def _candidate_embedder(text: str) -> list[float] | None:
                    return embed_text(
                        text,
                        enabled=True,
                        model=config.embedding_model,
                        host=config.embedding_host,
                        request_gate=embedding_request_gate,
                    )

                candidate_embedder_fn = _candidate_embedder

            retriever = ConceptRetriever(
                registry,
                candidate_limit=config.candidate_limit,
                candidate_char_budget=config.candidate_char_budget,
                ann_candidate_pool=config.ann_candidate_pool,
                embedding_model=(
                    config.embedding_model if config.embeddings_enabled else ""
                ),
            )
            query_vector = None
            if candidate_embedder_fn is not None:
                query_vector = await asyncio.to_thread(
                    candidate_embedder_fn,
                    ConceptRetriever.query_text(source.title, source.content),
                )
            retrieval = retriever.retrieve(
                source.title,
                source.content,
                query_vector=query_vector,
            )
            synth = await _synthesize_with_retry(
                config,
                source_file,
                source,
                retrieval.prompt_lines,
                metrics=metrics,
                schema_policy=schema_policy,
                request_limiter=request_limiter,
            )

            if synth is None:
                result.errors.append(
                    f"synth:{source_file}: failure after complete-source retries"
                )
                return result

            _normalize_and_assign_synthesis_identity(synth, registry)
            save_synthesis(
                synth,
                config.llmwiki_dir,
                source_file,
                artifact_versions=versions.as_dict(),
            )
            registry.upsert_synthesis(source_file, source, synth, versions)
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
        try:
            metrics.finish_run()
            metrics.save()
        finally:
            release_lock(lock_file)


def _pipeline_checkpoint(
    config: Config, versions: ArtifactVersions,
    equivalences: tuple[tuple[str, str], ...] = (),
) -> str:
    """Fingerprint all non-secret build settings, including derived views."""
    settings = asdict(config)
    settings.pop("_vault", None)
    settings.pop("llm", None)  # effective synthesis model is in artifact versions
    for key in list(settings):
        if any(word in key for word in ("key", "secret", "proxy", "cookies")):
            settings.pop(key)
    return hash_content(json.dumps(
        {"pipeline": "integrity-v2", "versions": versions.as_dict(),
         "settings": settings, "equivalences": equivalences},
        sort_keys=True, default=str,
    ))


def _source_input_error(source: SourceDoc, config: Config) -> str:
    """Reject known incomplete input rather than publish a complete-looking prefix."""
    size = len(source.content.encode("utf-8"))
    if size < config.min_source_chars:
        return f"too short ({size} < {config.min_source_chars} bytes)"
    if size > config.max_source_chars or any(
        "truncated" in diagnostic.casefold() for diagnostic in source.provenance.diagnostics
    ):
        return (
            "incomplete extraction: source exceeds the byte cap or was truncated; "
            "increase MAX_SOURCE_CHARS if appropriate and re-extract the complete source"
        )
    return ""


def _detect_source_language(content: str, filename: str) -> str:
    """Detect the primary language of source content."""
    try:
        from obsidian_llm_wiki.synth.language import detect_language
        lang = detect_language(content)
        logger.debug("Detected language '%s' for '%s'", lang, filename)
        return lang
    except Exception:
        return ""


def _normalize_and_assign_synthesis_identity(
    synthesis: SourceSynthesis,
    registry: VaultRegistry,
) -> bool:
    """Normalize identities and report only durable, post-canonical changes."""
    validate_local_identities(synthesis)
    before = source_synthesis_to_dict(synthesis)
    _normalize_synthesis_identity(synthesis)
    registry.assign_canonical_ids(synthesis)
    return source_synthesis_to_dict(synthesis) != before


def _normalize_synthesis_identity(synthesis: SourceSynthesis) -> bool:
    """Normalize safe identities and dedupe model-returned duplicates."""
    from obsidian_llm_wiki.render.bilingual import normalize_bilingual_titles_and_slugs
    from obsidian_llm_wiki.synth.dedupe import merge_concepts

    before = (
        synthesis.source_title,
        tuple((concept.title, concept.slug) for concept in synthesis.concepts),
        tuple((moc.title, moc.slug, tuple(moc.concept_slugs)) for moc in synthesis.maps),
    )
    bundle = SynthesisBundle(
        sources=[synthesis], concepts=synthesis.concepts, maps=synthesis.maps,
    )
    normalize_bilingual_titles_and_slugs(bundle)
    synthesis.concepts = merge_concepts(synthesis.concepts)
    for moc in synthesis.maps:
        moc.concept_slugs = list(dict.fromkeys(moc.concept_slugs))
    after = (
        synthesis.source_title,
        tuple((concept.title, concept.slug) for concept in synthesis.concepts),
        tuple((moc.title, moc.slug, tuple(moc.concept_slugs)) for moc in synthesis.maps),
    )
    return before != after


def _ann_concept_candidates(
    registry: VaultRegistry,
    bundle: SynthesisBundle,
    embeddings: dict[str, list[float]],
    comparison_slugs: set[str],
    config: Config,
    *,
    index_slugs: set[str] | None = None,
    schema_version: str = "concept-embedding-v1",
) -> dict[str, list[str]]:
    """Index changed vectors and resolve bounded semantic candidate slugs."""
    concept_by_slug = {concept.slug: concept for concept in bundle.concepts}
    id_to_slug = {
        concept.canonical_id: concept.slug
        for concept in bundle.concepts
        if concept.canonical_id
    }
    registry.upsert_embeddings(
        "concept",
        (
            (
                concept.canonical_id,
                embeddings[slug],
                hash_content(f"{concept.title}\n{concept.summary}"),
            )
            for slug in sorted(
                index_slugs if index_slugs is not None else comparison_slugs
            )
            if (concept := concept_by_slug.get(slug)) is not None
            and concept.canonical_id
            and slug in embeddings
        ),
        model_id=config.embedding_model,
        schema_version=schema_version,
    )

    searches: dict[str, AnnSearchResult] = {}
    candidate_ids: set[str] = set()
    for slug in sorted(comparison_slugs):
        vector = embeddings.get(slug)
        if vector is None:
            continue
        search = registry.search_ann(
            "concept",
            vector,
            model_id=config.embedding_model,
            limit=config.candidate_limit + 1,
            candidate_pool=config.ann_candidate_pool,
            schema_version=schema_version,
        )
        searches[slug] = search
        candidate_ids.update(candidate.entity_id for candidate in search.candidates)
    unknown_ids = [entity_id for entity_id in candidate_ids if entity_id not in id_to_slug]
    for candidate in registry.concept_candidates_by_id(unknown_ids):
        id_to_slug[candidate.id] = candidate.slug
    for entity_id, vector in registry.load_embeddings(
        "concept", candidate_ids, model_id=config.embedding_model
    ).items():
        candidate_slug = id_to_slug.get(entity_id)
        if candidate_slug is not None:
            embeddings.setdefault(candidate_slug, vector)

    candidate_map: dict[str, list[str]] = {}
    for slug in sorted(comparison_slugs):
        search = searches.get(slug)
        if search is None:
            candidate_map[slug] = []
            continue
        candidate_map[slug] = [
            candidate_slug
            for candidate in search.candidates
            if (candidate_slug := id_to_slug.get(candidate.entity_id)) is not None
            and candidate_slug != slug
        ][:config.candidate_limit]
    return candidate_map


def _load_registry_concept_vectors(
    registry: VaultRegistry,
    concept_by_slug: dict[str, ConceptNote],
    slugs: set[str],
    embeddings: dict[str, list[float]],
    model_id: str,
) -> None:
    ids_to_slugs = {
        concept.canonical_id: slug
        for slug in slugs
        if (concept := concept_by_slug.get(slug)) is not None
        and concept.canonical_id
        and slug not in embeddings
    }
    for entity_id, vector in registry.load_embeddings(
        "concept", ids_to_slugs, model_id=model_id
    ).items():
        slug = ids_to_slugs.get(entity_id)
        if slug is not None:
            embeddings.setdefault(slug, vector)


def _ann_moc_candidates(
    registry: VaultRegistry,
    bundle: SynthesisBundle,
    embeddings: dict[str, list[float]],
    comparison_slugs: set[str],
    config: Config,
    *,
    schema_version: str = "moc-centroid-v1",
) -> dict[str, list[str]]:
    """Update changed MoC centroids and retrieve bounded orphan candidates."""
    concept_by_slug = {concept.slug: concept for concept in bundle.concepts}
    affected_mocs = {
        moc.slug
        for moc in bundle.maps
        if any(slug in comparison_slugs for slug in moc.concept_slugs)
    }
    expected_hashes = {
        moc.slug: hash_content("\x1f".join(sorted(moc.concept_slugs)))
        for moc in bundle.maps
    }
    current_count = registry.embedding_count(
        "moc",
        model_id=config.embedding_model,
        schema_version=schema_version,
    )
    mocs_to_validate = (
        set(expected_hashes)
        if current_count != len(expected_hashes)
        else set(affected_mocs)
    )
    current_hashes = registry.embedding_input_hashes(
        "moc",
        model_id=config.embedding_model,
        schema_version=schema_version,
        entity_ids=mocs_to_validate,
    )
    affected_mocs.update(
        slug
        for slug in mocs_to_validate
        if current_hashes.get(slug) != expected_hashes[slug]
    )

    affected_member_slugs = {
        slug
        for moc in bundle.maps
        if moc.slug in affected_mocs
        for slug in moc.concept_slugs
    }
    _load_registry_concept_vectors(
        registry,
        concept_by_slug,
        affected_member_slugs,
        embeddings,
        config.embedding_model,
    )
    centroid_records: list[tuple[str, list[float], str]] = []
    for moc in bundle.maps:
        if moc.slug not in affected_mocs:
            continue
        vectors = [
            embeddings[slug]
            for slug in moc.concept_slugs
            if slug in concept_by_slug and slug in embeddings
        ]
        if not vectors:
            continue
        dimensions = len(vectors[0])
        compatible = [vector for vector in vectors if len(vector) == dimensions]
        if not compatible:
            continue
        centroid = [
            sum(vector[index] for vector in compatible) / len(compatible)
            for index in range(dimensions)
        ]
        centroid_records.append(
            (
                moc.slug,
                centroid,
                hash_content("\x1f".join(sorted(moc.concept_slugs))),
            )
        )
    registry.upsert_embeddings(
        "moc",
        centroid_records,
        model_id=config.embedding_model,
        schema_version=schema_version,
    )

    candidate_map: dict[str, list[str]] = {}
    for slug in sorted(comparison_slugs):
        vector = embeddings.get(slug)
        if vector is None:
            candidate_map[slug] = []
            continue
        search = registry.search_ann(
            "moc",
            vector,
            model_id=config.embedding_model,
            limit=config.candidate_limit,
            candidate_pool=config.ann_candidate_pool,
            schema_version=schema_version,
        )
        candidate_map[slug] = [candidate.entity_id for candidate in search.candidates]
    target_mocs = {
        moc_slug for candidates in candidate_map.values() for moc_slug in candidates
    }
    candidate_member_slugs = {
        member_slug
        for moc in bundle.maps
        if moc.slug in target_mocs
        for member_slug in moc.concept_slugs
    }
    _load_registry_concept_vectors(
        registry,
        concept_by_slug,
        candidate_member_slugs,
        embeddings,
        config.embedding_model,
    )
    return candidate_map


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
    *,
    request_limiter: AsyncRequestLimiter | None = None,
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
            if request_limiter is None:
                return await resynthesize_concept(
                    config, existing, combined_content, combined_title,
                )
            async with request_limiter.slot():
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
