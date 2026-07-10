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
from datetime import UTC, datetime
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.cache import (
    delete_cached_synthesis,
    load_all_cached_syntheses,
    save_synthesis,
)
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.models import (
    CompileResult,
    SourceDoc,
    SourceSynthesis,
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
                    "Source '%s' is %d chars — truncating to %d",
                    filename, content_len, config.max_source_chars,
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

            # ── Synthesise: one LLM call per changed source ─────────────
            sem = asyncio.Semaphore(config.compile_concurrency)

            async def _synth_one(filename: str, source: SourceDoc):
                async with sem:
                    return await _synthesize_source(
                        config, filename, source, existing_concepts
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
                    continue
                if res is None:
                    logger.warning("Synthesis produced no output for '%s'", filename)
                    result.errors.append(f"synth:{filename}: no output")
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
        # After merging cached + fresh syntheses, old concepts may not
        # have reverse links to new concepts that reference them. This
        # pass walks all edges and adds missing reverse links so MoC
        # cross-reference diagrams show bidirectional arrows correctly.
        from obsidian_llm_wiki.synth.dedupe import propagate_backlinks
        propagate_backlinks(bundle)

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
        written = render_vault(bundle_dir, bundle, all_sources_for_render)
        result.pages.extend(written)
        result.concepts = bundle.concepts

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
