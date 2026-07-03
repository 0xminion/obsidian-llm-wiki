"""Single pipeline orchestrator — the heart of obsidian-llm-wiki.

Replaces the legacy dual-orchestrator design (compiler.py + create/orchestrator.py)
with a single coherent flow:

  1. Load config + acquire lock
  2. Read state + detect changes
  3. Normalise sources (ingest → SourceDoc)
  4. Synthesise: ONE LLM call per source → SourceSynthesis (structured JSON)
  5. Merge: corpus-level concept/tag dedup → SynthesisBundle
  6. Render: deterministic markdown from SynthesisBundle → Obsidian vault
  7. Persist state + release lock

The LLM only produces the synthesis intermediate.  All markdown generation
is pure functions in ``render.obsidian``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.models import (
    CompileResult,
    SourceDoc,
    WikiState,
)
from obsidian_llm_wiki.core.state import (
    hash_content,
    read_state,
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
        sources: Dict mapping source filename → SourceDoc.
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

        # ── Detect changes ─────────────────────────────────────────────
        to_compile: dict[str, SourceDoc] = {}
        if force:
            to_compile = dict(sources)
        else:
            for filename, source in sources.items():
                content_hash = hash_content(source.content)
                prev = state.sources.get(filename)
                if prev is None or prev.hash != content_hash:
                    to_compile[filename] = source

        if not to_compile:
            logger.info("Nothing to compile — already up-to-date.")
            result.skipped = len(sources)
            return result

        logger.info("Synthesising %d source(s)...", len(to_compile))

        # ── Build existing concept index for dedup context ─────────────
        existing_concepts = _existing_concept_slugs(state)

        # ── Synthesise: one LLM call per source ────────────────────────
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

        # ── Collect successful syntheses ───────────────────────────────
        from obsidian_llm_wiki.core.models import SourceSynthesis

        all_syntheses: list[SourceSynthesis] = []
        filenames_done: list[str] = []

        for i, res in enumerate(synth_results):
            filename = list(to_compile.keys())[i]
            if isinstance(res, Exception):
                logger.error("Synthesis failed for '%s': %s", filename, res)
                result.errors.append(f"synth:{filename}:{res}")
                continue
            if res is None:
                logger.warning("Synthesis produced no output for '%s'", filename)
                result.errors.append(f"synth:{filename}: no output")
                continue
            all_syntheses.append(res)
            filenames_done.append(filename)

        if not all_syntheses:
            logger.error("All syntheses failed.")
            return result

        # ── Merge: corpus-level dedup ──────────────────────────────────
        bundle = merge_bundle(all_syntheses)
        result.errors.extend(bundle.errors)

        # ── Render: deterministic markdown ─────────────────────────────
        logger.info("Rendering %d concepts, %d MOCs...",
                     len(bundle.concepts), len(bundle.maps))
        written = render_vault(bundle_dir, bundle, to_compile)
        result.pages.extend(written)
        result.concepts = bundle.concepts
        result.compiled = len(filenames_done)

        # ── Update state ───────────────────────────────────────────────
        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for filename in filenames_done:
            source = to_compile[filename]
            concept_slugs = [c.slug for c in bundle.concepts
                             if any(c.slug in s.concepts or True
                                    for s in all_syntheses
                                    if s.source_title == source.title)]
            # Simpler: just track all concept slugs from this source's synthesis
            source_synth = next(
                (s for s in all_syntheses if s.source_title == source.title), None
            )
            if source_synth:
                concept_slugs = [c.slug for c in source_synth.concepts]
            update_source_state(
                state, filename,
                hash_content(source.content),
                concept_slugs,
                compiled_at,
            )

        write_state(config.state_file, state)

        logger.info(
            "Done: %d compiled, %d concepts, %d MOCs, %d errors",
            result.compiled, len(bundle.concepts), len(bundle.maps),
            len(result.errors),
        )
        return result

    finally:
        release_lock(config.lock_file)


async def _synthesize_source(
    config: Config,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
):
    """Call the LLM to synthesise one source into a SourceSynthesis."""
    from obsidian_llm_wiki.providers.llm import acall_llm

    prompt = build_synthesis_prompt(
        source.title,
        source.content,
        existing_concepts=existing_concepts,
        language=config.output_language,
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

    # Ensure source title matches.
    if not synthesis.source_title:
        synthesis.source_title = source.title

    return synthesis


def _existing_concept_slugs(state: WikiState) -> list[str]:
    """Extract all known concept slugs from state for dedup context."""
    slugs: set[str] = set()
    for src_state in state.sources.values():
        slugs.update(src_state.concepts)
    return sorted(slugs)
