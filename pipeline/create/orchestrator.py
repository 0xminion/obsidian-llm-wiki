"""Stage 2 orchestrator — compile sources into wiki pages.

Processes each ingested source: renders entry, extracts concepts, renders
concept pages, then groups into MoCs.  Uses configurable concurrency.

Ported from llm-wiki-compiler/src/compiler/index.ts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from pipeline.config import Config
from pipeline.hasher import hash_content
from pipeline.llm_client import call_llm
from pipeline.markdown import atomic_write, safe_read_file, slugify
from pipeline.models import (
    CompileResult,
    ExtractedConcept,
    IngestedSource,
    PageSummary,
    ProvenanceState,
    WikiState,
)
from pipeline.page_renderer import render_concept, render_entry, render_moc
from pipeline.prompts import EXTRACTION_TOOLS
from pipeline.state import update_source_state

logger = logging.getLogger("llmwiki.orchestrator")


# ── Concept extraction prompt (tool-calling) ────────────────────────────


_EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge extraction engine. "
    "Analyse the source document and identify 3-8 distinct, meaningful concepts. "
    "For each concept, provide:\n"
    "- title: A concise, descriptive title (3-8 words).\n"
    "- summary: A 1-2 sentence summary capturing the essential meaning.\n"
    "- is_new: Whether this concept does NOT appear in the existing index below.\n"
    "- tags: 2-4 lowercase categorical tags.\n"
    "- confidence: Confidence in this extraction (0.0 - 1.0).\n"
    "- provenance_state: One of: extracted, merged, inferred, ambiguous.\n"
    "Review the existing index below to avoid extracting duplicate concepts. "
    "Prioritise genuinely novel concepts."
)


async def run_create(
    config: Config,
    sources: dict[str, IngestedSource],
    state: WikiState | None = None,
) -> CompileResult:
    """Run the Stage 2 compilation pipeline.

    For each source:
      1. Render entry note via LLM → write to entries_dir.
      2. Extract concepts via LLM tool-calling.
      3. Render each concept page (in parallel) → write to concepts_dir.
    Then:
      4. Group concepts by tags → determine MoC topics.
      5. Render each MoC → write to mocs_dir.

    Args:
        config: Pipeline configuration.
        sources: Dict mapping source filename → IngestedSource.
        state: Optional persisted state for incremental compilation.

    Returns:
        CompileResult with counts of compiled/skipped/errored items.
    """
    if state is None:
        state = WikiState()

    config.entries_dir.mkdir(parents=True, exist_ok=True)
    config.concepts_dir.mkdir(parents=True, exist_ok=True)
    config.mocs_dir.mkdir(parents=True, exist_ok=True)

    result = CompileResult()
    all_concept_summaries: list[PageSummary] = []
    tag_map: dict[str, list[PageSummary]] = {}

    # Build existing concept index for dedup
    existing_index = _build_existing_index(state)

    sem = asyncio.Semaphore(config.compile_concurrency)

    # ── Phase 1: Process each source ───────────────────────────────────
    source_filenames = list(sources.keys())

    async def _process_source(filename: str) -> None:
        async with sem:
            try:
                source = sources[filename]
                compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

                # 1. Render entry
                concept_context = _format_concept_context(existing_index)
                entry_md = await render_entry(
                    config,
                    filename,
                    source.title,
                    source.content,
                    concept_context,
                )
                entry_slug = slugify(source.title)
                entry_path = config.entries_dir / f"{entry_slug}.md"
                atomic_write(entry_path, entry_md)
                result.pages.append(str(entry_path))
                result.compiled += 1

                # 2. Extract concepts via tool-calling
                concepts = await _extract_concepts(config, source, existing_index)
                result.concepts.extend(concepts)

                # Build concept summaries for MoC grouping
                concept_names: list[str] = []
                for c in concepts:
                    slug = slugify(c.concept)
                    summary = PageSummary(
                        title=c.concept,
                        slug=slug,
                        summary=c.summary,
                        tags=c.tags,
                    )
                    concept_names.append(c.concept)
                    all_concept_summaries.append(summary)
                    for tag in c.tags:
                        tag_map.setdefault(tag, []).append(summary)

                # 3. Render concept pages (in parallel within this source scope)
                concept_tasks = []
                for c in concepts:
                    concept_tasks.append(
                        _render_and_write_concept(
                            config, c, source.content, existing_index
                        )
                    )
                concept_results = await asyncio.gather(*concept_tasks, return_exceptions=True)

                rendered_concepts: list[str] = []
                for i, cres in enumerate(concept_results):
                    if isinstance(cres, Exception):
                        logger.error(
                            "Failed to render concept '%s': %s",
                            concepts[i].concept,
                            cres,
                        )
                        result.errors.append(
                            f"concept:{concepts[i].concept}:{cres}"
                        )
                    else:
                        rendered_concepts.append(concepts[i].concept)

                # Update state
                file_hash = hash_content(source.content)
                update_source_state(
                    state, filename, file_hash, rendered_concepts, compiled_at
                )

                logger.info(
                    "Source '%s' compiled: %d concepts rendered",
                    source.title,
                    len(rendered_concepts),
                )

            except Exception as exc:
                logger.error("Failed to process source '%s': %s", filename, exc)
                result.errors.append(f"source:{filename}:{exc}")

    # Process all sources with concurrency
    await asyncio.gather(
        *[_process_source(f) for f in source_filenames],
        return_exceptions=True,
    )

    # ── Phase 2: Build MoCs from tag groups ─────────────────────────────
    moc_topics = _determine_moc_topics(tag_map)
    moc_tasks = []
    for topic, summaries in moc_topics.items():
        moc_tasks.append(
            _render_and_write_moc(config, topic, summaries)
        )
    moc_results = await asyncio.gather(*moc_tasks, return_exceptions=True)

    for mres in moc_results:
        if isinstance(mres, Exception):
            logger.error("Failed to render MoC: %s", mres)
            result.errors.append(f"moc:{mres}")

    logger.info(
        "Compilation complete: %d entries, %d concepts, %d MoCs, %d errors",
        result.compiled,
        len(result.concepts),
        len(moc_topics),
        len(result.errors),
    )

    return result


# ── Internal helpers ────────────────────────────────────────────────────


async def _extract_concepts(
    config: Config,
    source: IngestedSource,
    existing_index: str,
) -> list[ExtractedConcept]:
    """Extract concepts from a source using LLM tool-calling.

    Falls back to JSON parsing if tool-calling returns raw JSON.
    """
    system = (
        _EXTRACTION_SYSTEM_PROMPT
        + "\n\n--- EXISTING INDEX ---\n"
        + existing_index
        + "\n\n--- SOURCE DOCUMENT ---\n"
        + source.content
        + "\n\nCall the extract_concepts tool with the concepts you identify."
    )

    messages: list[dict] = [
        {
            "role": "user",
            "content": "Extract the 3-8 most important concepts from the source document above.",
        }
    ]

    try:
        raw = await call_llm(system, messages, config, tools=EXTRACTION_TOOLS)
    except Exception:
        logger.warning("LLM call for concept extraction failed", exc_info=True)
        return []

    # Try tool-call response first
    concepts = _parse_tool_response(raw)
    if concepts:
        return concepts

    # Fallback: try to parse as JSON directly
    concepts = _parse_raw_json_concepts(raw)
    if concepts:
        return concepts

    logger.warning("Could not parse extracted concepts from LLM response: %s", raw[:200])
    return []


def _parse_tool_response(raw: str) -> list[ExtractedConcept] | None:
    """Parse tool-call JSON from LLM response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    # Check for tool_calls wrapper
    if isinstance(data, dict):
        tool_calls = data.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "")
                if args_str:
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        return _build_concepts_from_list(args.get("concepts", []))
                    except json.JSONDecodeError:
                        continue

        # Direct concepts array
        concepts_list = data.get("concepts")
        if concepts_list:
            return _build_concepts_from_list(concepts_list)

    if isinstance(data, list):
        return _build_concepts_from_list(data)

    return None


def _parse_raw_json_concepts(raw: str) -> list[ExtractedConcept] | None:
    """Attempt to parse raw JSON array of concept objects."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON block from markdown
        import re
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if isinstance(data, dict) and "concepts" in data:
        return _build_concepts_from_list(data["concepts"])

    if isinstance(data, list):
        return _build_concepts_from_list(data)

    return None


def _build_concepts_from_list(raw_list: list[dict]) -> list[ExtractedConcept]:
    """Build ExtractedConcept objects from raw dict list."""
    concepts: list[ExtractedConcept] = []
    for item in raw_list:
        prov_raw = item.get("provenance_state", "extracted")
        try:
            prov = ProvenanceState(prov_raw)
        except ValueError:
            prov = ProvenanceState.EXTRACTED

        concepts.append(
            ExtractedConcept(
                concept=item.get("title", item.get("concept", "")),
                summary=item.get("summary", ""),
                is_new=item.get("is_new", True),
                tags=item.get("tags", []),
                confidence=item.get("confidence"),
                provenance_state=prov,
            )
        )
    return concepts


async def _render_and_write_concept(
    config: Config,
    concept: ExtractedConcept,
    source_content: str,
    existing_index: str,
) -> str:
    """Render a concept page and write to disk."""
    slug = slugify(concept.concept)

    # Check for existing page
    concept_path = config.concepts_dir / f"{slug}.md"
    existing_page = safe_read_file(concept_path)

    # Build related pages context
    related_pages = existing_index

    markdown = await render_concept(
        config,
        concept.concept,
        source_content,
        existing_page=existing_page,
        related_pages=related_pages,
    )

    atomic_write(concept_path, markdown)
    return slug


def _determine_moc_topics(
    tag_map: dict[str, list[PageSummary]],
) -> dict[str, list[PageSummary]]:
    """Determine MoC topics from tag groups.

    Groups concepts by shared tags. Skips single-concept tags (uninteresting MoCs).
    Only creates MoCs for tags that have 2+ concepts.
    """
    topics: dict[str, list[PageSummary]] = {}
    for tag, summaries in tag_map.items():
        if len(summaries) >= 2:
            # Capitalize tag for topic name
            topic_name = tag.replace("-", " ").replace("_", " ").title()
            topics[topic_name] = summaries

    return topics


async def _render_and_write_moc(
    config: Config,
    topic: str,
    summaries: list[PageSummary],
) -> str:
    """Render a MoC page and write to disk."""
    slug = slugify(topic)

    # Build concept dicts for prompt
    related = [
        {"title": s.title, "summary": s.summary, "tags": s.tags}
        for s in summaries
    ]

    # Read concept pages for deeper context
    concept_pages_text = ""
    for s in summaries:
        cp = config.concepts_dir / f"{s.slug}.md"
        page = safe_read_file(cp)
        if page:
            concept_pages_text += f"\n\n--- {s.title} ---\n{page[:2000]}"

    markdown = await render_moc(config, topic, related, concept_pages_text)

    moc_path = config.mocs_dir / f"{slug}.md"
    atomic_write(moc_path, markdown)
    return slug


def _build_existing_index(state: WikiState) -> str:
    """Build a text index of existing concepts from state."""
    if not state.sources:
        return "(No existing concepts)"

    all_concepts: set[str] = set()
    for src_state in state.sources.values():
        for c in src_state.concepts:
            all_concepts.add(c)

    if not all_concepts:
        return "(No existing concepts)"

    return "Existing concepts:\n" + "\n".join(
        f"- {c}" for c in sorted(all_concepts)
    )


def _format_concept_context(existing_index: str) -> str:
    """Format concept context for entry prompts."""
    return existing_index
