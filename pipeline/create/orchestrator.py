"""Stage 2 orchestrator — compile sources into wiki pages.

Processes each ingested source: renders entry, extracts concepts, renders
concept pages, then groups into MoCs.  Uses configurable concurrency.

Ported from obsidian-llm-wiki/src/compiler/index.ts.

OKF migration: page rendering uses the OKF-native renderers
(``pipeline.okf_renderer``) which produce OKF v0.1 frontmatter.  The LLM
call pattern (one call per item) is preserved — the LLM generates the
*body* content, and the OKF renderer wraps it with conformant frontmatter.
Data models are imported from ``pipeline.okf_models``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime

from pipeline.config import Config
from pipeline.hasher import hash_content
from pipeline.llm.providers import acall_llm
from pipeline.okf_markdown import atomic_write, parse_frontmatter, safe_read_file, slugify
from pipeline.okf_models import (
    CompileResult,
    ExtractedConcept,
    IngestedSource,
    PageSummary,
    ProvenanceState,
    WikiState,
)
from pipeline.okf_renderer import (
    render_concept_page,
    render_entry_page,
    render_moc_page,
)
from pipeline.prompts import (
    EXTRACTION_TOOLS,
    build_concept_prompt,
    build_entry_prompt,
    build_moc_prompt,
)
from pipeline.state import update_source_state

logger = logging.getLogger("obswiki.orchestrator")


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
      1. Render entry note via LLM → wrap with OKF frontmatter → write to entries_dir.
      2. Extract concepts via LLM tool-calling.
      3. Render each concept page (in parallel) → write to concepts_dir.
    Then:
      4. Group concepts by tags → determine MoC topics.
      5. Render each MoC → wrap with OKF frontmatter → write to mocs_dir.

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

                # 1. Render entry — LLM call produces body, OKF renderer wraps it
                concept_context = _format_concept_context(existing_index)
                entry_body = await _generate_entry_body(
                    config, filename, source.title, source.content, concept_context
                )
                entry_slug = slugify(source.title)
                entry_md = render_entry_page(
                    title=source.title,
                    summary=_extract_first_paragraph(entry_body),
                    source_concept_id=f"sources/{filename}",
                    body=entry_body,
                    timestamp=compiled_at,
                )
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
                        description=c.summary,
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


async def _generate_entry_body(
    config: Config,
    source_filename: str,
    source_title: str,
    source_content: str,
    concept_context: str,
) -> str:
    """Call the LLM to generate the entry note body (one LLM call per source).

    This preserves the legacy "one call per item" pattern.  The LLM produces
    raw markdown; the OKF renderer (:func:`render_entry_page`) wraps it with
    OKF frontmatter afterward.

    Raises:
        ValueError: If both attempts produce content below the minimum threshold.
    """
    system_prompt = build_entry_prompt(source_title, source_content, concept_context)

    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Analyse the source document '{source_title}' and write the entry note.",
        }
    ]

    result = await acall_llm(system_prompt, messages, config)

    body = _extract_body(result)
    if len(body) >= config.entry_min_body_chars:
        return result

    logger.warning(
        "Entry for '%s' is too short (%d chars, need %d). Retrying with stronger prompt.",
        source_title,
        len(body),
        config.entry_min_body_chars,
    )

    # Retry with stronger prompt
    stronger_system = (
        system_prompt
        + "\n\n**CRITICAL:** Your previous response was too short. "
        + "You MUST produce a comprehensive, detailed entry with substantive findings. "
        + f"The body must be at least {config.entry_min_body_chars} characters. "
        + "Do NOT produce stubs or shallow summaries. Surface ALL available insights."
    )

    result2 = await acall_llm(stronger_system, messages, config)
    body2 = _extract_body(result2)

    if len(body2) < config.entry_min_body_chars:
        raise ValueError(
            f"Entry for '{source_title}' failed minimum body check "
            f"after retry ({len(body2)} < {config.entry_min_body_chars})"
        )

    return result2


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
        raw = await acall_llm(system, messages, config, tools=EXTRACTION_TOOLS)
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
    """Build ExtractedConcept objects from raw dict list.

    The OKF :class:`ExtractedConcept` does not have a ``provenance_state``
    field (unlike the legacy model), so we validate the value but do not
    pass it to the constructor.  This keeps the parsing logic backward
    compatible with LLM responses that include ``provenance_state``.
    """
    concepts: list[ExtractedConcept] = []
    for item in raw_list:
        # Validate provenance_state for logging / future use, but don't
        # pass it to the OKF ExtractedConcept (no such field).
        prov_raw = item.get("provenance_state", "extracted")
        with suppress(ValueError):
            ProvenanceState(prov_raw)  # validate

        concepts.append(
            ExtractedConcept(
                concept=item.get("title", item.get("concept", "")),
                summary=item.get("summary", ""),
                is_new=item.get("is_new", True),
                tags=item.get("tags", []),
                confidence=item.get("confidence"),
            )
        )
    return concepts


async def _render_and_write_concept(
    config: Config,
    concept: ExtractedConcept,
    source_content: str,
    existing_index: str,
) -> str:
    """Render a concept page and write to disk.

    The LLM call generates the concept body (one call per concept), then the
    OKF renderer (:func:`render_concept_page`) wraps it with OKF frontmatter.
    """
    slug = slugify(concept.concept)

    # Check for existing page
    concept_path = config.concepts_dir / f"{slug}.md"
    existing_page = safe_read_file(concept_path)

    # Build related pages context
    related_pages = existing_index

    # One LLM call per concept — generates the body markdown.
    concept_body = await _generate_concept_body(
        config, concept.concept, source_content, existing_page, related_pages
    )

    # Wrap with OKF frontmatter
    markdown = render_concept_page(
        title=concept.concept,
        summary=concept.summary,
        body=_extract_body(concept_body),
        tags=concept.tags,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    atomic_write(concept_path, markdown)
    return slug


async def _generate_concept_body(
    config: Config,
    concept_name: str,
    source_content: str,
    existing_page: str,
    related_pages: str,
) -> str:
    """Call the LLM to generate a concept page body (one call per concept).

    Preserves the legacy retry-with-stronger-prompt pattern.

    Raises:
        ValueError: If both attempts produce content below the minimum threshold.
    """
    system_prompt = build_concept_prompt(
        concept_name, source_content, existing_page, related_pages
    )

    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Write the concept note for '{concept_name}'.",
        }
    ]

    result = await acall_llm(system_prompt, messages, config)

    body = _extract_body(result)
    if len(body) >= config.concept_min_body_chars:
        return result

    logger.warning(
        "Concept '%s' is too short (%d chars, need %d). Retrying with stronger prompt.",
        concept_name,
        len(body),
        config.concept_min_body_chars,
    )

    stronger_system = (
        system_prompt
        + "\n\n**CRITICAL:** Your previous response was a stub — it was far too short. "
        + "You MUST produce a comprehensive concept page with substantive prose. "
        + f"The body must be at least {config.concept_min_body_chars} characters. "
        + "Write 2-4 flowing prose paragraphs for each section. "
        + "This is evergreen content — standalone, self-contained, insightful."
    )

    result2 = await acall_llm(stronger_system, messages, config)
    body2 = _extract_body(result2)

    if len(body2) < config.concept_min_body_chars:
        raise ValueError(
            f"Concept '{concept_name}' failed minimum body check "
            f"after retry ({len(body2)} < {config.concept_min_body_chars})"
        )

    return result2


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
    """Render a MoC page and write to disk.

    The LLM call generates the MoC body (one call per MoC), then the OKF
    renderer (:func:`render_moc_page`) wraps it with OKF frontmatter.
    """
    slug = slugify(topic)

    # Build concept dicts for prompt
    related = [
        {"title": s.title, "summary": s.description, "tags": s.tags}
        for s in summaries
    ]

    # Read concept pages for deeper context
    concept_pages_text = ""
    for s in summaries:
        cp = config.concepts_dir / f"{s.slug}.md"
        page = safe_read_file(cp)
        if page:
            concept_pages_text += f"\n\n--- {s.title} ---\n{page[:2000]}"

    # One LLM call per MoC — generates the body markdown
    moc_body = await _generate_moc_body(config, topic, related, concept_pages_text)

    # Build concept_links for OKF renderer: (concept_id, display_text)
    concept_links: list[tuple[str, str]] = [
        (f"concepts/{s.slug}", s.title) for s in summaries
    ]

    markdown = render_moc_page(
        title=topic,
        summary=_extract_first_paragraph(_extract_body(moc_body)) or topic,
        concept_links=concept_links,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    moc_path = config.mocs_dir / f"{slug}.md"
    atomic_write(moc_path, markdown)
    return slug


async def _generate_moc_body(
    config: Config,
    topic: str,
    related_concepts: list[dict],
    concept_pages: str,
) -> str:
    """Call the LLM to generate a MoC body (one call per MoC).

    Preserves the legacy retry-with-stronger-prompt pattern.

    Raises:
        ValueError: If the generated MoC is a stub.
    """
    system_prompt = build_moc_prompt(topic, related_concepts, concept_pages)

    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Create the Map of Content for '{topic}'.",
        }
    ]

    result = await acall_llm(system_prompt, messages, config)

    # Must NOT be stub — enforce meaningful content
    body = _extract_body(result)
    if not _is_stub(body):
        return result

    logger.warning(
        "MoC '%s' appears to be a stub. Retrying with stronger prompt.", topic
    )

    stronger_system = (
        system_prompt
        + "\n\n**CRITICAL:** Your previous response was a stub — it was far too short. "
        + "You MUST provide genuine synthesis, showing how concepts relate to each other. "
        + "Each concept needs a summary, its relation to the topic, and connections to "
        + "other concepts. This is NOT a simple list — it's a cartographic overview."
    )

    result2 = await acall_llm(stronger_system, messages, config)
    body2 = _extract_body(result2)

    if _is_stub(body2):
        raise ValueError(
            f"MoC '{topic}' is still a stub after retry — "
            "could not generate meaningful content"
        )

    return result2


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


# ── Markdown helpers (OKF-aware) ─────────────────────────────────────────


def _extract_body(markdown: str) -> str:
    """Extract body text from markdown, stripping frontmatter."""
    _meta, body = parse_frontmatter(markdown)
    return body.strip()


def _extract_first_paragraph(markdown: str) -> str:
    """Extract the first non-empty, non-header paragraph from ``markdown``.

    Used as a summary/description field for OKF frontmatter when the LLM
    output doesn't already provide an explicit summary.
    """
    if not markdown:
        return ""
    for para in markdown.strip().split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        # Skip headers, blockquotes, code fences, horizontal rules
        if stripped.startswith("#") or stripped.startswith(">"):
            continue
        if stripped.startswith("```") or stripped.startswith("---"):
            continue
        # Truncate to a reasonable description length
        if len(stripped) > 200:
            return stripped[:197] + "..."
        return stripped
    return ""


def _is_stub(body: str) -> bool:
    """Check if the body text looks like a stub (very short or placeholder)."""
    return len(body) < 200
