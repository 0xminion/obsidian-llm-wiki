"""Single-page LLM generation for entries, concepts, and MoCs.

One LLM call per item. Retry stubs once with stronger prompt.
Ported from llm-wiki-compiler/src/compiler/page-renderer.ts.
"""

from __future__ import annotations

import logging

from pipeline.config import Config
from pipeline.llm_client import call_llm
from pipeline.markdown import parse_frontmatter
from pipeline.prompts import build_concept_prompt, build_entry_prompt, build_moc_prompt

logger = logging.getLogger("llmwiki.page_renderer")


# ── Entry rendering ─────────────────────────────────────────────────────


async def render_entry(
    config: Config,
    source_filename: str,
    source_title: str,
    source_content: str,
    concept_context: str,
) -> str:
    """Render a single source entry note via LLM.

    Args:
        config: Pipeline configuration.
        source_filename: The source .md filename (for citations).
        source_title: Display title for the source.
        source_content: Full source document text.
        concept_context: Previously extracted concepts for linking context.

    Returns:
        Complete markdown for the entry note.

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

    result = await call_llm(system_prompt, messages, config)

    # Validate body length
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

    result2 = await call_llm(stronger_system, messages, config)
    body2 = _extract_body(result2)

    if len(body2) < config.entry_min_body_chars:
        raise ValueError(
            f"Entry for '{source_title}' failed minimum body check "
            f"after retry ({len(body2)} < {config.entry_min_body_chars})"
        )

    return result2


# ── Concept rendering ───────────────────────────────────────────────────


async def render_concept(
    config: Config,
    concept_name: str,
    source_content: str,
    existing_page: str = "",
    related_pages: str = "",
) -> str:
    """Render a single concept page via LLM.

    Args:
        config: Pipeline configuration.
        concept_name: The concept title.
        source_content: Source material relevant to this concept.
        existing_page: Current page content if the concept already exists (for merging).
        related_pages: Summaries of related concept pages for cross-linking.

    Returns:
        Complete markdown for the concept note (evergreen atomic format).

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

    result = await call_llm(system_prompt, messages, config)

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

    result2 = await call_llm(stronger_system, messages, config)
    body2 = _extract_body(result2)

    if len(body2) < config.concept_min_body_chars:
        raise ValueError(
            f"Concept '{concept_name}' failed minimum body check "
            f"after retry ({len(body2)} < {config.concept_min_body_chars})"
        )

    return result2


# ── MoC rendering ───────────────────────────────────────────────────────


async def render_moc(
    config: Config,
    topic: str,
    related_concepts: list[dict],
    concept_pages: str,
) -> str:
    """Render a Map of Content page via LLM.

    Args:
        config: Pipeline configuration.
        topic: The MOC topic name.
        related_concepts: List of concept dicts with 'title' and 'summary' keys.
        concept_pages: Concatenated concept page content for reference.

    Returns:
        Complete markdown for the MoC.

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

    result = await call_llm(system_prompt, messages, config)

    # Must NOT be stub — enforce meaningful content
    body = _extract_body(result)
    if _is_stub(body):
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

        result2 = await call_llm(stronger_system, messages, config)
        body2 = _extract_body(result2)

        if _is_stub(body2):
            raise ValueError(
                f"MoC '{topic}' is still a stub after retry — "
                "could not generate meaningful content"
            )

        return result2

    return result


# ── Helpers ─────────────────────────────────────────────────────────────


def _extract_body(markdown: str) -> str:
    """Extract body text from markdown, stripping frontmatter."""
    _meta, body = parse_frontmatter(markdown)
    return body.strip()


def _is_stub(body: str) -> bool:
    """Check if the body text looks like a stub (very short or placeholder)."""
    return len(body) < 200
