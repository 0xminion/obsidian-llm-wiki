"""Prompt builders for the LLM knowledge compilation pipeline.

Each function constructs a system prompt + user message pair suitable for
sending to the LLM via ``pipeline.llm.providers.call_llm``.

Ported from obsidian-llm-wiki/src/compiler/prompts.ts.
"""

from __future__ import annotations

import json
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions (OpenAI function-calling format)
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "extract_concepts",
            "description": "Extract distinct, meaningful concepts from a source document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "description": (
                        "3-8 distinct, meaningful concepts "
                        "extracted from the source."
                    ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": (
                            "A concise, descriptive title for "
                            "the concept (3-8 words)."
                        ),
                                },
                                "summary": {
                                    "type": "string",
                                    "description": (
                            "A 1-2 sentence summary capturing "
                            "the essential meaning."
                        ),
                                },
                                "is_new": {
                                    "type": "boolean",
                                    "description": (
                            "True if this concept does NOT appear "
                            "in the existing index."
                        ),
                                },
                                "tags": {
                                    "type": "array",
                                    "description": "2-4 lowercase categorical tags.",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 4,
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence in this extraction (0.0 - 1.0).",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "provenance_state": {
                                    "type": "string",
                                    "description": (
                            "Provenance state: extracted, merged, "
                            "inferred, or ambiguous."
                        ),
                                    "enum": ["extracted", "merged", "inferred", "ambiguous"],
                                },
                            },
                            "required": [
                                "title",
                                "summary",
                                "is_new",
                                "tags",
                                "confidence",
                                "provenance_state",
                            ],
                        },
                    }
                },
                "required": ["concepts"],
            },
        },
    }
]


# ──────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────────────────────────────────────


def build_extraction_prompt(
    source_content: str,
    existing_index: str,
) -> tuple[str, list[dict], list[dict[str, Any]]]:
    """Build the extraction prompt for a source document.

    Args:
        source_content: The full text content of the source to analyse.
        existing_index: String representation of the current concept index,
            used to avoid extracting duplicate concepts.

    Returns:
        A ``(system, messages, tools)`` tuple ready for ``call_llm``.
    """
    system = (
        "You are a knowledge extraction engine. "
        "Analyse the source document and identify 3-8 distinct, meaningful concepts. "
        "\n\n"
        "For each concept, provide:\n"
        "- **title**: A concise, descriptive title (3-8 words).\n"
        "- **summary**: A 1-2 sentence summary capturing the essential meaning.\n"
        "- **is_new**: Whether this concept does NOT appear in the existing index below.\n"
        "- **tags**: 2-4 lowercase categorical tags.\n"
        "- **confidence**: Confidence in this extraction (0.0 - 1.0).\n"
        "- **provenance_state**: One of: extracted, merged, inferred, ambiguous.\n"
        "\n"
        "Review the existing index below to avoid extracting duplicate concepts. "
        "Prioritise genuinely novel concepts.\n"
        "\n"
        "--- EXISTING INDEX ---\n"
        f"{existing_index}\n"
        "\n"
        "--- SOURCE DOCUMENT ---\n"
        f"{source_content}\n"
        "\n"
        "Call the extract_concepts tool with the concepts you identify."
    )

    # Use a user message so the system prompt remains clean.
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Extract the 3-8 most important concepts from the source document above.",
        }
    ]

    return system, messages, EXTRACTION_TOOLS


# ──────────────────────────────────────────────────────────────────────────────
# Source entry prompt
# ──────────────────────────────────────────────────────────────────────────────


def build_entry_prompt(
    source_title: str,
    source_content: str,
    concept_context: str,
) -> str:
    """Build the source-entry synthesis prompt.

    Creates the system prompt.  The entire content returned can be passed as
    ``system`` to ``call_llm`` with an empty messages list or a single
    ``"Analyse the source document above."`` user message.

    Args:
        source_title: The title of the source file (e.g. ``"DeepSeek-R1 Paper"``).
        source_content: The full source document text.
        concept_context: Previously extracted concepts (names/summaries) for
            linking context.

    Returns:
        Complete system prompt string for entry generation.
    """
    return (
        f"You are a knowledge synthesis engine. "
        f"Write a comprehensive entry note for source '{source_title}'.\n"
        "\n"
        "Structure the entry as follows:\n"
        "\n"
        "## Summary\n"
        "A high-level overview (2-3 sentences) of the source's key contributions.\n"
        "\n"
        "## Core Findings\n"
        "The most important, substantive findings. Each must be evidence-backed "
        "and at least 80 characters. Surface ALL available insights — there is no "
        "cap on the number of findings. Do NOT produce stubs or shallow bullet points.\n"
        "\n"
        "## Other Takeaways\n"
        "Secondary but still meaningful insights. Same quality bar as Core Findings.\n"
        "\n"
        "## Open Questions\n"
        "Unanswered questions, limitations, or areas for further exploration raised "
        "by this source.\n"
        "\n"
        "## Linked Concepts\n"
        "Cross-reference with relevant concepts from the context below using "
        "standard markdown links of the form [text](/concepts/concept-name.md).\n"
        "\n"
        "**Requirements:**\n"
        "- Every output document MUST have YAML frontmatter with a non-empty type field "
        "(e.g. type: Entry).\n"
        "- Each finding must be substantive (80+ characters), evidence-backed.\n"
        "- Must NOT be a stub or shallow summary — provide deep, insightful analysis.\n"
        "- Use a # Citations section at the bottom of the document with a numbered list "
        "of sources referenced. Do NOT use inline ^[...] footnotes.\n"
        "- Use the source title as the citation reference text.\n"
        "- Use absolute bundle-relative links starting with / for all cross-references "
        "(e.g. [text](/concepts/foo.md)).\n"
        "- Do NOT use Obsidian-specific features (wikilinks, aliases, #hashtag tags, etc). "
        "Use standard markdown only.\n"
        "\n"
        "--- CONCEPT CONTEXT ---\n"
        f"{concept_context}\n"
        "\n"
        "--- SOURCE DOCUMENT ---\n"
        f"{source_content}\n"
        "\n"
        "Now write the entry note."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Concept page prompt
# ──────────────────────────────────────────────────────────────────────────────


def build_concept_prompt(
    concept_name: str,
    source_content: str,
    existing_page: str,
    related_pages: str,
) -> str:
    """Build the atomic concept-page generation prompt.

    Args:
        concept_name: The concept title (e.g. ``"Mixture of Experts"``).
        source_content: Source material relevant to this concept.
        existing_page: Current page content if the concept already exists
            (for merging), or empty string.
        related_pages: Summaries of related concept pages for cross-linking.

    Returns:
        Complete system prompt string.
    """
    merge_note = ""
    if existing_page.strip():
        merge_note = (
            "\n**Existing page for this concept:**\n"
            f"{existing_page}\n"
            "\nMerge new information into the existing page rather than replacing it. "
            "Do NOT create duplicate content.\n"
        )

    return (
        f"You are a wiki author. "
        f"Write an evergreen atomic concept note for '{concept_name}'.\n"
        "\n"
        "The page must use this format:\n"
        "\n"
        "```yaml\n"
        "---\n"
        "type: Concept\n"
        "confidence: <0.0-1.0>\n"
        "provenance_state: extracted|merged|inferred|ambiguous\n"
        "tags:\n"
        "  - tag1\n"
        "  - tag2\n"
        "---\n"
        "```\n"
        "\n"
        "## Core concept\n"
        "A concise, clear definition of the concept (2-4 flowing prose paragraphs, "
        "at least 800 characters of substantive body). "
        "This is evergreen content — standalone, self-contained, and does not assume "
        "any specific source context.\n"
        "\n"
        "## Context\n"
        "Additional context, nuance, examples, or historical background. "
        "2-4 flowing prose paragraphs.\n"
        "\n"
        "## Sources\n"
        "List derived sources with brief annotations.\n"
        "\n"
        "# Citations\n"
        "A numbered list of source URLs or filenames at the bottom of the document.\n"
        "Do NOT use inline ^[...] footnotes.\n"
        "\n"
        "## Links\n"
        "Cross-reference related concepts using standard markdown links "
        "[text](/concepts/concept-name.md).\n"
        "\n"
        "**Requirements:**\n"
        "- Every output document MUST have YAML frontmatter with a non-empty type field.\n"
        "- Must be 800+ characters of substantive body — NOT a stub.\n"
        "- Evergreen: self-contained, can be read in isolation.\n"
        "- Use absolute bundle-relative links starting with / for all cross-references "
        "(e.g. [text](/concepts/foo.md)).\n"
        "- Do NOT use Obsidian-specific features (wikilinks, aliases, #hashtag tags, etc). "
        "Use standard markdown only.\n"
        "- Surface ALL relevant information from the source.\n"
        + merge_note
        + "\n"
        "\n--- RELATED CONCEPTS FOR CROSS-REFERENCE ---\n"
        f"{related_pages}\n"
        "\n"
        "--- SOURCE CONTENT ---\n"
        f"{source_content}\n"
        "\n"
        f"Now write the concept note for '{concept_name}'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Map of Content (MOC) prompt
# ──────────────────────────────────────────────────────────────────────────────


def build_moc_prompt(
    topic: str,
    related_concepts: list[dict],
    concept_pages: str,
) -> str:
    """Build the Map-of-Content page generation prompt.

    Args:
        topic: The MOC topic name (e.g. ``"Machine Learning Architecture"``).
        related_concepts: List of concept dicts with at least ``"title"`` and
            ``"summary"`` keys.
        concept_pages: Concatenated concept page content for reference.

    Returns:
        Complete system prompt string.
    """
    concepts_list = json.dumps(related_concepts, indent=2, ensure_ascii=False)

    return (
        f"You are a knowledge cartographer. "
        f"Create a Map of Content for '{topic}'.\n"
        "\n"
        "Structure the MOC as follows:\n"
        "\n"
        "## Purpose\n"
        "1-2 sentences explaining what this MOC covers and why the topic matters.\n"
        "\n"
        "## Key Concepts\n"
        "For each related concept, provide:\n"
        "- A concise summary (1-2 sentences).\n"
        "- How it relates to the broader topic.\n"
        "- How it connects to other concepts in this map.\n"
        "\n"
        "## Cross-References\n"
        "A section showing how concepts interrelate — grouping related ideas, "
        "highlighting contrasts, or noting dependencies.\n"
        "\n"
        "**Requirements:**\n"
        "- Meaningful, not a stub — show how concepts relate to each other.\n"
        "- Every output document MUST have YAML frontmatter with a non-empty type field "
        "(e.g. type: Map of Content).\n"
        "- Use standard markdown links [text](/concepts/concept-name.md) for all "
        "concept references. Use absolute bundle-relative links starting with "
        "/ (e.g. [text](/concepts/foo.md)).\n"
        "- Do NOT use Obsidian-specific features (wikilinks, aliases, #hashtag tags, etc). "
        "Use standard markdown only.\n"
        "- Provide genuine synthesis, not a simple list.\n"
        "\n"
        "--- RELATED CONCEPTS ---\n"
        f"{concepts_list}\n"
        "\n"
        "--- CONCEPT PAGES (for reference) ---\n"
        f"{concept_pages}\n"
        "\n"
        f"Now write the MOC for '{topic}'."
    )
