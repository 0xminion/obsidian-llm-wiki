"""Synthesis prompt builder — produces a single LLM call per source.

The prompt asks the LLM to return a JSON object matching the SourceSynthesis
schema: source summary, tags, key points, concepts (with sections, tags,
claims, relationships), and MOCs.  All markdown rendering is done
deterministically by the renderers — the LLM never writes markdown.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from obsidian_llm_wiki.core.schema import Granularity, SchemaPolicy, format_schema_guidance

__all__ = [
    "build_synthesis_prompt",
    "SYNTHESIS_SCHEMA",
    "RELATIONSHIP_FEWSHOT",
]


# ── Few-shot examples for relationship types ────────────────────────────

RELATIONSHIP_FEWSHOT = """\
Few-shot examples for choosing the right relation type:
  Bitcoin `enables` Proof of Work
    — Bitcoin's consensus mechanism makes PoW practically useful for settlement.
  AMM `evolves_into` Concentrated Liquidity
    — Uniswap V2 AMMs were extended into concentrated liquidity (V3).
  Prediction Markets `competes_with` Opinion Polls
    — Both forecast future events; markets use stakes, polls use sampling.
  Clearinghouse `part_of` Exchange Infrastructure
    — A clearinghouse is a subsystem within the broader exchange stack.
  Kelly Criterion `measures` Optimal Position Size
    — The Kelly formula quantifies the mathematically optimal bet fraction.
  Futarchy `supersedes` Voting
    — Futarchy replaces direct preference voting with market-based decision mechanisms.\
"""


# ── JSON schema shown to the LLM ────────────────────────────────────────

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "source_title": "string — the title of the source document",
    "source_summary": "string — 2-3 sentence high-level overview of the source's key contributions",
    "source_tags": ["string — 2-4 lowercase categorical tags for the source"],
    "key_points": ["string — the most important substantive findings (each 80+ chars, evidence-backed)"],
    "open_questions": ["string — unanswered questions, limitations, or areas for further exploration"],
    "language": "string — ISO 639-1 language code of the source (en, zh, etc.)",
    "concepts": [
        {
            "title": "string — concise descriptive title (3-8 words)",
            "slug": "string — filename-safe slug (lowercase, hyphens, no spaces)",
            "summary": "string — 1-2 sentence summary capturing the essential meaning",
            "tags": ["string — 2-4 lowercase categorical tags"],
            "aliases": ["string — alternative names for this concept"],
            "sections": [
                {
                    "heading": "string — section heading (e.g. 'Core concept', 'Context')",
                    "points": ["string — substantive bullet points for this section"],
                    "prose": "string — optional flowing prose instead of points (use one or the other)"
                }
            ],
            "claims": [
                {
                    "text": "string — a factual claim derived from the source",
                    "source_ref": "string — where in the source this claim appears"
                }
            ],
            "related": [
                {
                    "slug": "string — slug of a related concept",
                    "relation": "depends_on | prerequisite_of | example_of | variant_of | contrasts_with | complements | supersedes | part_of | evolves_into | measures | enables | competes_with | related_to",
                    "display": "string — display text for the link (optional)"
                }
            ],
            "confidence": "number — confidence in this extraction (0.0-1.0)",
            "provenance": "extracted | merged | inferred | ambiguous",
            "is_new": "boolean — true if this concept does not appear in the existing index"
        }
    ],
    "maps": [
        {
            "title": "string — MOC topic title",
            "slug": "string — filename-safe slug",
            "summary": "string — 3-5 sentences explaining what this MOC covers, the key tension or question it addresses, and how the concepts within it relate to each other. Do NOT write a generic one-liner like 'Concepts relating to X'. Write an insightful synthesis that demonstrates deep understanding of the source material.",
            "tags": ["string — lowercase tags"],
            "concept_slugs": ["string — slugs of concepts grouped under this MOC"]
        }
    ]
}


# ── Prompt builder ──────────────────────────────────────────────────────


def build_synthesis_prompt(
    source_title: str,
    source_content: str,
    existing_concepts: list[str] | None = None,
    language: str = "",
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
) -> str:
    """Build the single-call synthesis prompt for one source.

    Args:
        source_title: Title of the source document.
        source_content: Full text content of the source.
        existing_concepts: Slugs of already-known concepts (for dedup).
        language: ISO 639-1 language code — controls the language instruction
            injected into the prompt. If empty, auto-detected by detect_language().
        schema_policy: Optional bounded, vault-local user preferences. This never
            changes the fixed JSON response contract.
        granularity: Optional concise/standard/detailed extraction preference.
            When omitted, no granularity guidance is added (backward compatible).

    Returns:
        The complete system prompt string.
    """
    existing_str = (
        "\n".join(f"  - {s}" for s in existing_concepts)
        if existing_concepts
        else "(none yet — this is the first source)"
    )

    schema_json = json.dumps(SYNTHESIS_SCHEMA, indent=2, ensure_ascii=False)

    # Language instruction
    lang_instruction = ""
    if language:
        from obsidian_llm_wiki.synth.language import get_language_instruction
        li = get_language_instruction(language)
        if li:
            lang_instruction = f"\n{li}"

    # User policy is deliberately isolated from fixed rules and source content.
    schema_guidance = format_schema_guidance(schema_policy, granularity)

    # Key findings instruction — surface ALL, no cap
    findings_instruction = (
        "\n* Surface ALL key findings — do not limit to 5. Include every "
        "substantive claim, data point, and insight in the source. "
        "It is better to surface more than fewer."
    )

    return f"""You are a knowledge synthesis engine.  Analyse the source document \
and produce a structured JSON synthesis.

Your output must be a single JSON object matching this schema (return ONLY \
the JSON, no prose, no code fences):

{schema_json}

Rules:
* Identify 3-8 distinct, meaningful concepts from the source.
* Each concept must have a slug (lowercase, hyphens, no spaces, no special chars).
* Each concept must have at least one section with substantive content \
(either points or prose — not both empty).
* Tags must be lowercase, 2-4 per concept.
* Use the "related" field to link concepts to each other by slug.  Be \
generous with cross-references — a good knowledge graph has many edges.
* Create 1-3 MOCs (Maps of Content) that group related concepts by topic. \
Only create a MOC if 2+ concepts share a theme. Each MOC summary must be \
3-5 sentences of insightful synthesis — NOT a generic one-liner. Explain \
the key tension, how the concepts interact, and why this grouping matters.
* Review the existing concept index below to avoid duplicates.  Set \
"is_new" to false for concepts that already exist.
* Surface ALL available insights — do not produce stubs or shallow summaries.
* Claims should be specific, evidence-backed statements from the source.
{findings_instruction}{lang_instruction}
{schema_guidance}
{RELATIONSHIP_FEWSHOT}

--- EXISTING CONCEPT INDEX ---
{existing_str}

--- SOURCE DOCUMENT ---
Title: {source_title}

{source_content}

Now produce the JSON synthesis for this source."""
