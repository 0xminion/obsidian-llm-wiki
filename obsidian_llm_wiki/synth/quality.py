"""Two-pass quality synthesis — produces deep, evidence-backed concepts.

When ``SYNTHESIS_MODE=two_pass`` is set, the pipeline uses a two-pass approach
instead of the default single-shot synthesis:

  **Pass 1 — Extract:** A lightweight LLM call identifies concepts only
  (title, slug, rationale) without writing body content.

  **Pass 2 — Expand:** For each concept, a focused LLM call sends the source
  excerpt + the full concept list and asks the model to write a deep section
  (300-500 words minimum) with specific evidence from the source.

  **Quality gate:** After expansion, any concept whose total body chars
  fall below ``config.concept_min_body_chars`` gets ``confidence: 0.3`` and
  a warning is logged.

The two-pass mode is opt-in.  The default single-pass mode is unchanged and
the golden test exercises it.  Two-pass trades latency/cost for depth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    normalize_relation,
)
from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis

logger = logging.getLogger("obswiki.synth.quality")

__all__ = [
    "quality_synthesize_source",
    "build_extract_prompt",
    "build_expand_prompt",
    "concept_body_chars",
    "filter_thin_concepts",
]


# ── Pass 1: Extract concept skeleton ─────────────────────────────────────


_EXTRACT_SCHEMA: dict[str, Any] = {
    "source_title": "string — title of the source",
    "source_summary": "string — 2-3 sentence high-level overview",
    "source_tags": ["string — 2-4 lowercase tags"],
    "key_points": ["string — key findings (80+ chars each)"],
    "open_questions": ["string — unanswered questions"],
    "language": "string — ISO 639-1 code",
    "concepts": [
        {
            "title": "string — concise title (3-8 words)",
            "slug": "string — filename-safe slug",
            "summary": "string — 1-2 sentence summary",
            "tags": ["string — 2-4 lowercase tags"],
            "rationale": "string — why this concept matters in the source (50+ chars)",
        }
    ],
    "maps": [
        {
            "title": "string — MOC title",
            "slug": "string — slug",
            "summary": (
                "string — 3-5 sentences of insightful synthesis. "
                "Explain the key tension or question, how the concepts "
                "interact, and why this grouping matters. "
                "Do NOT write a generic one-liner."
            ),
            "tags": ["string"],
            "concept_slugs": ["string — slugs grouped under this MOC"],
        }
    ],
}


def build_extract_prompt(
    source_title: str,
    source_content: str,
    existing_concepts: list[str] | None = None,
    language: str = "",
) -> str:
    """Build the Pass 1 extraction prompt — concept skeleton only."""
    existing_str = (
        "\n".join(f"  - {s}" for s in existing_concepts)
        if existing_concepts
        else "(none yet — this is the first source)"
    )
    schema_json = json.dumps(_EXTRACT_SCHEMA, indent=2, ensure_ascii=False)
    lang_instruction = f"\nWrite all content in **{language}**." if language else ""

    return f"""You are a knowledge extraction engine.  Analyse the source document \
and identify the key concepts it covers.  Return ONLY a JSON object — no prose, \
no code fences:

{schema_json}

Rules:
* Identify 3-8 distinct, meaningful concepts.
* Each concept needs a slug (lowercase, hyphens, no spaces).
* Each concept needs a rationale explaining why it matters in THIS source.
* Do NOT write concept body content — just identify and summarise.
* Create 1-3 MOCs grouping related concepts. Each MOC summary must be \
3-5 sentences of insightful synthesis — NOT a generic one-liner. Explain \
the key tension, how the concepts interact, and why this grouping matters.
* Tags must be lowercase, 2-4 per concept.{lang_instruction}

--- EXISTING CONCEPT INDEX ---
{existing_str}

--- SOURCE DOCUMENT ---
Title: {source_title}

{source_content}

Now produce the JSON extraction."""


# ── Pass 2: Expand each concept with deep content ────────────────────────


_EXPAND_SCHEMA: dict[str, Any] = {
    "title": "string — concept title",
    "slug": "string — concept slug",
    "summary": "string — 1-2 sentence summary",
    "sections": [
        {
            "heading": "string — section heading",
            "points": ["string — substantive bullet points with evidence"],
            "prose": "string — optional flowing prose (use points OR prose)",
        }
    ],
    "claims": [
        {
            "text": "string — factual claim from the source",
            "source_ref": "string — where in the source",
        }
    ],
    "related": [
        {
            "slug": "string — related concept slug",
            "relation": "variant_of | depends_on | contrasts_with | related_to",
            "display": "string — display text (optional)",
        }
    ],
    "aliases": ["string — alternative names"],
}


def build_expand_prompt(
    concept_title: str,
    concept_slug: str,
    concept_rationale: str,
    source_title: str,
    source_content: str,
    all_concepts: list[dict[str, str]],
    language: str = "",
) -> str:
    """Build the Pass 2 expansion prompt for one concept."""
    schema_json = json.dumps(_EXPAND_SCHEMA, indent=2, ensure_ascii=False)
    concepts_list = "\n".join(
        f"  - {c['slug']}: {c['title']}"
        for c in all_concepts
        if c.get("slug") != concept_slug
    )
    lang_instruction = f"\nWrite all content in **{language}**." if language else ""

    return f"""You are a knowledge synthesis engine.  Write a deep, evidence-backed \
section for the concept "{concept_title}" based on the source document below.

Return ONLY a JSON object matching this schema — no prose, no code fences:

{schema_json}

Rules:
* Write at least 300 words of substantive content across sections.
* Use specific evidence, statistics, quotes, or examples from the source.
* Every claim must be grounded in the source — cite where it appears.
* Link to other concepts from this source using their slugs.
* Do NOT repeat what the source says verbatim — synthesise and explain.
* Sections must have substantive points OR prose — never both empty.\
{lang_instruction}

--- CONCEPT TO EXPAND ---
Title: {concept_title}
Slug: {concept_slug}
Rationale: {concept_rationale}

--- OTHER CONCEPTS IN THIS SOURCE (for cross-linking) ---
{concepts_list}

--- SOURCE DOCUMENT ---
Title: {source_title}

{source_content}

Now produce the JSON expansion for "{concept_title}"."""


# ── Two-pass orchestrator ──────────────────────────────────────────────────


async def quality_synthesize_source(
    config: Any,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
) -> SourceSynthesis | None:
    """Run the two-pass quality synthesis for one source.

    Pass 1: Extract concept skeleton (title, slug, rationale, summary, tags, MOCs).
    Pass 2: For each concept, expand with a focused prompt producing deep sections.

    Returns a complete SourceSynthesis with expanded concept bodies.
    """
    from obsidian_llm_wiki.providers.llm import acall_llm

    # Detect source language for correct prompt instructions
    source_lang = ""
    try:
        from obsidian_llm_wiki.synth.language import detect_language
        source_lang = detect_language(source.content)
    except Exception:
        pass

    # ── Pass 1: Extract skeleton ──────────────────────────────────────
    extract_prompt = build_extract_prompt(
        source.title,
        source.content,
        existing_concepts=existing_concepts,
        language=source_lang or config.output_language,
    )
    messages = [{"role": "user", "content": "Extract concepts from the source document above."}]

    try:
        response = await acall_llm(extract_prompt, messages, config)
    except Exception as exc:
        logger.error("Pass 1 (extract) failed for '%s': %s", filename, exc)
        raise

    skeleton = parse_single_source_synthesis(response)
    if skeleton is None:
        logger.warning("Pass 1 produced no parseable JSON for '%s'", filename)
        return None

    if not skeleton.source_title:
        skeleton.source_title = source.title
    skeleton.source_file = filename

    if not skeleton.concepts:
        logger.warning("Pass 1 extracted no concepts for '%s'", filename)
        return skeleton

    # Extract rationale from Pass 1 JSON (separate from summary).
    # The Pass 1 schema has both 'rationale' (why this concept matters)
    # and 'summary' (1-2 sentence summary). We store rationales by slug
    # so Pass 2 can use the richer rationale instead of the short summary.
    rationales: dict[str, str] = {}
    try:
        from obsidian_llm_wiki.synth.parser import _extract_json
        raw_data = _extract_json(response)
        if isinstance(raw_data, dict):
            for c in raw_data.get("concepts", []):
                if isinstance(c, dict) and c.get("slug"):
                    rat = c.get("rationale", "")
                    if rat:
                        rationales[c["slug"]] = rat
    except Exception:
        pass

    logger.info(
        "Pass 1 done for '%s': %d concepts extracted",
        filename, len(skeleton.concepts),
    )

    # ── Pass 2: Expand each concept ─────────────────────────────────────
    all_concept_dicts = [
        {"slug": c.slug, "title": c.title} for c in skeleton.concepts
    ]

    sem = asyncio.Semaphore(config.compile_concurrency)

    async def _run_with_sem(task_coro):
        async with sem:
            return await task_coro

    expand_results = await asyncio.gather(
        *[
            _run_with_sem(
                _expand_one_concept(
                    config, concept, source, all_concept_dicts, source_lang,
                    rationale=rationales.get(concept.slug, ""),
                )
            )
            for concept in skeleton.concepts
        ],
        return_exceptions=True,
    )

    # ── Merge expanded concepts back into skeleton ─────────────────────
    for i, result in enumerate(expand_results):
        original = skeleton.concepts[i]

        if isinstance(result, BaseException):
            logger.warning(
                "Pass 2 (expand) failed for '%s' concept '%s': %s",
                filename, original.slug, result,
            )
            original.confidence = 0.3
            continue

        if result is None:
            logger.warning(
                "Pass 2 produced no output for '%s' concept '%s'",
                filename, original.slug,
            )
            original.confidence = 0.3
            continue

        # Replace skeleton concept with expanded version.
        expanded = result
        expanded.tags = original.tags or expanded.tags
        expanded.confidence = original.confidence
        expanded.provenance = original.provenance
        expanded.is_new = original.is_new

        skeleton.concepts[i] = expanded

    # ── Post-loop quality sweep ────────────────────────────────────────
    # Catch both failed expansions (empty skeleton) and thin successful
    # expansions in one pass. Don't clobber already-lowered confidence.
    for concept in skeleton.concepts:
        body = _concept_body_chars(concept)
        if body < config.concept_min_body_chars and concept.confidence >= 1.0:
            concept.confidence = 0.3
            logger.warning(
                "Concept '%s' body is %d chars (threshold %d) — confidence set to 0.3",
                concept.slug, body, config.concept_min_body_chars,
            )

    logger.info(
        "Pass 2 done for '%s': %d concepts expanded",
        filename, len(skeleton.concepts),
    )
    return skeleton


async def _expand_one_concept(
    config: Any,
    concept: ConceptNote,
    source: SourceDoc,
    all_concepts: list[dict[str, str]],
    source_lang: str = "",
    rationale: str = "",
) -> ConceptNote | None:
    """Expand a single concept via a focused LLM call."""
    from obsidian_llm_wiki.providers.llm import acall_llm

    prompt = build_expand_prompt(
        concept_title=concept.title,
        concept_slug=concept.slug,
        concept_rationale=rationale or concept.summary,
        source_title=source.title,
        source_content=source.content,
        all_concepts=all_concepts,
        language=source_lang or config.output_language,
    )
    messages = [{"role": "user", "content": f'Expand the concept "{concept.title}".'}]

    try:
        response = await acall_llm(prompt, messages, config)
    except Exception as exc:
        logger.error("Pass 2 LLM call failed for '%s': %s", concept.slug, exc)
        raise

    expanded_data = _parse_concept_json(response)
    if expanded_data is None:
        return None

    return _build_concept_from_expand(expanded_data, concept.slug)


# ── Helpers ─────────────────────────────────────────────────────────────


def _concept_body_chars(concept: ConceptNote) -> int:
    """Count total body characters across all sections."""
    total = 0
    for section in concept.sections:
        if section.prose:
            total += len(section.prose)
        for point in section.points:
            total += len(point)
    return total


# Public alias for pipeline / tests
concept_body_chars = _concept_body_chars


def filter_thin_concepts(
    synthesis: SourceSynthesis,
    min_body_chars: int,
) -> SourceSynthesis:
    """Hard quality gate: drop concepts below threshold from a SourceSynthesis.

    This is a strict-mode gate, NOT applied by default in the pipeline.
    The default single-pass and two-pass paths retain all concepts — thin
    concepts get ``confidence: 0.3`` via the two-pass quality gate, not
    silent deletion.

    This function is available for future ``QUALITY_GATE_MODE=strict`` or
    CI validation use. It prunes:
    - ``related`` links on surviving concepts whose target slug was dropped.
    - Dropped slugs from each MOC's ``concept_slugs``.
    - MOCs with fewer than 2 surviving concept members.

    Returns a new SourceSynthesis (does not mutate the input).
    """
    import dataclasses

    kept: list[ConceptNote] = []
    dropped_slugs: set[str] = set()

    for concept in synthesis.concepts:
        bc = _concept_body_chars(concept)
        if bc >= min_body_chars:
            kept.append(concept)
        else:
            dropped_slugs.add(concept.slug)
            logger.warning(
                "Quality gate: dropping concept '%s' — body is %d chars (threshold %d)",
                concept.slug, bc, min_body_chars,
            )

    if not dropped_slugs:
        return synthesis  # nothing to prune

    # Prune related links to dropped slugs on surviving concepts (non-mutating)
    kept_final: list[ConceptNote] = []
    for concept in kept:
        if concept.related:
            pruned_related = [
                link for link in concept.related
                if link.slug not in dropped_slugs
            ]
            concept = dataclasses.replace(concept, related=pruned_related)
        kept_final.append(concept)

    # Prune MOC concept_slugs and drop MOCs with < 2 surviving members (non-mutating)
    kept_mocs: list[MapOfContent] = []
    for moc in synthesis.maps:
        pruned_slugs = [
            s for s in moc.concept_slugs if s not in dropped_slugs
        ]
        if len(pruned_slugs) >= 2:
            kept_mocs.append(dataclasses.replace(moc, concept_slugs=pruned_slugs))
        else:
            logger.warning(
                "Quality gate: dropping MOC '%s' — only %d concept(s) left after pruning",
                moc.slug, len(pruned_slugs),
            )

    logger.info(
        "Quality gate: dropped %d/%d concepts, %d/%d MOCs",
        len(dropped_slugs), len(synthesis.concepts),
        len(synthesis.maps) - len(kept_mocs), len(synthesis.maps),
    )

    # Return a new SourceSynthesis with pruned data
    return SourceSynthesis(
        source_title=synthesis.source_title,
        source_summary=synthesis.source_summary,
        source_tags=synthesis.source_tags,
        key_points=synthesis.key_points,
        open_questions=synthesis.open_questions,
        language=synthesis.language,
        concepts=kept_final,
        maps=kept_mocs,
        source_file=synthesis.source_file,
    )


def _parse_concept_json(response: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from an LLM response."""
    if not response or not response.strip():
        return None

    text = response.strip()
    # Strip code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { and try progressive parsing.
    start = text.find("{")
    if start == -1:
        return None
    text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for end in range(len(text), 1, -1):
        try:
            return json.loads(text[:end])
        except json.JSONDecodeError:
            continue

    return None


def _build_concept_from_expand(data: dict[str, Any], fallback_slug: str) -> ConceptNote:
    """Build a ConceptNote from a Pass 2 expansion response."""
    sections = [
        BodySection(
            heading=s.get("heading", ""),
            points=list(s.get("points", []) or []),
            prose=s.get("prose", ""),
        )
        for s in data.get("sections", [])
        if isinstance(s, dict)
    ]
    # Drop empty sections.
    sections = [s for s in sections if s.points or s.prose.strip()]

    claims = [
        Claim(
            text=c.get("text", ""),
            concept_slug=fallback_slug,
            source_ref=c.get("source_ref", ""),
        )
        for c in data.get("claims", [])
        if isinstance(c, dict)
    ]

    related = [
        ConceptLink(
            slug=r.get("slug", ""),
            relation=normalize_relation(r.get("relation", "related_to")),
            display=r.get("display", ""),
        )
        for r in data.get("related", [])
        if isinstance(r, dict)
    ]

    return ConceptNote(
        title=data.get("title", ""),
        slug=data.get("slug", fallback_slug),
        summary=data.get("summary", ""),
        tags=[],
        aliases=list(data.get("aliases", []) or []),
        sections=sections,
        claims=claims,
        related=related,
        confidence=1.0,
    )


def _extract_rationale(data: dict) -> str:
    """Extract rationale from Pass 1 JSON (separate from summary).

    The Pass 1 schema has both 'rationale' (why this concept matters)
    and 'summary' (1-2 sentence summary). We use rationale for the
    Pass 2 expand prompt and summary for the concept page.
    """
    return data.get("rationale", "") or data.get("summary", "")
