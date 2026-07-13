"""Two-pass quality synthesis — produces deep, evidence-backed concepts.

When ``SYNTHESIS_MODE=two_pass`` is set, the pipeline uses a two-pass approach
instead of the default single-shot synthesis:

  **Pass 1 — Extract:** A lightweight LLM call identifies concepts only
  (title, slug, rationale) without writing body content.

  **Pass 2 — Expand:** For each concept, a focused LLM call sends the source
  excerpt + the full concept list and asks the model to write a deep section
  (500-800 words minimum, at least 3 sections) with specific evidence.

  **Quality gate:** After expansion, any concept whose total body chars
  fall below ``config.concept_min_body_chars`` gets a gradient confidence
  score (see ``gradient_confidence``) and a warning is logged.

  **Content chunking:** Sources above ``config.chunk_size`` (default 30K chars)
  are split into chunks. Pass 1 runs on each chunk independently, and the
  resulting skeletons are merged (union concepts by slug, union MoCs by slug,
  union key_points) before Pass 2 expansion proceeds on the merged skeleton.

The two-pass mode is opt-in.  The default single-pass mode is unchanged and
the golden test exercises it.  Two-pass trades latency/cost for depth.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
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
from obsidian_llm_wiki.core.schema import Granularity, SchemaPolicy, format_schema_guidance
from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis

logger = logging.getLogger("obswiki.synth.quality")

# System prompts for LLM calls — kept short to avoid proxy system-prompt
# truncation. The full instructions go in the user message.
_SYSTEM_EXTRACT = (
    "You are a knowledge extraction engine. "
    "Return ONLY a JSON object, no prose, no code fences."
)
_SYSTEM_SYNTH = (
    "You are a knowledge synthesis engine. "
    "Return ONLY a JSON object, no prose, no code fences."
)

__all__ = [
    "quality_synthesize_source",
    "multi_model_entry_synthesize_source",
    "merge_entry_syntheses",
    "build_extract_prompt",
    "build_expand_prompt",
    "concept_body_chars",
    "filter_thin_concepts",
    "gradient_confidence",
    "chunk_content",
    "merge_skeletons",
]


# ── Content chunking ────────────────────────────────────────────────────


def chunk_content(content: str, chunk_size: int = 30_000) -> list[str]:
    """Split content into chunks of approximately ``chunk_size`` chars.

    Splits at paragraph boundaries (double newlines) when possible to avoid
    cutting mid-sentence. If a single paragraph exceeds chunk_size, it is
    hard-split at chunk_size boundaries.

    Returns a list of content chunks. If content <= chunk_size, returns a
    single-element list containing the original content.
    """
    if len(content) <= chunk_size:
        return [content]

    chunks: list[str] = []
    paragraphs = content.split("\n\n")
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    # Hard-split any chunk that still exceeds chunk_size (very long paragraphs)
    final: list[str] = []
    for chunk in chunks:
        while len(chunk) > chunk_size:
            final.append(chunk[:chunk_size])
            chunk = chunk[chunk_size:]
        if chunk:
            final.append(chunk)

    return final


def merge_skeletons(skeletons: list[SourceSynthesis]) -> SourceSynthesis:
    """Merge multiple Pass 1 skeletons into one.

    Unions:
    - concepts (by slug — first occurrence wins for body fields, tags merged)
    - maps/MoCs (by slug — concept_slugs unioned)
    - key_points (concatenated, de-duplicated by string equality)
    - open_questions (concatenated, de-duplicated)
    - source_tags (union)
    """
    if not skeletons:
        return SourceSynthesis(source_title="", source_summary="")

    if len(skeletons) == 1:
        return skeletons[0]

    merged = SourceSynthesis(
        source_title=skeletons[0].source_title,
        source_summary=skeletons[0].source_summary,
        source_file=skeletons[0].source_file,
        language=skeletons[0].language,
    )

    # Union source_tags
    all_tags: set[str] = set()
    for s in skeletons:
        all_tags.update(s.source_tags)
    merged.source_tags = sorted(all_tags)

    # Union key_points (dedup by string equality)
    seen_points: set[str] = set()
    for s in skeletons:
        for kp in s.key_points:
            if kp not in seen_points:
                seen_points.add(kp)
                merged.key_points.append(kp)

    # Union open_questions (dedup)
    seen_q: set[str] = set()
    for s in skeletons:
        for q in s.open_questions:
            if q not in seen_q:
                seen_q.add(q)
                merged.open_questions.append(q)

    # Union concepts by slug — first occurrence wins for body/summary,
    # but tags are unioned across chunks.
    concept_by_slug: dict[str, ConceptNote] = {}
    for s in skeletons:
        for concept in s.concepts:
            if concept.slug in concept_by_slug:
                # Merge tags into existing
                existing = concept_by_slug[concept.slug]
                for tag in concept.tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)
            else:
                concept_by_slug[concept.slug] = concept
    merged.concepts = list(concept_by_slug.values())

    # Union MoCs by slug — concept_slugs unioned
    moc_by_slug: dict[str, MapOfContent] = {}
    for s in skeletons:
        for moc in s.maps:
            if moc.slug in moc_by_slug:
                existing = moc_by_slug[moc.slug]
                for slug in moc.concept_slugs:
                    if slug not in existing.concept_slugs:
                        existing.concept_slugs.append(slug)
                for tag in moc.tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)
            else:
                moc_by_slug[moc.slug] = moc
    merged.maps = list(moc_by_slug.values())

    return merged


# ── Multi-model entry synthesis (section merging) ──────────────────────


def _section_key(section: BodySection) -> str:
    """Normalised heading used to match sections across models."""
    return (section.heading or "").strip().lower()


def merge_entry_syntheses(
    primary: SourceSynthesis,
    secondary: SourceSynthesis,
    *,
    concept_min_body_chars: int = 0,
) -> SourceSynthesis:
    """Merge two per-source syntheses by taking the deepest sections.

    For each concept slug present in *primary*, the merged concept keeps
    all sections from *primary*.  For sections also present in *secondary*
    (matched by normalised heading), the **deeper** version wins — deeper
    meaning more total body chars (points + prose).  Sections unique to
    *secondary* are appended.

    Concepts that only appear in *secondary* (model B found a concept the
    default model missed) are appended to the result.

    Entry-level fields (``source_summary``, ``key_points``,
    ``open_questions``, ``source_tags``) are unioned, keeping the primary
    model's values first.

    MoCs are merged the same way as in ``merge_skeletons``: union by slug,
    ``concept_slugs`` unioned.

    Parameters
    ----------
    primary
        Synthesis from the default model (gemma4:31b-cloud).
    secondary
        Synthesis from the pass-2 model (GLM-5.2:cloud).
    concept_min_body_chars
        If > 0, used only for logging which model's section won.
    """
    merged = SourceSynthesis(
        source_title=primary.source_title or secondary.source_title,
        source_summary=primary.source_summary or secondary.source_summary,
        source_file=primary.source_file or secondary.source_file,
        language=primary.language or secondary.language,
    )

    # ── Union entry-level metadata ──────────────────────────────────
    all_tags: set[str] = set(primary.source_tags) | set(secondary.source_tags)
    merged.source_tags = sorted(all_tags)

    seen_points: set[str] = set()
    for kp in primary.key_points:
        if kp not in seen_points:
            seen_points.add(kp)
            merged.key_points.append(kp)
    for kp in secondary.key_points:
        if kp not in seen_points:
            seen_points.add(kp)
            merged.key_points.append(kp)

    seen_q: set[str] = set()
    for q in primary.open_questions:
        if q not in seen_q:
            seen_q.add(q)
            merged.open_questions.append(q)
    for q in secondary.open_questions:
        if q not in seen_q:
            seen_q.add(q)
            merged.open_questions.append(q)

    # ── Merge concepts by slug ──────────────────────────────────────
    concept_by_slug: dict[str, ConceptNote] = {}

    for concept in primary.concepts:
        concept_by_slug[concept.slug] = concept

    for concept in secondary.concepts:
        if concept.slug not in concept_by_slug:
            # Secondary model found a concept the primary missed — keep it.
            concept_by_slug[concept.slug] = concept
            continue

        existing = concept_by_slug[concept.slug]

        # Build a section map from primary, keyed by normalised heading.
        merged_sections: list[BodySection] = []
        section_map: dict[str, int] = {}  # heading key → index in merged_sections

        for sec in existing.sections:
            merged_sections.append(sec)
            section_map[_section_key(sec)] = len(merged_sections) - 1

        # Merge in secondary sections: replace if deeper, else append.
        for sec in concept.sections:
            key = _section_key(sec)
            if key in section_map:
                idx = section_map[key]
                primary_chars = _section_body_chars(merged_sections[idx])
                secondary_chars = _section_body_chars(sec)
                if secondary_chars > primary_chars:
                    if concept_min_body_chars > 0:
                        logger.debug(
                            "merge_entry: concept '%s' section '%s' — "
                            "secondary deeper (%d > %d chars)",
                            concept.slug, sec.heading or "(untitled)",
                            secondary_chars, primary_chars,
                        )
                    merged_sections[idx] = sec
            else:
                merged_sections.append(sec)
                section_map[key] = len(merged_sections) - 1

        # Union tags.
        merged_tags = list(existing.tags)
        for tag in concept.tags:
            if tag not in merged_tags:
                merged_tags.append(tag)

        # Union claims (dedup by text).
        merged_claims = list(existing.claims)
        seen_claim_texts: set[str] = {c.text for c in existing.claims}
        for claim in concept.claims:
            if claim.text not in seen_claim_texts:
                seen_claim_texts.add(claim.text)
                merged_claims.append(claim)

        # Union related (dedup by slug).
        merged_related = list(existing.related)
        seen_rel_slugs: set[str] = {r.slug for r in existing.related}
        for link in concept.related:
            if link.slug not in seen_rel_slugs:
                seen_rel_slugs.add(link.slug)
                merged_related.append(link)

        # Union aliases.
        merged_aliases = list(existing.aliases)
        for alias in concept.aliases:
            if alias not in merged_aliases:
                merged_aliases.append(alias)

        # Use the longer summary.
        chosen_summary = (
            existing.summary
            if len(existing.summary) >= len(concept.summary)
            else concept.summary
        )

        concept_by_slug[concept.slug] = ConceptNote(
            title=existing.title or concept.title,
            slug=existing.slug,
            summary=chosen_summary,
            tags=merged_tags,
            aliases=merged_aliases,
            sections=merged_sections,
            claims=merged_claims,
            related=merged_related,
            confidence=max(existing.confidence, concept.confidence),
            provenance="merged",
            is_new=existing.is_new and concept.is_new,
        )

    merged.concepts = list(concept_by_slug.values())

    # ── Merge MoCs by slug (same logic as merge_skeletons) ───────────
    moc_by_slug: dict[str, MapOfContent] = {}
    for moc in primary.maps:
        moc_by_slug[moc.slug] = moc
    for moc in secondary.maps:
        if moc.slug in moc_by_slug:
            existing_moc = moc_by_slug[moc.slug]
            for slug in moc.concept_slugs:
                if slug not in existing_moc.concept_slugs:
                    existing_moc.concept_slugs.append(slug)
            for tag in moc.tags:
                if tag not in existing_moc.tags:
                    existing_moc.tags.append(tag)
        else:
            moc_by_slug[moc.slug] = moc
    merged.maps = list(moc_by_slug.values())

    return merged


def _section_body_chars(section: BodySection) -> int:
    """Count body characters in a single section."""
    total = len(section.prose)
    for point in section.points:
        total += len(point)
    return total


# ── Gradient confidence scoring ─────────────────────────────────────────


def gradient_confidence(body_chars: int, concept_min_body_chars: int) -> float:
    """Compute a gradient confidence score based on body length.

    - If body_chars >= concept_min_body_chars: confidence = 1.0
    - If body_chars >= concept_min_body_chars * 0.5:
        confidence = 0.5 + 0.5 * (body_chars / concept_min_body_chars)
    - If body_chars < concept_min_body_chars * 0.5:
        confidence = 0.1 + 0.4 * (body_chars / (concept_min_body_chars * 0.5))
    - Clamped to [0.1, 1.0]

    This replaces the old binary 0.3/1.0 threshold.
    """
    if concept_min_body_chars <= 0:
        return 1.0

    if body_chars >= concept_min_body_chars:
        return 1.0

    half_threshold = concept_min_body_chars * 0.5
    if body_chars >= half_threshold:
        return 0.5 + 0.5 * (body_chars / concept_min_body_chars)

    # body_chars < half_threshold
    if half_threshold <= 0:
        return 0.1
    return 0.1 + 0.4 * (body_chars / half_threshold)


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
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
) -> str:
    """Build the Pass 1 extraction prompt — concept skeleton only."""
    existing_str = (
        "\n".join(f"  - {s}" for s in existing_concepts)
        if existing_concepts
        else "(none yet — this is the first source)"
    )
    schema_json = json.dumps(_EXTRACT_SCHEMA, indent=2, ensure_ascii=False)
    lang_instruction = f"\nWrite all content in **{language}**." if language else ""
    schema_guidance = format_schema_guidance(schema_policy, granularity)

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
{schema_guidance}

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
            "relation": (
                "depends_on | prerequisite_of | example_of | variant_of | "
                "contrasts_with | complements | supersedes | part_of | "
                "evolves_into | measures | enables | competes_with | "
                "related_to"
            ),
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
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
) -> str:
    """Build the Pass 2 expansion prompt for one concept."""
    from obsidian_llm_wiki.synth.prompts import RELATIONSHIP_FEWSHOT

    schema_json = json.dumps(_EXPAND_SCHEMA, indent=2, ensure_ascii=False)
    concepts_list = "\n".join(
        f"  - {c['slug']}: {c['title']}"
        for c in all_concepts
        if c.get("slug") != concept_slug
    )
    lang_instruction = f"\nWrite all content in **{language}**." if language else ""
    schema_guidance = format_schema_guidance(schema_policy, granularity)

    return f"""You are a knowledge synthesis engine.  Write a deep, evidence-backed \
section for the concept "{concept_title}" based on the source document below.

Return ONLY a JSON object matching this schema — no prose, no code fences:

{schema_json}

Rules:
* Write at least 500-800 words of substantive content across sections.
* Produce at least 3 distinct sections, each covering a different facet \
of the concept (e.g. definition, mechanism, evidence, implications, \
comparison, limitations).
* Use specific evidence, statistics, quotes, or examples from the source.
* Every claim must be grounded in the source — cite where it appears.
* Link to other concepts from this source using their slugs.
* Do NOT repeat what the source says verbatim — synthesise and explain.
* Sections must have substantive points OR prose — never both empty.\
{lang_instruction}
{schema_guidance}

{RELATIONSHIP_FEWSHOT}

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


def _diagnose_pass2_failure(
    exc: BaseException,
    concept_slug: str,
    source: SourceDoc,
    config: Any,
    *,
    prompt: str | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """Capture structured diagnostic info for a Pass 2 synthesis failure.

    Determines the failure type:
    - timeout: exception is a timeout-related error
    - empty_response: response was empty or whitespace-only
    - json_parse_error: response was non-empty but unparseable as JSON
    - context_window_overflow: prompt length exceeds context_window
    - exception: other exception with its type and message

    Returns a structured dict suitable for logging and error messages.
    """
    diag: dict[str, Any] = {
        "concept_slug": concept_slug,
        "source_file": source.source_file or "",
        "source_title": source.title,
        "source_content_len": len(source.content),
    }

    # Check for timeout errors
    exc_type_name = type(exc).__name__
    exc_msg = str(exc)
    is_timeout = (
        "timeout" in exc_type_name.lower()
        or "timeout" in exc_msg.lower()
        or "timed out" in exc_msg.lower()
    )

    if is_timeout:
        diag["failure_type"] = "timeout"
        diag["exception_type"] = exc_type_name
        diag["reason"] = f"LLM call timed out: {exc_msg}"
        return diag

    # Check for context window overflow
    context_window = getattr(getattr(config, "llm", None), "context_window", None)
    if context_window and prompt:
        # Rough token estimate: ~4 chars per token
        est_tokens = len(prompt) // 4
        if est_tokens > context_window:
            diag["failure_type"] = "context_window_overflow"
            diag["estimated_prompt_tokens"] = est_tokens
            diag["context_window"] = context_window
            diag["reason"] = (
                f"Prompt estimated at {est_tokens} tokens exceeds "
                f"context window of {context_window} tokens"
            )
            return diag

    # Check response-based failures (when we have the response)
    if response is not None:
        if not response.strip():
            diag["failure_type"] = "empty_response"
            diag["reason"] = "LLM returned an empty or whitespace-only response"
            return diag

        # Try JSON parse to see if that was the issue
        try:
            json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            diag["failure_type"] = "json_parse_error"
            diag["response_len"] = len(response)
            diag["response_preview"] = response[:200]
            diag["reason"] = "Response could not be parsed as JSON"
            return diag

    # Generic exception fallback
    diag["failure_type"] = "exception"
    diag["exception_type"] = exc_type_name
    diag["reason"] = f"{exc_type_name}: {exc_msg}"
    return diag


async def quality_synthesize_source(
    config: Any,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
    *,
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
) -> SourceSynthesis | None:
    """Run the two-pass quality synthesis for one source.

    Pass 1: Extract concept skeleton (title, slug, rationale, summary, tags, MOCs).
        If source content exceeds the chunking threshold (~40K chars), it is
        split into chunks of ~chunk_size chars each. Each chunk is independently
        synthesised, and the resulting skeletons are merged.
    Pass 2: For each concept, expand with a focused prompt producing deep sections.
        Failures are captured with diagnostic info (timeout, empty response,
        JSON parse error, context window overflow).

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

    # ── Pass 1: Extract skeleton (with chunking) ──────────────────────
    # Gate chunking on config.chunk_size itself: the CHUNK_SIZE knob controls
    # *when* chunking happens, not just chunk width. (A separate hardcoded
    # 40K gate silently left 30-40K sources unchunked, contradicting both this
    # module's docstring and pipeline.py's user-facing warning.)
    chunk_size = getattr(config, "chunk_size", 30_000)
    content_len = len(source.content)

    if content_len > chunk_size:
        chunks = chunk_content(source.content, chunk_size)
        logger.info(
            "Source '%s' is %d chars — chunking into %d chunk(s) of ~%d chars",
            filename, content_len, len(chunks), chunk_size,
        )

        # Run Pass 1 on each chunk concurrently
        sem = asyncio.Semaphore(config.compile_concurrency)

        async def _extract_chunk(chunk_content: str) -> str:
            async with sem:
                prompt = build_extract_prompt(
                    source.title,
                    chunk_content,
                    existing_concepts=existing_concepts,
                    language=source_lang or config.output_language,
                    schema_policy=schema_policy,
                    granularity=granularity,
                )
                msgs = [
                    {"role": "system", "content": _SYSTEM_EXTRACT},
                    {"role": "user", "content": prompt},
                ]
                return await acall_llm(prompt, msgs, config, task="ingest")

        # Let every chunk finish, then fail loudly if ANY chunk failed. A
        # partial skeleton must never propagate as success: the pipeline would
        # cache it and hash-stamp the source, permanently losing the failed
        # chunks' content with no error recorded and no retry on later runs.
        chunk_responses = await asyncio.gather(
            *[_extract_chunk(c) for c in chunks],
            return_exceptions=True,
        )

        # Parse each chunk's response into a skeleton
        skeletons: list[SourceSynthesis] = []
        all_rationales: dict[str, str] = {}
        chunk_errors: list[str] = []
        for chunk_idx, chunk_resp in enumerate(chunk_responses, start=1):
            if isinstance(chunk_resp, BaseException):
                chunk_errors.append(
                    f"chunk {chunk_idx}/{len(chunks)}: "
                    f"{type(chunk_resp).__name__}: {chunk_resp}"
                )
                continue
            chunk_skeleton = parse_single_source_synthesis(chunk_resp)
            if chunk_skeleton is None:
                chunk_errors.append(
                    f"chunk {chunk_idx}/{len(chunks)}: unparseable response"
                )
                continue
            skeletons.append(chunk_skeleton)
            # Extract rationales from each chunk's JSON
            try:
                from obsidian_llm_wiki.synth.parser import _extract_json
                raw_data = _extract_json(chunk_resp)
                if isinstance(raw_data, dict):
                    for c in raw_data.get("concepts", []):
                        if isinstance(c, dict) and c.get("slug"):
                            rat = c.get("rationale", "")
                            if rat:
                                all_rationales[c["slug"]] = rat
            except Exception:
                pass

        if chunk_errors:
            raise RuntimeError(
                f"Pass 1 chunked extraction incomplete for '{filename}' "
                f"({len(skeletons)}/{len(chunks)} chunks succeeded): "
                + "; ".join(chunk_errors)
            )

        skeleton = merge_skeletons(skeletons)
        rationales = all_rationales
    else:
        # Single-chunk path (original logic)
        extract_prompt = build_extract_prompt(
            source.title,
            source.content,
            existing_concepts=existing_concepts,
            language=source_lang or config.output_language,
            schema_policy=schema_policy,
            granularity=granularity,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_EXTRACT},
            {"role": "user", "content": extract_prompt},
        ]

        try:
            response = await acall_llm(extract_prompt, messages, config, task="ingest")
        except Exception as exc:
            logger.error("Pass 1 (extract) failed for '%s': %s", filename, exc)
            raise

        skeleton = parse_single_source_synthesis(response)
        if skeleton is None:
            logger.warning("Pass 1 produced no parseable JSON for '%s'", filename)
            return None

        # Extract rationales from Pass 1 JSON
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

    if not skeleton.source_title:
        skeleton.source_title = source.title
    skeleton.source_file = filename

    if not skeleton.concepts:
        logger.warning("Pass 1 extracted no concepts for '%s'", filename)
        return skeleton

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
                    schema_policy=schema_policy,
                    granularity=granularity,
                )
            )
            for concept in skeleton.concepts
        ],
        return_exceptions=True,
    )

    # ── Merge expanded concepts back into skeleton ─────────────────────
    # Track diagnostic info for failures
    diagnostics: list[dict[str, Any]] = []

    for i, result in enumerate(expand_results):
        original = skeleton.concepts[i]

        if isinstance(result, BaseException):
            diag = _diagnose_pass2_failure(
                result, original.slug, source, config,
                prompt=None, response=None,
            )
            diagnostics.append(diag)
            logger.warning(
                "Pass 2 (expand) failed for '%s' concept '%s': %s — diagnostic: %s",
                filename, original.slug, result, diag,
            )
            original.confidence = gradient_confidence(
                _concept_body_chars(original), config.concept_min_body_chars,
            )
            continue

        if result is None:
            diag: dict[str, Any] = {
                "concept_slug": original.slug,
                "failure_type": "empty_response",
                "reason": "Pass 2 LLM call returned None (no parseable JSON in response)",
            }
            diagnostics.append(diag)
            logger.warning(
                "Pass 2 produced no output for '%s' concept '%s' — diagnostic: %s",
                filename, original.slug, diag,
            )
            original.confidence = gradient_confidence(
                _concept_body_chars(original), config.concept_min_body_chars,
            )
            continue

        # Replace skeleton concept with expanded version.
        expanded = result
        expanded.tags = original.tags or expanded.tags
        expanded.confidence = original.confidence
        expanded.provenance = original.provenance
        expanded.is_new = original.is_new

        skeleton.concepts[i] = expanded

    # ── Post-loop quality sweep with gradient confidence ───────────────
    # Catch both failed expansions (empty skeleton) and thin successful
    # expansions. Use gradient confidence instead of binary 0.3.
    for concept in skeleton.concepts:
        body = _concept_body_chars(concept)
        if body < config.concept_min_body_chars and concept.confidence >= 1.0:
            concept.confidence = gradient_confidence(
                body, config.concept_min_body_chars,
            )
            logger.warning(
                "Concept '%s' body is %d chars (threshold %d) — "
                "gradient confidence set to %.3f",
                concept.slug, body, config.concept_min_body_chars,
                concept.confidence,
            )

    if diagnostics:
        logger.info(
            "Pass 2 done for '%s': %d concepts expanded, %d failure(s) with diagnostics: %s",
            filename, len(skeleton.concepts), len(diagnostics), diagnostics,
        )
    else:
        logger.info(
            "Pass 2 done for '%s': %d concepts expanded",
            filename, len(skeleton.concepts),
        )
    return skeleton


def _swap_llm(
    config: Any,
    *,
    model: str | None = None,
    ingest_model: str | None = None,
    expand_model: str | None = None,
) -> Any:
    """Return a config copy with the LLM model overrides swapped.

    Works with both dataclass Config (via ``dataclasses.replace``) and
    plain test stubs (via shallow copy + attribute assignment).
    """
    import copy
    import dataclasses

    new_llm = dataclasses.replace(
        config.llm,
        model=model if model is not None else config.llm.model,
        ingest_model=ingest_model,
        expand_model=expand_model,
    )

    if dataclasses.is_dataclass(config):
        return dataclasses.replace(config, llm=new_llm)

    # Fallback for non-dataclass test stubs: shallow copy + attribute set.
    new_config = copy.copy(config)
    new_config.llm = new_llm
    return new_config


async def multi_model_entry_synthesize_source(
    config: Any,
    filename: str,
    source: SourceDoc,
    existing_concepts: list[str],
    *,
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
) -> SourceSynthesis | None:
    """Run entry synthesis with both the default and PASS2 model, then merge.

    When ``config.llm.expand_model`` is set (e.g. ``PASS2_MODEL=GLM-5.2:cloud``),
    this function runs the full two-pass synthesis **twice**:

    1. **Primary run** — both Pass 1 (extract) and Pass 2 (expand) use the
       default model (``config.llm.model``).
    2. **Secondary run** — both passes use the expand model override
       (``config.llm.expand_model``).

    The two syntheses are merged with :func:`merge_entry_syntheses`, which
    takes the deepest section for each concept (by body char count) and
    unions entry-level metadata, tags, claims, and related links.

    If no ``expand_model`` is configured, falls back to the standard
    :func:`quality_synthesize_source` (which uses the default model for
    Pass 1 and the expand model for Pass 2 — or the default model for both
    when there is no override).

    If either run fails (returns ``None``), the other run's result is used
    as-is. If both fail, ``None`` is returned.
    """
    expand_model = getattr(config.llm, "expand_model", None)
    if not expand_model:
        return await quality_synthesize_source(
            config, filename, source, existing_concepts,
            schema_policy=schema_policy, granularity=granularity,
        )

    logger.info(
        "Multi-model entry synthesis for '%s': primary=%s, secondary=%s",
        filename, config.llm.model, expand_model,
    )

    default_model = config.llm.model

    # ── Primary run: default model for both passes ──────────────────
    # Override model overrides to None so Pass 1 and Pass 2 both use
    # config.llm.model.
    config_primary = _swap_llm(
        config,
        model=default_model,
        ingest_model=None,
        expand_model=None,
    )

    primary = await quality_synthesize_source(
        config_primary, filename, source, existing_concepts,
        schema_policy=schema_policy, granularity=granularity,
    )

    # ── Secondary run: PASS2_MODEL for both passes ───────────────────
    config_secondary = _swap_llm(
        config,
        model=expand_model,
        ingest_model=None,
        expand_model=expand_model,
    )

    secondary = await quality_synthesize_source(
        config_secondary, filename, source, existing_concepts,
        schema_policy=schema_policy, granularity=granularity,
    )

    # ── Merge or fall back ──────────────────────────────────────────
    if primary is None and secondary is None:
        return None
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    merged = merge_entry_syntheses(
        primary, secondary,
        concept_min_body_chars=getattr(config, "concept_min_body_chars", 0),
    )

    logger.info(
        "Multi-model merge for '%s': %d primary + %d secondary → %d concepts, "
        "%d sections total",
        filename,
        len(primary.concepts), len(secondary.concepts),
        len(merged.concepts),
        sum(len(c.sections) for c in merged.concepts),
    )
    return merged


async def _expand_one_concept(
    config: Any,
    concept: ConceptNote,
    source: SourceDoc,
    all_concepts: list[dict[str, str]],
    source_lang: str = "",
    rationale: str = "",
    *,
    schema_policy: SchemaPolicy | Mapping[str, Any] | None = None,
    granularity: Granularity | str | None = None,
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
        schema_policy=schema_policy,
        granularity=granularity,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_SYNTH},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await acall_llm(prompt, messages, config, task="expand")
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
    concepts get a gradient confidence score via ``gradient_confidence()``,
    not silent deletion.

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
    from obsidian_llm_wiki.synth.parser import _extract_json, _sanitize_latex_artifacts

    if not response or not response.strip():
        return None
    data = _extract_json(response)
    if not isinstance(data, dict):
        return None
    return _sanitize_latex_artifacts(data)


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
        slug=fallback_slug,
        summary=data.get("summary", ""),
        tags=[],
        aliases=list(data.get("aliases", []) or []),
        sections=sections,
        claims=claims,
        related=related,
        confidence=1.0,
    )


# ── Incremental concept re-synthesis ────────────────────────────────────


async def resynthesize_concept(
    config: Any,
    concept: ConceptNote,
    new_source_content: str,
    new_source_title: str = "",
) -> ConceptNote | None:
    """Re-synthesize an existing concept using new source content.

    When a new source references an existing concept, this function produces
    an updated concept that integrates information from both the original
    and the new source. Unlike merge_concepts (which only appends sections),
    this produces a coherent re-written concept body.

    Args:
        config: Pipeline config with LLM settings.
        concept: The existing concept to re-synthesize.
        new_source_content: Content from the new source that references this concept.
        new_source_title: Title of the new source (for context).

    Returns:
        Updated ConceptNote with re-synthesized body, or None if LLM fails.
    """
    from obsidian_llm_wiki.providers.llm import acall_llm

    # Build the re-synthesis prompt
    existing_sections = "\n\n".join(
        f"## {s.heading}\n{s.prose or chr(10).join(f'- {p}' for p in s.points)}"
        for s in concept.sections
    )

    prompt = f"""You are updating an existing knowledge wiki concept \
with new information from a newly ingested source.

## Existing Concept: {concept.title}

### Current Summary
{concept.summary}

### Current Sections
{existing_sections}

## New Source: {new_source_title or 'Untitled'}

{new_source_content[:15000]}

## Task

Re-write the concept by integrating the new source's information with the existing content.
- Keep the same title and slug: "{concept.title}" / "{concept.slug}"
- Update the summary to reflect the combined understanding
- Merge sections — don't just append. Rewrite for coherence.
- Add new claims from the new source
- Keep existing tags and add new ones if relevant
- Preserve all existing related links and add new ones if the new source references other concepts

Return JSON:
```json
{{
    "title": "{concept.title}",
    "slug": "{concept.slug}",
    "summary": "Updated 1-2 sentence summary",
    "tags": ["tag1", "tag2"],
    "sections": [
        {{"heading": "Section Name", "points": ["point1", "point2"], "prose": ""}}
    ],
    "claims": [
        {{"text": "factual claim", "source_ref": "from new source"}}
    ]
}}
```
"""

    messages = [
        {"role": "system", "content": _SYSTEM_SYNTH},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await acall_llm(prompt, messages, config, task="ingest")
    except Exception as exc:
        logger.error("Concept re-synthesis failed for '%s': %s", concept.slug, exc)
        return None

    if not response or not response.strip():
        return None

    # Use the shared hardened extractor (fences, prose-wrapped, truncated
    # output) rather than a bespoke fence parser that fails on responses the
    # rest of the pipeline handles fine.
    from obsidian_llm_wiki.synth.parser import _extract_json

    try:
        data = _extract_json(response)
    except Exception:
        data = None
    if not isinstance(data, dict):
        logger.warning("Could not parse re-synthesis JSON for '%s'", concept.slug)
        return None

    # Build updated concept
    from obsidian_llm_wiki.core.models import BodySection, Claim

    sections = []
    for s in data.get("sections", []):
        if isinstance(s, dict):
            sections.append(BodySection(
                heading=s.get("heading", ""),
                points=s.get("points", []),
                prose=s.get("prose", ""),
            ))

    claims = []
    for c in data.get("claims", []):
        if isinstance(c, dict):
            claims.append(Claim(
                text=c.get("text", ""),
                concept_slug=concept.slug,
                source_ref=c.get("source_ref", ""),
            ))

    from obsidian_llm_wiki.synth.dedupe import normalise_tags

    return ConceptNote(
        title=data.get("title", concept.title),
        slug=concept.slug,
        summary=data.get("summary", concept.summary),
        tags=normalise_tags(list(dict.fromkeys(concept.tags + data.get("tags", [])))),
        aliases=concept.aliases,
        sections=sections,
        claims=claims,
        related=concept.related,
        confidence=concept.confidence,
        provenance="resynthesized",
        is_new=False,
    )
