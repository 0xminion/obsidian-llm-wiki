"""Corpus-level concept and tag reconciliation.

After synthesis, multiple sources may produce concepts with overlapping
slugs or similar titles.  This module merges them into a single canonical
concept per slug, combining tags, sections, and claims from all sources.

Tag normalisation:
  * Lowercase all tags
  * Strip whitespace
  * Deduplicate
  * Merge near-identical tags (e.g. "ml" → "machine-learning" via alias map)

Concept merging (same slug):
  * Combine tags (union)
  * Combine aliases (union)
  * Append sections from subsequent sources
  * Append claims from subsequent sources
  * Append related links (union by slug)
  * Keep the highest confidence
  * Set is_new=False if any source says is_new=False
"""

from __future__ import annotations

import logging
import re

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)

__all__ = [
    "merge_bundle",
    "merge_concepts",
    "normalise_tags",
    "slugify",
]

logger = logging.getLogger("obswiki.synth.dedupe")


# ── Tag normalisation ───────────────────────────────────────────────────

# Common abbreviations → canonical form.
_TAG_ALIASES: dict[str, str] = {
    "ml": "machine-learning",
    "ai": "artificial-intelligence",
    "nlp": "natural-language-processing",
    "llm": "large-language-model",
    "dl": "deep-learning",
    "rl": "reinforcement-learning",
    "nn": "neural-network",
    "cnn": "convolutional-neural-network",
    "rnn": "recurrent-neural-network",
    "gpu": "graphics-processing-unit",
    "api": "application-programming-interface",
}


def normalise_tags(tags: list[str]) -> list[str]:
    """Normalise a list of tags: lowercase, strip, dedupe, resolve aliases."""
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        t = tag.strip().lower().replace(" ", "-").replace("_", "-")
        t = re.sub(r"[^a-z0-9-]", "", t)
        t = re.sub(r"-+", "-", t).strip("-")
        if not t:
            continue
        t = _TAG_ALIASES.get(t, t)
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ── Slugify ─────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert arbitrary text to a filename-safe slug."""
    cleaned = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    slug = cleaned.strip("-").lower()
    return slug if slug else "untitled"


# ── Concept merging ─────────────────────────────────────────────────────


def merge_concepts(concepts: list[ConceptNote]) -> list[ConceptNote]:
    """Merge concepts with the same slug into single canonical entries.

    The first occurrence establishes the base; subsequent sources append
    their sections, claims, and related links.  Tags and aliases are
    unioned.  Confidence is maxed.  is_new is ANDed (false wins).
    """
    by_slug: dict[str, ConceptNote] = {}

    for concept in concepts:
        # Ensure slug is set.
        if not concept.slug:
            concept.slug = slugify(concept.title)

        existing = by_slug.get(concept.slug)
        if existing is None:
            # First occurrence — normalise tags.
            concept.tags = normalise_tags(concept.tags)
            by_slug[concept.slug] = concept
            continue

        # Merge into existing.
        existing.tags = normalise_tags(existing.tags + concept.tags)
        existing.aliases = list(dict.fromkeys(existing.aliases + concept.aliases))
        existing.sections.extend(concept.sections)
        existing.claims.extend(concept.claims)
        existing_related_slugs = {r.slug for r in existing.related}
        for link in concept.related:
            if link.slug not in existing_related_slugs:
                existing.related.append(link)
                existing_related_slugs.add(link.slug)
        existing.confidence = max(existing.confidence, concept.confidence)
        existing.is_new = existing.is_new and concept.is_new

    return list(by_slug.values())


# ── Bundle merging ──────────────────────────────────────────────────────


def merge_bundle(sources: list[SourceSynthesis]) -> SynthesisBundle:
    """Merge multiple SourceSynthesis into a single SynthesisBundle.

    Applies corpus-level concept dedup and tag normalisation.
    """
    all_concepts: list[ConceptNote] = []
    all_maps: dict[str, MapOfContent] = {}
    errors: list[str] = []

    for src in sources:
        all_concepts.extend(src.concepts)
        for moc in src.maps:
            if not moc.slug:
                moc.slug = slugify(moc.title)
            existing = all_maps.get(moc.slug)
            if existing is None:
                moc.tags = normalise_tags(moc.tags)
                all_maps[moc.slug] = moc
            else:
                # Merge concept slugs.
                existing_slugs = set(existing.concept_slugs)
                for s in moc.concept_slugs:
                    if s not in existing_slugs:
                        existing.concept_slugs.append(s)
                        existing_slugs.add(s)

    merged_concepts = merge_concepts(all_concepts)
    merged_maps = list(all_maps.values())

    return SynthesisBundle(
        sources=sources,
        concepts=merged_concepts,
        maps=merged_maps,
        errors=errors,
    )
