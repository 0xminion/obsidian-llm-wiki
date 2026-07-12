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
    "propagate_backlinks",
    "semantic_dedupe_concepts",
    "assign_orphans_to_mocs",
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


# ── Backlink propagation ────────────────────────────────────────────────


def propagate_backlinks(bundle: SynthesisBundle) -> None:
    """Ensure bidirectional relationships across all concepts.

    After merging, concept A may have a ``related`` link to concept B,
    but B may not have a link back to A — especially when A and B were
    synthesized in different runs (B is cached, A is new).

    This pass walks all concepts in the bundle and for every forward
    edge A→B, ensures B has a reverse edge B→A with the same relation
    type (or ``related_to`` as a fallback). It also enriches MoC
    concept lists: if a concept in a MoC links to a concept in another
    MoC, the target concept's MoCs are checked for reciprocal links.

    This is a pure data operation — no LLM needed. It runs in O(E)
    where E is the total number of edges.
    """
    concept_map: dict[str, ConceptNote] = {c.slug: c for c in bundle.concepts}
    if not concept_map:
        return

    added = 0

    for concept in bundle.concepts:
        for link in concept.related or []:
            target = concept_map.get(link.slug)
            if target is None:
                continue  # Link points outside the bundle

            # Check if target already has a link back to this concept
            existing_back = any(
                r.slug == concept.slug
                for r in (target.related or [])
            )
            if existing_back:
                continue

            # Add reverse link
            from obsidian_llm_wiki.core.models import ConceptLink
            reverse_relation = link.relation or "related_to"
            if target.related is None:
                target.related = []
            target.related.append(
                ConceptLink(
                    slug=concept.slug,
                    relation=reverse_relation,
                    display=concept.title,
                )
            )
            added += 1

    if added:
        logger.info(
            "Backlink propagation: added %d reverse edges across %d concepts",
            added, len(concept_map),
        )


# ── Semantic concept deduplication ──────────────────────────────────────


def semantic_dedupe_concepts(
    bundle: SynthesisBundle,
    threshold: float = 0.85,
) -> None:
    """Merge same-language concepts with high embedding cosine similarity.

    For every pair of concepts in the *same language* whose cosine similarity
    exceeds ``threshold``, the lower-confidence concept is merged into the
    higher-confidence one:

      * Tags, aliases, sections, claims, and related links are unioned.
      * All MoC ``concept_slugs`` referencing the merged slug are updated to
        point to the surviving slug.
      * All ``ConceptLink`` targets across remaining concepts are updated.

    Gated behind ``EMBEDDINGS_ENABLED`` — no-ops if embeddings are unavailable
    (the Ollama embedding service is down or the env var is not set).
    """
    from obsidian_llm_wiki.synth.embedding import cosine_similarity, embed_text
    from obsidian_llm_wiki.synth.language import detect_language

    if not bundle.concepts or len(bundle.concepts) < 2:
        return

    # Build embeddings for all concepts.
    embeddings: dict[str, list[float]] = {}
    concept_langs: dict[str, str] = {}

    for concept in bundle.concepts:
        text = f"{concept.title}. {concept.summary or ''}"
        emb = embed_text(text)
        if emb:
            embeddings[concept.slug] = emb
            concept_langs[concept.slug] = detect_language(text)

    if len(embeddings) < 2:
        logger.info(
            "Semantic dedup: not enough embeddings (%d) — skipping",
            len(embeddings),
        )
        return

    # Find mergeable pairs (same language, high similarity).
    concept_map: dict[str, ConceptNote] = {c.slug: c for c in bundle.concepts}
    slugs = list(embeddings.keys())
    merge_map: dict[str, str] = {}  # merged_slug → surviving_slug
    merged_slugs: set[str] = set()

    for i, slug_a in enumerate(slugs):
        if slug_a in merged_slugs:
            continue
        for slug_b in slugs[i + 1:]:
            # slug_a can become a victim mid-inner-loop; without this check it
            # would be picked as a *survivor* for a later pair, merging that
            # pair's content into an already-deleted concept.
            if slug_a in merged_slugs:
                break
            if slug_b in merged_slugs:
                continue
            lang_a = concept_langs.get(slug_a, "en")
            lang_b = concept_langs.get(slug_b, "en")
            if lang_a != lang_b:
                continue  # Only merge same-language pairs

            sim = cosine_similarity(embeddings[slug_a], embeddings[slug_b])
            if sim < threshold:
                continue

            # Determine survivor: higher confidence, tie-break by slug order.
            ca = concept_map[slug_a]
            cb = concept_map[slug_b]
            if cb.confidence > ca.confidence:
                survivor, victim = slug_b, slug_a
            else:
                survivor, victim = slug_a, slug_b

            _merge_into(concept_map[survivor], concept_map[victim])

            merge_map[victim] = survivor
            merged_slugs.add(victim)
            logger.info(
                "Semantic dedup: merging '%s' into '%s' (sim=%.3f)",
                victim, survivor, sim,
            )

    if not merge_map:
        return

    # Flatten transitive chains (A→B, B→C ⇒ A→C): a survivor of an early merge
    # can itself be merged away later, and a single-hop lookup would remap
    # references onto a deleted slug.
    for victim in merge_map:
        target = merge_map[victim]
        while target in merge_map:
            target = merge_map[target]
        merge_map[victim] = target

    # Remove merged concepts from the bundle.
    bundle.concepts = [
        c for c in bundle.concepts if c.slug not in merged_slugs
    ]

    # Update MoC concept_slugs — remap merged slugs to survivors.
    for moc in bundle.maps:
        remapped = [merge_map.get(s, s) for s in moc.concept_slugs]
        # Deduplicate while preserving order (merging may produce dupes).
        seen: set[str] = set()
        moc.concept_slugs = [
            s for s in remapped
            if not (s in seen or seen.add(s))
        ]

    # Update ConceptLink targets on remaining concepts.
    for concept in bundle.concepts:
        for link in concept.related:
            if link.slug in merge_map:
                link.slug = merge_map[link.slug]

    # Remap source-local concept slugs too. Entry pages build their wikilinks
    # from bundle.sources[*].concepts, and the pipeline persists state.json
    # from the same per-source objects — leaving victims here would render
    # [[victim]] links to pages that no longer exist and feed deleted slugs
    # back into the next run's dedup context.
    for synthesis in bundle.sources:
        seen_source_slugs: set[str] = set()
        remapped_concepts = []
        for concept in synthesis.concepts:
            concept.slug = merge_map.get(concept.slug, concept.slug)
            if concept.slug in seen_source_slugs:
                continue  # two of this source's concepts merged into one
            seen_source_slugs.add(concept.slug)
            remapped_concepts.append(concept)
        synthesis.concepts = remapped_concepts

    logger.info(
        "Semantic dedup: merged %d concept(s), %d remaining",
        len(merged_slugs), len(bundle.concepts),
    )


def _merge_into(survivor: ConceptNote, victim: ConceptNote) -> None:
    """Merge *victim* concept fields into *survivor* in-place."""
    survivor.tags = normalise_tags(survivor.tags + victim.tags)
    survivor.aliases = list(dict.fromkeys(survivor.aliases + victim.aliases))
    survivor.sections.extend(victim.sections)
    survivor.claims.extend(victim.claims)
    existing_rel_slugs = {r.slug for r in survivor.related}
    for link in victim.related:
        if link.slug not in existing_rel_slugs:
            survivor.related.append(link)
            existing_rel_slugs.add(link.slug)
    survivor.confidence = max(survivor.confidence, victim.confidence)
    survivor.is_new = survivor.is_new and victim.is_new


# ── Embedding-based MoC assignment for orphans ───────────────────────────


def assign_orphans_to_mocs(
    bundle: SynthesisBundle,
    threshold: float = 0.55,
) -> None:
    """Assign concepts not in any MoC to the most semantically similar MoC.

    For each orphan concept (not referenced by any MoC), compute its embedding
    and compare it to the average embedding of each MoC's member concepts.
    If the cosine similarity to a MoC's average exceeds ``threshold``, the
    orphan is added to that MoC's ``concept_slugs``.

    Gated behind ``EMBEDDINGS_ENABLED`` — no-ops if embeddings are unavailable.
    """
    from obsidian_llm_wiki.synth.embedding import cosine_similarity, embed_text

    if not bundle.concepts or not bundle.maps:
        return

    # Build set of all concept slugs that are in at least one MoC.
    moced_slugs: set[str] = set()
    for moc in bundle.maps:
        moced_slugs.update(moc.concept_slugs)

    orphans = [c for c in bundle.concepts if c.slug not in moced_slugs]
    if not orphans:
        return

    # Compute embeddings for orphans.
    orphan_embs: dict[str, list[float]] = {}
    for orphan in orphans:
        text = f"{orphan.title}. {orphan.summary or ''}"
        emb = embed_text(text)
        if emb:
            orphan_embs[orphan.slug] = emb

    if not orphan_embs:
        logger.info("MoC assignment: no embeddings for orphans — skipping")
        return

    # Compute average embedding for each MoC.
    moc_avg_embs: dict[str, list[float]] = {}
    for moc in bundle.maps:
        moc_embeddings: list[list[float]] = []
        for slug in moc.concept_slugs:
            concept = next((c for c in bundle.concepts if c.slug == slug), None)
            if concept:
                text = f"{concept.title}. {concept.summary or ''}"
                emb = embed_text(text)
                if emb:
                    moc_embeddings.append(emb)
        if moc_embeddings:
            dim = len(moc_embeddings[0])
            avg = [0.0] * dim
            for emb in moc_embeddings:
                for j, val in enumerate(emb):
                    avg[j] += val
            n = len(moc_embeddings)
            moc_avg_embs[moc.slug] = [v / n for v in avg]

    if not moc_avg_embs:
        logger.info("MoC assignment: no MoC embeddings — skipping")
        return

    # Assign orphans to MoCs.
    added = 0
    moc_map = {m.slug: m for m in bundle.maps}

    for slug, emb in orphan_embs.items():
        best_moc: str | None = None
        best_sim = threshold  # Must exceed threshold

        for moc_slug, moc_avg in moc_avg_embs.items():
            sim = cosine_similarity(emb, moc_avg)
            if sim > best_sim:
                best_sim = sim
                best_moc = moc_slug

        if best_moc:
            moc = moc_map[best_moc]
            if slug not in moc.concept_slugs:
                moc.concept_slugs.append(slug)
                added += 1
                logger.info(
                    "MoC assignment: orphan '%s' → MoC '%s' (sim=%.3f)",
                    slug, best_moc, best_sim,
                )

    if added:
        logger.info(
            "MoC assignment: assigned %d orphan(s) to MoCs", added,
        )
