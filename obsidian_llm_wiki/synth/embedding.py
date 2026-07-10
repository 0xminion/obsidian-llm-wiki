"""Embedding-based cross-lingual concept linking.

Uses nomic-embed-text via Ollama to compute semantic similarity between
concepts across languages. When a Chinese concept and an English concept
have high cosine similarity (>0.85), they are automatically linked as
cross-lingual aliases in the MoC and concept pages.

Architecture:
  - _embed(text) → list[float] — call Ollama /api/embeddings
  - _cosine_sim(a, b) → float — cosine similarity
  - find_cross_lingual_links(concepts) → dict[slug, list[(target_slug, score)]]
  - The results are injected into concept.related and MoC concept_slugs
    during rendering.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("obswiki.synth.embedding")

__all__ = [
    "find_cross_lingual_links",
    "embed_text",
    "cosine_similarity",
]

_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:v1.5")
_OLLAMA_HOST = os.environ.get("LLM_HOST", "http://localhost:11435")
_SIMILARITY_THRESHOLD = 0.85
_EMBEDDINGS_ENABLED = (
    os.environ.get("EMBEDDINGS_ENABLED", "false").strip().lower()
    in ("true", "1", "yes")
)
_EMBED_TIMEOUT = 10  # seconds — short to avoid hanging renders


def embed_text(text: str) -> list[float] | None:
    """Generate embedding for a text string via Ollama.

    Returns None if embeddings are disabled or the service is unavailable.
    """
    if not _EMBEDDINGS_ENABLED:
        return None

    if not text.strip():
        return None

    try:
        with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
            resp = client.post(
                f"{_OLLAMA_HOST}/api/embeddings",
                json={
                    "model": _EMBEDDING_MODEL,
                    "prompt": text[:2000],  # Truncate to avoid timeout
                },
            )
            if resp.status_code != 200:
                logger.debug(
                    "Embedding API returned %d — model may not be loaded",
                    resp.status_code,
                )
                return None
            data = resp.json()
            return data.get("embedding")
    except Exception as exc:
        logger.debug("Embedding API unavailable: %s", exc)
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_cross_lingual_links(
    concepts: list[Any],
    threshold: float = _SIMILARITY_THRESHOLD,
) -> dict[str, list[tuple[str, float, str]]]:
    """Find cross-lingual concept pairs with high semantic similarity.

    Returns empty dict if embeddings are disabled or unavailable.
    """
    from obsidian_llm_wiki.synth.language import detect_language

    # Build embeddings for all concepts
    embeddings: dict[str, list[float]] = {}
    concept_langs: dict[str, str] = {}

    for concept in concepts:
        # Embed title + summary (best semantic representation)
        text = f"{concept.title}. {concept.summary or ''}"
        emb = embed_text(text)
        if emb:
            embeddings[concept.slug] = emb
            # Detect language from title + summary
            lang = detect_language(text)
            concept_langs[concept.slug] = lang

    if len(embeddings) < 2:
        logger.info("Cross-lingual linking: not enough embeddings (%d)", len(embeddings))
        return {}

    # Compare all pairs
    links: dict[str, list[tuple[str, float, str]]] = {}
    slugs = list(embeddings.keys())

    for i, slug_a in enumerate(slugs):
        for slug_b in slugs[i + 1:]:
            lang_a = concept_langs.get(slug_a, "en")
            lang_b = concept_langs.get(slug_b, "en")

            # Only link cross-lingual pairs (different languages)
            if lang_a == lang_b:
                continue

            sim = cosine_similarity(embeddings[slug_a], embeddings[slug_b])
            if sim >= threshold:
                # Find display text — prefer the target's title in its language
                target_concept = next(
                    (c for c in concepts if c.slug == slug_b), None
                )
                display = target_concept.title if target_concept else slug_b

                links.setdefault(slug_a, []).append((slug_b, sim, display))
                links.setdefault(slug_b, []).append((slug_a, sim, ""))

                logger.info(
                    "Cross-lingual link: %s (%s) ↔ %s (%s) sim=%.3f",
                    slug_a, lang_a, slug_b, lang_b, sim,
                )

    return links
