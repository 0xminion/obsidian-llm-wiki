"""Embedding-based cross-lingual concept linking.

Uses a configured Ollama embedding model to compute semantic similarity between
concepts across languages. When a Chinese concept and an English concept
have high cosine similarity (>0.85), they are automatically linked as
cross-lingual aliases in the MoC and concept pages.

Architecture:
  - embed_text(text) → list[float] | None — call Ollama /api/embeddings
  - cosine_similarity(a, b) → float — cosine similarity
  - find_cross_lingual_links(concepts) → dict[slug, list[(target_slug, score, display)]]
  - The results are injected into concept.related and MoC concept_slugs
    during rendering.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import httpx

logger = logging.getLogger("obswiki.synth.embedding")

__all__ = [
    "find_cross_lingual_links",
    "embed_text",
    "cosine_similarity",
]

_SIMILARITY_THRESHOLD = 0.60
_EMBED_TIMEOUT = 30  # seconds — allow for cold model loading


def _embedding_model() -> str:
    """Resolve the model after the vault ``.env`` has been loaded."""
    return os.environ.get("EMBEDDING_MODEL", "embeddinggemma:300m").strip()


def _ollama_host() -> str:
    """Resolve the host at call time instead of freezing import-time settings."""
    return os.environ.get("LLM_HOST", "http://localhost:11434").rstrip("/")


def embed_text(
    text: str,
    *,
    enabled: bool | None = None,
    model: str | None = None,
    host: str | None = None,
) -> list[float] | None:
    """Generate embedding for a text string via Ollama.

    Returns None if embeddings are disabled or the service is unavailable.
    """
    # Explicit config wins; environment remains a backwards-compatible fallback
    # for direct library callers.
    if enabled is None:
        enabled = os.environ.get("EMBEDDINGS_ENABLED", "false").strip().lower() in (
            "true", "1", "yes"
        )
    if not enabled:
        return None

    model = model or _embedding_model()
    host = (host or _ollama_host()).rstrip("/")

    if not text.strip():
        return None

    try:
        with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
            # Use /api/embed (Ollama 0.4+) with fallback to /api/embeddings (older)
            resp = client.post(
                f"{host}/api/embed",
                json={
                    "model": model,
                    "input": text[:2000],  # Truncate to avoid timeout
                },
            )
            if resp.status_code != 200:
                # Fallback to old API
                resp = client.post(
                    f"{_ollama_host()}/api/embeddings",
                    json={
                        "model": _embedding_model(),
                        "prompt": text[:2000],
                    },
                )
                if resp.status_code != 200:
                    logger.debug(
                        "Embedding API returned %d — model may not be loaded",
                        resp.status_code,
                    )
                    return None
            data = resp.json()
            # /api/embed returns {"embeddings": [...]}, old API returns
            # {"embedding": [...]} — handle both response shapes.
            if "embeddings" in data:
                embeddings = data["embeddings"]
                return embeddings[0] if embeddings else None
            return data.get("embedding")
    except Exception as exc:
        logger.debug("Embedding API unavailable: %s", exc)
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Returns 0.0 if vectors have mismatched lengths or zero magnitude.
    """
    if len(a) != len(b):
        return 0.0
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    return dot / (norm_a * norm_b)


def find_cross_lingual_links(
    concepts: list[Any],
    threshold: float = _SIMILARITY_THRESHOLD,
    *,
    enabled: bool | None = None,
    model: str | None = None,
    host: str | None = None,
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
        emb = embed_text(text, enabled=enabled, model=model, host=host)
        if emb:
            embeddings[concept.slug] = emb
            # Detect language from title + summary
            lang = detect_language(text)
            concept_langs[concept.slug] = lang

    if len(embeddings) < 2:
        logger.info("Cross-lingual linking: not enough embeddings (%d)", len(embeddings))
        return {}

    # Build concept lookup by slug for O(1) access
    concept_by_slug = {c.slug: c for c in concepts}

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
                target_concept = concept_by_slug.get(slug_b)
                display = target_concept.title if target_concept else slug_b
                links.setdefault(slug_a, []).append((slug_b, sim, display))
                links.setdefault(slug_b, []).append((slug_a, sim, ""))

                logger.info(
                    "Cross-lingual link: %s (%s) ↔ %s (%s) sim=%.3f",
                    slug_a, lang_a, slug_b, lang_b, sim,
                )

    return links
