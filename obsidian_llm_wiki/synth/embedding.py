"""Embedding-based cross-lingual concept linking.

Uses a configured Ollama embedding model to compute semantic similarity between
concepts across languages. When a Chinese concept and an English concept
have high cosine similarity (>0.85), they are automatically linked as
cross-lingual aliases in the MoC and concept pages.

Architecture:
  - embed_text(text) → list[float] | None — call Ollama /api/embed
  - embed_batch(texts) → list[list[float] | None] — batched embedding (8 per call)
  - load_embeddings_cache(path) → dict[str, list[float]] — load persisted embeddings
  - save_embeddings_cache(path, cache) → None — persist embeddings to disk
  - cosine_similarity(a, b) → float — cosine similarity
  - find_cross_lingual_links(concepts) → dict[slug, list[(target_slug, score, display)]]
  - The results are injected into concept.related and MoC concept_slugs
    during rendering.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("obswiki.synth.embedding")

__all__ = [
    "find_cross_lingual_links",
    "embed_text",
    "embed_batch",
    "load_embeddings_cache",
    "save_embeddings_cache",
    "cosine_similarity",
]

_SIMILARITY_THRESHOLD = 0.60
_EMBED_TIMEOUT = 30  # seconds — allow for cold model loading
_BATCH_SIZE = 8  # texts per Ollama /api/embed call — keeps RAM under ~2GB
_ROUND_DECIMALS = 8
_CACHE_VERSION = 2


def _canonical_embedding(value: object) -> list[float] | None:
    """Return one finite, cache-stable embedding vector or ``None``.

    Persisted and fresh vectors use identical precision so threshold decisions
    do not change merely because one side came from the on-disk cache.
    """
    if not isinstance(value, list) or not value:
        return None
    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            return None
        number = float(component)
        if not math.isfinite(number):
            return None
        vector.append(round(number, _ROUND_DECIMALS))
    return vector


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
                    f"{host}/api/embeddings",
                    json={
                        "model": model,
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
                return _canonical_embedding(embeddings[0]) if embeddings else None
            return _canonical_embedding(data.get("embedding"))
    except Exception as exc:
        logger.debug("Embedding API unavailable: %s", exc)
        return None


def embed_batch(
    texts: list[str],
    *,
    enabled: bool | None = None,
    model: str | None = None,
    host: str | None = None,
    batch_size: int = _BATCH_SIZE,
) -> list[list[float] | None]:
    """Embed multiple texts efficiently using Ollama's batched /api/embed.

    Sends ``batch_size`` texts per API call.  Returns one embedding per input
    text (``None`` for texts that failed or when embeddings are disabled).
    """
    if enabled is None:
        enabled = os.environ.get("EMBEDDINGS_ENABLED", "false").strip().lower() in (
            "true", "1", "yes",
        )
    if not enabled or not texts:
        return [None] * len(texts)

    model = model or _embedding_model()
    host = (host or _ollama_host()).rstrip("/")

    results: list[list[float] | None] = [None] * len(texts)

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        truncated = [t[:2000] for t in batch]
        try:
            with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
                resp = client.post(
                    f"{host}/api/embed",
                    json={"model": model, "input": truncated},
                )
                if resp.status_code != 200:
                    for i, text in enumerate(batch):
                        idx = start + i
                        if results[idx] is None:
                            results[idx] = embed_text(
                                text, enabled=enabled, model=model, host=host,
                            )
                    continue
                data = resp.json()
                embeddings = data.get("embeddings", [])
                for i, emb in enumerate(embeddings):
                    idx = start + i
                    canonical = _canonical_embedding(emb)
                    if idx < len(results) and canonical is not None:
                        results[idx] = canonical
        except Exception as exc:
            logger.debug("Batch embedding failed at offset %d: %s", start, exc)
            for i, text in enumerate(batch):
                idx = start + i
                if results[idx] is None:
                    results[idx] = embed_text(
                        text, enabled=enabled, model=model, host=host,
                    )

    return results


def load_embeddings_cache(
    cache_path: Path, *, model: str | None = None
) -> dict[str, list[float]]:
    """Load persisted embeddings from ``embeddings.json``.

    Returns an empty dict if the file is missing, corrupt, or was produced by
    a different embedding model.
    """
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("Embeddings cache at %s is corrupt — ignoring", cache_path)
        return {}

    if not isinstance(data, dict) or data.get("_cache_version") != _CACHE_VERSION:
        logger.info("Embeddings cache format is stale — rebuilding")
        return {}
    cached_model = data.get("_model", "")
    current_model = model or _embedding_model()
    if cached_model != current_model:
        logger.info(
            "Embeddings cache model mismatch (cached=%s, current=%s) — rebuilding",
            cached_model, current_model,
        )
        return {}

    embeddings: dict[str, list[float]] = {}
    raw_embeddings = data.get("embeddings", {})
    if not isinstance(raw_embeddings, dict):
        return {}
    for slug, vec in raw_embeddings.items():
        canonical = _canonical_embedding(vec)
        if isinstance(slug, str) and canonical is not None:
            embeddings[slug] = canonical
    logger.info("Loaded %d cached embeddings from %s", len(embeddings), cache_path)
    return embeddings


def save_embeddings_cache(
    cache_path: Path, embeddings: dict[str, list[float]], *, model: str | None = None
) -> None:
    """Persist canonical embeddings atomically without shared temp-file races."""
    if not embeddings:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rounded = {
        slug: canonical
        for slug, vec in embeddings.items()
        if isinstance(slug, str)
        if (canonical := _canonical_embedding(vec)) is not None
    }
    if not rounded:
        return
    data = {
        "_cache_version": _CACHE_VERSION,
        "_model": model or _embedding_model(),
        "embeddings": rounded,
    }
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(data, handle, ensure_ascii=False)
        Path(tmp_name).replace(cache_path)
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
    logger.info("Saved %d embeddings to %s", len(rounded), cache_path)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Returns 0.0 if vectors have mismatched lengths or zero magnitude.
    """
    if len(a) != len(b):
        return 0.0
    try:
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
    except (TypeError, ValueError):
        return 0.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    try:
        similarity = sum(x * y for x, y in zip(a, b, strict=False)) / (norm_a * norm_b)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return similarity if math.isfinite(similarity) else 0.0


def find_cross_lingual_links(
    concepts: list[Any],
    threshold: float = _SIMILARITY_THRESHOLD,
    *,
    enabled: bool | None = None,
    model: str | None = None,
    host: str | None = None,
    embeddings_cache: dict[str, list[float]] | None = None,
) -> dict[str, list[tuple[str, float, str]]]:
    """Find cross-lingual concept pairs with high semantic similarity.

    Returns empty dict if embeddings are disabled or unavailable.

    When ``embeddings_cache`` is provided, uses and updates it instead of
    recomputing embeddings from scratch.
    """
    from obsidian_llm_wiki.synth.language import detect_language

    # Build embeddings for live concepts only. The shared cache can still hold
    # a victim slug until persistence prunes it after semantic dedup; allowing
    # that stale key into this comparison would emit links to deleted pages.
    cache = embeddings_cache if embeddings_cache is not None else {}
    live_slugs = {concept.slug for concept in concepts}
    embeddings: dict[str, list[float]] = {
        slug: vector for slug, vector in cache.items() if slug in live_slugs
    }
    concept_langs: dict[str, str] = {}

    for concept in concepts:
        if concept.slug in embeddings:
            lang = detect_language(f"{concept.title}. {concept.summary or ''}")
            concept_langs[concept.slug] = lang
            continue
        text = f"{concept.title}. {concept.summary or ''}"
        emb = embed_text(text, enabled=enabled, model=model, host=host)
        if emb:
            cache[concept.slug] = emb
            embeddings[concept.slug] = emb
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
