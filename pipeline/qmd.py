"""Shared semantic search module.

Uses Ollama for embeddings (qwen3-embedding:0.6b) with cosine similarity.
Replaces the previous qmd CLI dependency which was broken by node-llama-cpp
compilation failures.

Flow:
  1. Generate query embedding via Ollama /api/embeddings
  2. Compare against cached concept embeddings (generated on first use)
  3. Return top-k matches by cosine similarity
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


from pipeline.models import ConceptMatch
from pipeline.llm_client import LLMClient

log = logging.getLogger(__name__)

# Legacy env vars (still respected if no explicit embed_base_url configured)
_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("QMD_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = 1024  # qwen3-embedding-0.6b output dimension
BATCH_SIZE = 32   # Max texts per /api/embed batch call

# Module-level cache: concept_name -> embedding vector
_concept_embedding_cache: dict[str, list[float]] = {}
_cache_loaded = False
_cache_key: str | None = None


def _get_embed_client(base_url: str = "") -> LLMClient:
    """Return an LLMClient configured for embeddings (defaults to Ollama)."""
    url = base_url or _DEFAULT_OLLAMA_URL
    return LLMClient(provider="ollama", embed_model=EMBED_MODEL, embed_base_url=url, timeout=120)


def _ollama_embed(text: str) -> list[float] | None:
    """Get embedding vector from Ollama. Returns None on failure or wrong dims.

    Backward-compatible wrapper; delegates to LLMClient then validates dimension.
    """
    client = _get_embed_client()
    embedding = client.embed(text)
    if embedding and len(embedding) == EMBED_DIM:
        return embedding
    if embedding:
        log.warning("Ollama returned embedding with wrong dims: %d", len(embedding))
    return None


def _ollama_embed_batch(texts: list[str]) -> dict[str, list[float]]:
    """Batch embed multiple texts via Ollama /api/embed.

    Backward-compatible wrapper; delegates to LLMClient then validates dimensions.
    """
    client = _get_embed_client()
    batch = client.embed_batch(texts)
    results: dict[str, list[float]] = {}
    for text, emb in batch.items():
        if emb and len(emb) == EMBED_DIM:
            results[text] = emb
        elif emb:
            log.warning("Ollama /api/embed returned wrong dim: %d", len(emb))
    return results


def _embed_concepts_batch(md_files: list[Path]) -> dict[str, list[float]]:
    """Embed concept files using batched /api/embed with ThreadPoolExecutor fallback.

    Strategy:
      1. Try Ollama /api/embed in batches of BATCH_SIZE
      2. If batch fails, fallback to ThreadPoolExecutor with individual /api/embeddings
    """
    # Build (stem, text) pairs
    items: list[tuple[str, str]] = []
    for md in md_files:
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
            title_match = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", content, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else md.stem
            body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)[:1000]
            text = f"{title}\n{body}".strip()
            items.append((md.stem, text))
        except Exception as e:
            log.debug("Failed to read %s: %s", md.name, e)

    if not items:
        return {}

    # ── Strategy 1: Batched /api/embed ──────────────────────────────────────
    batched_results: dict[str, list[float]] = {}
    batch_worked = False

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        texts = [text for _, text in batch]
        embeddings = _ollama_embed_batch(texts)
        if embeddings:
            batch_worked = True
            for stem, text in batch:
                if text in embeddings:
                    batched_results[stem] = embeddings[text]
        else:
            # Batch failed — stop trying batched approach
            break

    if batch_worked and len(batched_results) == len(items):
        log.info("Batch embedding: %d/%d concepts via /api/embed", len(batched_results), len(items))
        return batched_results

    # ── Strategy 2: ThreadPoolExecutor fallback ─────────────────────────────
    log.info("Falling back to ThreadPoolExecutor for %d concepts", len(items))
    fallback_results: dict[str, list[float]] = {}

    def _embed_one(item: tuple[str, str]) -> tuple[str, list[float] | None]:
        stem, text = item
        emb = _ollama_embed(text)
        return stem, emb

    with ThreadPoolExecutor(max_workers=4) as executor:
        for stem, embedding in executor.map(_embed_one, items):
            if embedding:
                fallback_results[stem] = embedding

    return fallback_results


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_concept_embeddings(concepts_dir: Path) -> None:
    """Load/generate embeddings for all concept files. Cached per process."""
    global _cache_loaded, _concept_embedding_cache, _cache_key
    key = str(concepts_dir.resolve())
    if _cache_loaded and (_cache_key is None or _cache_key == key):
        return

    if _cache_key is not None and _cache_key != key:
        _concept_embedding_cache = {}
        _cache_loaded = False
    _cache_key = key

    if not concepts_dir.is_dir():
        _cache_loaded = True
        return

    md_files = list(concepts_dir.glob("*.md"))
    if not md_files:
        _cache_loaded = True
        return

    log.info("Generating embeddings for %d concepts via Ollama (%s)...", len(md_files), EMBED_MODEL)

    t0 = time.monotonic()
    results = _embed_concepts_batch(md_files)
    _concept_embedding_cache = results

    elapsed = time.monotonic() - t0
    log.info("Embedded %d/%d concepts in %.1fs", len(_concept_embedding_cache), len(md_files), elapsed)
    _cache_loaded = True


def _keyword_fallback(query: str, concepts_dir: Path) -> list[ConceptMatch]:
    """Keyword-based fallback when embeddings are unavailable."""
    if not concepts_dir.is_dir() or not query:
        return []

    keywords = [w.lower() for w in re.split(r"[^\w]+", query) if len(w) > 2]
    if not keywords:
        return []

    matches: dict[str, float] = {}
    for md in concepts_dir.glob("*.md"):
        name = md.stem.lower()
        score = sum(1 for kw in keywords if kw in name) * 0.5
        try:
            body = md.read_text(encoding="utf-8", errors="replace").lower()
            score += sum(1 for kw in keywords if kw in body) * 0.1
        except OSError:
            continue
        if score > 0:
            matches[md.stem] = score

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    return [ConceptMatch(concept=name, score=round(score, 3)) for name, score in sorted_matches[:5]]


def run_qmd_query(
    query: str,
    qmd_cmd: str,
    collection: str,
    timeout: int = 300,
    n_results: int = 5,
    min_score: float = 0.2,
    no_rerank: bool = False,
    concepts_dir: Path | None = None,
) -> list[ConceptMatch]:
    """Semantic concept search using Ollama embeddings.

    Ignores qmd_cmd/collection (kept for API compatibility).
    """
    if not query or not query.strip():
        return []

    query_embedding = _ollama_embed(query)
    if query_embedding is None:
        return []

    concepts_dir = concepts_dir or (Path.home() / "MyVault" / "04-Wiki" / "concepts")
    _load_concept_embeddings(concepts_dir)

    if not _concept_embedding_cache:
        return []

    scores: list[tuple[str, float]] = []
    for name, emb in _concept_embedding_cache.items():
        sim = _cosine_similarity(query_embedding, emb)
        if sim >= min_score:
            scores.append((name, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [ConceptMatch(concept=name, score=round(score, 3)) for name, score in scores[:n_results]]


def run_qmd_concept_search(
    queries: dict[str, str],
    cfg,
    no_rerank: bool = False,
) -> dict[str, list[ConceptMatch]]:
    """Run semantic search for multiple sources in parallel."""
    concepts_dir = cfg.vault_path / "04-Wiki" / "concepts"

    # Pre-load concept embeddings once
    _load_concept_embeddings(concepts_dir)

    results: dict[str, list[ConceptMatch]] = {}

    def _search_one(h: str, query: str) -> tuple[str, list[ConceptMatch]]:
        matches = run_qmd_query(
            query, "", "", n_results=5, min_score=0.2,
            concepts_dir=concepts_dir,
        )
        return h, matches

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search_one, h, q): h
            for h, q in queries.items() if q.strip()
        }
        for h, q in queries.items():
            if not q.strip():
                results[h] = []
        for future in futures:
            h = futures[future]
            try:
                _, matches = future.result()
            except Exception as e:
                log.debug("Search failed for %s: %s", h, e)
                matches = _keyword_fallback(queries[h], concepts_dir)
            if not matches and queries[h].strip():
                matches = _keyword_fallback(queries[h], concepts_dir)
            results[h] = matches

    return results


def run_qmd_convergence(
    plans: list,
    cfg,
) -> dict[str, list[dict]]:
    """Concept convergence for creation stage."""
    extract_dir = cfg.resolved_extract_dir
    queries: dict[str, str] = {}

    for plan in plans:
        h = plan.hash
        content_preview = ""
        extract_file = extract_dir / f"{h}.json"
        if extract_file.exists():
            try:
                ext = json.loads(extract_file.read_text(encoding="utf-8"))
                content_preview = ext.get("content", "")[:500]
            except (json.JSONDecodeError, OSError):
                pass

        query_parts = (
            [plan.title]
            + plan.concept_new
            + plan.concept_updates
            + [content_preview]
        )
        queries[h] = " ".join(p for p in query_parts if p)[:800]

    matches = run_qmd_concept_search(queries, cfg, no_rerank=True)

    return {
        h: [{"concept": m.concept, "score": m.score} for m in ml]
        for h, ml in matches.items()
    }
