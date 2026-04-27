"""Semantic concept search — QMD MCP integration."""

from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.models import ConceptMatch
from pipeline.qmd_mcp import QMDMCPClient, _qmd_results_to_concept_matches

log = logging.getLogger(__name__)

_DEFAULT_MCP_URL = os.environ.get("QMD_MCP_URL", "http://localhost:8181")


def _get_client(base_url: str = "") -> QMDMCPClient | None:
    """Return a fresh QMD client, checking server health each time.

    The QMD *server* keeps embedding models loaded across requests,
    so we don't need to cache the client object.  Re-checking health
    each call means we automatically recover if the server restarts.
    """
    url = base_url or _DEFAULT_MCP_URL
    client = QMDMCPClient(base_url=url, timeout=60)
    try:
        h = client.health()
        if h.get("status") != "ok":
            log.debug("QMD MCP server not healthy at %s", url)
            return None
    except Exception:
        log.debug("QMD MCP server unreachable at %s", url)
        return None
    if client.ensure_session():
        return client
    return None


def _keyword_fallback(query: str, concepts_dir: Path) -> list[ConceptMatch]:
    """Keyword-based fallback — works without QMD or embeddings."""
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
    """Run semantic concept search via QMD MCP.

    Args:
        qmd_cmd, collection, timeout — kept for backward compatibility but unused.
        no_rerank — unused with QMD (QMD handles reranking internally).
    """
    if not query or not query.strip():
        return []

    client = _get_client()
    if client is not None:
        # client.query() already performs vec → lex fallback internally
        results = client.query(
            query_text=query.strip(),
            n_results=n_results,
            min_score=min_score,
            collections=[collection or "concepts"],
        )
        if results:
            return _qmd_results_to_concept_matches(results, collection_filter=collection or "concepts")

    # Fallback: local keyword fallback
    if concepts_dir is not None:
        return _keyword_fallback(query, concepts_dir)
    return []


def run_qmd_concept_search(
    queries: dict[str, str],
    cfg,
    no_rerank: bool = False,
) -> dict[str, list[ConceptMatch]]:
    """Run semantic search for multiple sources in parallel.

    With QMD MCP, each query is a lightweight HTTP call to a pre-loaded model.
    Queries are batched via ThreadPoolExecutor at PARALLEL workers.
    """
    concepts_dir = cfg.vault_path / "04-Wiki" / "concepts"
    client = _get_client()
    results: dict[str, list[ConceptMatch]] = {}

    if client is None:
        # QMD unavailable — pure keyword fallback
        for h, q in queries.items():
            results[h] = _keyword_fallback(q, concepts_dir) if q.strip() else []
        return results

    def _query_one(h: str, q: str) -> tuple[str, list[ConceptMatch]]:
        if not q.strip():
            return h, []
        qmd_results = client.query(
            query_text=q.strip(),
            n_results=5,
            min_score=0.2,
            collections=[cfg.qmd_collection or "concepts"],
        )
        matches = _qmd_results_to_concept_matches(
            qmd_results, collection_filter=cfg.qmd_collection or "concepts"
        )
        if not matches:
            matches = _keyword_fallback(q, concepts_dir)
        return h, matches

    with ThreadPoolExecutor(max_workers=cfg.parallel) as executor:
        futures = [
            executor.submit(_query_one, h, q) for h, q in queries.items()
        ]
        for future in as_completed(futures):
            h, matches = future.result()
            results[h] = matches

    return results


def run_qmd_convergence(
    plans: list,
    cfg,
) -> dict[str, list[dict]]:
    """Concept convergence for creation stage."""
    extract_dir = cfg.resolved_extract_dir
    queries: dict[str, str] = {}
    import json as _json

    for plan in plans:
        h = plan.hash
        content_preview = ""
        extract_file = extract_dir / f"{h}.json"
        if extract_file.exists():
            try:
                ext = _json.loads(extract_file.read_text(encoding="utf-8"))
                content_preview = ext.get("content", "")[:500]
            except (_json.JSONDecodeError, OSError):
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


# ─── Embedding helpers (Rec 3) ──────────────────────────────────────────────


def batch_embed(texts: list[str], client=None) -> dict[str, list[float]]:
    """Embed a batch of texts using QMD MCP.

    Falls back to sequential single-text embeds if the batch endpoint
    is not supported by the QMD server."""
    if not texts:
        return {}
    _client = client or _get_client()
    if _client is None:
        return {}
    try:
        result = _client.embed_batch(texts)
        if result:
            return result
    except Exception:
        log.debug("QMD batch embed failed, falling back to sequential")

    # Sequential fallback
    results: dict[str, list[float]] = {}
    for text in texts:
        try:
            emb = _client.embed(text)
            if emb:
                results[text] = emb
        except Exception:
            continue
    return results


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two text embeddings via QMD.

    Returns 0.0 if QMD is unreachable or either embedding fails."""
    if not text_a or not text_b:
        return 0.0
    embeddings = batch_embed([text_a, text_b])
    emb_a = embeddings.get(text_a)
    emb_b = embeddings.get(text_b)
    if not emb_a or not emb_b:
        return 0.0
    dot = sum(x * y for x, y in zip(emb_a, emb_b))
    norm_a = math.sqrt(sum(x * x for x in emb_a))
    norm_b = math.sqrt(sum(x * x for x in emb_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
