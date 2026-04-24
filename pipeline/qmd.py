"""Semantic concept search — QMD MCP integration.

Replaces the previous Ollama embedding backend with QMD's MCP HTTP server.
QMD runs as a long-lived daemon on port 8181, avoiding repeated model loading.

Search priority (per user requirement):
  1. QMD query (hybrid / vec semantic) — default
  2. QMD lex (BM25 keyword fallback)              — fallback1
  3. Local keyword fallback (no QMD server)       — fallback2

Environment:
  QMD_MCP_URL — override base URL (default http://localhost:8181)
  USE_QMD_MCP — "true" to use QMD (default "true" if QMD server is reachable)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from pipeline.models import ConceptMatch
from pipeline.qmd_mcp import QMDMCPClient, _qmd_results_to_concept_matches

log = logging.getLogger(__name__)

_DEFAULT_MCP_URL = os.environ.get("QMD_MCP_URL", "http://localhost:8181")

# Module-level QMD client (lazy init)
_qmd_client: QMDMCPClient | None = None
_qmd_available: bool | None = None


def _get_client(base_url: str = "") -> QMDMCPClient | None:
    """Return the module-level QMD client, or None if QMD is unavailable."""
    global _qmd_client, _qmd_available
    if _qmd_available is False:
        return None
    if _qmd_client is None:
        url = base_url or _DEFAULT_MCP_URL
        client = QMDMCPClient(base_url=url, timeout=60)
        if client.ensure_session():
            _qmd_client = client
            _qmd_available = True
            log.info("QMD MCP client connected to %s", url)
        else:
            _qmd_available = False
            log.warning("QMD MCP server not reachable at %s", url)
            return None
    return _qmd_client


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
        # Default: QMD query (vec+rerank), fallback to lex
        results = client.query(
            query_text=query.strip(),
            n_results=n_results,
            min_score=min_score,
        )
        if results:
            return _qmd_results_to_concept_matches(results, collection_filter="concepts")
        # Fallback 1: BM25 keyword via QMD
        results = client._query_raw(
            searches=[{"type": "lex", "query": query.strip()}],
            n=n_results,
            min_score=min_score,
        )
        matches = _qmd_results_to_concept_matches(results, collection_filter="concepts")
        if matches:
            return matches

    # Fallback 2: local keyword fallback
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
    No local embedding cache or ThreadPoolExecutor is needed.
    """
    concepts_dir = cfg.vault_path / "04-Wiki" / "concepts"
    client = _get_client()
    results: dict[str, list[ConceptMatch]] = {}

    if client is None:
        # QMD unavailable — pure keyword fallback
        for h, q in queries.items():
            results[h] = _keyword_fallback(q, concepts_dir) if q.strip() else []
        return results

    for h, q in queries.items():
        if not q.strip():
            results[h] = []
            continue
        qmd_results = client.query(
            query_text=q.strip(),
            n_results=5,
            min_score=0.2,
        )
        matches = _qmd_results_to_concept_matches(qmd_results, collection_filter="concepts")
        if not matches:
            matches = _keyword_fallback(q, concepts_dir)
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
