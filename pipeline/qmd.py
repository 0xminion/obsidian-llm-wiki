"""Shared qmd semantic search module.

Consolidates qmd query logic from plan.py and create/agent.py into
a single source of truth. Both planning and creation stages use this.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from pipeline.models import ConceptMatch
from pipeline.utils import strip_qmd_noise

log = logging.getLogger(__name__)

def _keyword_fallback(query: str, concepts_dir: Path) -> list[ConceptMatch]:
    """Keyword-based fallback when qmd is unavailable or returns no results.

    Matches query words against concept filenames and content.
    Returns sorted list of ConceptMatch by simple relevance score.
    """
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
) -> list[ConceptMatch]:
    """Run a single qmd query and return concept matches.

    Handles cmake/Vulkan noise stripping, JSON parsing, and error recovery.
    Falls back to keyword search if qmd is not installed or returns no results.
    """
    if not query or not query.strip():
        return []

    cmd = [
        qmd_cmd, "query", query,
        "--json", "-n", str(n_results),
        "--min-score", str(min_score),
        "-c", collection,
    ]
    if no_rerank:
        cmd.append("--no-rerank")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout_clean = strip_qmd_noise(result.stdout)

        if result.returncode != 0 or not stdout_clean.strip().startswith("["):
            if result.returncode != 0:
                log.warning("qmd exited with code %d: %s", result.returncode, result.stderr[:200])
                # Raise so caller can fall back to keyword search
                raise OSError(f"qmd failed with code {result.returncode}")
            return []

        data = json.loads(stdout_clean)
        matches = []
        for item in data:
            if not isinstance(item, dict):
                continue
            score = item.get("score", 0)
            if score < min_score:
                continue
            f = item.get("file", item.get("path", ""))
            name = f.split("/")[-1].replace(".md", "") if "/" in f else f.replace(".md", "")
            if name:
                matches.append(ConceptMatch(concept=name, score=round(score, 3)))
        return matches

    except subprocess.TimeoutExpired:
        log.warning("qmd timeout for query: %s", query[:80])
        return []
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("qmd parse error: %s", e)
        return []
    except OSError as e:
        log.warning("qmd error: %s", e)
        return []


def run_qmd_concept_search(
    queries: dict[str, str],
    cfg,
    no_rerank: bool = False,
) -> dict[str, list[ConceptMatch]]:
    """Run qmd queries in parallel for multiple sources.

    Args:
        queries: mapping of hash -> query string
        cfg: Config object with qmd_cmd, qmd_collection, plan_timeout
        no_rerank: pass --no-rerank flag

    Returns:
        mapping of hash -> list of ConceptMatch
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    qmd_cmd = os.environ.get("QMD_CMD", cfg.qmd_cmd)
    collection = os.environ.get("QMD_COLLECTION", cfg.qmd_collection)

    results: dict[str, list[ConceptMatch]] = {}
    concepts_dir = cfg.vault_path / "04-Wiki" / "concepts"

    def _run_one(h: str, query: str) -> tuple[str, list[ConceptMatch]]:
        matches = run_qmd_query(
            query, qmd_cmd, collection,
            timeout=cfg.plan_timeout,
            no_rerank=no_rerank,
        )
        return h, matches

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_one, h, q): h
            for h, q in queries.items() if q.strip()
        }
        for h, q in queries.items():
            if not q.strip():
                results[h] = []
        for future in as_completed(futures):
            try:
                h, matches = future.result()
            except OSError:
                # qmd failed (not installed or crashed) — fall back to keyword search
                h = futures[future]
                matches = _keyword_fallback(queries[h], concepts_dir)
            if not matches and queries[h].strip():
                matches = _keyword_fallback(queries[h], concepts_dir)
            results[h] = matches

    return results


def run_qmd_convergence(
    plans: list,
    cfg,
) -> dict[str, list[dict]]:
    """Run concept convergence search for creation stage.

    Same as run_qmd_concept_search but returns dict format
    (list of {concept, score} instead of ConceptMatch objects)
    for backward compatibility with create/agent.py.
    """
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

    # Convert ConceptMatch list to dict format for backward compat
    return {
        h: [{"concept": m.concept, "score": m.score} for m in ml]
        for h, ml in matches.items()
    }
