"""Deterministic lexical and personalized-PageRank retrieval."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from obsidian_llm_wiki.query.graph import WikiGraph, WikiPage

__all__ = [
    "GraphMaturity",
    "RetrievedPage",
    "RetrievalResult",
    "RetrievalTrace",
    "assess_graph_maturity",
    "personalized_pagerank",
    "retrieve",
    "tokenize",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GraphMaturity:
    """Explicit guardrails before graph signals may outrank lexical signals."""

    min_pages: int = 4
    min_edges: int = 3
    min_connected_pages: int = 3


_DEFAULT_MATURITY = GraphMaturity()


@dataclass(frozen=True, slots=True)
class RetrievedPage:
    path: str
    title: str
    score: float
    lexical_score: float
    pagerank_score: float


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    strategy: str
    graph_mature: bool
    page_count: int
    edge_count: int
    lexical_terms: tuple[str, ...]
    seed_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[RetrievedPage, ...]
    trace: RetrievalTrace


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Latin text plus CJK words and characters without a segmenter."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.casefold()):
        token = match.group(0)
        tokens.append(token)
        if _contains_cjk(token):
            for length in range(2, len(token)):
                chunks = (
                    token[index : index + length]
                    for index in range(len(token) - length + 1)
                )
                tokens.extend(chunks)
            tokens.extend(token)
    return tuple(tokens)


def assess_graph_maturity(
    graph: WikiGraph, threshold: GraphMaturity = _DEFAULT_MATURITY
) -> bool:
    """Return whether ``graph`` has enough connected evidence for graph-first ranking."""
    connected = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
    return (
        graph.page_count >= threshold.min_pages
        and graph.edge_count >= threshold.min_edges
        and len(connected) >= threshold.min_connected_pages
    )


def personalized_pagerank(
    graph: WikiGraph,
    seeds: Mapping[str, float] | None = None,
    *,
    damping: float = 0.85,
    iterations: int = 40,
) -> Mapping[str, float]:
    """Compute stable personalized PageRank over the graph without NumPy.

    Dangling pages redistribute their rank to the personalization distribution.
    Sorting paths at every transition makes results reproducible across runs.
    """
    paths = tuple(sorted(graph.pages))
    if not paths:
        return MappingProxyType({})
    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be between zero and one")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    personalization = _normalize_seeds(paths, seeds)
    scores = dict(personalization)
    for _ in range(iterations):
        next_scores = {path: (1.0 - damping) * personalization[path] for path in paths}
        for source in paths:
            targets = graph.outbound[source]
            if targets:
                contribution = damping * scores[source] / len(targets)
                for target in targets:
                    next_scores[target] += contribution
            else:
                contribution = damping * scores[source]
                for target in paths:
                    next_scores[target] += contribution * personalization[target]
        scores = next_scores
    return MappingProxyType(scores)


def retrieve(
    query: str,
    graph: WikiGraph,
    *,
    max_results: int = 10,
    maturity: GraphMaturity = _DEFAULT_MATURITY,
) -> RetrievalResult:
    """Retrieve pages with lexical fallback, seeded PPR, or mature graph-first PPR."""
    if max_results < 1:
        raise ValueError("max_results must be positive")
    terms = tuple(dict.fromkeys(tokenize(query)))
    lexical = {path: _lexical_score(page, terms) for path, page in graph.pages.items()}
    seed_scores = {path: score for path, score in lexical.items() if score > 0.0}
    graph_mature = assess_graph_maturity(graph, maturity)
    has_edges = graph.edge_count > 0

    if seed_scores and has_edges:
        pagerank = personalized_pagerank(graph, seed_scores)
        if graph_mature:
            strategy = "graph_first_ppr"
            combined = {
                path: pagerank[path] + lexical[path] * 0.01 for path in graph.pages
            }
        else:
            strategy = "seeded_ppr"
            combined = {
                path: lexical[path] + pagerank[path] * 0.25 for path in graph.pages
            }
    else:
        strategy = "lexical"
        pagerank = dict.fromkeys(graph.pages, 0.0)
        combined = lexical

    candidates = [
        RetrievedPage(
            path=path,
            title=graph.pages[path].title,
            score=combined[path],
            lexical_score=lexical[path],
            pagerank_score=pagerank[path],
        )
        for path in graph.pages
        if combined[path] > 0.0
    ]
    candidates.sort(key=lambda page: (-page.score, -page.lexical_score, page.path))
    trace = RetrievalTrace(
        strategy=strategy,
        graph_mature=graph_mature,
        page_count=graph.page_count,
        edge_count=graph.edge_count,
        lexical_terms=terms,
        seed_paths=tuple(sorted(seed_scores)),
    )
    return RetrievalResult(candidates=tuple(candidates[:max_results]), trace=trace)


def _contains_cjk(token: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in token)


def _lexical_score(page: WikiPage, terms: tuple[str, ...]) -> float:
    title_terms = tokenize(page.title)
    alias_terms = tokenize(" ".join(page.aliases))
    body_terms = tokenize(page.body)
    return float(
        sum(
            4 * title_terms.count(term) + 3 * alias_terms.count(term) + body_terms.count(term)
            for term in terms
        )
    )


def _normalize_seeds(paths: tuple[str, ...], seeds: Mapping[str, float] | None) -> dict[str, float]:
    weighted = {
        path: max(0.0, float((seeds or {}).get(path, 0.0)))
        for path in paths
    }
    total = sum(weighted.values())
    if total == 0.0:
        return {path: 1.0 / len(paths) for path in paths}
    return {path: weighted[path] / total for path in paths}
