"""Bounded lexical and ANN retrieval for synthesis prompt context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from obsidian_llm_wiki.core.registry import ConceptCandidate, VaultRegistry

__all__ = ["ConceptRetriever", "RetrievalResult"]


@dataclass(frozen=True)
class RetrievalResult:
    """Candidates and audited resource counters for one source."""

    candidates: list[ConceptCandidate]
    prompt_lines: list[str]
    candidate_chars: int
    ann_evaluated: int


class ConceptRetriever:
    """Retrieve only the most relevant existing concepts for one source."""

    def __init__(
        self,
        registry: VaultRegistry,
        *,
        candidate_limit: int = 100,
        candidate_char_budget: int = 16_000,
        ann_candidate_pool: int = 1_000,
        embedding_model: str = "",
        embedder: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if candidate_char_budget < 256:
            raise ValueError("candidate_char_budget must be at least 256")
        if ann_candidate_pool < candidate_limit:
            raise ValueError("ann_candidate_pool must be >= candidate_limit")
        self.registry = registry
        self.candidate_limit = candidate_limit
        self.candidate_char_budget = candidate_char_budget
        self.ann_candidate_pool = ann_candidate_pool
        self.embedding_model = embedding_model
        self.embedder = embedder

    def retrieve(self, source_title: str, source_content: str) -> RetrievalResult:
        """Union bounded FTS and ANN results and format stable-ID prompt lines."""
        query = f"{source_title}\n{source_content[:8_000]}"
        lexical = self.registry.search_concepts(query, limit=self.candidate_limit)
        ordered: list[ConceptCandidate] = list(lexical)
        seen = {candidate.id for candidate in ordered}
        ann_evaluated = 0

        if self.embedder is not None and self.embedding_model:
            vector = self.embedder(query)
            if vector:
                ann = self.registry.search_ann(
                    "concept",
                    vector,
                    model_id=self.embedding_model,
                    limit=self.candidate_limit,
                    candidate_pool=self.ann_candidate_pool,
                )
                ann_evaluated = ann.evaluated
                semantic = self.registry.concept_candidates_by_id(
                    [candidate.entity_id for candidate in ann.candidates]
                )
                for candidate in semantic:
                    if candidate.id not in seen:
                        ordered.append(candidate)
                        seen.add(candidate.id)

        lines: list[str] = []
        accepted: list[ConceptCandidate] = []
        characters = 0
        for candidate in ordered:
            if len(accepted) >= self.candidate_limit:
                break
            title = " ".join(candidate.title.split())[:240]
            line = f"{candidate.id} | {candidate.slug} | {title}"
            added = len(line) + (1 if lines else 0)
            if characters + added > self.candidate_char_budget:
                continue
            lines.append(line)
            accepted.append(candidate)
            characters += added

        return RetrievalResult(
            candidates=accepted,
            prompt_lines=lines,
            candidate_chars=characters,
            ann_evaluated=ann_evaluated,
        )
