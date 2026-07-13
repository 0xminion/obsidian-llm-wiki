"""Behavior tests for grounded query context and citations."""

from __future__ import annotations

import pytest


def test_citation_validator_rejects_paths_not_in_retrieved_candidates():
    from obsidian_llm_wiki.query.context import CitationError, require_valid_citations
    from obsidian_llm_wiki.query.retrieval import RetrievedPage

    candidates = (
        RetrievedPage("concepts/attention.md", "Attention", 2.0, 2.0, 0.0),
    )

    with pytest.raises(CitationError, match="sources/uncited.md"):
        require_valid_citations(
            "The answer is unsupported [[sources/uncited.md]].", candidates
        )


def test_snippet_is_limited_to_the_matching_section_and_bound():
    from obsidian_llm_wiki.query.context import extract_snippet

    markdown = """---
title: Attention
---
# Attention
Introductory material.
## Retrieval
This section explains lexical retrieval in enough detail to be useful.
## Unrelated
This text must not leak into the chosen snippet.
"""

    snippet = extract_snippet(markdown, "lexical retrieval", max_chars=75)

    assert "lexical retrieval" in snippet.casefold()
    assert "Unrelated" not in snippet
    assert len(snippet) <= 75
