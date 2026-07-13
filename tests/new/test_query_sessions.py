"""Behavior tests for durable, JSON-backed query sessions."""

from __future__ import annotations


def test_query_session_store_roundtrips_provenance_and_answer(tmp_path):
    from obsidian_llm_wiki.query.sessions import QuerySession, QuerySessionStore

    store = QuerySessionStore(tmp_path / ".llmwiki" / "query-sessions.json")
    saved = store.save(
        QuerySession(
            session_id="session-001",
            query="How does attention work?",
            retrieved_paths=("concepts/attention.md",),
            retrieval_trace={"strategy": "lexical", "edge_count": 0},
            profile="research",
            instructions="Be concise.",
            answer="Attention weighs token relationships [[concepts/attention.md]].",
            citation_paths=("concepts/attention.md",),
            created_at="2026-07-13T00:00:00Z",
        )
    )

    assert saved == store.load("session-001")
    assert store.list() == (saved,)
