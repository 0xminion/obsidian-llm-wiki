import pytest

from obsidian_llm_wiki.query import retrieval
from obsidian_llm_wiki.query.graph import build_graph


def test_dangling_rank_distribution_is_linear_and_preserves_mass(monkeypatch):
    graph = build_graph({f"p{i}.md": "needle " + ("[[p1]]" if i == 0 else "") for i in range(200)})
    normalize = retrieval._normalize_seeds
    accesses = []

    class Counted(dict):
        def __getitem__(self, key):
            accesses.append(key)
            return super().__getitem__(key)

    monkeypatch.setattr(
        retrieval, "_normalize_seeds", lambda paths, seeds: Counted(normalize(paths, seeds))
    )
    scores = retrieval.personalized_pagerank(graph, {"p0.md": 1}, iterations=2)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["p0.md"] == pytest.approx(0.8725)
    assert scores["p1.md"] == pytest.approx(0.1275)
    assert len(accesses) <= 200 * 2 * 3
