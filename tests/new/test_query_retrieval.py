"""Behavior tests for the additive deterministic query retrieval engine."""

from __future__ import annotations


def test_sparse_graph_falls_back_to_lexical_retrieval():
    from obsidian_llm_wiki.query.graph import build_graph
    from obsidian_llm_wiki.query.retrieval import retrieve

    graph = build_graph(
        {
            "concepts/attention.md": (
                "---\ntitle: Attention\naliases: [Self Attention]\n---\n"
                "Attention routes information between tokens."
            ),
            "concepts/optimizer.md": (
                "---\ntitle: Optimizer\n---\nAn optimizer updates model weights."
            ),
        }
    )

    result = retrieve("self attention", graph, max_results=2)

    assert [candidate.path for candidate in result.candidates] == ["concepts/attention.md"]
    assert result.trace.strategy == "lexical"
    assert result.trace.graph_mature is False


def test_graph_builder_resolves_frontmatter_relations_from_a_vault(tmp_path):
    from obsidian_llm_wiki.query.graph import build_graph_from_vault

    concepts = tmp_path / "concepts"
    concepts.mkdir()
    relation_page = (
        "---\ntitle: Attention\nrelations:\n  - target: optimizer\n"
        "    relation: depends_on\n---\nBody"
    )
    (concepts / "attention.md").write_text(relation_page, encoding="utf-8")
    (concepts / "optimizer.md").write_text("# Optimizer", encoding="utf-8")

    graph = build_graph_from_vault(tmp_path)

    assert [(edge.source, edge.target, edge.relation) for edge in graph.edges] == [
        ("concepts/attention.md", "concepts/optimizer.md", "depends_on")
    ]


def test_sparse_linked_graph_uses_seeded_pagerank_deterministically():
    from obsidian_llm_wiki.query.graph import build_graph
    from obsidian_llm_wiki.query.retrieval import personalized_pagerank, retrieve

    graph = build_graph(
        {
            "concepts/seed.md": "# Seed Topic\n[[bridge]]",
            "concepts/bridge.md": "# Bridge\n[[leaf]]",
            "concepts/leaf.md": "# Leaf",
        }
    )

    result = retrieve("seed topic", graph)
    first = personalized_pagerank(graph, {"concepts/seed.md": 1.0})
    second = personalized_pagerank(graph, {"concepts/seed.md": 1.0})

    assert result.trace.strategy == "seeded_ppr"
    assert result.candidates[1].path == "concepts/bridge.md"
    assert first == second
    assert first["concepts/bridge.md"] > 0


def test_mature_graph_uses_graph_first_pagerank():
    from obsidian_llm_wiki.query.graph import build_graph
    from obsidian_llm_wiki.query.retrieval import retrieve

    graph = build_graph(
        {
            "concepts/a.md": "# Seed Topic\n[[b]]",
            "concepts/b.md": "# Bridge\n[[c]]",
            "concepts/c.md": "# Context\n[[d]]",
            "concepts/d.md": "# Destination",
        }
    )

    result = retrieve("seed topic", graph)

    assert result.trace.graph_mature is True
    assert result.trace.strategy == "graph_first_ppr"
    assert result.candidates[0].path == "concepts/a.md"
    assert result.candidates[0].pagerank_score > result.candidates[0].lexical_score * 0.1


def test_cjk_aliases_are_tokenized_for_lexical_retrieval():
    from obsidian_llm_wiki.query.graph import build_graph
    from obsidian_llm_wiki.query.retrieval import retrieve, tokenize

    graph = build_graph(
        {
            "concepts/attention.md": "---\ntitle: 注意力\naliases: [自注意力机制]\n---\n模型机制。",
        }
    )

    result = retrieve("自注意力", graph)

    assert "自注意力" in tokenize("自注意力机制")
    assert result.candidates[0].path == "concepts/attention.md"


def test_cjk_tokenization_uses_bounded_ngram_sizes():
    from obsidian_llm_wiki.query.retrieval import tokenize

    tokens = tokenize("知" * 1000)

    # Whole-token + unigrams + each 2/3/4-gram: linear, not every substring.
    assert len(tokens) <= 4_000
    assert "知知" in tokens and "知知知知" in tokens
