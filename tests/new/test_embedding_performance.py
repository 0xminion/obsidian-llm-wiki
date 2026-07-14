"""Tests for embedding persistence, cache sharing, batch embedding, and
incremental (new-vs-existing) semantic dedup comparison."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from obsidian_llm_wiki.synth.embedding import (
    cosine_similarity,
    embed_batch,
    embed_text,
    find_cross_lingual_links,
    load_embeddings_cache,
    save_embeddings_cache,
)


# ── Fix 1: Embedding persistence ──────────────────────────────────────


def test_save_and_load_embeddings_round_trip(tmp_path):
    """Saved embeddings can be loaded back with the same values (6-decimal rounded)."""
    cache_path = tmp_path / "embeddings.json"
    cache = {
        "concept-a": [0.123456789, 0.987654321, -0.5],
        "concept-b": [0.0, 1.0, 0.5],
    }
    save_embeddings_cache(cache_path, cache)
    assert cache_path.exists()

    loaded = load_embeddings_cache(cache_path)
    assert "concept-a" in loaded
    assert "concept-b" in loaded
    # Values should be rounded to 6 decimals
    assert loaded["concept-a"][0] == pytest.approx(0.123457, abs=1e-6)
    assert loaded["concept-a"][1] == pytest.approx(0.987654, abs=1e-6)
    assert loaded["concept-a"][2] == pytest.approx(-0.5, abs=1e-6)


def test_load_returns_empty_when_file_missing(tmp_path):
    """Missing cache file returns empty dict."""
    assert load_embeddings_cache(tmp_path / "nonexistent.json") == {}


def test_load_returns_empty_when_corrupt(tmp_path):
    """Corrupt JSON file returns empty dict."""
    cache_path = tmp_path / "embeddings.json"
    cache_path.write_text("not valid json{{{")
    assert load_embeddings_cache(cache_path) == {}


def test_load_returns_empty_on_model_mismatch(tmp_path):
    """Cache from a different embedding model is invalidated."""
    cache_path = tmp_path / "embeddings.json"
    cache_path.write_text(json.dumps({
        "_model": "old-model",
        "embeddings": {"concept-a": [0.1, 0.2]},
    }))
    with mock.patch(
        "obsidian_llm_wiki.synth.embedding._embedding_model",
        return_value="new-model",
    ):
        assert load_embeddings_cache(cache_path) == {}


def test_save_does_nothing_for_empty_cache(tmp_path):
    """Empty cache should not create a file."""
    cache_path = tmp_path / "embeddings.json"
    save_embeddings_cache(cache_path, {})
    assert not cache_path.exists()


def test_save_includes_model_key(tmp_path):
    """Saved cache includes _model for invalidation on model switch."""
    cache_path = tmp_path / "embeddings.json"
    with mock.patch(
        "obsidian_llm_wiki.synth.embedding._embedding_model",
        return_value="test-model-xyz",
    ):
        save_embeddings_cache(cache_path, {"a": [0.1]})
    data = json.loads(cache_path.read_text())
    assert data["_model"] == "test-model-xyz"


# ── Fix 2: Shared cache in find_cross_lingual_links ─────────────────────


def test_find_cross_lingual_links_reuses_existing_embeddings():
    """When embeddings_cache has a concept's embedding, embed_text is NOT called for it."""
    from obsidian_llm_wiki.core.models import ConceptNote

    concept_a = ConceptNote(title="Test A", slug="a", summary="summary a")
    concept_b = ConceptNote(title="Test B", slug="b", summary="summary b")

    # Pre-populate cache with embeddings
    pre_cache = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
    }

    with mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
    ) as mock_embed:
        links = find_cross_lingual_links(
            [concept_a, concept_b],
            enabled=True,
            embeddings_cache=pre_cache,
        )
        # embed_text should not be called since both are in cache
        mock_embed.assert_not_called()

    # The cache should still have both entries
    assert "a" in pre_cache
    assert "b" in pre_cache


def test_find_cross_lingual_links_computes_missing_embeddings():
    """When a concept is missing from cache, embed_text IS called for it."""
    from obsidian_llm_wiki.core.models import ConceptNote

    concept_a = ConceptNote(title="Test A", slug="a", summary="summary a")
    concept_b = ConceptNote(title="Test B", slug="b", summary="summary b")

    pre_cache = {"a": [1.0, 0.0]}

    def fake_embed(text, **kw):
        return [0.0, 1.0]

    with mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ) as mock_embed:
        links = find_cross_lingual_links(
            [concept_a, concept_b],
            enabled=True,
            embeddings_cache=pre_cache,
        )
        # embed_text should be called once for concept_b (missing from cache)
        assert mock_embed.call_count == 1
    assert "b" in pre_cache


# ── Fix 3: Incremental (new-vs-existing) dedup ─────────────────────────


def test_dedup_only_compares_new_concepts_against_all():
    """When new_slugs is provided, existing-vs-existing pairs are NOT compared."""
    from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts
    from obsidian_llm_wiki.core.models import ConceptNote, SynthesisBundle

    # Create 3 concepts with known embeddings
    # a and b are similar (should merge), c is different
    concepts = [
        ConceptNote(title="Concept A", slug="a", summary="alpha", confidence=0.9),
        ConceptNote(title="Concept B", slug="b", summary="alpha variant", confidence=0.8),
        ConceptNote(title="Concept C", slug="c", summary="completely different", confidence=1.0),
    ]
    bundle = SynthesisBundle(
        sources=[], concepts=concepts, maps=[],
        errors=[],
    )

    embeddings = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.99, 0.01, 0.0],  # very similar to a
        "c": [0.0, 0.0, 1.0],   # orthogonal
    }

    # Track which pairs cosine_similarity is called with
    compared_pairs: list[tuple[str, str]] = []
    original_cosine = cosine_similarity

    def tracking_cosine(a_vec, b_vec):
        # Find which slugs these belong to
        slug_a = None
        slug_b = None
        for slug, vec in embeddings.items():
            if vec == a_vec:
                slug_a = slug
            if vec == b_vec:
                slug_b = slug
        compared_pairs.append((slug_a, slug_b))
        return original_cosine(a_vec, b_vec)

    with mock.patch(
        "obsidian_llm_wiki.synth.dedupe._embed_concept",
        side_effect=lambda c, cache, opts: embeddings.get(c.slug),
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding.cosine_similarity",
        side_effect=tracking_cosine,
    ):
        # Only 'b' is new — should only compare b vs {a, b, c}
        semantic_dedupe_concepts(
            bundle,
            threshold=0.85,
            new_slugs={"b"},
        )

    # Extract comparisons involving b
    b_comparisons = [p for p in compared_pairs if "b" in p]
    # Extract comparisons involving a (should NOT happen since a is existing)
    a_initiated = [p for p in compared_pairs if p[0] == "a" or p[1] == "a"]
    # a-initiated comparisons should only include a-vs-b (where b initiated)
    # NOT a-vs-c (both existing, should be skipped)
    a_vs_c = any(
        (p[0] == "a" and p[1] == "c") or (p[0] == "c" and p[1] == "a")
        for p in compared_pairs
    )
    assert not a_vs_c, "existing-vs-existing pair (a, c) should not be compared"


def test_dedup_compares_all_when_new_slugs_none():
    """When new_slugs is None (default), all pairs are compared (backward compat)."""
    from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts
    from obsidian_llm_wiki.core.models import ConceptNote, SynthesisBundle

    concepts = [
        ConceptNote(title="A", slug="a", summary="alpha", confidence=0.9),
        ConceptNote(title="B", slug="b", summary="alpha variant", confidence=0.8),
        ConceptNote(title="C", slug="c", summary="completely different", confidence=1.0),
    ]
    bundle = SynthesisBundle(sources=[], concepts=concepts, maps=[], errors=[])

    embeddings = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],
        "c": [0.0, 1.0],
    }

    compare_count = 0
    original_cosine = cosine_similarity

    def counting_cosine(a, b):
        nonlocal compare_count
        compare_count += 1
        return original_cosine(a, b)

    with mock.patch(
        "obsidian_llm_wiki.synth.dedupe._embed_concept",
        side_effect=lambda c, cache, opts: embeddings.get(c.slug),
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding.cosine_similarity",
        side_effect=counting_cosine,
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85)

    # With 3 concepts, default behaviour should compare multiple pairs
    assert compare_count > 0


def test_dedup_new_concept_merges_with_existing():
    """A new concept that's very similar to an existing one should merge."""
    from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts
    from obsidian_llm_wiki.core.models import ConceptNote, SynthesisBundle

    concepts = [
        ConceptNote(title="Existing", slug="existing", summary="prediction market", confidence=1.0),
        ConceptNote(title="New", slug="new", summary="prediction markets", confidence=0.8),
    ]
    bundle = SynthesisBundle(sources=[], concepts=concepts, maps=[], errors=[])

    embeddings = {
        "existing": [1.0, 0.0, 0.0],
        "new": [0.99, 0.01, 0.0],  # very similar
    }

    with mock.patch(
        "obsidian_llm_wiki.synth.dedupe._embed_concept",
        side_effect=lambda c, cache, opts: embeddings.get(c.slug),
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85, new_slugs={"new"})

    # "new" should have been merged into "existing" (higher confidence)
    remaining = [c.slug for c in bundle.concepts]
    assert "existing" in remaining
    assert "new" not in remaining


# ── Fix 4: Batch embedding ─────────────────────────────────────────────


def test_embed_batch_returns_none_for_empty():
    """Empty input returns empty list."""
    assert embed_batch([], enabled=True) == []


def test_embed_batch_disabled_returns_nones():
    """When disabled, returns None for each input text."""
    result = embed_batch(["text1", "text2"], enabled=False)
    assert result == [None, None]


def test_embed_batch_calls_api_with_batches():
    """Batch embedding sends multiple texts in one API call."""
    texts = [f"concept {i}" for i in range(5)]

    mock_response = mock.Mock(status_code=200)
    mock_response.json.return_value = {
        "embeddings": [[0.1 * i, 0.2] for i in range(5)],
    }

    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.post.return_value = mock_response

    with mock.patch(
        "obsidian_llm_wiki.synth.embedding.httpx.Client",
        return_value=client,
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding._embedding_model",
        return_value="test-model",
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding._ollama_host",
        return_value="http://test:11434",
    ):
        result = embed_batch(texts, enabled=True, batch_size=8)

    assert len(result) == 5
    assert all(r is not None for r in result)
    # Should have made 1 API call (5 texts fit in 1 batch of 8)
    assert client.post.call_count == 1


def test_embed_batch_fallback_on_api_error():
    """When batch API fails, falls back to individual embed_text calls."""
    texts = ["text1", "text2"]

    mock_response = mock.Mock(status_code=500)
    mock_response.json.return_value = {}

    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.post.return_value = mock_response

    with mock.patch(
        "obsidian_llm_wiki.synth.embedding.httpx.Client",
        return_value=client,
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding._embedding_model",
        return_value="test-model",
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding._ollama_host",
        return_value="http://test:11434",
    ), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        return_value=[0.5, 0.5],
    ) as mock_embed_text:
        result = embed_batch(texts, enabled=True, batch_size=8)

    # Should have fallen back to individual calls
    assert mock_embed_text.call_count == 2
    assert result == [[0.5, 0.5], [0.5, 0.5]]