"""Tests for semantic concept deduplication and MoC orphan assignment."""

from __future__ import annotations

from unittest import mock

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.synth.dedupe import (
    assign_orphans_to_mocs,
    merge_bundle,
    semantic_dedupe_concepts,
)

# ── Semantic dedup: gating (embeddings disabled) ────────────────────────


def test_semantic_dedupe_noop_when_embeddings_disabled():
    """With EMBEDDINGS_ENABLED=false, semantic_dedupe must be a no-op."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Digital currency")
    c2 = ConceptNote(title="BTC", slug="btc", summary="Digital currency")
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[])

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "false"}, clear=False):
        semantic_dedupe_concepts(bundle)

    assert len(bundle.concepts) == 2  # Nothing merged


def test_semantic_dedupe_merges_high_similarity_pairs():
    """Same-language concepts with sim > threshold should be merged."""
    c1 = ConceptNote(
        title="Bitcoin", slug="bitcoin", summary="Digital currency",
        tags=["crypto"], confidence=0.9,
    )
    c2 = ConceptNote(
        title="Bitcoin", slug="bitcoin-protocol", summary="Digital currency network",
        tags=["blockchain"], confidence=0.7,
    )
    moc = MapOfContent(
        title="Crypto", slug="crypto", summary="",
        concept_slugs=["bitcoin-protocol"],
    )
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[moc])

    fake_emb_a = [1.0, 0.0, 0.0]
    fake_emb_b = [0.99, 0.01, 0.0]

    def fake_embed(text):
        if "Bitcoin" in text and "network" in text:
            return fake_emb_b
        return fake_emb_a

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ), mock.patch(
        "obsidian_llm_wiki.synth.language.detect_language",
        return_value="en",
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85)

    # c1 (higher confidence) should survive, c2 merged into c1.
    assert len(bundle.concepts) == 1
    survivor = bundle.concepts[0]
    assert survivor.slug == "bitcoin"
    # Tags should be unioned.
    assert "crypto" in survivor.tags
    assert "blockchain" in survivor.tags
    # MoC should point to the surviving slug.
    assert "bitcoin" in bundle.maps[0].concept_slugs
    assert "bitcoin-protocol" not in bundle.maps[0].concept_slugs


def test_semantic_dedupe_does_not_merge_different_languages():
    """Cross-lingual pairs should not be merged by semantic_dedupe."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Digital currency")
    c2 = ConceptNote(title="比特币", slug="bi-te-bi", summary="数字货币")
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[])

    fake_emb = [1.0, 0.0]

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        return_value=fake_emb,
    ), mock.patch(
        "obsidian_llm_wiki.synth.language.detect_language",
        side_effect=lambda text: "zh" if "比特币" in text else "en",
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85)

    assert len(bundle.concepts) == 2  # Not merged


def test_semantic_dedupe_updates_concept_links():
    """ConceptLink targets should be remapped to surviving slug."""
    c1 = ConceptNote(
        title="Bitcoin", slug="bitcoin", summary="Digital currency",
        confidence=0.9,
    )
    c2 = ConceptNote(
        title="Bitcoin Protocol", slug="bitcoin-protocol",
        summary="Digital currency", confidence=0.7,
    )
    c3 = ConceptNote(
        title="Mining", slug="mining", summary="Mining",
        related=[ConceptLink(slug="bitcoin-protocol", relation="depends_on")],
    )
    bundle = SynthesisBundle(concepts=[c1, c2, c3], maps=[])

    fake_emb_a = [1.0, 0.0]
    fake_emb_b = [0.99, 0.01]
    fake_emb_c = [0.0, 1.0]

    def fake_embed(text):
        if "Mining" in text:
            return fake_emb_c
        if "Protocol" in text:
            return fake_emb_b
        return fake_emb_a

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ), mock.patch(
        "obsidian_llm_wiki.synth.language.detect_language",
        return_value="en",
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85)

    # c3's link to bitcoin-protocol should now point to bitcoin.
    mining = next(c for c in bundle.concepts if c.slug == "mining")
    assert any(r.slug == "bitcoin" for r in mining.related)
    assert not any(r.slug == "bitcoin-protocol" for r in mining.related)


def test_semantic_dedupe_below_threshold_no_merge():
    """Pairs below the similarity threshold should not be merged."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Digital currency")
    c2 = ConceptNote(title="Ethereum", slug="ethereum", summary="Smart contracts")
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[])

    fake_emb_a = [1.0, 0.0]
    fake_emb_b = [0.0, 1.0]

    def fake_embed(text):
        if "Ethereum" in text:
            return fake_emb_b
        return fake_emb_a

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ), mock.patch(
        "obsidian_llm_wiki.synth.language.detect_language",
        return_value="en",
    ):
        semantic_dedupe_concepts(bundle, threshold=0.85)

    assert len(bundle.concepts) == 2  # Not merged


# ── MoC orphan assignment: gating ────────────────────────────────────────


def test_assign_orphans_noop_when_embeddings_disabled():
    """With EMBEDDINGS_ENABLED=false, assignment must be a no-op."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Crypto")
    c2 = ConceptNote(title="Orphan", slug="orphan", summary="No MoC")
    moc = MapOfContent(
        title="Crypto", slug="crypto", summary="",
        concept_slugs=["bitcoin"],
    )
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[moc])

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "false"}, clear=False):
        assign_orphans_to_mocs(bundle)

    assert "orphan" not in bundle.maps[0].concept_slugs


def test_assign_orphans_assigns_to_similar_moc():
    """Orphan with high similarity to a MoC's average should be assigned."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Digital currency")
    c2 = ConceptNote(title="Ethereum", slug="ethereum", summary="Smart contracts")
    c3 = ConceptNote(title="Litecoin", slug="litecoin", summary="Digital currency")
    moc = MapOfContent(
        title="Crypto", slug="crypto", summary="",
        concept_slugs=["bitcoin", "ethereum"],
    )
    bundle = SynthesisBundle(concepts=[c1, c2, c3], maps=[moc])

    # c3 (litecoin) is an orphan. Give it similar embedding to c1 (bitcoin).
    emb_crypto = [1.0, 0.0, 0.0]
    emb_smart = [0.0, 1.0, 0.0]
    emb_litecoin = [0.9, 0.1, 0.0]

    def fake_embed(text):
        if "Litecoin" in text:
            return emb_litecoin
        if "Ethereum" in text:
            return emb_smart
        return emb_crypto

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ):
        assign_orphans_to_mocs(bundle, threshold=0.55)

    assert "litecoin" in bundle.maps[0].concept_slugs


def test_assign_orphans_does_not_assign_below_threshold():
    """Orphan with low similarity to all MoCs should not be assigned."""
    c1 = ConceptNote(title="Bitcoin", slug="bitcoin", summary="Digital currency")
    c2 = ConceptNote(title="Pizza", slug="pizza", summary="Italian food")
    moc = MapOfContent(
        title="Crypto", slug="crypto", summary="",
        concept_slugs=["bitcoin"],
    )
    bundle = SynthesisBundle(concepts=[c1, c2], maps=[moc])

    fake_emb_crypto = [1.0, 0.0]
    fake_emb_food = [0.0, 1.0]

    def fake_embed(text):
        if "Pizza" in text:
            return fake_emb_food
        return fake_emb_crypto

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        side_effect=fake_embed,
    ):
        assign_orphans_to_mocs(bundle, threshold=0.55)

    assert "pizza" not in bundle.maps[0].concept_slugs


def test_assign_orphans_no_orphans():
    """If all concepts are in MoCs, assignment is a no-op."""
    c1 = ConceptNote(title="A", slug="a", summary="A")
    moc = MapOfContent(title="M", slug="m", summary="", concept_slugs=["a"])
    bundle = SynthesisBundle(concepts=[c1], maps=[moc])

    with mock.patch.dict("os.environ", {"EMBEDDINGS_ENABLED": "true"}, clear=False), mock.patch(
        "obsidian_llm_wiki.synth.embedding.embed_text",
        return_value=[1.0],
    ):
        assign_orphans_to_mocs(bundle)

    assert bundle.maps[0].concept_slugs == ["a"]


# ── Integration: render_vault calls new functions ────────────────────────


def test_render_vault_calls_semantic_dedupe_and_moc_assignment(tmp_path):
    """render_vault should call semantic_dedupe and assign_orphans without error."""
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.render.obsidian import render_vault

    sources = {
        "paper.md": SourceDoc(title="Paper", content="Content", url="https://x.com"),
    }
    synth = SourceSynthesis(
        source_title="Paper", source_summary="Summary",
        concepts=[
            ConceptNote(
                title="Concept A", slug="concept-a", summary="A",
                sections=[BodySection(heading="Core", points=["Detail"])],
            ),
        ],
        maps=[
            MapOfContent(title="Topic", slug="topic", summary="MOC",
                         concept_slugs=["concept-a"]),
        ],
    )
    bundle = merge_bundle([synth])
    written = render_vault(tmp_path / "04-Wiki", bundle, sources)

    # Graph export files should be written.
    graph_json = tmp_path / "04-Wiki" / ".llmwiki" / "graph.json"
    graph_mmd = tmp_path / "04-Wiki" / ".llmwiki" / "graph.mmd"
    assert graph_json.exists()
    assert graph_mmd.exists()
    assert any(str(graph_json) in w for w in written)
