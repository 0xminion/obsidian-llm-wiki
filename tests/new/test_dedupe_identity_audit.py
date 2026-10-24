"""Explicit identity, rather than embedding similarity, authorizes merges."""

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptLink,
    ConceptNote,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.synth.dedupe import merge_bundle, normalise_tags, semantic_dedupe_concepts


def test_cosine_similarity_and_equal_titles_do_not_authorize_destructive_merge():
    concepts = [ConceptNote("Shared title", "a", "A"), ConceptNote("Shared title", "b", "B")]
    bundle = SynthesisBundle(concepts=concepts)
    semantic_dedupe_concepts(bundle, embeddings_cache={"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert [c.slug for c in bundle.concepts] == ["a", "b"]


def test_explicit_merge_remaps_claims_and_excludes_self_edges():
    a = ConceptNote("A", "a", "", canonical_id="same", confidence=0.9)
    b = ConceptNote(
        "B",
        "b",
        "",
        canonical_id="same",
        confidence=0.8,
        claims=[Claim("Fact", concept_slug="b")],
        related=[ConceptLink("a")],
    )
    bundle = SynthesisBundle(concepts=[a, b])
    semantic_dedupe_concepts(bundle, embeddings_cache={"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert len(bundle.concepts) == 1
    assert bundle.concepts[0].claims[0].concept_slug == "a"
    assert not bundle.concepts[0].related


def test_merge_aggregation_does_not_contaminate_source_local_content():
    a = SourceSynthesis(
        "A",
        "",
        concepts=[
            ConceptNote("Shared", "shared", "", sections=[BodySection("A", prose="Original A")])
        ],
    )
    b = SourceSynthesis(
        "B",
        "",
        concepts=[
            ConceptNote("Shared", "shared", "", sections=[BodySection("B", prose="Original B")])
        ],
    )
    once = merge_bundle([a, b])
    twice = merge_bundle([a, b])
    assert [s.heading for s in a.concepts[0].sections] == ["A"]
    assert once.concepts == twice.concepts


def test_chinese_tags_are_preserved():
    assert normalise_tags(["机器学习", "知识图谱"]) == ["机器学习", "知识图谱"]
