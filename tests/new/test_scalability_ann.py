from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.core.pipeline import _normalize_synthesis_identity
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.synth.dedupe import (
    assign_orphans_to_mocs,
    semantic_dedupe_concepts,
)
from obsidian_llm_wiki.synth.embedding import (
    find_cross_lingual_links,
    load_embeddings_cache,
    save_embeddings_cache,
)


def _unit(angle: float, dimensions: int = 16) -> list[float]:
    vector = [0.0] * dimensions
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector


def test_registry_persists_float32_vectors_and_ann_reranks_a_bounded_pool(
    tmp_path: Path,
) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_embedding(
            "concept",
            "target",
            _unit(0.01),
            model_id="embed-v1",
            input_hash="target-hash",
            schema_version="embed-schema-v1",
        )
        for index in range(200):
            registry.upsert_embedding(
                "concept",
                f"noise-{index:03d}",
                _unit(0.5 + index * 0.01),
                model_id="embed-v1",
                input_hash=f"noise-{index}",
                schema_version="embed-schema-v1",
            )

        loaded = registry.load_embedding("concept", "target", model_id="embed-v1")
        result = registry.search_ann(
            "concept",
            _unit(0.0),
            model_id="embed-v1",
            limit=5,
            candidate_pool=40,
        )

    assert loaded == pytest.approx(_unit(0.01), abs=1e-6)
    assert result.candidates[0].entity_id == "target"
    assert len(result.candidates) <= 5
    assert result.evaluated <= 40


def test_ann_index_is_versioned_by_model_and_dimensions(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_embedding(
            "concept",
            "concept-a",
            [1.0, 0.0, 0.0],
            model_id="embed-v1",
            input_hash="hash-v1",
            schema_version="schema-v1",
        )

        wrong_model = registry.search_ann(
            "concept", [1.0, 0.0, 0.0], model_id="embed-v2", limit=10
        )
        wrong_dimensions = registry.search_ann(
            "concept", [1.0, 0.0], model_id="embed-v1", limit=10
        )

    assert wrong_model.candidates == []
    assert wrong_dimensions.candidates == []


def test_semantic_operations_inspect_only_supplied_ann_candidates() -> None:
    concepts = [
        ConceptNote(
            "Alpha", "alpha", "same meaning", confidence=1.0,
            canonical_id="concept-alpha",
        ),
        ConceptNote(
            "Alpha Duplicate", "alpha-duplicate", "same meaning", confidence=0.9,
            canonical_id="concept-duplicate",
        ),
        ConceptNote("Excluded Duplicate", "excluded", "same meaning", confidence=0.8),
    ]
    synthesis = SourceSynthesis(
        source_title="Source",
        source_summary="summary",
        source_file="source.md",
        concepts=list(concepts),
    )
    bundle = SynthesisBundle(sources=[synthesis], concepts=concepts)
    embeddings = {
        "alpha": [1.0, 0.0],
        "alpha-duplicate": [1.0, 0.0],
        "excluded": [1.0, 0.0],
    }

    semantic_dedupe_concepts(
        bundle,
        threshold=0.9,
        embeddings_cache=embeddings,
        new_slugs={"alpha"},
        candidate_slugs_by_slug={"alpha": ["alpha-duplicate"]},
    )

    assert {concept.slug for concept in bundle.concepts} == {"alpha", "excluded"}
    source_alpha = next(
        concept for concept in synthesis.concepts if concept.slug == "alpha"
    )
    assert source_alpha.canonical_id == "concept-alpha"


def test_orphan_assignment_and_cross_lingual_links_use_ann_candidate_maps() -> None:
    concepts = [
        ConceptNote("Orphan", "orphan", "orphan concept"),
        ConceptNote("Member One", "member-one", "first map"),
        ConceptNote("Member Two", "member-two", "second map"),
    ]
    bundle = SynthesisBundle(
        concepts=concepts,
        maps=[
            MapOfContent("Map One", "map-one", "First.", concept_slugs=["member-one"]),
            MapOfContent("Map Two", "map-two", "Second.", concept_slugs=["member-two"]),
        ],
    )
    embeddings = {
        "orphan": [1.0, 0.0],
        "member-one": [1.0, 0.0],
        "member-two": [1.0, 0.0],
    }
    assign_orphans_to_mocs(
        bundle,
        threshold=0.5,
        embeddings_cache=embeddings,
        candidate_mocs_by_slug={"orphan": ["map-two"]},
    )
    assert "orphan" not in bundle.maps[0].concept_slugs
    assert "orphan" in bundle.maps[1].concept_slugs

    bilingual = [
        ConceptNote("Knowledge Graph", "knowledge-graph", "graph structure"),
        ConceptNote("知识图谱", "knowledge-graph-zh", "图谱结构"),
        ConceptNote("知识网络", "excluded-zh", "图谱结构"),
    ]
    links = find_cross_lingual_links(
        bilingual,
        threshold=0.9,
        embeddings_cache={concept.slug: [1.0, 0.0] for concept in bilingual},
        comparison_slugs={"knowledge-graph"},
        candidate_slugs_by_slug={"knowledge-graph": ["knowledge-graph-zh"]},
    )
    assert {target for target, _score, _display in links["knowledge-graph"]} == {
        "knowledge-graph-zh"
    }


def test_embedding_batch_write_and_subset_load_are_bounded(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        written = registry.upsert_embeddings(
            "concept",
            (
                (f"concept-{index}", [1.0, float(index % 7), 0.5], f"hash-{index}")
                for index in range(513)
            ),
            model_id="batch-model",
            schema_version="embedding-v1",
            batch_size=64,
        )
        loaded = registry.load_embeddings(
            "concept",
            ["concept-1", "concept-257", "concept-512", "missing"],
            model_id="batch-model",
        )
        embedding_count = registry.connection.execute(
            "SELECT COUNT(*) FROM embedding WHERE model_id='batch-model'"
        ).fetchone()[0]
        bucket_count = registry.connection.execute(
            "SELECT COUNT(*) FROM ann_bucket WHERE model_id='batch-model'"
        ).fetchone()[0]

    assert written == 513
    assert set(loaded) == {"concept-1", "concept-257", "concept-512"}
    assert embedding_count == 513
    assert bucket_count == 513 * 8


def test_incremental_semantic_work_embeds_only_initiator_and_candidates(
    monkeypatch,
) -> None:
    import obsidian_llm_wiki.synth.dedupe as dedupe_module

    concepts = [
        ConceptNote(f"Concept {index}", f"concept-{index}", "meaning")
        for index in range(100)
    ]
    embedded: list[str] = []

    def fake_batch(concepts_to_embed, cache, _options):
        for concept in concepts_to_embed:
            embedded.append(concept.slug)
            cache[concept.slug] = [1.0, 0.0]

    monkeypatch.setattr(dedupe_module, "_embed_concepts_batch", fake_batch)
    semantic_dedupe_concepts(
        SynthesisBundle(concepts=concepts),
        embeddings_cache={},
        new_slugs={"concept-99"},
        candidate_slugs_by_slug={"concept-99": ["concept-1"]},
    )

    assert set(embedded) == {"concept-99", "concept-1"}


def test_embedding_cache_invalidates_on_pipeline_artifact_version_change(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "embeddings.json"
    versions = {
        "extractor": "extract-v1",
        "prompt": "prompt-v1",
        "schema": "schema-v1",
        "synthesis_model": "model-v1",
    }
    save_embeddings_cache(
        cache_path,
        {"concept": [1.0, 0.0]},
        model="embed-v1",
        input_hashes={"concept": "input-v1"},
        artifact_versions=versions,
    )

    current = load_embeddings_cache(
        cache_path,
        model="embed-v1",
        artifact_versions=versions,
    )
    changed = load_embeddings_cache(
        cache_path,
        model="embed-v1",
        artifact_versions={**versions, "prompt": "prompt-v2"},
    )

    assert current == {"concept": [1.0, 0.0]}
    assert changed == {}


def test_source_identity_normalization_merges_duplicate_model_concepts() -> None:
    synthesis = SourceSynthesis(
        source_title="Duplicate Source",
        source_summary="summary",
        source_file="duplicate.md",
        concepts=[
            ConceptNote("Self Attention", "self-attention", "first"),
            ConceptNote("Self Attention", "self-attention", "second"),
        ],
        maps=[
            MapOfContent(
                "Attention",
                "attention",
                "summary",
                concept_slugs=["self-attention", "self-attention"],
            )
        ],
    )

    changed = _normalize_synthesis_identity(synthesis)

    assert changed
    assert [concept.slug for concept in synthesis.concepts] == ["self-attention"]
    assert synthesis.maps[0].concept_slugs == ["self-attention"]


def test_candidate_ids_collapsing_distinct_model_slugs_are_merged(
    tmp_path: Path,
) -> None:
    versions = ArtifactVersions("extract", "prompt", "schema", "model")
    existing = SourceSynthesis(
        source_title="Existing",
        source_summary="summary",
        source_file="existing.md",
        concepts=[ConceptNote("Flash Attention", "flash-attention", "summary")],
    )
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis(
            "existing.md", SourceDoc("Existing", "body"), existing, versions
        )
        canonical_id = existing.concepts[0].canonical_id
        synthesis = SourceSynthesis(
            source_title="New",
            source_summary="summary",
            source_file="new.md",
            concepts=[
                ConceptNote(
                    "FlashAttention",
                    "flashattention",
                    "first",
                    canonical_id=canonical_id,
                ),
                ConceptNote(
                    "GPU Flash Attention",
                    "gpu-flash-attention",
                    "second",
                    canonical_id=canonical_id,
                ),
            ],
        )
        registry.assign_canonical_ids(synthesis)
        _normalize_synthesis_identity(synthesis)

    assert [concept.slug for concept in synthesis.concepts] == ["flash-attention"]
    assert synthesis.concepts[0].canonical_id == canonical_id


def test_partial_embedding_namespace_resumes_missing_vectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """One surviving vector must not make an interrupted backfill look complete."""
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline as pipeline_module

    sources = {
        "alpha.md": SourceDoc("Alpha Source", "alpha body " * 50, source_file="alpha.md"),
        "beta.md": SourceDoc("Beta Source", "beta body " * 50, source_file="beta.md"),
    }

    async def fake_retry(_config, filename, source, *_args, **_kwargs):
        slug = filename.removesuffix(".md")
        return SourceSynthesis(
            source_title=source.title,
            source_summary="summary",
            source_file=filename,
            concepts=[ConceptNote(slug.title(), slug, f"{slug} summary")],
        )

    def fake_embed_text(text, **_kwargs):
        return _unit(0.0 if "alpha" in text.casefold() else math.pi / 2)

    def fake_embed_batch(texts, **_kwargs):
        return [fake_embed_text(text) for text in texts]

    monkeypatch.setattr(pipeline_module, "_synthesize_with_retry", fake_retry)
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.embed_text", fake_embed_text
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.embed_batch", fake_embed_batch
    )
    config = Config(
        vault_path=str(tmp_path),
        embeddings_enabled=True,
        similarity_dedup_threshold=0.99,
    )

    asyncio.run(pipeline_module.run_pipeline(tmp_path, sources, config))
    with VaultRegistry(config.llmwiki_dir) as registry:
        rows = registry.connection.execute(
            """
            SELECT entity_id, schema_version FROM embedding
            WHERE entity_kind='concept' ORDER BY entity_id
            """
        ).fetchall()
        assert len(rows) == 2
        missing_id = str(rows[0][0])
        schema_version = str(rows[0][1])
        with registry.connection:
            registry.connection.execute(
                "DELETE FROM ann_bucket WHERE entity_kind='concept' AND entity_id=?",
                (missing_id,),
            )
            registry.connection.execute(
                "DELETE FROM embedding WHERE entity_kind='concept' AND entity_id=?",
                (missing_id,),
            )

    asyncio.run(pipeline_module.run_pipeline(tmp_path, sources, config))

    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.embedding_count(
            "concept",
            model_id=config.embedding_model,
            schema_version=schema_version,
        ) == 2
