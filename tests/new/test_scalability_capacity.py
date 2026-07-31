from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.render.dataview import render_dataview_views
from obsidian_llm_wiki.render.graph_export import export_graph
from obsidian_llm_wiki.render.obsidian import _render_directory_indexes
from obsidian_llm_wiki.synth.dedupe import merge_bundle

SOURCE_CAPACITY = 5_000
CONCEPT_CAPACITY = 50_000
MOC_CAPACITY = 20_000


def test_registry_capacity_queries_and_queues_remain_bounded(tmp_path: Path) -> None:
    now = datetime.now(UTC).isoformat()
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        with registry.connection:
            registry.connection.executemany(
                """
                INSERT INTO source(
                    filename, content_hash, extractor_version, title,
                    created_at, updated_at
                ) VALUES(?, ?, 'capacity-v1', ?, ?, ?)
                """,
                (
                    (f"source-{index:05d}.md", f"hash-{index}", f"Source {index}", now, now)
                    for index in range(SOURCE_CAPACITY)
                ),
            )
            registry.connection.executemany(
                """
                INSERT INTO concept(
                    id, slug, title, summary, payload_json, content_hash,
                    prompt_version, schema_version, model_id, updated_at
                ) VALUES(?, ?, ?, ?, '{}', ?, 'p1', 's1', 'm1', ?)
                """,
                (
                    (
                        f"concept-id-{index:05d}",
                        f"capacity-concept-{index:05d}",
                        f"Capacity Concept {index}",
                        f"capacity marker {index}",
                        f"concept-hash-{index}",
                        now,
                    )
                    for index in range(CONCEPT_CAPACITY)
                ),
            )
            registry.connection.executemany(
                """
                INSERT INTO concept_fts(
                    concept_id, slug, title, aliases, tags, summary
                ) VALUES(?, ?, ?, '', '', ?)
                """,
                (
                    (
                        f"concept-id-{index:05d}",
                        f"capacity-concept-{index:05d}",
                        f"Capacity Concept {index}",
                        f"capacity marker {index}",
                    )
                    for index in range(CONCEPT_CAPACITY)
                ),
            )
            registry.connection.executemany(
                """
                INSERT INTO moc(
                    id, slug, title, payload_json, content_hash, updated_at
                ) VALUES(?, ?, ?, '{}', ?, ?)
                """,
                (
                    (
                        f"moc-id-{index:05d}",
                        f"capacity-moc-{index:05d}",
                        f"Capacity MoC {index}",
                        f"moc-hash-{index}",
                        now,
                    )
                    for index in range(MOC_CAPACITY)
                ),
            )
            registry.connection.executemany(
                """
                INSERT INTO dirty_queue(
                    entity_type, entity_id, reason, state, next_attempt_at,
                    created_at, updated_at
                ) VALUES('source', ?, 'capacity', 'pending', ?, ?, ?)
                """,
                (
                    (f"source-{index:05d}.md", now, now, now)
                    for index in range(SOURCE_CAPACITY)
                ),
            )

        counts = {
            table: registry.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("source", "concept", "moc")
        }
        candidates = registry.search_concepts("capacity marker 49999", limit=100)
        claimed = registry.claim_dirty(limit=32, worker_id="capacity-worker")

    assert counts == {
        "source": SOURCE_CAPACITY,
        "concept": CONCEPT_CAPACITY,
        "moc": MOC_CAPACITY,
    }
    assert len(candidates) <= 100
    assert any(candidate.slug == "capacity-concept-49999" for candidate in candidates)
    assert len(claimed) == 32


def test_capacity_artifacts_are_sharded_at_fixed_budgets(tmp_path: Path) -> None:
    concepts = [
        ConceptNote(
            title=f"Capacity Concept {index}",
            slug=f"capacity-concept-{index:05d}",
            summary="Bounded capacity fixture.",
        )
        for index in range(CONCEPT_CAPACITY)
    ]
    maps = [
        MapOfContent(
            title=f"Capacity MoC {index}",
            slug=f"capacity-moc-{index:05d}",
            summary="Capacity map.",
            concept_slugs=[f"capacity-concept-{index:05d}"],
        )
        for index in range(MOC_CAPACITY)
    ]
    source_syntheses = [
        SourceSynthesis(
            source_title=f"Source {index}",
            source_summary="Capacity source.",
            source_file=f"source-{index:05d}.md",
        )
        for index in range(SOURCE_CAPACITY)
    ]
    sources = {
        synthesis.source_file: SourceDoc(
            synthesis.source_title,
            "Capacity source body.",
            source_file=synthesis.source_file,
        )
        for synthesis in source_syntheses
    }
    bundle = SynthesisBundle(
        sources=source_syntheses,
        concepts=concepts,
        maps=maps,
    )

    graph_paths = export_graph(
        bundle,
        tmp_path / ".llmwiki",
        shard_size=5_000,
        monolith_threshold=10_000,
    )
    graph_manifest = json.loads(
        (tmp_path / ".llmwiki" / "graph" / "manifest.json").read_text()
    )
    index_paths = _render_directory_indexes(
        "concepts",
        tmp_path / "04-Wiki" / "concepts",
        [(concept.slug, concept.title) for concept in concepts],
        tmp_path / "04-Wiki",
        shard_size=2_000,
    )
    report_paths = render_dataview_views(
        tmp_path / "04-Wiki",
        bundle,
        sources,
        shard_size=2_000,
    )

    assert graph_manifest["node_count"] == (
        SOURCE_CAPACITY + CONCEPT_CAPACITY + MOC_CAPACITY
    )
    assert all(shard["count"] <= 5_000 for shard in graph_manifest["shards"])
    assert all(shard["count"] <= 5_000 for shard in graph_manifest["edge_shards"])
    assert len(list((tmp_path / "04-Wiki" / "concepts").glob("index-*.md"))) == 25
    assert len(
        list((tmp_path / "04-Wiki" / "views").glob("concepts-by-confidence-*.md"))
    ) == 25
    assert str(tmp_path / ".llmwiki" / "graph" / "manifest.json") in graph_paths
    assert index_paths
    assert report_paths


def test_normal_merge_path_handles_exact_five_year_capacity() -> None:
    syntheses: list[SourceSynthesis] = []
    for source_index in range(SOURCE_CAPACITY):
        concept_start = source_index * 10
        moc_start = source_index * 4
        concepts = [
            ConceptNote(
                title=f"Capacity Concept {index}",
                slug=f"capacity-concept-{index:05d}",
                summary="Capacity merge fixture.",
            )
            for index in range(concept_start, concept_start + 10)
        ]
        maps = [
            MapOfContent(
                title=f"Capacity MoC {index}",
                slug=f"capacity-moc-{index:05d}",
                summary="Capacity merge map.",
                concept_slugs=[concepts[index % 10].slug],
            )
            for index in range(moc_start, moc_start + 4)
        ]
        syntheses.append(
            SourceSynthesis(
                source_title=f"Source {source_index}",
                source_summary="Capacity source.",
                source_file=f"source-{source_index:05d}.md",
                concepts=concepts,
                maps=maps,
            )
        )

    bundle = merge_bundle(syntheses)

    assert len(bundle.sources) == SOURCE_CAPACITY
    assert len(bundle.concepts) == CONCEPT_CAPACITY
    assert len(bundle.maps) == MOC_CAPACITY
