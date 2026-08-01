from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.cache import load_synthesis, save_synthesis
from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry


def test_registry_creates_the_v2_schema_in_wal_mode(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        tables = {
            row[0]
            for row in registry.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        journal_mode = registry.connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert {
        "source",
        "source_version",
        "synthesis",
        "concept",
        "concept_alias",
        "moc",
        "membership",
        "embedding",
        "ann_bucket",
        "dirty_queue",
        "operation",
        "ingest_hash",
        "health_finding",
        "meta",
    } <= tables
    assert journal_mode == "wal"


@pytest.mark.parametrize("marker", ["999", "not-a-version"])
def test_registry_rejects_future_or_malformed_schema_versions(
    tmp_path: Path,
    marker: str,
) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    with VaultRegistry(llmwiki_dir):
        pass
    with sqlite3.connect(llmwiki_dir / "registry.db") as connection:
        connection.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (marker,)
        )

    with pytest.raises(RuntimeError, match="schema version|newer than this binary"):
        VaultRegistry(llmwiki_dir)

    with sqlite3.connect(llmwiki_dir / "registry.db") as connection:
        persisted = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert persisted == marker

    with sqlite3.connect(tmp_path / ".llmwiki" / "registry.db") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_canonical_id_assignment_remaps_slugs_links_and_moc_memberships(
    tmp_path: Path,
) -> None:
    versions = ArtifactVersions("extract-v2", "prompt-v2", "schema-v2", "model-a")
    source = SourceDoc(title="English", content="body " * 30, source_file="en.md")
    existing = SourceSynthesis(
        source_title="English",
        source_summary="summary",
        source_file="en.md",
        concepts=[
            ConceptNote(
                title="Transformer Architecture",
                slug="transformer-architecture",
                summary="architecture",
            ),
            ConceptNote(
                title="Positional Encoding",
                slug="positional-encoding",
                summary="positions",
            ),
        ],
    )
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis("en.md", source, existing, versions)
        ids = {concept.slug: concept.canonical_id for concept in existing.concepts}
        incoming = SourceSynthesis(
            source_title="中文",
            source_summary="摘要",
            source_file="zh.md",
            concepts=[
                ConceptNote(
                    title="Transformer Architecture (Transformer 架构)",
                    slug="transformer-architecture-transformer-架构",
                    summary="架构",
                    canonical_id=ids["transformer-architecture"],
                    related=[
                        ConceptLink(
                            slug="positional-encoding-位置编码",
                            relation="depends_on",
                        )
                    ],
                ),
                ConceptNote(
                    title="Positional Encoding (位置编码)",
                    slug="positional-encoding-位置编码",
                    summary="位置",
                    canonical_id=ids["positional-encoding"],
                ),
            ],
            maps=[
                MapOfContent(
                    title="Transformer 核心",
                    slug="transformer-core",
                    summary="核心",
                    concept_slugs=[
                        "transformer-architecture-transformer-架构",
                        "positional-encoding-位置编码",
                    ],
                )
            ],
        )

        registry.assign_canonical_ids(incoming)

    assert [concept.slug for concept in incoming.concepts] == [
        "transformer-architecture",
        "positional-encoding",
    ]
    assert incoming.concepts[0].related[0].slug == "positional-encoding"
    assert incoming.maps[0].concept_slugs == [
        "transformer-architecture",
        "positional-encoding",
    ]


def test_canonical_concept_payload_aggregates_all_current_source_owners(
    tmp_path: Path,
) -> None:
    versions = ArtifactVersions("extract", "prompt", "schema", "model")
    source_one = SourceDoc("One", "one " * 30, source_file="one.md")
    source_two = SourceDoc("Two", "two " * 30, source_file="two.md")
    synthesis_one = SourceSynthesis(
        source_title="One",
        source_summary="one",
        source_file="one.md",
        concepts=[
            ConceptNote(
                title="Shared",
                slug="shared",
                summary="short",
                aliases=["alias-one"],
                tags=["one"],
            )
        ],
    )
    synthesis_two = SourceSynthesis(
        source_title="Two",
        source_summary="two",
        source_file="two.md",
        concepts=[
            ConceptNote(
                title="Shared",
                slug="shared",
                summary="a much longer summary from the second owner",
                aliases=["alias-two"],
                tags=["two"],
            )
        ],
    )

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis("one.md", source_one, synthesis_one, versions)
        registry.upsert_synthesis("two.md", source_two, synthesis_two, versions)
        row = registry.connection.execute(
            "SELECT payload_json FROM concept WHERE slug='shared'"
        ).fetchone()
        aggregate = json.loads(row[0])
        assert set(aggregate["aliases"]) == {"alias-one", "alias-two"}
        assert set(aggregate["tags"]) == {"one", "two"}
        assert aggregate["summary"] == "a much longer summary from the second owner"

        registry.upsert_synthesis(
            "two.md",
            source_two,
            SourceSynthesis(
                source_title="Two",
                source_summary="two",
                source_file="two.md",
            ),
            versions,
        )
        row = registry.connection.execute(
            "SELECT payload_json FROM concept WHERE slug='shared'"
        ).fetchone()
        aggregate = json.loads(row[0])
        assert aggregate["aliases"] == ["alias-one"]
        assert aggregate["tags"] == ["one"]
        assert aggregate["summary"] == "short"


def test_registry_round_trips_synthesis_and_enqueues_version_invalidation(
    tmp_path: Path,
) -> None:
    source = SourceDoc(
        title="Scaling Knowledge Vaults",
        content="Bounded retrieval keeps prompt context stable as the corpus grows.",
        url="https://example.com/scaling",
    )
    synthesis = SourceSynthesis(
        source_title=source.title,
        source_summary="A note about bounded retrieval.",
        source_file="scaling.md",
        concepts=[
            ConceptNote(
                title="Bounded Retrieval",
                slug="bounded-retrieval",
                summary="Retrieval with an explicit candidate budget.",
            )
        ],
    )
    versions = ArtifactVersions(
        extractor="web-v1",
        prompt="prompt-v1",
        schema="schema-v1",
        synthesis_model="model-a",
    )

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis("scaling.md", source, synthesis, versions)

        loaded = registry.load_synthesis("scaling.md")
        assert loaded is not None
        assert loaded.source_file == "scaling.md"
        assert loaded.concepts[0].slug == "bounded-retrieval"
        assert registry.synthesis_is_current("scaling.md", source.content, versions)

        changed = ArtifactVersions(
            extractor="web-v1",
            prompt="prompt-v2",
            schema="schema-v1",
            synthesis_model="model-a",
        )
        assert not registry.synthesis_is_current("scaling.md", source.content, changed)

        dirty = registry.list_dirty(limit=10)
        assert len(dirty) == 1
        assert dirty[0].entity_type == "source"
        assert dirty[0].entity_id == "scaling.md"
        assert dirty[0].reason == "prompt_version_changed"


def test_dirty_queue_claims_are_bounded_and_survive_reopen(tmp_path: Path) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    with VaultRegistry(llmwiki_dir) as registry:
        for index in range(5):
            registry.enqueue_dirty("source", f"source-{index}.md", "retry")
        first = registry.claim_dirty(limit=2, worker_id="worker-a")
        assert [item.entity_id for item in first] == ["source-0.md", "source-1.md"]
        assert all(item.state == "running" and item.attempts == 1 for item in first)
        registry.complete_dirty(first[0].id, worker_id="worker-a")
        registry.fail_dirty(
            first[1].id,
            "temporary model failure",
            worker_id="worker-a",
            next_attempt_at="2999-01-01T00:00:00+00:00",
        )

    with VaultRegistry(llmwiki_dir) as registry:
        pending = registry.claim_dirty(limit=10, worker_id="worker-b")
        assert [item.entity_id for item in pending] == [
            "source-2.md",
            "source-3.md",
            "source-4.md",
        ]
        assert registry.list_dirty(state="completed", limit=10)[0].entity_id == "source-0.md"


def test_stale_dirty_worker_cannot_finish_a_reclaimed_lease(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        record_id = registry.enqueue_dirty("source", "leased.md", "retry")
        assert registry.start_dirty(record_id, worker_id="worker-a")
        assert not registry.start_dirty(record_id, worker_id="worker-b")
        registry.connection.execute(
            "UPDATE dirty_queue SET lease_expires_at=? WHERE id=?",
            ("2000-01-01T00:00:00+00:00", record_id),
        )
        registry.connection.commit()

        assert registry.start_dirty(record_id, worker_id="worker-b")
        assert not registry.complete_dirty(record_id, worker_id="worker-a")
        assert not registry.fail_dirty(
            record_id,
            "stale failure",
            worker_id="worker-a",
        )
        assert registry.complete_dirty(record_id, worker_id="worker-b")


def test_ingest_hash_index_normalizes_urls_and_resolves_content_duplicates(
    tmp_path: Path,
) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_ingest_hash(
            requested_url="HTTPS://Example.COM/article?b=2&a=1#section",
            canonical_url="https://example.com/article?a=1&b=2",
            content_hash="sha256-content",
            output_file="article.md",
        )

        by_url = registry.find_ingest_duplicate(
            requested_url="https://example.com/article?a=1&b=2#other"
        )
        by_content = registry.find_ingest_duplicate(content_hash="sha256-content")

    assert by_url is not None and by_url.output_file == "article.md"
    assert by_content is not None and by_content.canonical_url == (
        "https://example.com/article?a=1&b=2"
    )


def test_concepts_receive_stable_ids_and_are_retrieved_through_fts(tmp_path: Path) -> None:
    source = SourceDoc(title="Retrieval", content="Candidate budgets preserve context quality.")
    synthesis = SourceSynthesis(
        source_title=source.title,
        source_summary="Bounded candidate retrieval.",
        source_file="retrieval.md",
        concepts=[
            ConceptNote(
                title="Bounded Candidate Retrieval",
                slug="bounded-candidate-retrieval",
                summary="Limits prompt context without losing the most relevant concepts.",
                tags=["retrieval", "scalability"],
                aliases=["bounded retrieval"],
            )
        ],
    )
    versions = ArtifactVersions("web-v1", "prompt-v1", "schema-v1", "model-a")

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis("retrieval.md", source, synthesis, versions)
        first_id = synthesis.concepts[0].canonical_id
        candidates = registry.search_concepts(
            "prompt context candidate retrieval", limit=10
        )
        registry.upsert_synthesis("retrieval.md", source, synthesis, versions)

    assert first_id.startswith("concept-")
    assert synthesis.concepts[0].canonical_id == first_id
    assert [(item.id, item.slug) for item in candidates] == [
        (first_id, "bounded-candidate-retrieval")
    ]


def test_legacy_state_and_cache_migrate_once_into_registry(tmp_path: Path) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    cache_dir = llmwiki_dir / "cache"
    cache_dir.mkdir(parents=True)
    synthesis = SourceSynthesis(
        source_title="Legacy Source",
        source_summary="Migrated synthesis.",
        source_file="legacy.md",
        concepts=[ConceptNote("Legacy Concept", "legacy-concept", "Migrated concept.")],
    )
    (llmwiki_dir / "state.json").write_text(
        json.dumps(
            {
                "sources": {
                    "legacy.md": {
                        "hash": "legacy-content-hash",
                        "concepts": ["legacy-concept"],
                        "compiled_at": "2026-07-01T00:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "legacy.md.json").write_text(
        json.dumps(source_synthesis_to_dict(synthesis)),
        encoding="utf-8",
    )
    versions = ArtifactVersions("legacy", "legacy", "legacy", "legacy")

    with VaultRegistry(llmwiki_dir) as registry:
        assert registry.migrate_legacy(versions)
        assert registry.load_synthesis("legacy.md") is not None
        assert not registry.migrate_legacy(versions)

    (cache_dir / "legacy.md.json").unlink()
    with VaultRegistry(llmwiki_dir) as registry:
        assert not registry.migrate_legacy(versions)
        assert registry.load_synthesis("legacy.md") is not None


def test_portable_json_cache_has_a_versioned_envelope(tmp_path: Path) -> None:
    synthesis = SourceSynthesis(
        source_title="Versioned Cache",
        source_summary="Portable companion artifact.",
        source_file="versioned.md",
    )
    versions = {
        "extractor": "extract-v2",
        "prompt": "prompt-v2",
        "schema": "schema-v2",
        "synthesis_model": "model-a",
    }

    save_synthesis(
        synthesis,
        tmp_path,
        "versioned.md",
        artifact_versions=versions,
    )
    raw = json.loads((tmp_path / "cache" / "versioned.md.json").read_text())

    assert raw["_artifact"]["schema_version"] == 2
    assert raw["_artifact"]["versions"] == versions
    assert raw["payload"]["source_file"] == "versioned.md"
    assert load_synthesis(tmp_path, "versioned.md") is not None


def test_maintenance_cursor_cycles_through_bounded_render_batches(
    tmp_path: Path,
) -> None:
    manifest = {
        f"concept-{index}": (f"concepts/concept-{index}.md", f"hash-{index}")
        for index in range(5)
    }
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        diff = registry.render_diff("concept-page", manifest)
        registry.commit_render_manifest("concept-page", manifest, diff)

        first, first_complete = registry.next_maintenance_batch(limit=2)
        uncommitted_retry, _ = registry.next_maintenance_batch(limit=2)
        registry.commit_maintenance_batch(first, (), cycle_complete=first_complete)
        second, second_complete = registry.next_maintenance_batch(limit=2)
        registry.commit_maintenance_batch(second, (), cycle_complete=second_complete)
        third, third_complete = registry.next_maintenance_batch(limit=2)
        registry.commit_maintenance_batch(third, (), cycle_complete=third_complete)
        restarted, _ = registry.next_maintenance_batch(limit=2)

    assert first == ["concepts/concept-0.md", "concepts/concept-1.md"]
    assert uncommitted_retry == first
    assert second == ["concepts/concept-2.md", "concepts/concept-3.md"]
    assert third == ["concepts/concept-4.md"]
    assert restarted == first
    assert (first_complete, second_complete, third_complete) == (False, False, True)


def test_registry_removes_stale_entities_but_retains_shared_ownership(
    tmp_path: Path,
) -> None:
    versions = ArtifactVersions("extract", "prompt", "schema", "model")

    def synthesis(filename: str, slugs: list[str]) -> SourceSynthesis:
        return SourceSynthesis(
            source_title=filename,
            source_summary="summary",
            source_file=filename,
            concepts=[
                ConceptNote(
                    title=slug.title(), slug=slug, summary=f"marker {slug}"
                )
                for slug in slugs
            ],
            maps=[
                MapOfContent(
                    title="Shared Map",
                    slug="shared-map",
                    summary="Shared membership map.",
                    concept_slugs=slugs,
                )
            ],
        )

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        registry.upsert_synthesis(
            "one.md",
            SourceDoc("One", "first revision"),
            synthesis("one.md", ["shared", "obsolete"]),
            versions,
        )
        registry.upsert_synthesis(
            "two.md",
            SourceDoc("Two", "second source"),
            synthesis("two.md", ["shared", "current"]),
            versions,
        )
        registry.upsert_synthesis(
            "one.md",
            SourceDoc("One", "replacement revision"),
            synthesis("one.md", ["shared", "replacement"]),
            versions,
        )

        slugs = {
            str(row[0]) for row in registry.connection.execute("SELECT slug FROM concept")
        }
        members = {
            str(row[0])
            for row in registry.connection.execute(
                """
                SELECT c.slug FROM membership AS ms
                JOIN concept AS c ON c.id=ms.concept_id
                JOIN moc AS m ON m.id=ms.moc_id
                WHERE m.slug='shared-map'
                """
            )
        }
        registry.delete_source("two.md")
        after_delete = {
            str(row[0]) for row in registry.connection.execute("SELECT slug FROM concept")
        }
        obsolete_hits = registry.search_concepts("obsolete", limit=10)

    assert slugs == {"shared", "current", "replacement"}
    assert members == {"shared", "current", "replacement"}
    assert after_delete == {"shared", "replacement"}
    assert all(candidate.slug != "obsolete" for candidate in obsolete_hits)


def test_v2_registry_rebuilds_source_ownership_during_schema_migration(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / ".llmwiki"
    versions = ArtifactVersions("extract", "prompt", "schema", "model")
    synthesis = SourceSynthesis(
        source_title="Legacy",
        source_summary="summary",
        source_file="legacy.md",
        concepts=[ConceptNote("Legacy Concept", "legacy-concept", "summary")],
        maps=[
            MapOfContent(
                "Legacy Map",
                "legacy-map",
                "summary",
                concept_slugs=["legacy-concept"],
            )
        ],
    )
    with VaultRegistry(registry_dir) as registry:
        registry.upsert_synthesis(
            "legacy.md", SourceDoc("Legacy", "body"), synthesis, versions
        )

    with sqlite3.connect(registry_dir / "registry.db") as connection:
        connection.execute("DELETE FROM membership_source")
        connection.execute("DELETE FROM concept_source")
        connection.execute("DELETE FROM moc_source")
        connection.execute("UPDATE meta SET value='2' WHERE key='schema_version'")

    with VaultRegistry(registry_dir) as migrated:
        counts = {
            table: migrated.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in ("concept_source", "moc_source", "membership_source")
        }

    assert counts == {
        "concept_source": 1,
        "moc_source": 1,
        "membership_source": 1,
    }


def test_expired_dirty_queue_lease_is_reclaimed_after_worker_crash(
    tmp_path: Path,
) -> None:
    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        record_id = registry.enqueue_dirty("source", "crashed.md", "retry")
        first = registry.claim_dirty(limit=1, worker_id="worker-a")
        blocked = registry.claim_dirty(limit=1, worker_id="worker-b")
        registry.connection.execute(
            "UPDATE dirty_queue SET lease_expires_at='2000-01-01T00:00:00+00:00' "
            "WHERE id=?",
            (record_id,),
        )
        reclaimed = registry.claim_dirty(limit=1, worker_id="worker-b")
        row = registry.connection.execute(
            "SELECT claimed_by, attempts, lease_expires_at FROM dirty_queue WHERE id=?",
            (record_id,),
        ).fetchone()

    assert [record.id for record in first] == [record_id]
    assert blocked == []
    assert [record.id for record in reclaimed] == [record_id]
    assert row[0] == "worker-b"
    assert row[1] == 2
    assert row[2]


def test_parallel_dirty_claimers_never_receive_the_same_record(
    tmp_path: Path,
) -> None:
    registry_dir = tmp_path / ".llmwiki"
    with VaultRegistry(registry_dir) as registry:
        for index in range(20):
            registry.enqueue_dirty("source", f"source-{index}.md", "retry")

    def claim(worker_id: str) -> set[int]:
        with VaultRegistry(registry_dir) as registry:
            return {
                record.id
                for record in registry.claim_dirty(limit=20, worker_id=worker_id)
            }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(claim, "worker-a")
        second_future = executor.submit(claim, "worker-b")
        first = first_future.result()
        second = second_future.result()

    assert first.isdisjoint(second)
    assert first | second == set(range(1, 21))
