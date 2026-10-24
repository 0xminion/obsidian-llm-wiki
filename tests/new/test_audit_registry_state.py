import pytest

from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry

VERSIONS = ArtifactVersions("e1", "p1", "s1", "m1")


def synthesis(filename="a.md", title="Alpha", slug="alpha"):
    return SourceSynthesis(
        source_title=filename,
        source_summary="summary",
        source_file=filename,
        concepts=[ConceptNote(title, slug, "A useful summary.")],
    )


@pytest.mark.parametrize("mutation", ["upsert", "update", "delete", "metadata"])
def test_pipeline_checkpoint_is_durable_and_invalidated_by_source_mutations(tmp_path, mutation):
    with VaultRegistry(tmp_path) as registry:
        source = SourceDoc("A", "body", source_file="a.md")
        registry.upsert_synthesis("a.md", source, synthesis(), VERSIONS)
        assert registry.get_pipeline_checkpoint() == ""
        registry.set_pipeline_checkpoint("successful-render-config-hash")
    with VaultRegistry(tmp_path) as registry:
        assert registry.get_pipeline_checkpoint() == "successful-render-config-hash"
        if mutation == "upsert":
            registry.upsert_synthesis("a.md", source, synthesis(), VERSIONS)
        elif mutation == "update":
            registry.update_synthesis("a.md", synthesis(), VERSIONS)
        elif mutation == "delete":
            registry.delete_source("a.md")
        else:
            registry.refresh_source_metadata("a.md", source, VERSIONS)
        assert registry.get_pipeline_checkpoint() == ""


def test_unrelated_titles_sharing_slug_get_distinct_stable_canonical_identity(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        first = synthesis("a.md", "Java programming language", "java")
        second = synthesis("b.md", "Java island geography", "java")
        registry.upsert_synthesis("a.md", SourceDoc("A", "programming"), first, VERSIONS)
        registry.upsert_synthesis("b.md", SourceDoc("B", "geography"), second, VERSIONS)
        assert second.concepts[0].canonical_id != first.concepts[0].canonical_id
        assert second.concepts[0].slug != first.concepts[0].slug
        repeat = synthesis("b.md", "Java island geography", "java")
        registry.assign_canonical_ids(repeat)
        assert repeat.concepts[0].canonical_id == second.concepts[0].canonical_id
        assert registry.load_synthesis("a.md").concepts[0].title == "Java programming language"
        assert registry.connection.execute("SELECT count(*) FROM concept").fetchone()[0] == 2


def test_source_metadata_follows_current_revision_after_content_revert(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        for body, tag in [("original", "first"), ("changed", "second"), ("original", "reverted")]:
            registry.upsert_synthesis(
                "a.md",
                SourceDoc("A", body, source_file="a.md", tags=[tag]),
                synthesis(),
                VERSIONS,
            )
        registry.record_source_file_state("a.md", mtime_ns=1, size_bytes=8)
        stub = registry.source_file_states()["a.md"][2]
        assert stub.tags == ["reverted"]


def test_v4_search_identity_migration_preserves_search_and_point_update(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        registry.upsert_synthesis("a.md", SourceDoc("A", "body"), synthesis(), VERSIONS)
        with registry.connection:
            registry.connection.execute("DROP TABLE concept_fts_identity")
            registry.connection.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
    with VaultRegistry(tmp_path) as registry:
        assert registry.search_concepts("Alpha", limit=10)[0].slug == "alpha"
        registry.update_synthesis("a.md", synthesis(), VERSIONS)
        assert registry.connection.execute("SELECT count(*) FROM concept_fts").fetchone()[0] == 1
        assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        registry.delete_source("a.md")
        assert registry.search_concepts("Alpha", limit=10) == []
        assert (
            registry.connection.execute("SELECT count(*) FROM concept_fts_identity").fetchone()[0]
            == 0
        )
