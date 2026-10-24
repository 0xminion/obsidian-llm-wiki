"""Current-owner aggregates must lose every deleted contribution."""
import hashlib
import json
import sqlite3
from itertools import permutations

import pytest

from obsidian_llm_wiki.core.models import (
    BodySection,
    Claim,
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry

OWNER_TEXT = {
    "a.md": "Copper conducts heat.",
    "b.md": "Silver conducts heat in a much longer description of a different experiment.",
}


def add_owners(registry, order=("a.md", "b.md")):
    canonical_id = ""
    for filename in order:
        token = OWNER_TEXT[filename].split()[0]
        synthesis = SourceSynthesis(
            source_title=filename, source_summary=OWNER_TEXT[filename], source_file=filename,
            concepts=[ConceptNote(
                title="Metals", slug="metals", summary=OWNER_TEXT[filename],
                canonical_id=canonical_id, aliases=[f"{token}Alias"], tags=[token.lower()],
                sections=[BodySection(token, prose=f"{token} body")],
                claims=[Claim(f"{token} claim", "metals", filename)],
            )],
        )
        registry.upsert_synthesis(
            filename, SourceDoc(filename, OWNER_TEXT[filename]), synthesis,
            ArtifactVersions("extractor", f"prompt-{token}", "schema", f"model-{token}"),
        )
        canonical_id = synthesis.concepts[0].canonical_id
    assert registry.connection.execute("SELECT count(*) FROM concept").fetchone()[0] == 1
    assert registry.connection.execute("SELECT count(*) FROM concept_source").fetchone()[0] == 2
    return canonical_id


@pytest.mark.parametrize("order", list(permutations(OWNER_TEXT)))
@pytest.mark.parametrize("deleted", list(OWNER_TEXT))
def test_delete_rebuilds_current_owner_payload_aliases_and_fts(tmp_path, order, deleted):
    survivor = next(filename for filename in OWNER_TEXT if filename != deleted)
    kept = OWNER_TEXT[survivor].split()[0]
    removed = OWNER_TEXT[deleted].split()[0]
    with VaultRegistry(tmp_path) as registry:
        canonical_id = add_owners(registry, order)
        owner_synthesis = registry.load_synthesis(survivor)
        assert owner_synthesis is not None
        owner_payload = owner_synthesis.concepts[0]
        registry.set_pipeline_checkpoint("published")
        registry.delete_source(deleted)
        row = registry.connection.execute("SELECT * FROM concept").fetchone()
        payload = json.loads(row["payload_json"])
        assert removed not in row["payload_json"]
        assert row["id"] == canonical_id
        assert row["summary"] == owner_payload.summary
        assert payload["aliases"] == [f"{kept}Alias"]
        assert payload["tags"] == [kept.lower()]
        assert [section["prose"] for section in payload["sections"]] == [f"{kept} body"]
        assert [claim["text"] for claim in payload["claims"]] == [f"{kept} claim"]
        assert row["prompt_version"] == f"prompt-{kept}"
        assert row["model_id"] == f"model-{kept}"
        aliases = {r[0] for r in registry.connection.execute("SELECT alias FROM concept_alias")}
        assert f"{removed.lower()}alias" not in aliases
        assert f"{kept.lower()}alias" in aliases
        for query in (removed, f"{removed}Alias"):
            assert registry.search_concepts(query, limit=10) == []
            assert registry.connection.execute(
                "SELECT concept_id FROM concept_fts WHERE concept_fts MATCH ?", (query,)
            ).fetchall() == []
        assert [c.id for c in registry.search_concepts(kept, limit=10)] == [canonical_id]
        assert registry.load_synthesis(survivor) == owner_synthesis
        assert registry.get_pipeline_checkpoint() == ""
        before = list(registry.connection.iterdump())
        registry.delete_source(deleted)
        assert list(registry.connection.iterdump()) == before
        assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert registry.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("deleted", list(OWNER_TEXT))
def test_delete_invalidates_only_stale_embedding_inputs_and_ann(tmp_path, deleted):
    survivor = next(filename for filename in OWNER_TEXT if filename != deleted)
    with VaultRegistry(tmp_path) as registry:
        canonical_id = add_owners(registry)
        # Pipeline may already have embedded the post-deletion bundle before
        # committing deletion. Do not throw away that freshly computed vector.
        hashes = {
            "old": hashlib.sha256(f"Metals\n{OWNER_TEXT['b.md']}".encode()).hexdigest(),
            "fresh": hashlib.sha256(f"Metals\n{OWNER_TEXT[survivor]}".encode()).hexdigest(),
            "obsolete": "obsolete input from an older model",
        }
        for model, input_hash in hashes.items():
            registry.upsert_embedding(
                "concept", canonical_id, [1., 0.], model_id=model,
                input_hash=input_hash, schema_version="vectors-v1",
            )
        registry.upsert_embedding(
            "concept", "untouched", [0., 1.], model_id="other",
            input_hash="unrelated", schema_version="vectors-v1",
        )
        unrelated = tuple(registry.connection.execute(
            "SELECT * FROM embedding WHERE entity_id='untouched'"
        ).fetchone())
        registry.delete_source(deleted)
        models = {row[0] for row in registry.connection.execute(
            "SELECT model_id FROM embedding WHERE entity_id=?", (canonical_id,)
        )}
        expected = {model for model, value in hashes.items() if value == hashes["fresh"]}
        assert models == expected
        assert {row[0] for row in registry.connection.execute(
            "SELECT DISTINCT model_id FROM ann_bucket WHERE entity_id=?", (canonical_id,)
        )} == expected
        assert tuple(registry.connection.execute(
            "SELECT * FROM embedding WHERE entity_id='untouched'"
        ).fetchone()) == unrelated
        registry.delete_source(survivor)
        for table in ("embedding", "ann_bucket"):
            assert registry.connection.execute(
                f"SELECT 1 FROM {table} WHERE entity_id=?", (canonical_id,)
            ).fetchall() == []


@pytest.mark.parametrize("failure", ["fts", "embedding"])
def test_delete_rolls_back_ownership_payload_indexes_and_checkpoint(tmp_path, monkeypatch, failure):
    with VaultRegistry(tmp_path) as registry:
        canonical_id = add_owners(registry)
        registry.upsert_embedding(
            "concept", canonical_id, [1., 0.], model_id="old",
            input_hash="obsolete", schema_version="v1",
        )
        registry.set_pipeline_checkpoint("published")
        if failure == "fts":
            original = registry._replace_concept_fts_unlocked

            def fail_fts(concept):
                original(concept)
                raise sqlite3.IntegrityError("injected index failure")

            monkeypatch.setattr(registry, "_replace_concept_fts_unlocked", fail_fts)
        else:
            registry.connection.execute(
                "CREATE TRIGGER abort_vector_delete BEFORE DELETE ON embedding "
                "BEGIN SELECT RAISE(ABORT, 'injected index failure'); END"
            )
        before = list(registry.connection.iterdump())
        with pytest.raises(sqlite3.IntegrityError, match="injected index failure"):
            registry.delete_source("b.md")
        assert list(registry.connection.iterdump()) == before
        assert registry.get_pipeline_checkpoint() == "published"
        assert registry.source_filenames() == {"a.md", "b.md"}
        assert registry.search_concepts("Silver", limit=10)
        assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_delete_work_is_bounded_by_touched_owners_not_corpus(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core import registry as module

    counts = []
    for size in (100, 2000):
        with VaultRegistry(tmp_path / str(size)) as registry:
            canonical_id = add_owners(registry)
            for index in range(size):
                filename = f"unrelated-{index}.md"
                note = ConceptNote(f"Unrelated {index}", f"unrelated-{index}", "Unrelated summary.")
                registry.upsert_synthesis(
                    filename, SourceDoc(filename, "Unrelated body"),
                    SourceSynthesis(filename, "Unrelated summary.", concepts=[note]),
                    ArtifactVersions("e", "p", "s", "m"),
                )
                registry.upsert_embedding(
                    "concept", note.canonical_id, [1., 0.], model_id="model",
                    input_hash="unused noise hash", schema_version="v1",
                )
            registry.upsert_embedding(
                "concept", canonical_id, [1., 0.], model_id="model",
                input_hash="obsolete", schema_version="v1",
            )
            touched_payloads = []
            original = module.source_synthesis_from_dict

            def tracked_load(payload, touched_payloads=touched_payloads, original=original):
                touched_payloads.append(payload["source_file"])
                return original(payload)

            with monkeypatch.context() as patcher:
                patcher.setattr(module, "source_synthesis_from_dict", tracked_load)
                steps = []
                registry.connection.set_progress_handler(
                    lambda steps=steps: steps.append(1) or 0, 100,
                )
                registry.delete_source("b.md")
                registry.connection.set_progress_handler(None, 0)
            counts.append(len(steps))
            assert touched_payloads == ["a.md"]
            row = registry.connection.execute("SELECT count(*) FROM concept").fetchone()
            assert row[0] == size + 1
    print(f"deletion SQLite steps/100 for 100 and 2000 unrelated concepts: {counts}")
    assert counts[1] < 3 * counts[0], counts
