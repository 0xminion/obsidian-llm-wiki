"""MoC canonical payloads are projections of current, ordered owners."""

import hashlib
import json
import sqlite3
from copy import deepcopy
from itertools import permutations

import pytest

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    moc_to_dict,
)
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.synth.dedupe import merge_bundle

V = ArtifactVersions("e", "p", "s", "m")
OWNERS = ("a.md", "b.md")


def contribution(filename):
    token = "Copper" if filename == "a.md" else "Silver"
    member = token.lower()
    return SourceSynthesis(
        filename,
        f"{token} source summary.",
        source_file=filename,
        concepts=[
            ConceptNote("Heat", "heat", "Heat transfer."),
            ConceptNote(token, member, f"{token} experiment."),
        ],
        maps=[
            MapOfContent(
                "Thermal topic" if filename == "a.md" else "THERMAL TOPIC",
                "thermal-topic",
                f"{token} review of thermal conductivity.",
                tags=[member],
                concept_slugs=[member, "heat"],
            )
        ],
    )


def add_owners(registry, order=OWNERS):
    for filename in order:
        registry.upsert_synthesis(
            filename, SourceDoc(filename, filename), contribution(filename), V
        )
    assert registry.connection.execute("SELECT count(*) FROM moc").fetchone()[0] == 1
    assert registry.connection.execute("SELECT count(*) FROM moc_source").fetchone()[0] == 2


def assert_aggregate(registry, owners):
    sources = [registry.load_synthesis(filename) for filename in sorted(owners)]
    before = deepcopy(sources)
    expected = merge_bundle(deepcopy(sources)).maps[0]
    row = registry.connection.execute("SELECT * FROM moc").fetchone()
    assert json.loads(row["payload_json"]) == moc_to_dict(expected)
    assert row["title"] == expected.title
    assert row["summary"] == expected.summary
    assert json.loads(row["tags_json"]) == expected.tags
    assert row["content_hash"] == hashlib.sha256(row["payload_json"].encode()).hexdigest()
    members = registry.connection.execute(
        "SELECT c.slug, mb.position FROM membership mb JOIN concept c ON c.id=mb.concept_id "
        "WHERE mb.moc_id=? ORDER BY mb.position",
        (row["id"],),
    ).fetchall()
    assert [(r[0], r[1]) for r in members] == [
        (slug, index) for index, slug in enumerate(expected.concept_slugs)
    ]
    assert [registry.load_synthesis(filename) for filename in sorted(owners)] == before
    assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    return dict(row)


@pytest.mark.parametrize("order", list(permutations(OWNERS)))
@pytest.mark.parametrize("deleted", OWNERS)
def test_shared_moc_aggregate_matches_ordered_render_on_upsert_and_delete(tmp_path, order, deleted):
    with VaultRegistry(tmp_path) as registry:
        add_owners(registry, order)
        assert_aggregate(registry, OWNERS)
        survivor = next(filename for filename in OWNERS if filename != deleted)
        local_before = registry.load_synthesis(survivor)
        registry.delete_source(deleted)
        assert_aggregate(registry, [survivor])
        assert registry.load_synthesis(survivor) == local_before
        before = list(registry.connection.iterdump())
        registry.delete_source(deleted)
        assert list(registry.connection.iterdump()) == before
        registry.delete_source(survivor)
        for table in ("moc", "moc_source", "membership", "membership_source"):
            assert registry.connection.execute(f"SELECT * FROM {table}").fetchall() == []


@pytest.mark.parametrize("method", ["upsert", "update"])
@pytest.mark.parametrize("changed", OWNERS)
def test_changed_owner_and_noop_preserve_local_contributions(tmp_path, method, changed):
    with VaultRegistry(tmp_path) as registry:
        add_owners(registry)
        other = next(filename for filename in OWNERS if filename != changed)
        untouched = registry.load_synthesis(other)
        revised = registry.load_synthesis(changed)
        revised.maps[0].summary = "Revised experiment overview."
        revised.maps[0].tags = ["revised", "Thermal Conductivity"]
        revised.maps[0].concept_slugs.reverse()
        if method == "upsert":
            registry.upsert_synthesis(changed, SourceDoc(changed, "Revised body"), revised, V)
        else:
            registry.update_synthesis(changed, revised, V)
        expected = assert_aggregate(registry, OWNERS)
        assert registry.load_synthesis(other) == untouched
        registry.update_synthesis(changed, deepcopy(revised), V)
        assert assert_aggregate(registry, OWNERS) == expected
        revised.maps = []
        registry.update_synthesis(changed, revised, V)
        assert_aggregate(registry, [other])
        assert registry.load_synthesis(other) == untouched


def member_hash(slugs):
    return hashlib.sha256("\x1f".join(sorted(slugs)).encode()).hexdigest()


def put_centroid(registry, model, input_hash, slug="thermal-topic"):
    registry.upsert_embedding(
        "moc",
        slug,
        [1.0, 0.0],
        model_id=model,
        input_hash=input_hash,
        schema_version="moc-centroid-v2:test",
    )


@pytest.mark.parametrize("mutation", ["delete", "update", "upsert"])
@pytest.mark.parametrize("changed", OWNERS)
def test_moc_centroid_invalidation_keeps_fresh_membership_hash(tmp_path, mutation, changed):
    with VaultRegistry(tmp_path) as registry:
        add_owners(registry)
        removed = "copper" if changed == "a.md" else "silver"
        remaining = [slug for slug in ["heat", "copper", "silver"] if slug != removed]
        put_centroid(registry, "old", member_hash(["heat", "copper", "silver"]))
        put_centroid(registry, "fresh", member_hash(remaining))
        put_centroid(registry, "unrelated", "noise", slug="unrelated")
        unrelated = tuple(
            registry.connection.execute(
                "SELECT * FROM embedding WHERE entity_id='unrelated'"
            ).fetchone()
        )
        if mutation == "delete":
            registry.delete_source(changed)
        else:
            revised = registry.load_synthesis(changed)
            revised.maps[0].concept_slugs.remove(removed)
            if mutation == "upsert":
                registry.upsert_synthesis(changed, SourceDoc(changed, "Revision"), revised, V)
            else:
                registry.update_synthesis(changed, revised, V)
        for table in ("embedding", "ann_bucket"):
            assert {
                row[0]
                for row in registry.connection.execute(
                    f"SELECT model_id FROM {table} WHERE entity_kind='moc' "
                    "AND entity_id='thermal-topic'"
                )
            } == {"fresh"}
        assert (
            tuple(
                registry.connection.execute(
                    "SELECT * FROM embedding WHERE entity_id='unrelated'"
                ).fetchone()
            )
            == unrelated
        )
        for filename in OWNERS:
            registry.delete_source(filename)
        for table in ("embedding", "ann_bucket"):
            assert (
                registry.connection.execute(
                    f"SELECT * FROM {table} WHERE entity_kind='moc' AND entity_id='thermal-topic'"
                ).fetchall()
                == []
            )


def test_prose_only_change_does_not_invalidate_compatible_centroid(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        add_owners(registry)
        put_centroid(registry, "current", member_hash(["heat", "copper", "silver"]))
        before = tuple(registry.connection.execute("SELECT * FROM embedding").fetchone())
        revised = registry.load_synthesis("a.md")
        revised.maps[0].summary = "Corrected overview without changing member vectors."
        registry.update_synthesis("a.md", revised, V)
        assert tuple(registry.connection.execute("SELECT * FROM embedding").fetchone()) == before


@pytest.mark.parametrize("mutation", ["delete", "update"])
def test_removed_external_member_rebuilds_moc_not_owned_by_changed_source(tmp_path, mutation):
    with VaultRegistry(tmp_path) as registry:
        external = SourceSynthesis(
            "external",
            "External source.",
            source_file="external.md",
            concepts=[ConceptNote("External", "external", "External member.")],
        )
        registry.upsert_synthesis(
            "external.md", SourceDoc("External", "External body"), external, V
        )
        owner = contribution("a.md")
        owner.maps[0].concept_slugs.append("external")
        registry.upsert_synthesis("a.md", SourceDoc("Owner", "Owner body"), owner, V)
        saved = registry.load_synthesis("a.md")
        put_centroid(registry, "old", member_hash(["heat", "copper", "external"]))
        put_centroid(registry, "fresh", member_hash(["heat", "copper"]))
        if mutation == "delete":
            registry.delete_source("external.md")
        else:
            external.concepts = []
            registry.update_synthesis("external.md", external, V)
        row = registry.connection.execute("SELECT * FROM moc").fetchone()
        assert json.loads(row["payload_json"])["concept_slugs"] == ["copper", "heat"]
        assert registry.load_synthesis("a.md") == saved
        assert {
            r[0]
            for r in registry.connection.execute(
                "SELECT model_id FROM embedding WHERE entity_kind='moc'"
            )
        } == {"fresh"}
        assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("mutation", ["update", "upsert", "delete"])
def test_changed_member_vectors_invalidate_centroids_even_with_same_members(tmp_path, mutation):
    with VaultRegistry(tmp_path) as registry:
        # MoC is owned elsewhere: source c contributes a longer canonical
        # definition, and removing it changes the member vector, not membership.
        first = SourceSynthesis(
            "A",
            "A summary",
            source_file="a.md",
            concepts=[ConceptNote("Heat", "heat", "Short heat definition.")],
            maps=[
                MapOfContent(
                    "Thermal topic", "thermal-topic", "Thermal overview.", concept_slugs=["heat"]
                )
            ],
        )
        registry.upsert_synthesis("a.md", SourceDoc("A", "Body A"), first, V)
        cid = first.concepts[0].canonical_id
        second = SourceSynthesis(
            "C",
            "C summary",
            source_file="c.md",
            concepts=[
                ConceptNote("Heat", "heat", "A longer old heat definition.", canonical_id=cid)
            ],
        )
        registry.upsert_synthesis("c.md", SourceDoc("C", "Body C"), second, V)
        summary = (
            "Short heat definition."
            if mutation == "delete"
            else "New significantly longer corrected heat definition."
        )
        hashes = {
            "old": hashlib.sha256(b"Heat\nA longer old heat definition.").hexdigest(),
            "fresh": hashlib.sha256(f"Heat\n{summary}".encode()).hexdigest(),
        }
        for model in ("old", "fresh", "stale-centroid", "wrong-schema"):
            registry.upsert_embedding(
                "concept",
                cid,
                [0.0, 1.0],
                model_id=model,
                input_hash=hashes["old" if model == "old" else "fresh"],
                schema_version="concept-embedding-v2:other"
                if model == "wrong-schema"
                else "concept-embedding-v2:test",
            )
            registry.upsert_embedding(
                "moc",
                "thermal-topic",
                [1.0, 0.0] if model == "stale-centroid" else [0.0, 1.0],
                model_id=model,
                input_hash=member_hash(["heat"]),
                schema_version="moc-centroid-v2:test",
            )
        if mutation == "delete":
            registry.delete_source("c.md")
        else:
            second.concepts[0].summary = summary
            if mutation == "update":
                registry.update_synthesis("c.md", second, V)
            else:
                registry.upsert_synthesis("c.md", SourceDoc("C", "Revised C"), second, V)
        for table in ("embedding", "ann_bucket"):
            assert {
                row[0]
                for row in registry.connection.execute(
                    f"SELECT model_id FROM {table} WHERE entity_kind='moc' "
                    "AND entity_id='thermal-topic'"
                )
            } == {"fresh"}


@pytest.mark.parametrize("mutation", ["delete", "update", "upsert"])
@pytest.mark.parametrize("failure", ["aggregate", "membership", "centroid"])
def test_moc_mutation_rolls_back_all_state(tmp_path, mutation, failure):
    with VaultRegistry(tmp_path) as registry:
        add_owners(registry)
        put_centroid(registry, "old", member_hash(["heat", "copper", "silver"]))
        registry.set_pipeline_checkpoint("published")
        event, condition = {
            "aggregate": ("AFTER UPDATE ON moc", "1"),
            "membership": ("BEFORE INSERT ON membership", "1"),
            "centroid": ("BEFORE DELETE ON embedding", "OLD.entity_kind='moc'"),
        }[failure]
        registry.connection.execute(
            f"CREATE TRIGGER fail_moc {event} WHEN {condition} "
            "BEGIN SELECT RAISE(ABORT, 'injected moc failure'); END"
        )
        before = list(registry.connection.iterdump())
        with pytest.raises(sqlite3.IntegrityError, match="injected moc failure"):
            if mutation == "delete":
                registry.delete_source("a.md")
            else:
                revised = registry.load_synthesis("a.md")
                assert revised is not None
                revised.maps[0].summary = "New overview."
                revised.maps[0].concept_slugs.remove("copper")
                if mutation == "update":
                    registry.update_synthesis("a.md", revised, V)
                else:
                    registry.upsert_synthesis("a.md", SourceDoc("A", "Revised body"), revised, V)
        assert list(registry.connection.iterdump()) == before
        assert registry.get_pipeline_checkpoint() == "published"
        assert registry.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("mutation", ["delete", "update"])
def test_moc_rebuild_work_is_bounded_by_indexed_owners(tmp_path, monkeypatch, mutation):
    from obsidian_llm_wiki.core import registry as module

    work = []
    for size in (100, 1600):
        with VaultRegistry(tmp_path / str(size)) as registry:
            add_owners(registry)
            for index in range(size):
                slug = f"unrelated-{index}"
                noise = SourceSynthesis(
                    slug,
                    "Unrelated summary.",
                    source_file=slug + ".md",
                    concepts=[ConceptNote(slug, slug, "Noise definition.")],
                    maps=[MapOfContent(slug, slug, "Noise overview.", concept_slugs=[slug])],
                )
                registry.upsert_synthesis(
                    noise.source_file, SourceDoc(slug, "Noise body"), noise, V
                )
                put_centroid(registry, "noise", "unused", slug=slug)
            revised = registry.load_synthesis("b.md")
            assert revised is not None
            revised.maps[0].summary = "Revised second-owner summary."
            revised.maps[0].concept_slugs = ["heat"]
            touched = []
            steps = []
            original = module.source_synthesis_from_dict

            def tracked(payload, touched=touched, original=original):
                touched.append(payload["source_file"])
                return original(payload)

            with monkeypatch.context() as patcher:
                patcher.setattr(module, "source_synthesis_from_dict", tracked)
                registry.connection.set_progress_handler(
                    lambda steps=steps: steps.append(1) or 0, 100
                )
                try:
                    if mutation == "delete":
                        registry.delete_source("b.md")
                    else:
                        registry.update_synthesis("b.md", revised, V)
                finally:
                    registry.connection.set_progress_handler(None, 0)
            assert set(touched) <= set(OWNERS)
            assert len(touched) <= 6
            assert registry.connection.execute("SELECT count(*) FROM moc").fetchone()[0] == size + 1
            assert (
                registry.connection.execute(
                    "SELECT count(*) FROM embedding WHERE model_id='noise'"
                ).fetchone()[0]
                == size
            )
            work.append((len(touched), len(steps)))
    print(
        f"MoC {mutation}: (owner payload decodes, SQLite steps/100) "
        f"for 100/1600 noise owners: {work}"
    )
    assert work[0][0] == work[1][0]
    assert work[1][1] < 3 * work[0][1]
