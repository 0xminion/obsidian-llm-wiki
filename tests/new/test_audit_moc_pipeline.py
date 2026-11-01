"""Shared MoC aggregates remain fresh through real incremental builds.

Two sources share one MoC only because they share its member entity, and a
member entity is shared only through an explicit bilateral identity
assessment of both sources' reviewed text. The scripted assessor below is
keyed to the fact both sources state (``SHARED_FACT``), never to the equal
concept titles, slugs or summaries.
"""

import hashlib
import json
import re
import socket

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.ingest.sources import load_sources_incremental
from tests.new.bilateral_fixture_helpers import fact_verdict
from tests.new.grounding_doubles import identity_answer, review_answer
from tests.new.immutable_semantic_helpers import (
    file_state,
    input_registry,
    published_meta,
    published_registry,
    published_root,
)

SHARED_FACT = "Thermal conductivity relates heat flux to the temperature gradient."
SUMMARY = "Thermal conductivity relates flux to a temperature gradient."


def _candidate_id(prompt: str, slug: str) -> str:
    match = re.search(rf"^\s*-?\s*(concept-[0-9a-f]+) \| {re.escape(slug)} \|", prompt, re.M)
    return match.group(1) if match else ""


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", ["a.md", "b.md"])
@pytest.mark.parametrize("arrival", [("a.md", "b.md"), ("b.md", "a.md")])
async def test_shared_moc_deletion_noop_refresh_and_revision(
    tmp_path,
    monkeypatch,
    deleted,
    arrival,
):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, min_source_chars=1)
    config.sources_dir.mkdir(parents=True)
    summaries = {
        "a.md": "Copper review of thermal conductivity.",
        "b.md": "Silver review with a superseded conclusion.",
    }
    calls = []
    identity_calls = []
    verdict = fact_verdict(same=(SHARED_FACT,))

    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in MoC lifecycle test")

    def produce(prompt):
        filename = next(name for name in summaries if "MOC_LIFECYCLE_" + name in prompt)
        calls.append(filename)
        return json.dumps(
            {
                "source_title": filename,
                "source_summary": SUMMARY,
                "concepts": [
                    {
                        "title": "Conductivity",
                        "slug": "conductivity",
                        "summary": SUMMARY,
                        # The producer proposes the offered candidate (an untrusted
                        # hint); the registry binds it only once assessed.
                        "canonical_id": _candidate_id(prompt, "conductivity"),
                        "sections": [{"heading": "Mechanism", "prose": "Heat follows a gradient."}],
                    }
                ],
                "maps": [
                    {
                        "title": "Thermal topic",
                        "slug": "thermal-topic",
                        "summary": summaries[filename],
                        "tags": [filename[0]],
                        "concept_slugs": ["conductivity"],
                    }
                ],
            }
        )

    async def model(system, messages, config, **kwargs):
        prompt = messages[-1]["content"] if isinstance(messages, list) else str(messages)
        task = kwargs.get("task")
        if task == "review":
            return json.dumps(review_answer(prompt, kwargs["format"], {}))
        if task == "identity":
            identity_calls.append(prompt)
            return json.dumps(
                identity_answer(prompt, kwargs["format"], verdict), ensure_ascii=False
            )
        return produce(prompt)

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", model)

    async def build(**kwargs):
        with input_registry(config) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        result = await run_pipeline(
            tmp_path,
            loaded.sources,
            config,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
            **kwargs,
        )
        assert not result.errors
        return result

    def assert_current(*owners):
        # Lossless shared-MoC contract: the registry aggregate equals the
        # rendered merge — every CURRENT owner's reviewed summary paragraph in
        # filename order; a deleted owner's paragraph is gone.
        expected_summary = "\n\n".join(summaries[name] for name in sorted(owners))
        with published_registry(config) as registry:
            rows = registry.connection.execute("SELECT * FROM moc").fetchall()
            assert len(rows) == 1
            row = dict(rows[0])
            assert row["summary"] == expected_summary
            assert json.loads(row["payload_json"])["summary"] == expected_summary
            assert row["content_hash"] == hashlib.sha256(row["payload_json"].encode()).hexdigest()
            assert registry.connection.execute("SELECT count(*) FROM concept").fetchone()[0] == 1
            assert {
                str(r[0])
                for r in registry.connection.execute(
                    "SELECT s.filename FROM concept_source cs JOIN source s ON s.id=cs.source_id"
                )
            } == set(owners)
            assert registry.get_pipeline_checkpoint()
            assert registry.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        page = (published_root(config) / "mocs" / "thermal-topic.md").read_text()
        for name in summaries:
            assert (summaries[name] in page) == (name in owners)
        return row

    def write(filename, extra=""):
        (config.sources_dir / filename).write_text(
            "---\ntitle: Thermal observations\n---\nMOC_LIFECYCLE_"
            + filename
            + "\n"
            + summaries[filename]
            + "\n"
            + SHARED_FACT
            + extra
        )

    seen = []
    for filename in arrival:
        write(filename)
        assert (await build()).compiled == 1
        seen.append(filename)
        assert_current(*seen)
    assert len(identity_calls) == 1  # the second owner's proposal was assessed once
    with published_registry(config) as registry:
        second = registry.load_synthesis(arrival[1])
        records = second.review_receipt["identity"]["assessments"]
        assert [r["verdict"] for r in records] == ["same"]
        assert records[0]["candidate_context"]["owners"][0]["source_file"] == arrival[0]
        assert registry.load_synthesis(arrival[0]).review_receipt["publication_binding"]
    survivor = next(name for name in summaries if name != deleted)
    (config.sources_dir / deleted).unlink()
    assert (await build()).deleted == 1
    expected = assert_current(survivor)
    for kwargs in ({}, {"refresh_render": True}, {}):
        assert (await build(**kwargs)).compiled == 0
        assert assert_current(survivor) == expected
    assert len(calls) == 2
    assert len(identity_calls) == 1  # no re-assessment for an unchanged survivor
    summaries[survivor] = "Revised current-source conductivity review."
    # The revision keeps the identifying fact: continuity is assessed against the
    # survivor's own admitted revision (sole ownership alone would not suffice).
    write(survivor, "\nNew measurements change the conclusion.")
    assert (await build()).compiled == 1
    revised = assert_current(survivor)
    assert len(identity_calls) == 2
    for kwargs in ({}, {"refresh_render": True}, {}):
        assert (await build(**kwargs)).compiled == 0
        assert assert_current(survivor) == revised
    assert len(calls) == 3
    assert len(identity_calls) == 2


@pytest.mark.asyncio
async def test_unassessed_second_owner_keeps_its_own_moc_and_facts(tmp_path, monkeypatch):
    """Negative control: equal titles/summaries without a bilateral ``same`` split."""
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, min_source_chars=1)
    config.sources_dir.mkdir(parents=True)
    identity_calls = []

    async def model(system, messages, config, **kwargs):
        prompt = messages[-1]["content"]
        task = kwargs.get("task")
        if task == "review":
            return json.dumps(review_answer(prompt, kwargs["format"], {}))
        if task == "identity":
            identity_calls.append(prompt)
            return json.dumps(identity_answer(prompt, kwargs["format"], None))  # insufficient
        filename = "a.md" if "MOC_LIFECYCLE_a.md" in prompt else "b.md"
        return json.dumps(
            {
                "source_title": filename,
                "source_summary": SUMMARY,
                "concepts": [
                    {
                        "title": "Conductivity",
                        "slug": "conductivity",
                        "summary": SUMMARY,
                        "canonical_id": _candidate_id(prompt, "conductivity"),
                        "sections": [{"heading": "Mechanism", "prose": "Heat follows a gradient."}],
                    }
                ],
                "maps": [
                    {
                        "title": "Thermal topic",
                        "slug": "thermal-topic",
                        "summary": f"{filename} review.",
                        "tags": [],
                        "concept_slugs": ["conductivity"],
                    }
                ],
            }
        )

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", model)
    for filename in ("a.md", "b.md"):
        (config.sources_dir / filename).write_text(
            f"---\ntitle: T\n---\nMOC_LIFECYCLE_{filename}\nA {filename} review. {SHARED_FACT}\n"
        )
        with input_registry(config) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        result = await run_pipeline(
            tmp_path,
            loaded.sources,
            config,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
        )
        assert not result.errors
    assert len(identity_calls) == 1
    with published_registry(config) as registry:
        assert registry.connection.execute("SELECT count(*) FROM concept").fetchone()[0] == 2
        assert registry.connection.execute("SELECT count(*) FROM moc").fetchone()[0] == 2
    pages = {
        p.name: p.read_text()
        for p in (published_root(config) / "mocs").glob("*.md")
        if p.name != "index.md"
    }
    assert "thermal-topic.md" in pages and len(pages) == 2
    assert any("b.md review." in text for text in pages.values())


@pytest.mark.asyncio
async def test_refresh_render_after_assessed_revision_rewrites_nothing(tmp_path, monkeypatch):
    """Production repro: after an assessed sole-owner revision, the next full-path
    rebuild of an UNCHANGED vault must not re-persist the source or rewrite its
    MoC row. (Observed: ``publication_binding.basis`` flips from
    ``assessed same entity`` to ``unchanged revision binding`` on re-assignment,
    the frozen plan payload then differs from the committed row, and the source
    plus its MoC aggregate are written again with a new ``updated_at``.)"""
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, min_source_chars=1)
    config.sources_dir.mkdir(parents=True)
    identity_calls = []
    verdict = fact_verdict(same=(SHARED_FACT,))

    async def model(system, messages, config, **kwargs):
        prompt = messages[-1]["content"]
        task = kwargs.get("task")
        if task == "review":
            return json.dumps(review_answer(prompt, kwargs["format"], {}))
        if task == "identity":
            identity_calls.append(prompt)
            return json.dumps(identity_answer(prompt, kwargs["format"], verdict))
        return json.dumps(
            {
                "source_title": "a.md",
                "source_summary": SUMMARY,
                "concepts": [
                    {
                        "title": "Conductivity",
                        "slug": "conductivity",
                        "summary": SUMMARY,
                        "canonical_id": _candidate_id(prompt, "conductivity"),
                        "sections": [{"heading": "Mechanism", "prose": "Heat follows a gradient."}],
                    }
                ],
                "maps": [
                    {
                        "title": "Thermal topic",
                        "slug": "thermal-topic",
                        "summary": "Copper review.",
                        "tags": [],
                        "concept_slugs": ["conductivity"],
                    }
                ],
            }
        )

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", model)

    async def build(**kwargs):
        with input_registry(config) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        result = await run_pipeline(
            tmp_path,
            loaded.sources,
            config,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
            **kwargs,
        )
        assert not result.errors
        return result

    def snapshot():
        with published_registry(config) as registry:
            moc = dict(registry.connection.execute("SELECT * FROM moc").fetchone())
            payload = registry.connection.execute("SELECT payload_json FROM synthesis").fetchone()[
                0
            ]
            basis = {
                k: v["basis"]
                for k, v in registry.load_synthesis("a.md")
                .review_receipt["publication_binding"]
                .items()
            }
        marker = (published_meta(config) / "identity-publication.json").read_bytes()
        return moc, payload, basis, marker

    source = config.sources_dir / "a.md"
    source.write_text(f"---\ntitle: T\n---\nMOC_LIFECYCLE_a.md\nCopper review. {SHARED_FACT}\n")
    assert (await build()).compiled == 1
    source.write_text(source.read_text() + "New measurements.\n")
    assert (await build()).compiled == 1
    assert len(identity_calls) == 1  # sole-owner continuity was assessed once
    generation = published_root(config)
    old_files = file_state(generation)
    before = snapshot()
    assert set(before[2].values()) == {"assessed same entity"}
    assert (await build(refresh_render=True)).compiled == 0
    assert len(identity_calls) == 1
    after = snapshot()
    assert after[2] == before[2], "binding basis must be idempotent across unchanged rebuilds"
    assert after[1] == before[1], "unchanged source payload was re-persisted"
    assert after[0] == before[0], "MoC row rewritten (updated_at) on an unchanged rebuild"
    # Explicit refresh has a fresh staging/source path in its semantic marker;
    # the original committed marker AND every prior artifact remain immutable.
    assert file_state(generation) == old_files
    assert (generation / ".llmwiki/identity-publication.json").read_bytes() == before[3]
    assert json.loads(after[3])["generation"] == json.loads(before[3])["generation"]
    refreshed = published_root(config)
    refreshed_files = file_state(refreshed)
    assert (await build()).compiled == 0
    assert published_root(config) == refreshed
    assert file_state(refreshed) == refreshed_files
    assert len(identity_calls) == 1
