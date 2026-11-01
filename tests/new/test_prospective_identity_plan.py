"""Vertical assignment/publication regressions; no external model calls."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.models import (
    ConceptNote,
    SourceDoc,
    SourceSynthesis,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.synth.review import build_unchecked_receipt
from tests.new.test_immutable_publication import human_state


def committed(config):
    snapshot = current_publication(config.wiki_dir)
    assert snapshot is not None
    return snapshot[0]


def registry_syntheses(generation, filenames):
    raw = bytearray((generation / ".llmwiki/registry.db").read_bytes())
    assert raw.startswith(b"SQLite format 3\x00")
    raw[18] = raw[19] = 1
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bytes(raw))
        connection.execute("PRAGMA query_only=ON")
        return VaultRegistry.load_syntheses(SimpleNamespace(connection=connection), filenames)
    finally:
        connection.close()


def generation_state(generation):
    return {p: human_state(p) for p in generation.rglob("*") if p.is_file()}


def test_same_run_owner_retarget_has_one_committed_assignment(tmp_path, monkeypatch):
    config = Config(
        vault_path=str(tmp_path),
        grounding_review="off",
        embeddings_enabled=False,
        min_source_chars=1,
    )

    async def produce(config, filename, source, *args, **kwargs):
        synthesis = SourceSynthesis(
            source_title=source.title,
            source_file=filename,
            source_summary=source.content,
            concepts=[
                ConceptNote(
                    title="Packet Scheduler", slug="packet-scheduler", summary=source.content
                )
            ],
        )
        synthesis.review_receipt = build_unchecked_receipt(synthesis, source.content, filename)
        return synthesis

    monkeypatch.setattr(pipeline, "_synthesize_with_retry", produce)
    old = SourceDoc(title="Kernel", content="A kernel module that orders packets.")
    new = SourceDoc(title="Postal", content="A postal robot that sorts parcels.")
    first = asyncio.run(pipeline.run_pipeline(tmp_path, {"sched.md": old}, config))
    assert not first.errors
    previous = committed(config)
    previous_state = generation_state(previous)
    second = asyncio.run(
        pipeline.run_pipeline(tmp_path, {"sched.md": new, "sched2.md": old}, config)
    )
    assert not second.errors
    assert generation_state(previous) == previous_state
    generation = committed(config)
    persisted = registry_syntheses(generation, {"sched.md", "sched2.md"})
    ids = {name: item.concepts[0].canonical_id for name, item in persisted.items()}
    assert ids["sched.md"] != ids["sched2.md"], "same-run retarget mixed unrelated subjects"
    for name, item in persisted.items():
        cache = load_synthesis(generation / ".llmwiki", name)
        assert source_synthesis_to_dict(cache) == source_synthesis_to_dict(item)
        page = (generation / "concepts" / f"{item.concepts[0].slug}.md").read_text()
        assert item.concepts[0].summary in page
    marker = json.loads((generation / ".llmwiki/identity-publication.json").read_text())
    assert marker["version"] == "identity-publication-plan-v1"
    assert marker["status"] == "committed"
    assert marker["assignments"] == {name: [ids[name]] for name in ids}


def test_failed_cache_commit_restores_last_good_all_consumers(tmp_path, monkeypatch):
    config = Config(
        vault_path=str(tmp_path),
        grounding_review="off",
        embeddings_enabled=False,
        min_source_chars=1,
    )

    async def produce(config, filename, source, *args, **kwargs):
        synthesis = SourceSynthesis(
            source.title,
            source.content,
            source_file=filename,
            concepts=[ConceptNote("Subject", "subject", source.content)],
        )
        synthesis.review_receipt = build_unchecked_receipt(synthesis, source.content, filename)
        return synthesis

    monkeypatch.setattr(pipeline, "_synthesize_with_retry", produce)
    old = SourceDoc("Old", "The old subject.")
    asyncio.run(pipeline.run_pipeline(tmp_path, {"a.md": old}, config))
    generation = committed(config)
    before = (generation / "concepts/subject.md").read_bytes()
    all_before = generation_state(generation)
    cache_before = source_synthesis_to_dict(load_synthesis(generation / ".llmwiki", "a.md"))
    real_save = pipeline.save_synthesis

    def fail(*args, **kwargs):
        raise OSError("injected cache failure after render")

    monkeypatch.setattr(pipeline, "save_synthesis", fail)
    with pytest.raises(OSError, match="injected"):
        asyncio.run(
            pipeline.run_pipeline(
                tmp_path, {"a.md": SourceDoc("New", "An unrelated subject.")}, config
            )
        )
    assert committed(config) == generation
    assert (generation / "concepts/subject.md").read_bytes() == before
    assert generation_state(generation) == all_before
    assert (
        source_synthesis_to_dict(registry_syntheses(generation, {"a.md"})["a.md"])
        == cache_before
    )
    assert source_synthesis_to_dict(load_synthesis(generation / ".llmwiki", "a.md")) == cache_before
    monkeypatch.setattr(pipeline, "save_synthesis", real_save)
    result = asyncio.run(pipeline.run_pipeline(tmp_path, {"a.md": old}, config))
    assert not result.errors
    assert committed(config) == generation
    assert generation_state(generation) == all_before


def test_interrupted_generation_is_recovered_before_reuse(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    config = Config(
        vault_path=str(tmp_path),
        grounding_review="off",
        embeddings_enabled=False,
        min_source_chars=1,
    )

    async def produce(config, filename, source, *args, **kwargs):
        synthesis = SourceSynthesis(
            source.title,
            source.content,
            source_file=filename,
            concepts=[ConceptNote("Subject", "subject", source.content)],
        )
        synthesis.review_receipt = build_unchecked_receipt(synthesis, source.content, filename)
        return synthesis

    monkeypatch.setattr(pipeline, "_synthesize_with_retry", produce)
    old = SourceDoc("Old", "The old subject.")
    asyncio.run(pipeline.run_pipeline(tmp_path, {"a.md": old}, config))
    generation = committed(config)
    before = (generation / "concepts/subject.md").read_bytes()
    all_before = generation_state(generation)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    pointer_before = human_state(pointer)
    staging = pointer.parent / ".staging"
    stages_before = set(staging.iterdir())
    script = """import asyncio,os
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.core.models import SourceDoc,SourceSynthesis,ConceptNote
from obsidian_llm_wiki.synth.review import build_unchecked_receipt
async def produce(config,filename,source,*a,**kw):
    synthesis=SourceSynthesis(source.title,source.content,source_file=filename,
        concepts=[ConceptNote("Subject","subject",source.content)])
    synthesis.review_receipt=build_unchecked_receipt(synthesis,source.content,filename)
    return synthesis
pipeline._synthesize_with_retry=produce
pipeline.save_synthesis=lambda *a,**kw: os._exit(72)
config=Config(vault_path=VAULT,grounding_review="off",embeddings_enabled=False,min_source_chars=1)
asyncio.run(pipeline.run_pipeline(VAULT,{"a.md":SourceDoc("New","An unrelated subject.")},config))
""".replace("VAULT", repr(str(tmp_path)))
    child = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(pipeline.__file__).parents[2],
        env=dict(os.environ),
        capture_output=True,
    )
    assert child.returncode == 72, child.stderr.decode()
    assert committed(config) == generation
    assert human_state(pointer) == pointer_before
    assert generation_state(generation) == all_before
    assert set(staging.iterdir()) - stages_before, "interrupted workspace should be retained"
    result = asyncio.run(pipeline.run_pipeline(tmp_path, {"a.md": old}, config))
    assert not result.errors
    assert committed(config) == generation
    assert human_state(pointer) == pointer_before
    assert (generation / "concepts/subject.md").read_bytes() == before
    assert generation_state(generation) == all_before
