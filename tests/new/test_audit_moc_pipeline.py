"""Shared MoC aggregates remain fresh through real incremental builds."""

import hashlib
import json
import socket

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", ["a.md", "b.md"])
@pytest.mark.parametrize("arrival", [("a.md", "b.md"), ("b.md", "a.md")])
async def test_shared_moc_deletion_noop_refresh_and_revision(
    tmp_path, monkeypatch, deleted, arrival,
):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, min_source_chars=1)
    config.sources_dir.mkdir(parents=True)
    summaries = {
        "a.md": "Copper review of thermal conductivity.",
        "b.md": "Silver review with a superseded conclusion.",
    }
    calls = []

    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in MoC lifecycle test")

    async def model(prompt, *args, **kwargs):
        filename = next(name for name in summaries if "MOC_LIFECYCLE_" + name in prompt)
        calls.append(filename)
        return json.dumps(
            {
                "source_title": filename,
                "source_summary": "Thermal conductivity relates flux to a temperature gradient.",
                "concepts": [
                    {
                        "title": "Conductivity",
                        "slug": "conductivity",
                        "summary": "Thermal conductivity relates flux to a temperature gradient.",
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

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", model)

    async def build(**kwargs):
        with VaultRegistry(config.llmwiki_dir) as registry:
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

    def assert_current(filename):
        with VaultRegistry(config.llmwiki_dir) as registry:
            rows = registry.connection.execute("SELECT * FROM moc").fetchall()
            assert len(rows) == 1
            row = dict(rows[0])
            assert row["summary"] == summaries[filename]
            assert json.loads(row["payload_json"])["summary"] == summaries[filename]
            assert row["content_hash"] == hashlib.sha256(row["payload_json"].encode()).hexdigest()
            assert registry.get_pipeline_checkpoint()
            assert registry.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert registry.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        page = (config.mocs_dir / "thermal-topic.md").read_text()
        assert summaries[filename] in page
        return row

    seen = []
    for filename in arrival:
        (config.sources_dir / filename).write_text(
            "---\ntitle: Thermal observations\n---\nMOC_LIFECYCLE_"
            + filename
            + "\n"
            + summaries[filename]
        )
        assert (await build()).compiled == 1
        seen.append(filename)
        assert_current(min(seen))
    survivor = next(name for name in summaries if name != deleted)
    (config.sources_dir / deleted).unlink()
    assert (await build()).deleted == 1
    expected = assert_current(survivor)
    for kwargs in ({}, {"refresh_render": True}, {}):
        assert (await build(**kwargs)).compiled == 0
        assert assert_current(survivor) == expected
    assert len(calls) == 2
    summaries[survivor] = "Revised current-source conductivity review."
    path = config.sources_dir / survivor
    path.write_text(path.read_text() + "\nNew measurements change the conclusion.")
    assert (await build()).compiled == 1
    revised = assert_current(survivor)
    for kwargs in ({}, {"refresh_render": True}, {}):
        assert (await build(**kwargs)).compiled == 0
        assert assert_current(survivor) == revised
    assert len(calls) == 3
