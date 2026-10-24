import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import _synthesize_source, run_pipeline


@pytest.mark.asyncio
async def test_placeholder_chunk_rejects_complete_source(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), chunk_size=200, embeddings_enabled=False)
    calls = 0

    async def response(*args, **kwargs):
        nonlocal calls
        calls += 1
        return json.dumps({
            "source_summary": "Treatment increased observed growth." if calls == 1 else "TODO",
            "concepts": [], "maps": [],
        })

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", response)
    result = await run_pipeline(tmp_path, {"experiment.md": SourceDoc(
        "Experiment", "Treatment increased observed growth, whereas controls were unchanged. " * 15
    )}, config)
    assert result.compiled == 0
    assert result.errors
    assert not (config.entries_dir / "experiment.md").exists()


@pytest.mark.asyncio
async def test_chunk_tasks_are_bounded_not_only_provider_requests(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), chunk_size=60, compile_concurrency=2)
    initial_tasks = len(asyncio.all_tasks())
    live = []

    async def response(*args, **kwargs):
        live.append(len(asyncio.all_tasks()) - initial_tasks)
        await asyncio.sleep(0)
        return json.dumps({"source_summary": "Observation recorded.", "concepts": [], "maps": []})

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", response)
    result = await _synthesize_source(config, "experiment.md", SourceDoc(
        "Experiment", "An experimental observation was recorded. " * 200
    ), [])
    assert result is not None
    assert max(live) <= 4


@pytest.mark.asyncio
async def test_bounded_chunk_worker_propagates_cancellation():
    from obsidian_llm_wiki.synth.quality import _bounded_map

    async def cancelled(_item):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _bounded_map([1], cancelled, max_workers=1)


@pytest.mark.asyncio
async def test_metadata_only_title_edit_refreshes_entry_and_caches(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.cache import load_synthesis
    from obsidian_llm_wiki.core.registry import VaultRegistry
    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    sources = {"alpha.md": SourceDoc("Original title", "Measured observations. " * 8)}
    await run_pipeline(tmp_path, sources, config)
    sources["alpha.md"].title = "Corrected title"
    result = await run_pipeline(tmp_path, sources, config)
    assert not result.errors
    assert llm.call_count == 1
    assert "title: Corrected title" in (config.entries_dir / "alpha.md").read_text()
    assert load_synthesis(config.llmwiki_dir, "alpha.md").source_title == "Corrected title"
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.load_synthesis("alpha.md").source_title == "Corrected title"
