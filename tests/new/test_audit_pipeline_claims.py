from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from tests.new.test_incremental_cache import FAKE_SYNTH_A


@pytest.mark.asyncio
async def test_pipeline_claims_at_worker_start_not_all_before_workers(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), compile_concurrency=1, embeddings_enabled=False)
    observed = []

    async def provider(*args, **kwargs):
        with VaultRegistry(config.llmwiki_dir) as registry:
            observed.append(len(registry.list_dirty(state="running")))
        return FAKE_SYNTH_A

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    result = await run_pipeline(tmp_path, {
        f"source-{i}.md": SourceDoc(f"Source {i}", "Observed experimental evidence. " * 8)
        for i in range(3)
    }, config)
    assert not result.errors
    assert observed[0] == 1


@pytest.mark.asyncio
async def test_pipeline_reclaims_proven_abandoned_claim_under_compile_lock(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    with VaultRegistry(config.llmwiki_dir) as registry:
        item = registry.enqueue_dirty("source", "source.md", "synthesis_retry", {})
        assert registry.start_dirty(item, worker_id="pipeline:dead-process")
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )
    result = await run_pipeline(tmp_path, {
        "source.md": SourceDoc("Source", "Observed experimental evidence. " * 8)
    }, config)
    assert result.compiled == 1
    assert not result.errors


@pytest.mark.asyncio
async def test_force_build_explicitly_resets_retry_backoff(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"source.md": SourceDoc("Source", "Observed experimental evidence. " * 8)}
    llm = AsyncMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    assert (await run_pipeline(tmp_path, sources, config)).errors
    llm.side_effect = None
    llm.return_value = FAKE_SYNTH_A
    result = await run_pipeline(tmp_path, sources, config, force=True)
    assert result.compiled == 1
    assert not result.errors


def test_lease_renewal_only_extends_live_owned_claims(tmp_path):
    with VaultRegistry(tmp_path) as registry:
        own = registry.enqueue_dirty("source", "own", "retry", {})
        other = registry.enqueue_dirty("source", "other", "retry", {})
        expired = registry.enqueue_dirty("source", "expired", "retry", {})
        registry.start_dirty(own, worker_id="pipeline:owner")
        registry.start_dirty(other, worker_id="pipeline:other")
        registry.start_dirty(expired, worker_id="pipeline:owner")
        registry.connection.execute(
            "UPDATE dirty_queue SET lease_expires_at='2000-01-01' WHERE id=?", (expired,)
        )
        registry.connection.commit()
        assert registry.renew_dirty_worker("pipeline:owner") == 1
        assert not registry.complete_dirty(expired, worker_id="pipeline:owner")
        assert registry.complete_dirty(own, worker_id="pipeline:owner")
        assert registry.complete_dirty(other, worker_id="pipeline:other")


@pytest.mark.asyncio
async def test_heartbeat_reports_failure_instead_of_silently_stopping():
    from obsidian_llm_wiki.core.pipeline import _maintain_pipeline_leases

    class BrokenRegistry:
        def renew_dirty_worker(self, worker_id):
            raise OSError("database unavailable")

    errors = []
    await _maintain_pipeline_leases(BrokenRegistry(), "pipeline:owner", errors, interval=0)
    assert errors == ["queue: lease heartbeat failed: database unavailable"]
