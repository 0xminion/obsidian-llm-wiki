"""Failure must not be interpreted as permission to delete prior knowledge."""
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.state import read_state
from tests.new.test_incremental_cache import FAKE_SYNTH_A


@pytest.mark.asyncio
async def test_forced_failed_rebuild_preserves_last_good_notes(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    assert (await run_pipeline(tmp_path, sources, config)).compiled == 1
    page = config.concepts_dir / "concept-a1.md"
    before = page.read_bytes()
    llm.side_effect = RuntimeError("provider offline")
    result = await run_pipeline(tmp_path, sources, config, force=True)
    assert result.errors
    assert page.exists(), "failed regeneration deleted last successful concept"
    assert page.read_bytes() == before
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.load_syntheses({"alpha.md"})


@pytest.mark.asyncio
async def test_invalid_changed_source_preserves_last_good_notes(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )
    await run_pipeline(
        tmp_path, {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}, config
    )
    page = config.concepts_dir / "concept-a1.md"
    before = page.read_bytes()
    result = await run_pipeline(tmp_path, {"alpha.md": SourceDoc("Alpha", "")}, config)
    assert result.errors
    assert page.exists()
    assert page.read_bytes() == before


@pytest.mark.asyncio
async def test_final_source_deletion_commits_and_preserves_orphan(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )
    await run_pipeline(
        tmp_path, {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}, config
    )
    result = await run_pipeline(tmp_path, {}, config)
    assert result.deleted == 1
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.source_filenames() == set()
    assert read_state(config.state_file).sources == {}
    orphan = config.concepts_dir / "concept-a1.md"
    assert orphan.exists()
    assert "orphaned: true" in orphan.read_text()
    assert (await run_pipeline(tmp_path, {}, config)).deleted == 0


@pytest.mark.asyncio
async def test_render_failure_releases_synthesis_lease_for_immediate_retry(tmp_path, monkeypatch):
    from obsidian_llm_wiki.render.obsidian import render_vault

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )

    def fail_render(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.render_vault", fail_render)
    with pytest.raises(OSError, match="disk unavailable"):
        await run_pipeline(tmp_path, sources, config)
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert not registry.list_dirty(state="running")
    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.render_vault", render_vault)
    result = await run_pipeline(tmp_path, sources, config)
    assert result.compiled == 1
    assert not result.errors


@pytest.mark.asyncio
async def test_metrics_failure_does_not_leak_compile_lock(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.lock import acquire_lock, release_lock

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)

    def fail_metrics(*args, **kwargs):
        raise OSError("metrics unavailable")

    monkeypatch.setattr("obsidian_llm_wiki.core.metrics.MetricsCollector.save", fail_metrics)
    with pytest.raises(OSError, match="metrics unavailable"):
        await run_pipeline(tmp_path, {}, config)
    assert acquire_lock(config.lock_file), "metrics failure leaked lock"
    release_lock(config.lock_file)


@pytest.mark.asyncio
async def test_identity_migration_cannot_promote_stale_synthesis_versions(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await run_pipeline(tmp_path, sources, config)
    config.chunk_size += 100
    monkeypatch.setattr(
        "obsidian_llm_wiki.core.pipeline._normalize_and_assign_synthesis_identity",
        lambda *args: True,
    )
    result = await run_pipeline(tmp_path, sources, config)
    assert result.compiled == 1, "identity-only migration incorrectly blessed old model output"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_deleting_all_owners_preserves_shared_orphan(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )
    sources = {
        "alpha.md": SourceDoc("Alpha", "Alpha observations. " * 8),
        "beta.md": SourceDoc("Beta", "Beta observations. " * 8),
    }
    await run_pipeline(tmp_path, sources, config)
    orphan = config.concepts_dir / "concept-shared.md"
    assert orphan.exists()
    result = await run_pipeline(tmp_path, {}, config)
    assert result.deleted == 2
    assert orphan.exists(), "simultaneous owner deletion lost shared concept"
    assert "orphaned: true" in orphan.read_text()
