"""An unchanged build must not deserialize the complete synthesis corpus."""
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental
from tests.new.test_incremental_cache import FAKE_SYNTH_A


async def _build(config):
    with VaultRegistry(config.llmwiki_dir) as registry:
        loaded = load_sources_incremental(config.sources_dir, registry)
    return await run_pipeline(
        config.vault, loaded.sources, config,
        unchanged_filenames=loaded.unchanged_filenames,
        source_file_signatures=loaded.signatures,
    )


@pytest.mark.asyncio
async def test_unchanged_build_skips_corpus_loading_and_rendering(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8
    )
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    first = await _build(config)
    assert first.compiled == 1

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged run materialized the corpus")

    monkeypatch.setattr(VaultRegistry, "load_syntheses", forbidden)
    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.read_state", forbidden)
    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.render_vault", forbidden)
    second = await _build(config)
    assert second.skipped == 1
    assert second.compiled == 0
    assert second.errors == []
    assert second.pages == []
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_checkpoint_invalidated_by_recompile(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.pipeline import recompile_single_source

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8
    )
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await _build(config)
    await recompile_single_source(tmp_path, "alpha.md", config)
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.get_pipeline_checkpoint() == ""
    result = await _build(config)
    assert not result.errors
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.get_pipeline_checkpoint()


@pytest.mark.asyncio
async def test_refresh_render_repairs_missing_note_without_llm(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8
    )
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await _build(config)
    page = config.concepts_dir / "concept-a1.md"
    page.unlink()
    with VaultRegistry(config.llmwiki_dir) as registry:
        loaded = load_sources_incremental(config.sources_dir, registry)
    result = await run_pipeline(
        tmp_path, loaded.sources, config,
        unchanged_filenames=loaded.unchanged_filenames,
        source_file_signatures=loaded.signatures,
        refresh_render=True,
    )
    assert not result.errors
    assert page.exists()
    assert llm.call_count == 1
