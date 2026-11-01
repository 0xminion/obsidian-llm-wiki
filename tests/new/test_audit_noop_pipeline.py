"""An unchanged build must not deserialize the complete synthesis corpus."""

from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental
from tests.new.grounding_doubles import legacy_fixture_llm, producer_call_count
from tests.new.immutable_semantic_helpers import file_state, published_registry, published_root
from tests.new.test_incremental_cache import FAKE_SYNTH_A


async def _build(config):
    with VaultRegistry(config.llmwiki_dir) as registry:
        loaded = load_sources_incremental(config.sources_dir, registry)
    return await run_pipeline(
        config.vault,
        loaded.sources,
        config,
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
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
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
    assert producer_call_count(llm) == 1


@pytest.mark.asyncio
async def test_recompile_revalidates_complete_publication_before_checkpoint(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.pipeline import recompile_single_source

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8
    )
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await _build(config)
    retried = await recompile_single_source(tmp_path, "alpha.md", config)
    assert retried.compiled == 1 and not retried.errors
    from obsidian_llm_wiki.core.immutable_publication import current_publication

    with published_registry(config) as registry:
        # Recompile now uses the complete publication transaction, so a fresh
        # verified checkpoint is valid instead of requiring another render.
        checkpoint = registry.get_pipeline_checkpoint()
        assert checkpoint
    assert current_publication(config.wiki_dir)[0] == published_root(config)
    before = llm.call_count
    result = await _build(config)
    assert not result.errors and result.compiled == 0
    assert llm.call_count == before
    with published_registry(config) as registry:
        assert registry.get_pipeline_checkpoint()


@pytest.mark.asyncio
async def test_refresh_render_refuses_damaged_generation_without_llm(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text(
        "---\ntitle: Alpha\n---\n" + "Experimental observations justify the conclusion. " * 8
    )
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await _build(config)
    root = published_root(config)
    page = root / "concepts" / "concept-a1.md"
    page.unlink()
    before = file_state(root.parent)
    from obsidian_llm_wiki.ingest.sources import load_source_file

    source = load_source_file(config.sources_dir / "alpha.md")
    with pytest.raises(RuntimeError, match="generation inventory/digest mismatch"):
        await run_pipeline(tmp_path, {"alpha.md": source}, config, refresh_render=True)
    assert file_state(page.parents[2]) == before
    assert not page.exists()
    assert producer_call_count(llm) == 1
