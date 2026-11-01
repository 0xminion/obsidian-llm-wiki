"""Failure must not be interpreted as permission to delete prior knowledge."""

from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.state import read_state
from tests.new.grounding_doubles import legacy_fixture_llm, producer_call_count
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
)
from tests.new.test_incremental_cache import FAKE_SYNTH_A


@pytest.mark.asyncio
async def test_forced_failed_rebuild_preserves_last_good_notes(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    assert (await run_pipeline(tmp_path, sources, config)).compiled == 1
    page = published_root(config) / "concepts" / "concept-a1.md"
    before = page.read_bytes()
    llm.side_effect = RuntimeError("provider offline")
    result = await run_pipeline(tmp_path, sources, config, force=True)
    assert result.errors
    assert page.exists(), "failed regeneration deleted last successful concept"
    assert page.read_bytes() == before
    with published_registry(config) as registry:
        assert registry.load_syntheses({"alpha.md"})


@pytest.mark.asyncio
async def test_invalid_changed_source_preserves_last_good_notes(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A)),
    )
    await run_pipeline(
        tmp_path, {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}, config
    )
    page = published_root(config) / "concepts" / "concept-a1.md"
    before = page.read_bytes()
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    (config.sources_dir / "alpha.md").write_text("x")
    result = await run_pipeline(tmp_path, {"alpha.md": SourceDoc("Alpha", "x")}, config)
    assert result.errors
    assert page.exists()
    # Last-good knowledge is preserved verbatim, but because the source changed
    # and the new revision was not admitted, the page is labelled as retained
    # content (grounded-core review L3) instead of passing as current output.
    after = (published_root(config) / "concepts" / "concept-a1.md").read_text(encoding="utf-8")
    assert page.read_bytes() == before
    assert "> [!warning] Not current verified content" in after
    body_before = before.decode("utf-8").split("\n---\n", 1)[1]
    retained_marker = " *(retained from an earlier build; not current verified evidence)*"
    for line in body_before.splitlines():
        if not line.strip() or line.startswith("   \u2014"):
            continue  # evidence source line: anchor is withheld for retained evidence
        relabelled = line.replace(" [1]", " [1*]")
        if line.startswith("1. "):
            relabelled = line + retained_marker
        assert line in after or relabelled in after, line


@pytest.mark.asyncio
async def test_final_source_deletion_commits_and_preserves_orphan(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A)),
    )
    await run_pipeline(
        tmp_path, {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}, config
    )
    old_root = published_root(config)
    old_files = file_state(old_root)
    result = await run_pipeline(tmp_path, {}, config)
    assert result.deleted == 1
    assert published_root(config) != old_root
    assert file_state(old_root) == old_files
    with published_registry(config) as registry:
        assert registry.source_filenames() == set()
    assert read_state(published_meta(config) / "state.json").sources == {}
    orphan = published_root(config) / "concepts" / "concept-a1.md"
    assert orphan.exists()
    assert "orphaned: true" in orphan.read_text()
    assert (await run_pipeline(tmp_path, {}, config)).deleted == 0


@pytest.mark.asyncio
async def test_render_failure_releases_synthesis_lease_for_immediate_retry(tmp_path, monkeypatch):
    from obsidian_llm_wiki.render.obsidian import render_vault

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    sources = {"alpha.md": SourceDoc("Alpha", "Substantive source observations. " * 8)}
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A)),
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
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    await run_pipeline(tmp_path, sources, config)
    config.chunk_size += 100
    from obsidian_llm_wiki.core import pipeline

    normalize = pipeline._normalize_and_assign_synthesis_identity

    def normalize_and_report_migration(*args):
        normalize(*args)  # preserve real effective assignments / binding evidence
        return True

    monkeypatch.setattr(
        pipeline,
        "_normalize_and_assign_synthesis_identity",
        normalize_and_report_migration,
    )
    result = await run_pipeline(tmp_path, sources, config)
    assert result.compiled == 1, "identity-only migration incorrectly blessed old model output"
    assert producer_call_count(llm) == 2


@pytest.mark.asyncio
async def test_deleting_all_owners_preserves_shared_orphan(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A)),
    )
    sources = {
        "alpha.md": SourceDoc("Alpha", "Alpha observations. " * 8),
        "beta.md": SourceDoc("Beta", "Beta observations. " * 8),
    }
    await run_pipeline(tmp_path, sources, config)
    orphan = published_root(config) / "concepts" / "concept-shared.md"
    assert orphan.exists()
    old_files = file_state(orphan.parents[1])
    result = await run_pipeline(tmp_path, {}, config)
    assert result.deleted == 2
    assert orphan.exists(), "simultaneous owner deletion lost shared concept"
    assert file_state(orphan.parents[1]) == old_files
    assert (
        "orphaned: true" in (published_root(config) / "concepts" / "concept-shared.md").read_text()
    )
