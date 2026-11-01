"""Runtime receipts must describe admitted source bytes, not lazy placeholders."""
import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.cache import load_synthesis, synthesis_cache_path
from obsidian_llm_wiki.core.models import ConceptNote, SourceSynthesis
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.state import hash_content, read_state
from obsidian_llm_wiki.ingest.sources import load_sources_incremental
from tests.new.grounding_doubles import reviewed
from tests.new.immutable_lifecycle_helpers import compile_legacy
from tests.new.immutable_semantic_helpers import (
    assert_preserved,
    file_state,
    input_registry,
    published_meta,
    published_registry,
    published_root,
)


@pytest.mark.asyncio
async def test_repaired_cache_keeps_hash_of_unloaded_unchanged_source(tmp_path, monkeypatch):
    """Cache repair can persist a lazy source without changing its source revision."""
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    body = ("A bounded queue controls producer throughput. " * 10).strip()
    (config.sources_dir / "a.md").write_text(body)
    calls = []

    async def synthesize(cfg, filename, source, *_args, **_kwargs):
        calls.append(filename)
        return reviewed(SourceSynthesis(
            source_title=source.title, source_summary="Bounded queue.", source_file=filename,
            concepts=[ConceptNote("Bounded queue", "bounded-queue", "Controls throughput.")],
        ), source, filename, cfg)

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)

    async def build(*, legacy=False):
        with input_registry(config) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        runner = compile_legacy if legacy else run_pipeline
        args = (config, loaded.sources) if legacy else (tmp_path, loaded.sources, config)
        result = await runner(
            *args,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures,
        )
        assert not result.errors
        return loaded, result

    await build(legacy=True)  # repair a closed legacy cache, never damage published data
    original_hash = read_state(published_meta(config) / "state.json").sources["a.md"].hash
    assert original_hash == hash_content(body)
    # Missing portable cache forces repair from the authoritative registry,
    # without granting permission to resynthesize or hash the empty lazy body.
    synthesis_cache_path(config.llmwiki_dir, "a.md").unlink()
    loaded, result = await build()
    assert loaded.sources["a.md"].content == ""
    assert result.compiled == 0 and calls == ["a.md"]
    assert load_synthesis(published_meta(config), "a.md") is not None
    assert not synthesis_cache_path(config.llmwiki_dir, "a.md").exists(), "legacy input untouched"
    assert read_state(published_meta(config) / "state.json").sources["a.md"].hash == original_hash
    with published_registry(config) as registry:
        assert registry.connection.execute(
            "SELECT content_hash FROM source WHERE filename=?", ("a.md",),
        ).fetchone()[0] == original_hash
    previous = published_root(config)
    before = file_state(previous)
    _, noop = await build()
    assert noop.compiled == 0 and calls == ["a.md"]
    assert published_root(config) == previous
    assert_preserved(previous, before)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_boundary", ["missing-render", "omitted-contribution"])
async def test_incomplete_consumers_cannot_commit(tmp_path, monkeypatch, broken_boundary):
    """Success-shaped doubles must not bypass the complete-plan fence."""
    from obsidian_llm_wiki.core.models import SourceDoc

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    source = SourceDoc("Queue", "Bounded queues provide backpressure. " * 10)

    async def synthesize(cfg, filename, doc, *_args, **_kwargs):
        return reviewed(SourceSynthesis(
            source_title=doc.title, source_summary="Bounded queue.", source_file=filename,
            concepts=[ConceptNote("Bounded queue", "bounded-queue", "Controls throughput.")],
        ), doc, filename, cfg)

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)
    if broken_boundary == "missing-render":
        from obsidian_llm_wiki.core import pipeline

        real_render = pipeline.render_vault

        def missing_render(directory, *args, **kwargs):
            pages = real_render(directory, *args, **kwargs)
            # A fault after real rendering, only in the private workspace.
            (directory / "concepts/bounded-queue.md").unlink()
            return pages

        monkeypatch.setattr(pipeline, "render_vault", missing_render)
        message = "missing page"
    else:
        def omit(bundle, **_kwargs):
            bundle.concepts.clear()  # source still contributes the missing entity

        monkeypatch.setattr("obsidian_llm_wiki.synth.dedupe.semantic_dedupe_concepts", omit)
        message = "omits or retargets a contribution"
    with pytest.raises(RuntimeError, match=message):
        await run_pipeline(tmp_path, {"a.md": source}, config)
    from obsidian_llm_wiki.core.immutable_publication import current_publication

    assert current_publication(config.wiki_dir, allow_uncommitted=True) is None
    assert not (config.wiki_dir / "OLW-Publications/current.json").exists()
    assert not (config.llmwiki_dir / "registry.db").exists()
    assert load_synthesis(published_meta(config), "a.md") is None
    assert not read_state(published_meta(config) / "state.json").sources


@pytest.mark.asyncio
async def test_failed_revision_cache_repair_keeps_last_admitted_hash(tmp_path, monkeypatch):
    """Retained content/cache repair never blesses a failed new input revision."""
    from obsidian_llm_wiki.core.models import SourceDoc

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    old = SourceDoc("Queue", "Bounded queues provide backpressure. " * 10)

    async def synthesize(cfg, filename, doc, *_args, **_kwargs):
        return reviewed(SourceSynthesis(
            source_title=doc.title, source_summary="Bounded queue.", source_file=filename,
            concepts=[ConceptNote("Bounded queue", "bounded-queue", "Controls throughput.")],
        ), doc, filename, cfg)

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)
    assert (await compile_legacy(config, {"a.md": old})).compiled == 1
    synthesis_cache_path(config.llmwiki_dir, "a.md").unlink()
    legacy_before = file_state(config.llmwiki_dir)

    async def failed(*_args, **_kwargs):
        raise RuntimeError("offline fixture")

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", failed)
    new = SourceDoc("Queue", "Unbounded queues exhaust memory. " * 10)
    # This is the explicit new human revision; old evidence remains in legacy metadata.
    (config.sources_dir / "a.md").write_text(new.content)
    result = await run_pipeline(tmp_path, {"a.md": new}, config)
    assert result.compiled == 0 and any("offline fixture" in e for e in result.errors)
    state = read_state(published_meta(config) / "state.json")
    assert state.sources["a.md"].hash == hash_content(old.content)
    with published_registry(config) as registry:
        assert registry.connection.execute(
            "SELECT content_hash FROM source WHERE filename=?", ("a.md",),
        ).fetchone()[0] == hash_content(old.content)
    page = published_root(config) / "concepts/bounded-queue.md"
    assert "Not current verified content" in page.read_text()
    from obsidian_llm_wiki.core.models import source_synthesis_to_dict

    portable = load_synthesis(published_meta(config), "a.md")
    assert portable is not None, "missing legacy cache must be repaired in the new generation"
    with published_registry(config) as registry:
        assert source_synthesis_to_dict(portable) == source_synthesis_to_dict(
            registry.load_synthesis("a.md"),
        )
    assert_preserved(config.llmwiki_dir, legacy_before)
    assert (config.sources_dir / "a.md").read_text() == new.content
    assert (published_meta(config) / "source-inputs/a.md").read_text() == new.content
    view = (published_root(config) / "sources/a.md").read_text()
    assert old.content.strip() in view and new.content.strip() not in view


@pytest.mark.asyncio
async def test_failed_disk_revision_remains_retryable_and_visibly_stale(tmp_path, monkeypatch):
    from tests.new.grounding_doubles import legacy_fixture_llm

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    source_path = config.sources_dir / "a.md"
    source_path.write_text("A bounded queue limits memory. " * 10)
    available = True
    attempts = []

    async def synthesize(cfg, filename, doc, *_args, **_kwargs):
        attempts.append(doc.content)
        if not available:
            raise RuntimeError("offline fixture")
        return reviewed(SourceSynthesis(
            source_title=doc.title, source_summary="Queue", source_file=filename,
            concepts=[ConceptNote("Queue", "queue", "Queue properties.")],
        ), doc, filename, cfg)

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", legacy_fixture_llm(""))

    async def build(**kwargs):
        with input_registry(config) as registry:
            loaded = load_sources_incremental(config.sources_dir, registry)
        return await run_pipeline(
            tmp_path, loaded.sources, config,
            unchanged_filenames=loaded.unchanged_filenames,
            source_file_signatures=loaded.signatures, **kwargs,
        )

    assert (await build()).compiled == 1
    previous = published_root(config)
    before = file_state(previous)
    source_path.write_text("An unbounded queue can exhaust memory. " * 10)
    raw_new = source_path.read_bytes()
    available = False
    failed = await build()
    assert failed.errors and len(attempts) == 2
    with published_registry(config) as registry:
        pending = [
            r for r in registry.list_dirty(state="pending")
            if r.entity_id == "a.md" and r.reason == "synthesis_retry"
        ]
        assert len(pending) == 1 and pending[0].attempts == 1
        assert "offline fixture" in pending[0].error and pending[0].next_attempt_at
    again = await build()
    assert len(attempts) == 2, "backoff must not invoke the model again"
    with published_registry(config) as registry:
        retained = [
            r for r in registry.list_dirty(state="pending")
            if r.entity_id == "a.md" and r.reason == "synthesis_retry"
        ]
        assert retained == pending
    assert again.errors, "failed revision was incorrectly certified as an unchanged success"
    page = published_root(config) / "concepts/queue.md"
    assert "Not current verified content" in page.read_text()
    available = True
    recovered = await build(force=True)  # explicit retry resets queue backoff
    assert recovered.compiled == 1 and not recovered.errors
    assert "unbounded" in attempts[-1]
    assert len(attempts) == 3
    assert source_path.read_bytes() == raw_new
    from tests.new.immutable_lifecycle_helpers import assert_input_hash

    assert_input_hash(config, "a.md", raw_new)
    with published_registry(config) as registry:
        assert registry.source_content_hashes() == {"a.md": hash_content(attempts[-1])}
        assert not [
            r for r in registry.list_dirty(state="pending")
            if r.entity_id == "a.md" and r.reason == "synthesis_retry"
        ]
    assert_preserved(previous, before)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["cache-delete", "cache-change", "source-view-change"])
async def test_committed_generation_damage_is_refused_not_repaired(tmp_path, monkeypatch, damage):
    """Missing legacy caches are repairable; damaged published files are corruption."""
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.registry import VaultRegistry

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    source = SourceDoc("Queue", "A bounded queue controls producer throughput. " * 10)

    async def synthesize(cfg, name, doc, *_args, **_kwargs):
        return reviewed(SourceSynthesis(
            source_title=doc.title, source_summary="Bounded queue", source_file=name,
            concepts=[ConceptNote("Queue", "queue", "Controls throughput.")],
        ), doc, name, cfg)

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)
    result = await run_pipeline(tmp_path, {"a.md": source}, config)
    assert result.compiled == 1 and not result.errors
    root = published_root(config)
    relative = "sources/a.md" if damage == "source-view-change" else ".llmwiki/cache/a.md.json"
    target = root / relative
    assert target.is_file()
    if damage == "cache-delete":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"\nmanual corruption")
    before = file_state(config.vault)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("corrupt generation must refuse before model work")

    def no_registry(*_args, **_kwargs):
        pytest.fail("corruption refusal must not open any writable registry")

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", forbidden)
    monkeypatch.setattr(VaultRegistry, "__init__", no_registry)
    with pytest.raises(RuntimeError, match="generation inventory/digest mismatch"):
        await run_pipeline(tmp_path, {"a.md": source}, config, refresh_render=True)
    assert file_state(config.vault) == before
