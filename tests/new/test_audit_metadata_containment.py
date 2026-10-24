"""Metadata boundaries reject pre-existing symlinks before any outside access."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import lock as lock_module
from obsidian_llm_wiki.core.lock import acquire_lock, release_lock
from obsidian_llm_wiki.core.registry import VaultRegistry


def snapshot(root: Path):
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in root.rglob("*")
    }


@pytest.mark.parametrize("component", ["lock", "lock.guard", "parent", "ancestor"])
@pytest.mark.parametrize("dangling", [False, True])
def test_direct_lock_rejects_symlinks_without_outside_access(tmp_path, component, dangling):
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "vault" / "metadata"
    parent.mkdir(parents=True)
    lock = parent / "lock"
    if component in {"parent", "ancestor"}:
        link = parent if component == "parent" else parent.parent
        if component == "ancestor":
            parent.rmdir()
        link.rmdir()
        link.symlink_to(outside / "missing" if dangling else outside, target_is_directory=True)
    else:
        target = outside / "note"
        if not dangling:
            target.write_bytes(b"IRREPLACEABLE USER NOTE\n")
        (parent / component).symlink_to(target)
    before = snapshot(outside)
    try:
        with pytest.raises(ValueError, match="Unsafe generated path"):
            acquire_lock(lock)
    finally:
        release_lock(lock)
    assert snapshot(outside) == before
    assert not lock_module._held_locks


@pytest.mark.parametrize("replacement", ["lock", "parent", "guard"])
def test_release_closes_owned_descriptor_after_path_replacement(tmp_path, replacement):
    parent = tmp_path / "metadata"
    parent.mkdir()
    lock = parent / "lock"
    outside = tmp_path / "outside"
    outside.mkdir()
    note = outside / "lock"
    note.write_text(str(os.getpid()))
    assert acquire_lock(lock)
    fd = lock_module._held_locks[lock.absolute()]
    if replacement == "parent":
        parent.rename(tmp_path / "original")
        parent.symlink_to(outside, target_is_directory=True)
    else:
        victim = lock if replacement == "lock" else Path(str(lock) + ".guard")
        victim.unlink()
        victim.symlink_to(note)
    before = snapshot(outside)
    try:
        release_lock(lock)
        with pytest.raises(OSError):
            os.fstat(fd)
        assert not lock_module._held_locks
        assert snapshot(outside) == before
    finally:
        # Do not leak a descriptor on the intentionally failing baseline.
        for held in list(lock_module._held_locks.values()):
            os.close(held)
        lock_module._held_locks.clear()


@pytest.mark.parametrize("component", ["root", "ancestor", "registry.db", "registry.db-wal",
                                       "registry.db-shm", "registry.db-journal"])
@pytest.mark.parametrize("dangling", [False, True])
def test_registry_constructor_rejects_symlinks_before_sqlite_open(tmp_path, component, dangling):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "vault" / "metadata"
    root.mkdir(parents=True)
    if component in {"root", "ancestor"}:
        link = root if component == "root" else root.parent
        if component == "ancestor":
            root.rmdir()
        link.rmdir()
        link.symlink_to(outside / "missing" if dangling else outside, target_is_directory=True)
    else:
        target = outside / "user.db"
        if not dangling:
            with sqlite3.connect(target) as conn:
                conn.execute("CREATE TABLE user_data(value TEXT)")
                conn.execute("INSERT INTO user_data VALUES ('PRESERVE')")
        (root / component).symlink_to(target)
    before = snapshot(outside)
    with pytest.raises(ValueError, match="Unsafe generated path"), VaultRegistry(root):
        pass
    assert snapshot(outside) == before
    if component not in {"root", "ancestor", "registry.db"}:
        assert not (root / "registry.db").exists()


@pytest.mark.parametrize("property_name,relative", [
    ("wiki_dir", "04-Wiki"), ("sources_dir", "04-Wiki/sources"),
    ("entries_dir", "04-Wiki/entries"), ("concepts_dir", "04-Wiki/concepts"),
    ("mocs_dir", "04-Wiki/mocs"), ("clippings_dir", "02-Clippings"),
    ("annotations_dir", "03-Raw-Annotations"), ("queries_dir", "05-Queries"),
    ("llmwiki_dir", "04-Wiki/.llmwiki"), ("state_file", "04-Wiki/.llmwiki/state.json"),
    ("lock_file", "04-Wiki/.llmwiki/lock"),
    ("candidates_dir", "04-Wiki/.llmwiki/candidates"),
])
def test_config_generated_children_do_not_follow_symlinks(tmp_path, property_name, relative):
    config = Config(vault_path=str(tmp_path / "vault"))
    target = tmp_path / "outside"
    target.mkdir()
    link = config.vault / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe generated path"):
        getattr(config, property_name)
    assert snapshot(target) == {}


def test_authorized_vault_alias_resolves_once_and_contained_apis_work(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(vault, target_is_directory=True)
    config = Config(vault_path=str(alias))
    assert config.vault == vault
    assert config.llmwiki_dir == vault / "04-Wiki" / ".llmwiki"
    assert acquire_lock(config.lock_file)
    release_lock(config.lock_file)
    with VaultRegistry(config.llmwiki_dir) as registry:
        assert registry.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert registry.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("api", ["save", "latest", "history"])
@pytest.mark.parametrize("component", ["wiki", "metadata", "metrics.json"])
@pytest.mark.parametrize("dangling", [False, True])
def test_metrics_apis_reject_symlinks(tmp_path, api, component, dangling):
    from obsidian_llm_wiki.core.metrics import MetricsCollector, load_all_metrics, load_metrics

    vault = tmp_path / "vault"
    metadata = vault / "04-Wiki" / ".llmwiki"
    metadata.mkdir(parents=True)
    collector = MetricsCollector(vault)
    # Replacement after construction must also be rejected at save time.
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / ("missing" if dangling else "target")
    if component == "metrics.json":
        if not dangling:
            target.write_bytes(b'{"owner":"human"}\n')
        link = metadata / component
    else:
        if not dangling:
            target.mkdir()
        link = metadata if component == "metadata" else metadata.parent
        if component == "wiki":
            metadata.rmdir()
        link.rmdir()
    link.symlink_to(target, target_is_directory=component != "metrics.json")
    before = snapshot(outside)
    with pytest.raises(ValueError, match="Unsafe generated path"):
        if api == "save":
            collector.start_run()
            collector.finish_run()
            collector.save()
        elif api == "latest":
            load_metrics(vault)
        else:
            load_all_metrics(vault)
    assert snapshot(outside) == before


@pytest.mark.parametrize("api", ["save", "load", "all", "delete", "overlay_load", "overlay_save"])
@pytest.mark.parametrize("component", ["root", "cache", "leaf"])
def test_cache_apis_reject_symlinks(tmp_path, api, component):
    from obsidian_llm_wiki.core import cache
    from obsidian_llm_wiki.core.models import SourceSynthesis

    root = tmp_path / "metadata"
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "sentinel.json"
    target.write_bytes(b'{"source_file":"note.md","concepts":[]}\n')
    if component == "leaf":
        name = "_resynthesis_overlay.json" if api.startswith("overlay") else "note.md.json"
        (cache_dir / name).symlink_to(target)
    else:
        link = root if component == "root" else cache_dir
        if component == "root":
            cache_dir.rmdir()
        link.rmdir()
        link.symlink_to(outside, target_is_directory=True)
    before = snapshot(outside)
    with pytest.raises(ValueError, match="Unsafe generated path"):
        if api == "save":
            cache.save_synthesis(SourceSynthesis(source_title="Note", source_summary="Evidence"),
                                 root, "note.md")
        elif api == "load":
            cache.load_synthesis(root, "note.md")
        elif api == "all":
            cache.load_all_cached_syntheses(root)
        elif api == "delete":
            cache.delete_cached_synthesis(root, "note.md")
        elif api == "overlay_load":
            cache.load_resynthesis_overlay(root)
        else:
            cache.save_resynthesis_overlay(root, {})
    assert snapshot(outside) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["pipeline", "recompile"])
@pytest.mark.parametrize("component", ["wiki", "metadata", "lock", "lock.guard", "registry.db",
    "registry.db-wal", "registry.db-shm", "registry.db-journal", "metrics.json", "state.json",
    "schema.yaml", "equivalences.json", "operations.json", "cache", "cache/note.md.json"])
async def test_pipeline_preflights_metadata_before_reads_or_writes(
    tmp_path, monkeypatch, component, entrypoint,
):
    from obsidian_llm_wiki.core.pipeline import recompile_single_source, run_pipeline

    vault = tmp_path / "vault"
    config = Config(vault_path=str(vault))
    metadata = vault / "04-Wiki" / ".llmwiki"
    metadata.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"IRREPLACEABLE USER NOTE\n")
    if component in {"wiki", "metadata", "cache"}:
        link = metadata.parent if component == "wiki" else metadata
        if component == "wiki":
            metadata.rmdir()
        if component != "cache":
            link.rmdir()
        else:
            link = metadata / "cache"
        link.symlink_to(outside, target_is_directory=True)
    else:
        link = metadata / component
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(sentinel)
    before = snapshot(outside)
    reads = []
    original_read = Path.read_text

    def observed_read(path, *args, **kwargs):
        if path.resolve().is_relative_to(outside):
            reads.append(str(path))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observed_read)
    try:
        with pytest.raises(ValueError, match="Unsafe generated path"):
            if entrypoint == "pipeline":
                await run_pipeline(vault, {}, config)
            else:
                await recompile_single_source(vault, "note.md", config)
        assert not reads, f"metadata was read before validation: {reads}"
        assert snapshot(outside) == before
        assert not lock_module._held_locks
        assert not (metadata / "registry.db").exists() or component == "registry.db"
    finally:
        for held in list(lock_module._held_locks.values()):
            os.close(held)
        lock_module._held_locks.clear()


@pytest.mark.parametrize(
    "component", ["metadata", "registry.db", "registry.db-wal", "metrics.json"],
)
def test_public_cli_build_rejects_metadata_symlinks(tmp_path, monkeypatch, component):
    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app

    vault = tmp_path / "vault"
    metadata = vault / "04-Wiki" / ".llmwiki"
    metadata.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"IRREPLACEABLE USER NOTE\n")
    if component == "metadata":
        metadata.rmdir()
        metadata.symlink_to(outside, target_is_directory=True)
    else:
        (metadata / component).symlink_to(sentinel)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")
    before = snapshot(outside)
    result = CliRunner().invoke(app, ["build", str(vault)])
    assert result.exit_code != 0
    assert snapshot(outside) == before
    assert not lock_module._held_locks


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_pipeline_cleanup_releases_replaced_lock_even_when_metrics_fails(
    tmp_path, monkeypatch, cancelled,
):
    import asyncio

    from obsidian_llm_wiki.core.metrics import MetricsCollector
    from obsidian_llm_wiki.core.pipeline import run_pipeline

    vault = tmp_path / "vault"
    config = Config(vault_path=str(vault))
    lock = config.lock_file
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "note"
    sentinel.write_text(str(os.getpid()))
    before = snapshot(outside)
    captured = []
    exception = asyncio.CancelledError if cancelled else RuntimeError

    def failing_save(self):
        captured.append(lock_module._held_locks[lock.absolute()])
        lock.unlink()
        lock.symlink_to(sentinel)
        raise exception("metrics interrupted after path replacement")

    monkeypatch.setattr(MetricsCollector, "save", failing_save)
    with pytest.raises(exception):
        await run_pipeline(vault, {}, config)
    assert captured
    for fd in captured:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert not lock_module._held_locks
    assert snapshot(outside) == before


def test_metrics_explicit_path_cannot_override_authorized_root(tmp_path):
    from obsidian_llm_wiki.core.metrics import MetricsCollector

    outside = tmp_path / "outside"
    outside.mkdir()
    collector = MetricsCollector(tmp_path / "vault")
    collector.metrics_file = outside / "created.json"
    with pytest.raises(ValueError, match="Unsafe generated path"):
        collector.save()
    assert snapshot(outside) == {}


@pytest.mark.parametrize("api", ["lock", "registry", "metrics", "cache"])
def test_metadata_apis_reject_even_in_root_symlink_aliases(tmp_path, api):
    from obsidian_llm_wiki.core.cache import load_synthesis
    from obsidian_llm_wiki.core.metrics import load_metrics

    config = Config(vault_path=str(tmp_path))
    root = config.llmwiki_dir
    root.mkdir(parents=True)
    sentinel = root / "human.json"
    sentinel.write_bytes(b'{"owner":"human"}\n')
    paths = {"lock": root / "lock", "registry": root / "registry.db",
             "metrics": root / "metrics.json", "cache": root / "cache" / "note.md.json"}
    path = paths[api]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(sentinel)
    before = snapshot(root)
    with pytest.raises(ValueError, match="Unsafe generated path"):
        if api == "lock":
            acquire_lock(path)
        elif api == "registry":
            with VaultRegistry(root):
                pass
        elif api == "metrics":
            load_metrics(tmp_path)
        else:
            load_synthesis(root, "note.md")
    assert snapshot(root) == before


@pytest.mark.parametrize("component", ["operations.json", "state.json", "cache/note.md.json"])
def test_registry_preflights_legacy_inputs_before_any_read(tmp_path, monkeypatch, component):
    root = tmp_path / "metadata"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "private.json"
    sentinel.write_bytes(b'{"owner":"human"}\n')
    link = root / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(sentinel)
    reads = []
    read_text = Path.read_text

    def observed_read(path, *args, **kwargs):
        reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", observed_read)
    with pytest.raises(ValueError, match="Unsafe generated path"), VaultRegistry(root):
        pass
    assert not reads
    assert not (root / "registry.db").exists()
    assert snapshot(outside) == {"private.json": b'{"owner":"human"}\n'}
