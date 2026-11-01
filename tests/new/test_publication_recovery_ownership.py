"""Recovery owns app writes, never every path present in a vault snapshot."""
import json
import os
import stat
from types import SimpleNamespace

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.publication_transaction import (
    PublicationTransaction,
    recover_publication,
)
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.render.frontmatter import atomic_write


def test_outer_journal_retains_human_metadata_after_inner_render_failure(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.models import SynthesisBundle
    from obsidian_llm_wiki.render import obsidian

    config = config_at(tmp_path)
    path = config.wiki_dir / "graph.json"
    path.write_text("before")

    def fail_render(*_args, **_kwargs):
        obsidian._write_generated_artifact(path, "app", config.wiki_dir)
        path.chmod(0o640)  # human metadata change after the owned write
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(obsidian, "_render_vault", fail_render)
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        with pytest.raises(RuntimeError, match="injected render failure"):
            obsidian.render_vault(config.wiki_dir, SynthesisBundle(), {})
        with pytest.raises(RuntimeError, match="recovery conflict"):
            tx.rollback()
        assert path.read_bytes() == b"app" and stat.S_IMODE(path.stat().st_mode) == 0o640


def test_committed_journal_is_cleanup_only_even_if_cleanup_crashes(tmp_path, monkeypatch):
    import shutil

    config = config_at(tmp_path)
    path = config.wiki_dir / "generated.md"
    path.write_text("before")
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(path, "committed")
        real_remove = shutil.rmtree

        def fail_cleanup(*_a, **_k):
            raise OSError("cleanup interrupted")

        monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
        with pytest.raises(OSError, match="cleanup interrupted"):
            tx.commit()
        tx.rollback()  # inactive: no revocation after durable commit
        monkeypatch.setattr(shutil, "rmtree", real_remove)
        assert not recover_publication(registry, config)
        assert path.read_text() == "committed"


def test_legacy_snapshot_without_ownership_blocks_without_touching_files(tmp_path):
    config = config_at(tmp_path)
    path = config.wiki_dir / "human.md"
    path.write_bytes(b"human\r\n")
    stage = config.llmwiki_dir / "publication-stage"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps({
        "status": "pending", "files": {"human.md": "old-hash"},
    }))
    with (
        VaultRegistry(config.llmwiki_dir) as registry,
        pytest.raises(RuntimeError, match="lacks write ownership"),
    ):
        recover_publication(registry, config)
    assert path.read_bytes() == b"human\r\n" and stage.is_dir()


def config_at(tmp_path):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.wiki_dir.mkdir(parents=True)
    return config


def plan():
    return SimpleNamespace(version="identity-publication-plan-v1", plan_id="ownership-test")


@pytest.mark.parametrize("lazy", [False, True])
def test_render_snapshot_alone_does_not_own_a_human_edit(tmp_path, lazy):
    from obsidian_llm_wiki.render.obsidian import _RenderTransaction

    path = tmp_path / "human.md"
    path.write_bytes(b"before\r\n")
    tx = _RenderTransaction(tmp_path, lazy=lazy)
    tx.before_change(path)  # preparation is not evidence that a write happened
    path.write_bytes(b"human during failed preparation\r\n")
    tx.rollback()
    assert path.read_bytes() == b"human during failed preparation\r\n"


@pytest.mark.parametrize("action", ["edit", "new", "delete", "policy", "binary"])
def test_recovery_leaves_unowned_human_paths_in_place(tmp_path, action):
    config = config_at(tmp_path)
    path = config.wiki_dir / (".llmwiki/schema.yaml" if action == "policy" else "notes/human.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old\r\n")
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        if action == "new":
            path = path.with_name("new-idea.md")
        if action == "delete":
            path.unlink()
        else:
            path.write_bytes(b"\x89PNG\xff\x00" if action == "binary" else b"human\r\n")
            path.chmod(0o640)
        expected = path.read_bytes() if path.exists() else None
        expected_stat = path.stat() if path.exists() else None
        tx.rollback()
        assert (path.read_bytes() if path.exists() else None) == expected
        if expected_stat:
            assert path.stat().st_mtime_ns == expected_stat.st_mtime_ns
            assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize("body", [b"line1\r\nline2\r\n", b"\x89PNG\xff\x00"])
def test_owned_rollback_is_byte_and_metadata_exact(tmp_path, body):
    config = config_at(tmp_path)
    page = config.wiki_dir / "sources/index.md"
    page.parent.mkdir(parents=True)
    page.write_bytes(body)
    page.chmod(0o640)
    os.utime(page, ns=(1600000000000000000, 1600000000000000000))
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(page, "app replacement\n")
        tx.rollback()
    assert page.read_bytes() == body
    assert stat.S_IMODE(page.stat().st_mode) == 0o640
    assert page.stat().st_mtime_ns == 1600000000000000000


def test_owned_postimage_conflict_is_preserved_and_blocks_recovery(tmp_path):
    config = config_at(tmp_path)
    path = config.wiki_dir / "generated.md"
    path.write_bytes(b"before\n")
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        atomic_write(path, "app\n")
        path.write_bytes(b"human after app\r\n")
        with pytest.raises(RuntimeError, match="recovery conflict"):
            tx.rollback()
        assert path.read_bytes() == b"human after app\r\n"
        manifest = json.loads((tx.stage / "manifest.json").read_text())
        assert manifest["status"] == "conflicted"
        conflict = manifest["conflicts"][0]
        assert conflict["path"] == "generated.md"
        assert conflict["reason"] and conflict["before"] and conflict["observed"]
        with pytest.raises(RuntimeError, match="recovery conflict"):
            recover_publication(registry, config)
        assert path.read_bytes() == b"human after app\r\n"


@pytest.mark.asyncio
async def test_committed_marker_is_not_visible_until_durable_commit_decision(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.identity_plan import publication_current
    from obsidian_llm_wiki.core.models import ConceptNote, SourceDoc, SourceSynthesis
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from tests.new.grounding_doubles import reviewed

    config = config_at(tmp_path)

    async def synthesize(cfg, filename, source, *_a, **_k):
        return reviewed(SourceSynthesis(
            source_title=source.title, source_summary="Queue", source_file=filename,
            concepts=[ConceptNote("Queue", "queue", "Bounds memory.")],
        ), source, filename, cfg)

    commit = PublicationTransaction.commit
    observations = []

    def before_commit(tx):
        from obsidian_llm_wiki.core.immutable_publication import current_publication

        policy = tx.registry.get_pipeline_checkpoint()
        workspace = tx.config
        marker = json.loads((workspace.llmwiki_dir / "identity-publication.json").read_text())
        assert marker["status"] == "committed"  # exercise the historical crash window
        assert current_publication(config.wiki_dir, allow_uncommitted=True) is None
        observations.append(publication_current(tx.registry, workspace, policy))
        assert observations == [False], "a pending journal must not advertise committed output"
        commit(tx)
        assert publication_current(tx.registry, workspace, policy)
        assert not recover_publication(tx.registry, workspace)  # committed is never revoked
        assert current_publication(config.wiki_dir, allow_uncommitted=True) is None

    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline._synthesize_with_retry", synthesize)
    monkeypatch.setattr(PublicationTransaction, "commit", before_commit)
    result = await run_pipeline(
        tmp_path, {"a.md": SourceDoc("Queue", "A bounded queue limits memory. " * 10)}, config,
    )
    assert result.compiled == 1 and not result.errors and observations == [False]


@pytest.mark.parametrize("command", [["health", "--json"], ["validate"], ["build", "--dry-run"]])
def test_cli_readers_report_pending_without_registry_writes(tmp_path, command):
    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app

    config = config_at(tmp_path)
    with VaultRegistry(config.llmwiki_dir) as registry:
        tx = PublicationTransaction(registry, config, plan())
        before = list(registry.connection.iterdump())
        try:
            result = CliRunner().invoke(app, [command[0], str(tmp_path), *command[1:]])
            assert result.exit_code != 0, result.output
            assert "publication" in (result.output + str(result.exception)).lower()
            assert list(registry.connection.iterdump()) == before
        finally:
            tx.rollback()
