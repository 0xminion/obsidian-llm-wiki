"""Direct domain writers must not mutate an immutable publication or legacy files."""

import json
import os
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.metrics import MetricsCollector
from obsidian_llm_wiki.core.orphan import orphan_page
from obsidian_llm_wiki.core.pipeline import run_pipeline
from tests.new.test_immutable_publication import fixture, human_state


async def test_orphan_refuses_legacy_and_installed_generation(tmp_path, monkeypatch):
    config, sources, paths, _ = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    generation, _ = current_publication(config.wiki_dir)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    frozen = {p: human_state(p) for p in generation.rglob("*") if p.is_file()}
    legacy = paths[1]
    old_legacy, old_pointer = human_state(legacy), human_state(pointer)
    for page in (legacy, generation / "concepts/concept-a1.md"):
        with pytest.raises(RuntimeError, match="build|publication"):
            orphan_page(page.parent, page.stem, "deleted")
    assert human_state(legacy) == old_legacy
    assert human_state(pointer) == old_pointer
    assert frozen == {p: human_state(p) for p in frozen}
    assert current_publication(config.wiki_dir)[0] == generation


async def test_metrics_refuses_legacy_and_generation_without_any_writes(tmp_path, monkeypatch):
    config, sources, _, _ = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    generation, _ = current_publication(config.wiki_dir)
    sentinel = config.llmwiki_dir / "metrics.json"
    sentinel.write_bytes(b'{"human sentinel": true}')
    sentinel.chmod(0o640)
    original = human_state(sentinel)
    frozen = {p: human_state(p) for p in generation.rglob("*") if p.is_file()}
    for vault in (tmp_path, generation):
        collector = MetricsCollector(vault)
        collector.start_run()
        collector.finish_run()
        with pytest.raises(RuntimeError, match="build|publication"):
            collector.save()
    assert human_state(sentinel) == original
    assert not (config.llmwiki_dir / "registry.db").exists()
    assert frozen == {p: human_state(p) for p in frozen}
    assert current_publication(config.wiki_dir)[0] == generation


def test_legacy_domain_writers_remain_usable(tmp_path):
    concepts = tmp_path / "OLW-Publications-not-a-namespace" / "04-Wiki" / "concepts"
    concepts.mkdir(parents=True)
    page = concepts / "topic.md"
    page.write_text("---\ntitle: Topic\n---\n# Topic\n")
    assert orphan_page(concepts, "topic", "deleted")
    assert "orphaned: true" in page.read_text()
    collector = MetricsCollector(concepts.parent.parent)
    collector.start_run()
    collector.finish_run()
    collector.save()
    assert collector.metrics_file.is_file()


def test_collision_and_symlink_refuse_before_any_domain_writes(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    concepts = vault / "04-Wiki" / "concepts"
    concepts.mkdir(parents=True)
    page = concepts / "topic.md"
    page.write_text("# Human\n")
    namespace = concepts.parent / "OLW-Publications"
    namespace.mkdir()
    (namespace / "owner.json").write_text('{"owner":"human"}')
    original = human_state(page)
    with pytest.raises(RuntimeError, match="collision"):
        orphan_page(concepts, "topic", "deleted")
    collector = MetricsCollector(vault)
    collector.start_run()
    collector.finish_run()
    with pytest.raises(RuntimeError, match="collision"):
        collector.save()
    assert human_state(page) == original
    assert not (concepts.parent / ".llmwiki").exists()
    alias = tmp_path / "alias"
    alias.symlink_to(vault, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MetricsCollector(alias).save()
    monkeypatch.setenv("DOMAIN_ALIAS", str(alias))
    with pytest.raises(ValueError, match="symlink"):
        MetricsCollector("$DOMAIN_ALIAS").save()
    assert human_state(page) == original


@pytest.mark.parametrize("spelling", ["literal", "environment", "home"])
@pytest.mark.parametrize("dangling", [False, True])
def test_metrics_validates_original_path_before_config(tmp_path, monkeypatch, spelling, dangling):
    import obsidian_llm_wiki.core.metrics as metrics

    target = tmp_path / "target"
    if not dangling:
        target.mkdir()
        (target / "human.md").write_bytes(b"preserve me\r\n")
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    before = {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    if spelling == "environment":
        monkeypatch.setenv("DOMAIN_ALIAS", str(alias))
        argument = "$DOMAIN_ALIAS"
    elif spelling == "home":
        monkeypatch.setenv("HOME", str(tmp_path))
        argument = "~/alias"
    else:
        argument = alias

    def unexpected_config(**kwargs):
        pytest.fail("Config was reached before lexical symlink containment")

    monkeypatch.setattr(metrics, "Config", unexpected_config)
    with pytest.raises(ValueError, match="symlink"):
        metrics.MetricsCollector(argument)
    assert before == {p: human_state(p) for p in before}
    assert not (target / "04-Wiki").exists()


@pytest.mark.parametrize("writer", ["orphan", "metrics"])
@pytest.mark.parametrize("dangling", [False, True])
def test_namespace_symlink_is_rejected_before_following_stat(
    tmp_path, monkeypatch, writer, dangling,
):
    vault = tmp_path / "vault"
    concepts = vault / "04-Wiki" / "concepts"
    concepts.mkdir(parents=True)
    page = concepts / "topic.md"
    page.write_bytes(b"# Human\r\n")
    target = tmp_path / "outside"
    if not dangling:
        target.mkdir()
        (target / "owner.json").write_text('{"owner":"human"}')
    namespace = concepts.parent / "OLW-Publications"
    namespace.symlink_to(target, target_is_directory=True)
    before = {p: human_state(p) for p in (page,)}
    collector = MetricsCollector(vault)
    stat = Path.stat

    def no_follow(self, *args, **kwargs):
        if self == namespace and kwargs.get("follow_symlinks", True):
            pytest.fail("namespace metadata followed before containment")
        return stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", no_follow)
    with pytest.raises(ValueError, match="symlink"):
        if writer == "orphan":
            orphan_page(concepts, "topic", "deleted")
        else:
            collector.save()
    assert before == {p: human_state(p) for p in before}
    assert not (concepts.parent / ".llmwiki").exists()


def _tree_state(root):
    """Include directory entries; file state includes bytes/inode/mode/xattrs."""
    return {
        p.relative_to(root).as_posix(): (
            ("directory", p.stat().st_ino, p.stat().st_mode)
            if p.is_dir() else ("file", human_state(p))
        )
        for p in root.rglob("*")
    }


@pytest.mark.parametrize("status", ["pending", "installed", "committed"])
async def test_domain_refusal_preserves_complete_tree(tmp_path, monkeypatch, status):
    from obsidian_llm_wiki.core.models import SourceState, WikiState
    from obsidian_llm_wiki.core.orphan import mark_orphaned_concepts

    config, sources, paths, _ = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    generation, _ = current_publication(config.wiki_dir)
    root = generation.parent.parent
    if status != "committed":
        (root / "current.json").unlink()
    if status == "pending":
        pending = root / ".staging/abandoned/04-Wiki"
        pending.parent.mkdir()
        generation.rename(pending)
        # Keep the installed spelling as a forbidden absent destination too.
    sentinel = config.llmwiki_dir / "metrics.json"
    sentinel.write_bytes(b'{"human": "do not overwrite"}\r\n')
    for path in (paths[1], sentinel):
        path.chmod(0o640)
        os.setxattr(path, "user.domain-guard", b"\x00\xff")
    before = _tree_state(tmp_path)
    state = WikiState(sources={"gone.md": SourceState(hash="h", concepts=["concept-a1"])})
    for concepts in (config.concepts_dir, generation / "concepts"):
        with pytest.raises(RuntimeError, match="publication"):
            orphan_page(concepts, "concept-a1", "deleted")
        with pytest.raises(RuntimeError, match="publication"):
            mark_orphaned_concepts(concepts, "gone.md", state)
    for vault in (tmp_path, generation):
        collector = MetricsCollector(vault)
        collector.start_run()
        collector.finish_run()
        with pytest.raises(RuntimeError, match="publication"):
            collector.save()
    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("owner", ["missing", "malformed", "unowned", "symlink"])
@pytest.mark.parametrize("writer", ["orphan", "metrics"])
def test_staging_shape_does_not_bypass_namespace_owner(tmp_path, owner, writer):
    root = tmp_path / "04-Wiki/OLW-Publications"
    work = root / ".staging/private"
    concepts = work / "04-Wiki/concepts"
    concepts.mkdir(parents=True)
    page = concepts / "topic.md"
    page.write_bytes(b"# Human collision\r\n")
    marker = root / "owner.json"
    if owner == "malformed":
        marker.write_bytes(b"{broken")
    elif owner == "unowned":
        marker.write_text('{"owner":"human"}')
    elif owner == "symlink":
        from obsidian_llm_wiki.core.immutable_publication import OWNER

        target = tmp_path / "human-owner.json"
        target.write_text(json.dumps(OWNER))
        marker.symlink_to(target)
    before = human_state(page)
    expected = ValueError if owner == "symlink" else RuntimeError
    with pytest.raises(expected, match="symlink|collision"):
        if writer == "orphan":
            orphan_page(concepts, "topic", "deleted")
        else:
            MetricsCollector(work).save()
    assert human_state(page) == before
    assert not (work / "04-Wiki/.llmwiki").exists()


def test_legacy_metrics_collector_rechecks_after_first_publication(tmp_path):
    from obsidian_llm_wiki.core.immutable_publication import OWNER

    collector = MetricsCollector(tmp_path)
    collector.start_run()
    collector.save()
    root = tmp_path / "04-Wiki/OLW-Publications"
    root.mkdir()
    (root / "owner.json").write_text(json.dumps(OWNER))
    before = _tree_state(tmp_path)
    with pytest.raises(RuntimeError, match="publication"):
        collector.save()
    assert _tree_state(tmp_path) == before


def test_metrics_rechecks_input_ancestor_after_construction(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    collector = MetricsCollector(vault)
    retained = tmp_path / "retained"
    vault.rename(retained)
    vault.symlink_to(retained, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        collector.save()
    assert not list(retained.iterdir())


async def test_build_delete_and_failure_call_real_writers_only_in_staging(tmp_path, monkeypatch):
    import obsidian_llm_wiki.core.pipeline as pipeline
    from tests.new.test_immutable_publication_lifecycle import setup, source

    config, provider = setup(tmp_path, monkeypatch)
    seen_metrics, seen_orphans = [], []
    save, orphan = MetricsCollector.save, pipeline.orphan_page
    staging = config.wiki_dir / "OLW-Publications/.staging"

    def check_staging(path):
        parts = path.relative_to(staging).parts
        assert len(parts) >= 3 and parts[1] == "04-Wiki"

    def observed_save(self):
        check_staging(self.metrics_file)
        save(self)
        assert self.metrics_file.is_file()
        seen_metrics.append(self.metrics_file)

    def observed_orphan(concepts, slug, reason):
        check_staging(concepts)
        changed = orphan(concepts, slug, reason)
        assert changed
        assert "orphaned: true" in (concepts / f"{slug}.md").read_text()
        seen_orphans.append((concepts, slug))
        return changed

    monkeypatch.setattr(MetricsCollector, "save", observed_save)
    monkeypatch.setattr(pipeline, "orphan_page", observed_orphan)
    documents = source(config, "17")
    assert not (await run_pipeline(tmp_path, documents, config)).errors
    generation, _ = current_publication(config.wiki_dir)
    before = _tree_state(generation)
    (config.sources_dir / "amber.md").unlink()
    result = await run_pipeline(tmp_path, {}, config)
    assert not result.errors and result.deleted == 1
    assert len(seen_orphans) == 1 and len(seen_metrics) == 2
    assert _tree_state(generation) == before
    calls = len(provider.calls)
    assert not (await run_pipeline(tmp_path, {}, config)).errors
    assert len(seen_metrics) == 2 and len(provider.calls) == calls
    latest, _ = current_publication(config.wiki_dir)
    frozen = _tree_state(latest)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    pointer_before = human_state(pointer)

    def render_error(*args, **kwargs):
        raise RuntimeError("scripted domain caller render failure")

    monkeypatch.setattr(pipeline, "render_vault", render_error)
    with pytest.raises(RuntimeError, match="scripted domain caller render failure"):
        await run_pipeline(tmp_path, {}, config, refresh_render=True)
    assert len(seen_metrics) == 3  # real finally-block save, not just the success path
    assert _tree_state(latest) == frozen and human_state(pointer) == pointer_before
