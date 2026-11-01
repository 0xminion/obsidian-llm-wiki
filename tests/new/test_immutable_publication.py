"""Approved immutable publication contract; scripted providers, real IO/readers."""

import json
import os
from unittest.mock import AsyncMock, Mock

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from tests.new.grounding_doubles import legacy_fixture_llm
from tests.new.test_incremental_cache import FAKE_SYNTH_A


def human_state(path):
    st = path.stat()
    return (
        path.read_bytes(),
        st.st_ino,
        st.st_mode,
        st.st_mtime_ns,
        st.st_ctime_ns,
        st.st_uid,
        st.st_gid,
        {n: os.getxattr(path, n) for n in os.listxattr(path)},
    )


def fixture(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    source = SourceDoc("Alpha", "Experimental observations justify the conclusion. " * 8)
    paths = [
        config.sources_dir / "alpha.md",
        config.concepts_dir / "concept-a1.md",
        config.wiki_dir / "index.md",
        config.llmwiki_dir / "schema.yaml",
        config.annotations_dir / "human.md",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(source.content if p.name == "alpha.md" else "# Human retained\n")
    paths[-2].write_text("domain: science\n")
    llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    return config, {"alpha.md": source}, paths, llm


async def test_build_commits_exclusive_generation_query_and_repeat_noop(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    before = {p: human_state(p) for p in paths}
    result = await run_pipeline(tmp_path, sources, config)
    assert not result.errors
    assert before == {p: human_state(p) for p in paths}, "publisher mutated human paths"
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    assert pointer.is_file(), "build must publish through one committed manifest"
    manifest = json.loads(pointer.read_text())
    generation = config.wiki_dir / "OLW-Publications/generations" / manifest["generation"]
    assert (generation / "concepts/concept-a1.md").is_file()
    assert not (config.llmwiki_dir / "registry.db").exists()
    graph = build_graph_from_vault(config.wiki_dir)
    assert any("A1 detail one" in p.body for p in graph.pages.values())
    assert all("OLW-Publications/generations/" in name for name in graph.pages)
    assert not any("# Human retained" in p.body for p in graph.pages.values())
    import obsidian_llm_wiki.cli.query as query

    answer = Mock(return_value="A1 detail one.")
    monkeypatch.setattr(query, "call_llm", answer)
    query_result = CliRunner().invoke(
        app, ["query", str(tmp_path), "--ask", "Concept A1", "--json"]
    )
    assert query_result.exit_code == 0, query_result.output
    assert answer.call_count == 1
    committed = {p: human_state(p) for p in generation.rglob("*") if p.is_file()}
    pointer_before = human_state(pointer)
    calls = llm.call_count
    second = await run_pipeline(tmp_path, sources, config)
    assert not second.errors and second.compiled == 0 and second.skipped == 1
    assert llm.call_count == calls
    assert human_state(pointer) == pointer_before
    assert committed == {p: human_state(p) for p in committed}
    assert before == {p: human_state(p) for p in paths}


def test_cli_build_uses_readonly_inventory(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    before = {p: human_state(p) for p in paths}
    result = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not (config.llmwiki_dir / "registry.db").exists(), "inventory mutated legacy DB"
    assert before == {p: human_state(p) for p in paths}
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    first = human_state(pointer)
    calls = llm.call_count
    result = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert human_state(pointer) == first and llm.call_count == calls


async def test_annotations_are_stable_exclusive_and_legacy_pages_retained(tmp_path, monkeypatch):
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config, annotation_templates=True)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    first = json.loads(pointer.read_text())
    generation = pointer.parent / "generations" / first["generation"]
    page = generation / "concepts/concept-a1.md"
    cid = parse_frontmatter(page.read_text())[0]["canonical_id"]
    annotation = config.annotations_dir / "entities" / (cid + ".md")
    assert annotation.is_file()
    assert f"[[03-Raw-Annotations/entities/{cid}|Annotations]]" in page.read_text()
    assert "[[04-Wiki/concepts/concept-a1|Legacy page (retained in place)]]" in page.read_text()
    annotation.write_text("# Human annotation\n中文事实 / human context\n")
    os.chmod(annotation, 0o640)
    os.setxattr(annotation, "user.olw-test", b"\x00\xff")
    protected = {p: human_state(p) for p in [*paths, annotation]}
    old_generation = {p: human_state(p) for p in generation.rglob("*") if p.is_file()}
    calls = llm.call_count
    result = await run_pipeline(
        tmp_path, sources, config, refresh_render=True, annotation_templates=True
    )
    assert not result.errors and llm.call_count == calls
    second = json.loads(pointer.read_text())
    assert second["generation"] != first["generation"]
    new_page = pointer.parent / "generations" / second["generation"] / "concepts/concept-a1.md"
    assert parse_frontmatter(new_page.read_text())[0]["canonical_id"] == cid
    assert f"[[03-Raw-Annotations/entities/{cid}|Annotations]]" in new_page.read_text()
    assert protected == {p: human_state(p) for p in protected}
    assert old_generation == {p: human_state(p) for p in old_generation}


async def test_reader_rejects_first_uncommitted_generation(tmp_path, monkeypatch):
    import pytest

    import obsidian_llm_wiki.core.immutable_publication as publication

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    observed = []

    def pause(event, **context):
        if event == "after_install":
            with pytest.raises(RuntimeError, match="pending|committed"):
                build_graph_from_vault(config.wiki_dir)
            observed.append(event)

    monkeypatch.setattr(publication, "_checkpoint", pause)
    await run_pipeline(tmp_path, sources, config)
    assert observed == ["after_install"]


async def test_final_generation_gate_rechecks_effective_cache_coherence(tmp_path, monkeypatch):
    import pytest

    import obsidian_llm_wiki.core.immutable_publication as publication

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    original = publication._qualify_links

    def corrupt(wiki, *args):
        original(wiki, *args)
        cache = next((wiki / ".llmwiki/cache").glob("*.json"))
        cache.write_text("{}")

    monkeypatch.setattr(publication, "_qualify_links", corrupt)
    with pytest.raises(RuntimeError, match="cache disagreement"):
        await run_pipeline(tmp_path, sources, config)
    assert not (config.wiki_dir / "OLW-Publications/current.json").exists()


async def test_current_entrypoint_and_migration_inventory_are_readonly(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.identity_migration import preview_identity_migration

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config)
    before = {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls = llm.call_count
    result = CliRunner().invoke(app, ["current", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    current = json.loads(result.output)
    assert current["native_ui_verified"] is False
    assert (tmp_path / current["entrypoint"]).is_file()
    assert current["obsidian_uri"].startswith("obsidian://open?path=")
    report = preview_identity_migration(tmp_path)
    workflow = report["publication_workflow"]
    assert workflow["format"] == "immutable-generation-v1"
    assert workflow["current_generation"] == current["generation"]
    assert "04-Wiki/concepts/concept-a1.md" in workflow["legacy_pages_retained"]
    assert workflow["model_calls_made"] == 0 and llm.call_count == calls
    assert (
        workflow["legacy_action"] == "retain in place; link whole pages without attributing edits"
    )
    assert before == {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}


async def test_public_health_validation_and_dryrun_use_committed_snapshot(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    manifest = json.loads(pointer.read_text())
    generation = pointer.parent / "generations" / manifest["generation"]
    before = {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    for command in [
        ["health", str(tmp_path), "--json"],
        ["validate", str(tmp_path)],
        ["build", str(tmp_path), "--dry-run"],
    ]:
        result = CliRunner().invoke(app, command)
        assert manifest["generation"] in result.output, (command, result.output)
    assert before == {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    page = generation / "concepts/concept-a1.md"
    page.write_text("Human mistakenly edited immutable output; do not repair me")
    corrupted = human_state(page)
    for command in [
        ["health", str(tmp_path), "--json"],
        ["validate", str(tmp_path)],
        ["build", str(tmp_path), "--dry-run"],
        ["query", str(tmp_path), "--ask", "A1", "--json"],
    ]:
        result = CliRunner().invoke(app, command)
        assert result.exit_code != 0 and "generation" in result.output, (command, result.output)
    assert human_state(page) == corrupted


async def test_source_snapshot_and_rendered_source_are_separate(tmp_path, monkeypatch):
    from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    raw = human_state(paths[0])
    result = await run_pipeline(tmp_path, sources, config, annotation_templates=True)
    assert not result.errors
    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    directory = config.wiki_dir / "OLW-Publications/generations" / manifest["generation"]
    assert parse_frontmatter((directory / "sources/alpha.md").read_text())[0]["type"] == "Source"
    assert (directory / ".llmwiki/source-inputs/alpha.md").read_bytes() == raw[0]
    assert human_state(paths[0]) == raw
    validation = CliRunner().invoke(app, ["validate", str(tmp_path)])
    assert validation.exit_code == 0, validation.output


async def test_query_cannot_overwrite_generation_even_with_force(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config)
    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    page = (
        config.wiki_dir
        / "OLW-Publications/generations"
        / manifest["generation"]
        / "concepts/concept-a1.md"
    )
    before = human_state(page)
    answer = Mock(return_value="A1 detail one.")
    monkeypatch.setattr(query, "call_llm", answer)
    result = CliRunner().invoke(
        app,
        ["query", str(tmp_path), "--ask", "A1", "--json", "--save-answer", str(page), "--force"],
    )
    assert result.exit_code == 1 and "immutable" in result.output.lower(), result.output
    assert answer.call_count == 0 and human_state(page) == before
    result = CliRunner().invoke(app, ["query", str(tmp_path), "--ask", "A1", "--json"])
    assert result.exit_code == 0, result.output
    assert not (config.llmwiki_dir / "query-sessions.json").exists()
    assert (config.wiki_dir / "OLW-Publications/query-state/query-sessions.json").exists()


async def test_annotation_template_install_is_atomic_no_overwrite(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config)
    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    cid = next(iter(manifest["entities"]))
    target = config.annotations_dir / "entities" / (cid + ".md")
    target.parent.mkdir(parents=True)
    opened = Path.open
    import obsidian_llm_wiki.core.immutable_publication as publication

    linked = publication._install_exclusive
    states = []

    def edit():
        subprocess.run(
            [
                sys.executable,
                "-c",
                'import pathlib,sys; '
                'pathlib.Path(sys.argv[1]).write_bytes(b"Human editor won\\r\\n")',
                str(target),
            ],
            check=True,
        )
        states.append(human_state(target))

    def opening(self, mode="r", *args, **kwargs):
        fd = opened(self, mode, *args, **kwargs)
        if self == target and mode == "xb":
            edit()  # old template path exposes a partially created file
        return fd

    def linking(src, dst, *args, **kwargs):
        if Path(dst) == target:
            edit()  # atomic install loses cleanly to a just-created human file
        return linked(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(publication, "_install_exclusive", linking)
    await run_pipeline(tmp_path, sources, config, annotation_templates=True)
    assert states and human_state(target) == states[0]
