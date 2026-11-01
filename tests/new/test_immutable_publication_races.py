"""Separate-process editor/crash acceptance for immutable publication.

The L1 seam is the ORIGINAL durable record_replace -> os.replace boundary,
now inside private staging. The editor ignores all publisher locks. No xfails.
"""

import json
import multiprocessing
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from tests.new.test_immutable_publication import fixture, human_state


def _child_cli(vault, connection, point, hard_exit):
    import obsidian_llm_wiki.core.immutable_publication as publication
    import obsidian_llm_wiki.core.publication_transaction as transaction

    original_record = transaction.record_replace
    original_replace = os.replace
    fired = False
    target = None

    def pause():
        connection.send("paused")
        assert connection.recv() == "continue"

    def record(path, temporary):
        nonlocal fired, target
        original_record(path, temporary)
        if point == "L1" and not fired and Path(path).parts[-2:] == ("concepts", "concept-a1.md"):
            assert ".staging" in Path(path).parts
            fired = True
            target = Path(path)
            pause()

    def renamed(source, destination, *args, **kwargs):
        value = original_replace(source, destination, *args, **kwargs)
        if point == "L1" and target == Path(destination) and hard_exit:
            os._exit(86)
        return value

    def checkpoint(event, **context):
        nonlocal fired
        if event == point and not fired:
            fired = True
            pause()
            if hard_exit:
                os._exit(86)

    transaction.record_replace = record
    os.replace = renamed
    publication._checkpoint = checkpoint
    result = CliRunner().invoke(app, ["build", str(vault), "--refresh-render"])
    connection.send({"exit": result.exit_code, "output": result.output, "fired": fired})
    connection.close()
    os._exit(result.exit_code)


def start_child(vault, point="L1", hard_exit=True):
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=_child_cli, args=(vault, child, point, hard_exit))
    worker.start()
    child.close()
    assert parent.poll(30), "compiler did not reach fault seam"
    message = parent.recv()
    assert message == "paused", message
    return worker, parent


def finish_child(worker, pipe, expected=86):
    pipe.send("continue")
    worker.join(30)
    if worker.is_alive():
        worker.kill()
        worker.join()
        pytest.fail("child hung")
    assert worker.exitcode == expected
    pipe.close()


@pytest.mark.parametrize("human_kind", ["legacy", "annotation", "source", "policy"])
def test_original_l1_editor_bytes_and_extended_metadata_remain_in_place(
    tmp_path, monkeypatch, human_kind
):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    initial = CliRunner().invoke(app, ["build", str(tmp_path), "--annotation-templates"])
    assert initial.exit_code == 0, initial.output
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    committed = human_state(pointer)
    target = {
        "legacy": paths[1],
        "annotation": next((config.annotations_dir / "entities").glob("*.md")),
        "source": paths[0],
        "policy": paths[3],
    }[human_kind]
    worker, pipe = start_child(tmp_path)
    # Same byte payload/chmod as original L1. Hold an FD across compiler rename,
    # then continue writing through that FD after the compiler's hard exit.
    with target.open("r+b") as editor:
        editor.seek(0)
        editor.write(b"Human edited while compiler active\r\n")
        editor.truncate()
        editor.flush()
        os.fchmod(editor.fileno(), 0o640)
        os.setxattr(editor.fileno(), "user.editor", b"\x00\xffcontinued")
        first = human_state(target)
        finish_child(worker, pipe)
        assert human_state(target) == first
        editor.write(b"Still editing after compiler exit\r\n")
        editor.flush()
    edited = human_state(target)
    assert human_state(pointer) == committed
    assert build_graph_from_vault(config.wiki_dir).pages
    # Recovery has no undo operation. Rebuild may reject the now-malformed
    # source/policy, but may never repair or replace that human edit.
    for _ in range(2):
        CliRunner().invoke(app, ["build", str(tmp_path)])
        assert human_state(target) == edited


@pytest.mark.parametrize(
    "point", ["before_install", "after_install", "before_commit", "after_commit"]
)
def test_hardexit_never_revokes_a_reader_accepted_generation(tmp_path, monkeypatch, point):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    initial = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert initial.exit_code == 0, initial.output
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    first = json.loads(pointer.read_text())["generation"]
    protected = {p: human_state(p) for p in paths}
    worker, pipe = start_child(tmp_path, point)
    accepted = json.loads(pointer.read_text())["generation"]
    assert (accepted != first) == (point == "after_commit")
    pages = dict(build_graph_from_vault(config.wiki_dir).pages)
    finish_child(worker, pipe)
    assert dict(build_graph_from_vault(config.wiki_dir).pages) == pages
    for _ in range(2):
        result = CliRunner().invoke(app, ["build", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert json.loads(pointer.read_text())["generation"] == accepted
    assert protected == {p: human_state(p) for p in protected}


async def test_new_generation_name_collision_never_overwrites(tmp_path, monkeypatch):
    import obsidian_llm_wiki.core.immutable_publication as publication

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    await run_pipeline(tmp_path, sources, config)
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    accepted = human_state(pointer)
    collision = []

    def collide(event, **context):
        if event == "before_install":
            target = context["target"]
            target.mkdir()
            sentinel = target / "human.md"
            sentinel.write_bytes(b"Human owns this new path\r\n")
            collision.append((sentinel, human_state(sentinel)))

    monkeypatch.setattr(publication, "_checkpoint", collide)
    with pytest.raises(FileExistsError):
        await run_pipeline(tmp_path, sources, config, refresh_render=True)
    assert collision and all(human_state(p) == state for p, state in collision)
    assert human_state(pointer) == accepted


def test_competing_writer_is_refused_without_human_mutation(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    initial = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert initial.exit_code == 0, initial.output
    protected = {p: human_state(p) for p in paths}
    worker, pipe = start_child(tmp_path, "workspace_ready")
    result = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 1 and "lock" in result.output, result.output
    finish_child(worker, pipe)
    assert protected == {p: human_state(p) for p in protected}
