"""Resealed registry formats and exact-copy reader regressions (no live providers)."""

import hashlib
import json
import sqlite3
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.pipeline import run_pipeline
from tests.new.test_immutable_publication import fixture, human_state


def _reseal(config, sql):
    directory, manifest = current_publication(config.wiki_dir)
    db = directory / ".llmwiki/registry.db"
    connection = sqlite3.connect(db)
    try:
        connection.executescript(sql)
        connection.commit()
    finally:
        connection.close()
    manifest["files"][".llmwiki/registry.db"] = hashlib.sha256(db.read_bytes()).hexdigest()
    (config.wiki_dir / "OLW-Publications/current.json").write_text(json.dumps(manifest))
    return directory


@pytest.mark.parametrize("version", ["999", "0", "-1", "7.0", "07", "garbage"])
async def test_resealed_unsupported_registry_rejected_before_provider_or_writes(
    tmp_path, monkeypatch, version,
):
    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    _reseal(config, f"UPDATE meta SET value='{version}' WHERE key='schema_version';")
    before = {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    answer = Mock(side_effect=AssertionError("unsupported registry reached provider"))
    monkeypatch.setattr("obsidian_llm_wiki.cli.query.call_llm", answer)
    for command in (
        ["current", str(tmp_path), "--json"],
        ["query", str(tmp_path), "--ask", "Concept A1", "--json"],
        ["health", str(tmp_path), "--json"],
        ["validate", str(tmp_path)],
        ["build", str(tmp_path), "--dry-run"],
    ):
        result = CliRunner().invoke(app, command)
        assert result.exit_code != 0, (command, result.output)
        assert "schema" in result.output.lower(), (command, result.output, result.exception)
    assert answer.call_count == 0
    assert before == {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("sql", [
    "DROP TABLE render_state;",
    "ALTER TABLE source_file_state DROP COLUMN updated_at;",
    "DELETE FROM meta WHERE key='schema_version';",
])
async def test_required_registry_shape_is_not_replaced_by_resealed_hashes(
    tmp_path, monkeypatch, sql,
):
    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    _reseal(config, sql)
    with pytest.raises(RuntimeError, match="schema"):
        current_publication(config.wiki_dir)


async def test_unknown_schema_preview_inventories_but_never_certifies(tmp_path, monkeypatch):
    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    _reseal(config, "UPDATE meta SET value='999' WHERE key='schema_version';")
    before = {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = CliRunner().invoke(app, ["identity-migration", str(tmp_path), "--dry-run", "--json"])
    assert result.exit_code == 3, result.output
    report = json.loads(result.stdout)
    assert report["publication"]["verified"] is False
    assert "unknown_schema_version" in {b["kind"] for b in report["blockers"]}
    assert all(b["status"] == "unresolved" for b in report["bindings"])
    assert before == {p: human_state(p) for p in tmp_path.rglob("*") if p.is_file()}
