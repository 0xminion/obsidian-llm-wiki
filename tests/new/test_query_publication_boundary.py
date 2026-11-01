"""Public queries must not accept a rollback-eligible generation."""
import json
from unittest.mock import Mock

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app


def test_pending_query_refuses_before_provider_or_output_creation(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query

    wiki = tmp_path / "04-Wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "concepts/relay.md").write_text("# Amber Relay\nAmber Relay buffers 29 messages.")
    stage = wiki / ".llmwiki/publication-stage"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps({"status": "pending"}))
    (stage.parent / "identity-publication.json").write_text(json.dumps({"status": "committed"}))
    before = {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob("*") if p.is_file()}
    llm = Mock(return_value="29 messages [[concepts/relay.md]].")
    monkeypatch.setattr(query, "call_llm", llm)
    result = CliRunner().invoke(app, ["query", str(tmp_path), "--ask", "Amber Relay",
                                     "--json", "--save-answer", "pending/answer.md"])
    assert result.exit_code == 1, result.output
    assert "recovery" in json.loads(result.output)["errors"][0]
    llm.assert_not_called()
    assert not (tmp_path / "05-Queries").exists()
    assert before == {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in tmp_path.rglob("*") if p.is_file()}


def test_graph_excludes_internal_publication_and_backup_pages(tmp_path):
    from obsidian_llm_wiki.query.graph import build_graph_from_vault

    (tmp_path / "live.md").write_text("# Live\nCurrent evidence.")
    for directory in [".llmwiki/publication-stage/files", ".llmwiki/backups", ".obsidian"]:
        hidden = tmp_path / directory / "historical.md"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        hidden.write_text("# Historical\nNot live evidence.")
    graph = build_graph_from_vault(tmp_path)
    assert list(graph.pages) == ["live.md"]


def test_query_rechecks_generation_before_saving_answer(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query
    from tests.new.test_query_currentness import _build

    config, paths, _llm = _build(tmp_path, monkeypatch)
    wiki = config.wiki_dir
    calls = []

    def begin_publication(*_args, **_kwargs):
        calls.append(True)
        paths[0].write_text(paths[0].read_text() + 'Updated source.')
        built = CliRunner().invoke(app, ['build', str(tmp_path)])
        assert built.exit_code == 0, built.output
        return "A1 evidence."

    monkeypatch.setattr(query, "call_llm", begin_publication)
    result = CliRunner().invoke(app, ["query", str(tmp_path), "--ask", "A1", "--json"])
    assert result.exit_code == 1, result.output
    assert calls == [True]
    assert json.loads(result.output)["errors"]
    assert not (wiki / ".llmwiki/query-sessions.json").exists()
    assert not (wiki / 'OLW-Publications/query-state/query-sessions.json').exists()
