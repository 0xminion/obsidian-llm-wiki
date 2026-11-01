"""CLI behavior tests for grounded, structured wiki queries."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _vault_with_wiki(tmp_path, monkeypatch):
    from tests.new.grounding_doubles import legacy_fixture_llm

    sources = tmp_path / '04-Wiki/sources'
    sources.mkdir(parents=True)
    (sources / 'evidence.md').write_text('Seed evidence. Bridge evidence. Leaf evidence. ' * 20)
    synthesis = {'source_title': 'Evidence', 'source_summary': 'Seed evidence.', 'maps': [],
                 'concepts': [
                     {'title': title, 'slug': slug, 'summary': slug + ' evidence.',
                      'sections': [{'heading': 'Evidence', 'prose': slug + ' evidence.'}],
                      'related': related}
                     for title, slug, related in [
                         ('Seed Topic', 'seed', [{'slug': 'bridge', 'display': 'Bridge'}]),
                         ('Bridge', 'bridge', [{'slug': 'leaf', 'display': 'Leaf'}]),
                         ('Leaf', 'leaf', []),
                     ]]}
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.acall_llm',
                        legacy_fixture_llm(json.dumps(synthesis)))
    result = runner.invoke(app, ['build', str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path


def _citation(vault, page):
    from obsidian_llm_wiki.core.immutable_publication import current_publication

    generation = current_publication(vault / '04-Wiki')[1]['generation']
    return f'04-Wiki/OLW-Publications/generations/{generation}/{page}'


def test_json_query_emits_ppr_trace_and_only_grounded_citations(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path, monkeypatch)
    monkeypatch.setattr(
        query_module,
        "call_llm",
        lambda *_args, **_kwargs: 'Seed evidence is linked to bridge [[' +
        _citation(vault, 'concepts/seed.md') + ']].',
    )

    result = runner.invoke(app, ["query", str(vault), "--ask", "seed topic", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer"].startswith("Seed evidence")
    assert payload["citations"] == [_citation(vault, 'concepts/seed.md')]
    assert payload["retrieval_trace"]["strategy"] == "graph_first_ppr"
    assert payload["errors"] == []
    assert "🔍" not in result.output


def test_invalid_llm_citation_is_replaced_with_retrieved_references(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path, monkeypatch)
    monkeypatch.setattr(
        query_module,
        "call_llm",
        lambda *_args, **_kwargs: "Unsupported claim [[sources/not-retrieved.md]].",
    )

    result = runner.invoke(app, ["query", str(vault), "--ask", "seed topic", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "sources/not-retrieved.md" not in payload["answer"]
    assert "## References" in payload["answer"]
    assert set(payload["citations"]) == {_citation(vault, page) for page in [
        'concepts/seed.md', 'concepts/bridge.md', 'concepts/leaf.md',
        'entries/evidence.md', 'sources/evidence.md',
    ]}
    assert payload["errors"] == ["Ungrounded citations were replaced with retrieved references."]


def test_profile_and_instructions_are_applied_and_named_session_continues(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path, monkeypatch)
    calls = []

    def fake_call(system, messages, *_args, **_kwargs):
        calls.append((system, messages))
        return 'Grounded response [[' + _citation(vault, 'concepts/seed.md') + ']].'

    monkeypatch.setattr(query_module, "call_llm", fake_call)
    first = runner.invoke(
        app,
        [
            "query",
            str(vault),
            "--ask",
            "seed topic",
            "--profile",
            "research",
            "--instructions",
            "Use one sentence.",
            "--session",
            "study",
            "--json",
        ],
    )
    second = runner.invoke(
        app,
        ["query", str(vault), "--ask", "seed topic again", "--session", "study", "--json"],
    )

    assert first.exit_code == second.exit_code == 0
    assert json.loads(first.output)["session"] == "study"
    assert "Synthesize the retrieved evidence" in calls[0][0]
    assert "Use one sentence." in calls[0][0]
    assert any(
        message["content"] == json.loads(first.output)['answer']
        for message in calls[1][1]
    )


@pytest.mark.parametrize("force", [False, True])
def test_save_answer_refuses_collision_without_force(tmp_path, monkeypatch, force):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path, monkeypatch)
    destination = tmp_path / "05-Queries" / "answer.md"
    destination.parent.mkdir()
    destination.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        query_module,
        "call_llm",
        lambda *_args, **_kwargs: "Grounded response [[concepts/seed.md]].",
    )

    result = runner.invoke(
        app,
        [
            "query",
            str(vault),
            "--ask",
            "seed topic",
            "--save-answer",
            str(destination),
            *(["--force"] if force else []),
        ],
    )

    assert result.exit_code == 1
    assert destination.read_text(encoding="utf-8") == "original"
    assert "Refusing to overwrite" in result.output
