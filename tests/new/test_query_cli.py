"""CLI behavior tests for grounded, structured wiki queries."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _vault_with_wiki(tmp_path):
    wiki = tmp_path / "04-Wiki" / "concepts"
    wiki.mkdir(parents=True)
    (wiki / "seed.md").write_text("# Seed Topic\n[[bridge]]\nSeed evidence.", encoding="utf-8")
    (wiki / "bridge.md").write_text("# Bridge\n[[leaf]]\nBridge evidence.", encoding="utf-8")
    (wiki / "leaf.md").write_text("# Leaf\nLeaf evidence.", encoding="utf-8")
    return tmp_path


def test_json_query_emits_seeded_ppr_trace_and_only_grounded_citations(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path)
    monkeypatch.setattr(
        query_module,
        "call_llm",
        lambda *_args, **_kwargs: "Seed evidence is linked to bridge [[concepts/seed.md]].",
    )

    result = runner.invoke(app, ["query", str(vault), "--ask", "seed topic", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["answer"].startswith("Seed evidence")
    assert payload["citations"] == ["concepts/seed.md"]
    assert payload["retrieval_trace"]["strategy"] == "seeded_ppr"
    assert payload["errors"] == []
    assert "🔍" not in result.output


def test_invalid_llm_citation_is_replaced_with_retrieved_references(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path)
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
    assert payload["citations"] == [
        "concepts/seed.md",
        "concepts/bridge.md",
        "concepts/leaf.md",
    ]
    assert payload["errors"] == ["Ungrounded citations were replaced with retrieved references."]


def test_profile_and_instructions_are_applied_and_named_session_continues(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path)
    calls = []

    def fake_call(system, messages, *_args, **_kwargs):
        calls.append((system, messages))
        return "Grounded response [[concepts/seed.md]]."

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
        message["content"] == "Grounded response [[concepts/seed.md]]."
        for message in calls[1][1]
    )


def test_save_answer_refuses_collision_without_force(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query_module

    vault = _vault_with_wiki(tmp_path)
    destination = tmp_path / "answer.md"
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
        ],
    )

    assert result.exit_code == 1
    assert destination.read_text(encoding="utf-8") == "original"
    assert "Refusing to overwrite" in result.output
