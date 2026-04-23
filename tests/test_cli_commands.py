"""Regression tests for CLI commands and URL inbox parsing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from pipeline.cli import app, _collect_url_files


runner = CliRunner()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    vault_path = tmp_path / "vault"
    for rel in [
        "01-Raw",
        "03-Queries",
        "04-Wiki/concepts",
        "04-Wiki/entries",
        "04-Wiki/mocs",
        "05-Outputs",
        "06-Config",
    ]:
        (vault_path / rel).mkdir(parents=True, exist_ok=True)
    (vault_path / "06-Config" / "wiki-index.md").write_text(
        "# Wiki Index\n\n- [[test-concept]]: test concept summary (concept)\n",
        encoding="utf-8",
    )
    return vault_path


class TestCollectUrlFiles:
    def test_accepts_plain_url_file(self, tmp_path: Path):
        inbox = tmp_path / "01-Raw"
        inbox.mkdir()
        plain_file = inbox / "article.url"
        plain_file.write_text("https://example.com/article\n", encoding="utf-8")

        results = _collect_url_files(inbox)

        assert results == [(plain_file, "https://example.com/article")]

    def test_accepts_internet_shortcut_format(self, tmp_path: Path):
        inbox = tmp_path / "01-Raw"
        inbox.mkdir()
        shortcut_file = inbox / "article.url"
        shortcut_file.write_text(
            "[InternetShortcut]\nURL=https://example.com/article\n",
            encoding="utf-8",
        )

        results = _collect_url_files(inbox)

        assert results == [(shortcut_file, "https://example.com/article")]


class TestCliCommands:
    def test_ingest_dry_run_does_not_initialize_missing_vault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        vault = tmp_path / "dry-run-vault"

        monkeypatch.setattr("pipeline.cli.check_dependencies", lambda agent_cmd="hermes": [])

        result = runner.invoke(app, ["ingest", str(vault), "--dry-run"])

        assert result.exit_code == 0, result.stdout
        assert not vault.exists()
        assert "dry run" in result.stdout.lower()

    def test_ingest_dry_run_does_not_mutate_incomplete_vault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        vault = tmp_path / "incomplete-vault"
        (vault / "01-Raw").mkdir(parents=True)
        (vault / "01-Raw" / "article.url").write_text("https://example.com/article\n", encoding="utf-8")

        monkeypatch.setattr("pipeline.cli.check_dependencies", lambda agent_cmd="hermes": [])

        before = sorted(str(p.relative_to(vault)) for p in vault.rglob("*"))
        result = runner.invoke(app, ["ingest", str(vault), "--dry-run"])
        after = sorted(str(p.relative_to(vault)) for p in vault.rglob("*"))

        assert result.exit_code == 0, result.stdout
        assert before == after
        assert "would migrate" in result.stdout.lower() or "incomplete vault" in result.stdout.lower()

    def test_tags_command_completes(self, vault: Path):
        (vault / "04-Wiki" / "entries" / "entry.md").write_text(
            "---\ntags:\n  - ai\n  - infra\n---\n\n# Entry\n",
            encoding="utf-8",
        )
        (vault / "04-Wiki" / "concepts" / "concept.md").write_text(
            "---\ntags:\n  - ai\n---\n\n# Concept\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["tags", str(vault)])

        assert result.exit_code == 0, result.stdout
        registry = (vault / "06-Config" / "tag-registry.md").read_text(encoding="utf-8")
        assert "Entry Tags" in registry
        assert "Concept Tags" in registry
        assert "`ai` (1 uses)" in registry or "`ai` (2 uses)" in registry

    def test_tags_command_includes_moc_tags(self, vault: Path):
        (vault / "04-Wiki" / "mocs" / "moc.md").write_text(
            "---\ntags:\n  - moc-tag\n---\n\n# MoC\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["tags", str(vault)])

        assert result.exit_code == 0, result.stdout
        assert "MoC tags" in result.stdout
        registry = (vault / "06-Config" / "tag-registry.md").read_text(encoding="utf-8")
        assert "MoC Tags" in registry
        assert "`moc-tag` (1 uses)" in registry

    def test_query_command_completes_for_cli_question(self, vault: Path, monkeypatch: pytest.MonkeyPatch):
        (vault / "04-Wiki" / "entries" / "deep-dive.md").write_text(
            "# Deep Dive\n\nPrediction markets are useful for forecasting real-world outcomes.\n",
            encoding="utf-8",
        )
        (vault / "04-Wiki" / "sources").mkdir(parents=True, exist_ok=True)
        (vault / "04-Wiki" / "sources" / "primary-source.md").write_text(
            "# Primary Source\n\nThis source explains why prediction markets aggregate dispersed information.\n",
            encoding="utf-8",
        )
        (vault / "04-Wiki" / "concepts" / "test-concept.md").write_text(
            "# Test Concept\n\nAI infrastructure and prediction markets.\n",
            encoding="utf-8",
        )

        def fake_run(*args, **kwargs):
            prompt = args[0][3]
            assert "Deep Dive" in prompt
            assert "Primary Source" in prompt
            return SimpleNamespace(returncode=0, stdout="Use [[test-concept]].", stderr="")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = runner.invoke(app, ["query", str(vault), "--ask", "What is this vault about?"])

        assert result.exit_code == 0, result.stdout
        assert "Use [[test-concept]]." in result.stdout
        output_file = vault / "05-Outputs" / "cli-query.md"
        assert output_file.exists()
        assert "[[test-concept]]" in output_file.read_text(encoding="utf-8")

    def test_query_command_returns_non_zero_on_agent_failure(self, vault: Path, monkeypatch: pytest.MonkeyPatch):
        def fake_run(*_args, **_kwargs):
            return SimpleNamespace(returncode=42, stdout="", stderr="agent blew up")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = runner.invoke(app, ["query", str(vault), "--ask", "What failed?"])

        assert result.exit_code == 1
        assert "Agent failed for cli-query" in result.stderr
        assert not (vault / "05-Outputs" / "cli-query.md").exists()

    def test_query_command_handles_missing_agent_binary(self, vault: Path, monkeypatch: pytest.MonkeyPatch):
        def fake_run(*_args, **_kwargs):
            raise FileNotFoundError("missing agent binary")

        monkeypatch.setattr("subprocess.run", fake_run)

        result = runner.invoke(app, ["query", str(vault), "--ask", "What failed?"])

        assert result.exit_code == 127
        assert "Agent command not found" in result.stderr
        assert not (vault / "05-Outputs" / "cli-query.md").exists()
