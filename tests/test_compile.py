"""Tests for pipeline.compile module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.compile import _load_prompt, run_compile, CompileResult, VaultSnapshot, _parse_agent_metrics
from pipeline.utils import count_md
from pipeline.config import Config


@pytest.fixture
def cfg(tmp_path):
    for d in ["04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources", "06-Config", "Meta/Scripts"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Config(vault_path=tmp_path)


@pytest.fixture
def prompts_dir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "compile-pass.prompt").write_text(
        "Compile vault at {VAULT_PATH}. {ENTRY_COUNT} entries, {CONCEPT_COUNT} concepts, {MOC_COUNT} MoCs."
    )
    return d


class TestLoadPrompt:
    def test_loads_existing_prompt(self, prompts_dir):
        content = _load_prompt("compile-pass", prompts_dir)
        assert "{VAULT_PATH}" in content

    def test_returns_empty_for_missing(self, prompts_dir):
        content = _load_prompt("nonexistent", prompts_dir)
        assert content == ""


class TestCountMd:
    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert count_md(d) == 0

    def test_counts_md_files(self, tmp_path):
        d = tmp_path / "notes"
        d.mkdir()
        (d / "a.md").write_text("# A\n")
        (d / "b.md").write_text("# B\n")
        (d / "c.txt").write_text("not md\n")
        assert count_md(d) == 2

    def test_nonexistent_dir(self, tmp_path):
        assert count_md(tmp_path / "nope") == 0


class TestRunCompile:
    @patch("pipeline.compile._run_agent", return_value=(True, "Cross-links added: 3"))
    def test_success(self, mock_agent, cfg, prompts_dir, monkeypatch):
        import pipeline.compile as compile_mod
        monkeypatch.setattr(compile_mod, "_load_prompt", lambda name, d: "Test prompt {VAULT_PATH}")
        result = run_compile(cfg)
        assert result["success"] is True
        assert result["entries"] == 0
        assert result["concepts"] == 0
        assert result["agent_succeeded"] is True

    @patch("pipeline.compile._run_agent", return_value=(False, ""))
    def test_failure(self, mock_agent, cfg, monkeypatch):
        import pipeline.compile as compile_mod
        monkeypatch.setattr(compile_mod, "_load_prompt", lambda name, d: "Test prompt")
        result = run_compile(cfg)
        # Agent failed but deterministic ops may succeed (reindex on empty vault)
        assert result["agent_succeeded"] is False

    def test_missing_prompt(self, cfg, monkeypatch):
        import pipeline.compile as compile_mod
        monkeypatch.setattr(compile_mod, "_load_prompt", lambda name, d: "")
        result = run_compile(cfg)
        # Missing prompt means agent can't run, but deterministic ops still proceed
        assert result["agent_succeeded"] is False


class TestParseAgentMetrics:
    def test_parses_crosslinks(self):
        output = "Cross-links added: 5\nConcepts merged: 2\nMoCs updated: 3"
        metrics = _parse_agent_metrics(output)
        assert metrics["crosslinks_added"] == 5
        assert metrics["concepts_merged"] == 2
        assert metrics["mocs_updated"] == 3

    def test_handles_empty_output(self):
        metrics = _parse_agent_metrics("")
        assert metrics["crosslinks_added"] == 0
        assert metrics["concepts_merged"] == 0

    def test_handles_various_formats(self):
        output = "I added 7 new wikilinks. Merged 1 concept pair. Updated 2 MoCs."
        metrics = _parse_agent_metrics(output)
        assert metrics["crosslinks_added"] == 7
        assert metrics["concepts_merged"] == 1
        assert metrics["mocs_updated"] == 2


class TestCompileResult:
    def test_to_dict(self):
        result = CompileResult(success=True, entries_after=5, concepts_after=3)
        d = result.to_dict()
        assert d["success"] is True
        assert d["entries"] == 5
        assert d["concepts"] == 3

    def test_defaults(self):
        result = CompileResult()
        d = result.to_dict()
        assert d["success"] is True
        assert d["edges_added"] == 0
        assert d["error"] == ""


class TestVaultSnapshot:
    def test_capture_empty_vault(self, cfg):
        snap = VaultSnapshot.capture(cfg)
        assert snap.entries == 0
        assert snap.concepts == 0
        assert snap.mocs == 0

    def test_capture_with_content(self, cfg):
        (cfg.entries_dir / "test-entry.md").write_text("# Test\nContent\n")
        (cfg.concepts_dir / "test-concept.md").write_text("# Concept\nBody\n")
        snap = VaultSnapshot.capture(cfg)
        assert snap.entries == 1
        assert snap.concepts == 1
        assert snap.mocs == 0
