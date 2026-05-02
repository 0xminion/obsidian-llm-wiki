"""Tests for pipeline/qmd.py — Semantic Concept Search via QMD MCP."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import ConceptMatch
from pipeline.utils import strip_qmd_noise
from pipeline.qmd import (
    run_qmd_query,
    run_qmd_concept_search,
    run_qmd_convergence,
)


# ─── strip_qmd_noise (keep for backward compat) ──────────────────────────────

class TestStripQmdNoise:
    def test_clean_json_passthrough(self):
        text = '[{"file":"qmd://concepts/test.md","score":0.9}]'
        assert strip_qmd_noise(text) == text

    def test_strips_cmake_prefix(self):
        noisy = (
            "CMake Warning at /usr/share/cmake/Modules/FindVulkan.cmake:123\n"
            "Not searching for Vulkan ...\n"
            '[{"file":"qmd://concepts/prediction-markets.md","score":0.85}]'
        )
        result = strip_qmd_noise(noisy)
        assert result.startswith("[")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["file"] == "qmd://concepts/prediction-markets.md"

    def test_empty_string(self):
        assert strip_qmd_noise("") == ""


# ─── run_qmd_query ─────────────────────────────────────────────────────────

class TestRunQmdQuery:
    def test_empty_query_returns_empty(self):
        result = run_qmd_query("", "qmd", "concepts")
        assert result == []

    def test_whitespace_query_returns_empty(self):
        result = run_qmd_query("   ", "qmd", "concepts")
        assert result == []

    @patch("pipeline.qmd._get_client")
    def test_qmd_returns_results(self, mock_get_client):
        client = MagicMock()
        client.query.return_value = [
            MagicMock(file="concepts/prediction-markets.md", score=0.85, snippet="test", collection="concepts"),
            MagicMock(file="concepts/forecasting.md", score=0.6, snippet="", collection="concepts"),
        ]
        mock_get_client.return_value = client

        matches = run_qmd_query("prediction markets", "qmd", "concepts")
        assert len(matches) == 2
        assert matches[0].concept == "prediction-markets"
        assert matches[0].score == pytest.approx(0.85)
        assert matches[1].concept == "forecasting"

    @patch("pipeline.qmd._get_client")
    def test_qmd_empty_falls_back_to_keyword(self, mock_get_client):
        client = MagicMock()
        client.query.return_value = []
        client._query_raw.return_value = []
        mock_get_client.return_value = client

        # Create a fake concept file
        tmp_dir = Path("/tmp/qmd_test_concepts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "alpha-concept.md").write_text("title: Alpha\n---\nprediction markets content")

        matches = run_qmd_query("alpha concept", "qmd", "concepts", concepts_dir=tmp_dir)
        assert any(m.concept == "alpha-concept" for m in matches), matches

    def test_no_qmd_no_concepts_dir_returns_empty(self):
        with patch("pipeline.qmd._get_client", return_value=None):
            result = run_qmd_query("anything", "qmd", "concepts")
            assert result == []


# ─── run_qmd_concept_search ────────────────────────────────────────────────

class TestRunQmdConceptSearch:
    def test_empty_queries(self):
        cfg = MagicMock()
        cfg.vault_path = Path.home() / "MyVault"
        cfg.parallel = 4
        result = run_qmd_concept_search({"h1": "   "}, cfg)
        assert result == {"h1": []}

    @patch("pipeline.qmd._get_client")
    def test_parallel_queries(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.query.return_value = [
            MagicMock(file="concepts/test.md", score=0.9, snippet="", collection="concepts"),
        ]
        mock_get_client.return_value = mock_client

        cfg = MagicMock()
        cfg.vault_path = Path.home() / "MyVault"
        cfg.parallel = 4
        cfg.qmd_collection = "concepts"

        queries = {
            "hash1": "query one",
            "hash2": "query two",
            "hash3": "query three",
        }
        result = run_qmd_concept_search(queries, cfg)
        assert len(result) == 3
        for h in queries:
            assert len(result[h]) == 1
            assert result[h][0].concept == "test"

    @patch("pipeline.qmd._get_client")
    def test_qmd_unavailable_uses_keyword(self, mock_get_client, tmp_path: Path):
        mock_get_client.return_value = None
        cfg = MagicMock()
        cfg.vault_path = tmp_path / "vault"
        cfg.parallel = 4
        cfg.qmd_collection = "concepts"
        result = run_qmd_concept_search({"h1": "alpha"}, cfg)
        assert result == {"h1": []}  # no concepts dir in test


# ─── run_qmd_convergence ───────────────────────────────────────────────────

class TestRunQmdConvergence:
    def test_returns_dict_format(self, tmp_path):
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        cfg = MagicMock()
        cfg.vault_path = tmp_path
        cfg.resolved_extract_dir = extract_dir

        plan = MagicMock()
        plan.hash = "abc123"
        plan.title = "Test Article"
        plan.concept_new = ["New Concept"]
        plan.concept_updates = ["Existing"]

        with patch("pipeline.qmd.run_qmd_concept_search") as mock_search:
            mock_search.return_value = {"abc123": [ConceptMatch(concept="found", score=0.7)]}
            result = run_qmd_convergence([plan], cfg)
            assert "abc123" in result
            entry = result["abc123"]
            assert isinstance(entry, list)
            assert len(entry) == 1
            assert entry[0]["concept"] == "found"
            assert entry[0]["score"] == 0.7
