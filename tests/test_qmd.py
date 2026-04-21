"""Tests for pipeline/qmd.py — Semantic Concept Search module."""

import json
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import ConceptMatch
from pipeline.utils import strip_qmd_noise
from pipeline.qmd import run_qmd_query, run_qmd_concept_search, run_qmd_convergence


# ─── Helpers ────────────────────────────────────────────────────────────────

qmd_available = shutil.which("qmd") is not None

requires_qmd = pytest.mark.skipif(not qmd_available, reason="qmd binary not in PATH")


# ─── strip_qmd_noise (used by qmd module) ──────────────────────────────────

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

    def test_strips_vulkan_noise(self):
        noisy = (
            "VULKAN_SDK not set\n"
            '[{"file":"qmd://concepts/foo.md","score":0.5}]'
        )
        result = strip_qmd_noise(noisy)
        assert result.startswith("[")
        json.loads(result)  # should not raise

    def test_empty_string(self):
        assert strip_qmd_noise("") == ""

    def test_no_json_array_returns_unchanged(self):
        text = "some random text with no JSON"
        assert strip_qmd_noise(text) == text

    def test_nested_brackets(self):
        """strip_qmd_noise tracks bracket depth to find matching close."""
        data = [{"file": "qmd://concepts/deep.md", "score": 0.9, "meta": {"extra": [1, 2]}}]
        json_text = json.dumps(data)
        noisy = "cmake noise here\n" + json_text
        result = strip_qmd_noise(noisy)
        parsed = json.loads(result)
        assert parsed[0]["meta"]["extra"] == [1, 2]


# ─── run_qmd_query ─────────────────────────────────────────────────────────

class TestRunQmdQuery:
    def test_empty_query_returns_empty(self):
        """Empty or whitespace-only queries return [] without calling subprocess."""
        with patch("pipeline.qmd.subprocess.run") as mock_run:
            result = run_qmd_query("", "qmd", "concepts")
            assert result == []
            mock_run.assert_not_called()

    def test_whitespace_query_returns_empty(self):
        with patch("pipeline.qmd.subprocess.run") as mock_run:
            result = run_qmd_query("   ", "qmd", "concepts")
            assert result == []
            mock_run.assert_not_called()

    @patch("pipeline.qmd.subprocess.run")
    def test_successful_query(self, mock_run):
        qmd_output = json.dumps([
            {"file": "qmd://concepts/prediction-markets.md", "score": 0.92},
            {"file": "qmd://concepts/forecasting.md", "score": 0.75},
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=qmd_output, stderr="")
        matches = run_qmd_query("prediction markets", "qmd", "concepts")
        assert len(matches) == 2
        assert matches[0].concept == "prediction-markets"
        assert matches[0].score == 0.92
        assert matches[1].concept == "forecasting"
        assert matches[1].score == 0.75

    @patch("pipeline.qmd.subprocess.run")
    def test_query_with_noise_stripping(self, mock_run):
        noisy_output = (
            "CMake Warning at FindVulkan.cmake\n"
            '[{"file":"qmd://concepts/test.md","score":0.8}]'
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=noisy_output, stderr="")
        matches = run_qmd_query("test", "qmd", "concepts")
        assert len(matches) == 1
        assert matches[0].concept == "test"

    @patch("pipeline.qmd.subprocess.run")
    def test_filters_below_min_score(self, mock_run):
        qmd_output = json.dumps([
            {"file": "qmd://concepts/high.md", "score": 0.8},
            {"file": "qmd://concepts/low.md", "score": 0.05},
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=qmd_output, stderr="")
        matches = run_qmd_query("test", "qmd", "concepts", min_score=0.1)
        assert len(matches) == 1
        assert matches[0].concept == "high"

    @patch("pipeline.qmd.subprocess.run")
    def test_handles_path_field(self, mock_run):
        """qmd may return 'path' instead of 'file'."""
        qmd_output = json.dumps([
            {"path": "qmd://concepts/my-concept.md", "score": 0.6},
        ])
        mock_run.return_value = MagicMock(returncode=0, stdout=qmd_output, stderr="")
        matches = run_qmd_query("test", "qmd", "concepts")
        assert len(matches) == 1
        assert matches[0].concept == "my-concept"

    @patch("pipeline.qmd.subprocess.run")
    def test_nonzero_exit_returns_empty(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        matches = run_qmd_query("test", "qmd", "concepts")
        assert matches == []

    @patch("pipeline.qmd.subprocess.run")
    def test_timeout_returns_empty(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["qmd"], timeout=10)
        matches = run_qmd_query("test", "qmd", "concepts")
        assert matches == []

    @patch("pipeline.qmd.subprocess.run")
    def test_invalid_json_returns_empty(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="not json at all", stderr="")
        matches = run_qmd_query("test", "qmd", "concepts")
        assert matches == []

    @patch("pipeline.qmd.subprocess.run")
    def test_os_error_returns_empty(self, mock_run):
        mock_run.side_effect = OSError("No such file or directory")
        matches = run_qmd_query("test", "/nonexistent/qmd", "concepts")
        assert matches == []

    @patch("pipeline.qmd.subprocess.run")
    def test_command_construction(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        run_qmd_query("my query", "/usr/bin/qmd", "my-collection",
                       timeout=120, n_results=3, min_score=0.3, no_rerank=True)
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "/usr/bin/qmd"
        assert "my query" in cmd
        assert "-c" in cmd
        assert "my-collection" in cmd
        assert "-n" in cmd
        assert "3" in cmd
        assert "--min-score" in cmd
        assert "0.3" in cmd
        assert "--no-rerank" in cmd

    @patch("pipeline.qmd.subprocess.run")
    def test_no_rerank_flag_omitted_when_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")
        run_qmd_query("test", "qmd", "concepts", no_rerank=False)
        cmd = mock_run.call_args[0][0]
        assert "--no-rerank" not in cmd


# ─── run_qmd_concept_search ────────────────────────────────────────────────

class TestRunQmdConceptSearch:
    def test_empty_queries(self):
        """Empty query strings are skipped, returning empty list for that hash."""
        cfg = MagicMock()
        cfg.qmd_cmd = "qmd"
        cfg.qmd_collection = "concepts"
        cfg.plan_timeout = 60
        with patch("pipeline.qmd.run_qmd_query") as mock_qmd:
            result = run_qmd_concept_search({"h1": "   "}, cfg)
            assert result == {"h1": []}
            mock_qmd.assert_not_called()

    @patch("pipeline.qmd.run_qmd_query")
    def test_parallel_queries(self, mock_qmd):
        mock_qmd.return_value = [ConceptMatch(concept="test", score=0.8)]
        cfg = MagicMock()
        cfg.qmd_cmd = "qmd"
        cfg.qmd_collection = "concepts"
        cfg.plan_timeout = 60
        queries = {
            "hash1": "query one",
            "hash2": "query two",
            "hash3": "query three",
        }
        result = run_qmd_concept_search(queries, cfg)
        assert len(result) == 3
        assert all(len(v) == 1 for v in result.values())
        assert mock_qmd.call_count == 3


# ─── run_qmd_convergence ───────────────────────────────────────────────────

class TestRunQmdConvergence:
    def test_returns_dict_format(self, tmp_path):
        """run_qmd_convergence returns {hash: [{concept, score}]} dicts."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        cfg = MagicMock()
        cfg.qmd_cmd = "qmd"
        cfg.qmd_collection = "concepts"
        cfg.plan_timeout = 60
        cfg.resolved_extract_dir = extract_dir

        plan = MagicMock()
        plan.hash = "abc123"
        plan.title = "Test Article"
        plan.concept_new = ["New Concept"]
        plan.concept_updates = ["Existing"]

        with patch("pipeline.qmd.run_qmd_query") as mock_qmd:
            mock_qmd.return_value = [ConceptMatch(concept="found", score=0.7)]
            result = run_qmd_convergence([plan], cfg)
            assert "abc123" in result
            entry = result["abc123"]
            assert isinstance(entry, list)
            assert len(entry) == 1
            assert entry[0]["concept"] == "found"
            assert entry[0]["score"] == 0.7


# ─── QMD availability ─────────────────────────────────────────────────────

class TestQmdAvailability:
    def test_which_qmd(self):
        """Verify shutil.which can detect qmd binary (installed or not)."""
        result = shutil.which("qmd")
        # Either None or a valid path string
        assert result is None or isinstance(result, str)

    @requires_qmd
    def test_qmd_is_actually_installed(self):
        """If qmd exists, this test passes — otherwise it's skipped."""
        assert True
