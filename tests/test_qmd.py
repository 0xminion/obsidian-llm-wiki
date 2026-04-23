"""Tests for pipeline/qmd.py — Semantic Concept Search via Ollama."""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from pipeline.models import ConceptMatch
from pipeline.utils import strip_qmd_noise
from pipeline.qmd import (
    run_qmd_query,
    run_qmd_concept_search,
    run_qmd_convergence,
    _cosine_similarity,
    _ollama_embed,
)


# ─── strip_qmd_noise (still used for backward compat) ────────────────────────

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


# ─── _cosine_similarity ─────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ─── _ollama_embed ──────────────────────────────────────────────────────────

class TestOllamaEmbed:
    @patch("urllib.request.urlopen")
    def test_successful_embed(self, mock_urlopen):
        mock_resp = MagicMock()
        embedding = [0.1] * 1024
        mock_resp.read.return_value = json.dumps({"embedding": embedding}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = _ollama_embed("test query")
        assert result == embedding

    @patch("urllib.request.urlopen")
    def test_wrong_dims_returns_none(self, mock_urlopen):
        mock_resp = MagicMock()
        # qwen3-embedding outputs 1024 dims; 768 is wrong
        mock_resp.read.return_value = json.dumps({"embedding": [0.1] * 768}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = _ollama_embed("test")
        assert result is None

    @patch("urllib.request.urlopen")
    def test_connection_error_returns_none(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        result = _ollama_embed("test")
        assert result is None


# ─── run_qmd_query ─────────────────────────────────────────────────────────

class TestRunQmdQuery:
    def test_empty_query_returns_empty(self):
        result = run_qmd_query("", "qmd", "concepts")
        assert result == []

    def test_whitespace_query_returns_empty(self):
        result = run_qmd_query("   ", "qmd", "concepts")
        assert result == []

    @patch("pipeline.qmd._ollama_embed")
    def test_successful_query(self, mock_embed):
        # Query embedding
        mock_embed.return_value = [1.0, 0.0, 0.0]
        # Reset cache and inject concept embeddings
        import pipeline.qmd as qmd_module
        qmd_module._cache_loaded = True
        qmd_module._concept_embedding_cache = {
            "prediction-markets": [0.9, 0.1, 0.0],
            "forecasting": [0.5, 0.5, 0.0],
        }

        matches = run_qmd_query("prediction markets", "qmd", "concepts")
        assert len(matches) == 2
        assert matches[0].concept == "prediction-markets"
        assert matches[0].score > 0.8
        assert matches[1].concept == "forecasting"

    @patch("pipeline.qmd._ollama_embed")
    def test_filters_below_min_score(self, mock_embed):
        mock_embed.return_value = [1.0, 0.0, 0.0]
        import pipeline.qmd as qmd_module
        qmd_module._cache_loaded = True
        qmd_module._concept_embedding_cache = {
            "high": [0.9, 0.1, 0.0],
            "low": [0.1, 0.9, 0.0],
        }

        matches = run_qmd_query("test", "qmd", "concepts", min_score=0.5)
        assert len(matches) == 1
        assert matches[0].concept == "high"

    @patch("pipeline.qmd._ollama_embed")
    def test_ollama_failure_returns_empty(self, mock_embed):
        mock_embed.return_value = None
        result = run_qmd_query("test", "qmd", "concepts")
        assert result == []


# ─── run_qmd_concept_search ────────────────────────────────────────────────

class TestRunQmdConceptSearch:
    def test_empty_queries(self):
        cfg = MagicMock()
        cfg.vault_path = MagicMock()
        cfg.vault_path.__truediv__ = MagicMock(return_value=MagicMock(is_dir=MagicMock(return_value=False)))
        with patch("pipeline.qmd._ollama_embed"):
            result = run_qmd_concept_search({"h1": "   "}, cfg)
            assert result == {"h1": []}

    @patch("pipeline.qmd._ollama_embed")
    def test_parallel_queries(self, mock_embed):
        mock_embed.return_value = [1.0] + [0.0] * 1023
        import pipeline.qmd as qmd_module
        qmd_module._cache_loaded = True
        qmd_module._concept_embedding_cache = {
            "test": [0.9] + [0.1] * 1023,
        }

        cfg = MagicMock()
        cfg.vault_path = MagicMock()
        cfg.vault_path.__truediv__ = MagicMock(return_value=MagicMock(is_dir=MagicMock(return_value=False)))

        queries = {
            "hash1": "query one",
            "hash2": "query two",
            "hash3": "query three",
        }
        result = run_qmd_concept_search(queries, cfg)
        assert len(result) == 3


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

        with patch("pipeline.qmd.run_qmd_query") as mock_qmd:
            mock_qmd.return_value = [ConceptMatch(concept="found", score=0.7)]
            result = run_qmd_convergence([plan], cfg)
            assert "abc123" in result
            entry = result["abc123"]
            assert isinstance(entry, list)
            assert len(entry) == 1
            assert entry[0]["concept"] == "found"
            assert entry[0]["score"] == 0.7
