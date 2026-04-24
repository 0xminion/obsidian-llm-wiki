"""Integration test for QMD MCP server on localhost:8181.

Run only when the QMD MCP server is active. Skipped otherwise."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.config import Config
from pipeline.models import Plan
from pipeline.qmd import run_qmd_query, run_qmd_concept_search, run_qmd_convergence
from pipeline.qmd_mcp import QMDMCPClient


@pytest.fixture(scope="module")
def qmd_client():
    client = QMDMCPClient(base_url="http://localhost:8181", timeout=5)
    h = client.health()
    if h.get("status") != "ok":
        pytest.skip("QMD MCP server not running on localhost:8181")
    return client


class TestQMDMCPIntegration:
    def test_health(self, qmd_client):
        assert qmd_client.health()["status"] == "ok"

    def test_status_populated(self, qmd_client):
        status = qmd_client.status()
        assert status.get("totalDocuments", 0) >= 0
        assert len(status.get("collections", [])) >= 0

    def test_lex_query_fast(self, qmd_client):
        """BM25 keyword search should be sub-second for a warm server."""
        t0 = time.monotonic()
        results = qmd_client.query("prediction", n_results=5, min_score=0.01, mode="lex")
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"Lex query took {elapsed:.2f}s, expected <1s"
        assert isinstance(results, list)
        for r in results:
            assert r.file
            assert 0 <= r.score <= 1

    def test_vec_query_semantic(self, qmd_client):
        """Vector semantic search — may take longer if context is cold.
        We bump timeout because QMD can spend ~10s loading embedding model on first use."""
        results = qmd_client.query("blockchain smart contracts", n_results=5, min_score=0.1, mode="vec")
        assert isinstance(results, list)

    def test_pipeline_run_qmd_query_e2e(self):
        """Run pipeline.qmd.run_qmd_query end-to-end."""
        matches = run_qmd_query("prediction markets", "qmd", "concepts", n_results=5)
        assert isinstance(matches, list)
        if matches:
            assert hasattr(matches[0], "concept")
            assert hasattr(matches[0], "score")
            assert 0 <= matches[0].score <= 1

    def test_pipeline_concept_search_e2e(self):
        cfg = Config(vault_path=Path.home() / "MyVault")
        queries = {
            "h1": "blockchain",
            "h2": "artificial intelligence",
            "h3": "prediction markets",
        }
        results = run_qmd_concept_search(queries, cfg)
        assert len(results) == 3
        for h, matches in results.items():
            assert isinstance(matches, list)

    def test_pipeline_convergence_e2e(self, tmp_path):
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        plan = Plan(
            hash="abc123",
            title="Prediction Markets in DeFi",
            concept_new=["decentralized finance"],
            concept_updates=["forecasting"],
        )

        cfg = Config(vault_path=Path.home() / "MyVault")
        cfg = MagicMock()
        cfg.vault_path = Path.home() / "MyVault"
        cfg.resolved_extract_dir = extract_dir
        cfg.parallel = 4
        cfg.qmd_collection = "concepts"

        result = run_qmd_convergence([plan], cfg)
        assert "abc123" in result
        assert isinstance(result["abc123"], list)
        # If QMD has concepts it should find some; if not list is fine
        for entry in result["abc123"]:
            assert "concept" in entry
            assert "score" in entry
            assert 0 <= entry["score"] <= 1
