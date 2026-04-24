"""Tests for pipeline/qmd_mcp.py — QMD MCP HTTP client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.qmd_mcp import QMDMCPClient, QMDSearchResult, _qmd_results_to_concept_matches
from pipeline.models import ConceptMatch


# ─── QMDMCPClient ────────────────────────────────────────────────────────────

class TestQMDMCPClient:
    def test_initialization(self):
        client = QMDMCPClient(base_url="http://localhost:9999", timeout=5)
        assert client.base_url == "http://localhost:9999"
        assert client.timeout == 5
        assert client._session_id is None

    @patch.object(QMDMCPClient, "_call")
    def test_initialize_success(self, mock_call):
        mock_call.return_value = {
            "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "qmd"}},
            "jsonrpc": "2.0", "id": 1,
        }
        client = QMDMCPClient(base_url="http://test:8181")
        res = client.initialize()
        assert res["result"]["serverInfo"]["name"] == "qmd"
        # Session ID is set from HTTP headers, not body — mock that separately
        client._session_id = "sess-123"
        assert client._session_id == "sess-123"

    @patch.object(QMDMCPClient, "_call")
    def test_query_raw_parses_results(self, mock_call):
        mock_call.return_value = {
            "result": {
                "structuredContent": {
                    "results": [
                        {"file": "concepts/prediction-markets.md", "score": 0.85, "snippet": "test"},
                        {"file": "entries/foo.md", "score": 0.7},
                    ]
                }
            },
            "jsonrpc": "2.0", "id": 2,
        }
        client = QMDMCPClient(base_url="http://test:8181")
        client._session_id = "sess-456"

        results = client._query_raw(
            searches=[{"type": "lex", "query": "prediction markets"}],
            n=5,
            min_score=0.2,
        )
        assert len(results) == 2
        assert results[0].file == "concepts/prediction-markets.md"
        assert results[0].score == pytest.approx(0.85)
        assert results[0].collection == ""

    @patch.object(QMDMCPClient, "_call")
    def test_query_lex_mode_skips_vector_search(self, mock_call):
        mock_call.return_value = {
            "result": {
                "structuredContent": {
                    "results": [
                        {"file": "concepts/prediction-markets.md", "score": 0.91},
                    ]
                }
            },
            "jsonrpc": "2.0", "id": 2,
        }
        client = QMDMCPClient(base_url="http://test:8181")
        client._session_id = "sess-789"

        results = client.query("prediction", n_results=3, mode="lex")
        assert len(results) == 1
        assert results[0].file == "concepts/prediction-markets.md"
        assert mock_call.call_count == 1
        payload = mock_call.call_args.args[1]
        assert payload["name"] == "query"
        assert payload["arguments"]["searches"] == [{"type": "lex", "query": "prediction"}]

    @patch.object(QMDMCPClient, "_call")
    def test_query_falls_back_to_lex(self, mock_call):
        # vec returns empty, lex returns result
        mock_call.side_effect = [
            {
                "result": {"structuredContent": {"results": []}},
                "jsonrpc": "2.0", "id": 2,
            },
            {
                "result": {
                    "structuredContent": {
                        "results": [
                            {"file": "concepts/blockchain.md", "score": 0.75},
                        ]
                    }
                },
                "jsonrpc": "2.0", "id": 3,
            },
        ]
        client = QMDMCPClient(base_url="http://test:8181")
        client._session_id = "sess-789"

        results = client.query("blockchain", n_results=3)
        assert len(results) == 1
        assert results[0].file == "concepts/blockchain.md"

    @patch("urllib.request.urlopen")
    def test_health_ok(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"status": "ok", "uptime": 42}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = QMDMCPClient(base_url="http://test:8080")
        assert client.health() == {"status": "ok", "uptime": 42}

    def test_health_unavailable(self):
        client = QMDMCPClient(base_url="http://localhost:99999")
        res = client.health()
        assert "error" in res

    def test_query_rejects_invalid_mode(self):
        client = QMDMCPClient(base_url="http://test:8181")
        with pytest.raises(ValueError, match="Unsupported QMD query mode"):
            client.query("prediction", mode="turbo")


# ─── _qmd_results_to_concept_matches ───────────────────────────────────────

class TestQmdResultsToConceptMatches:
    def test_converts_and_filters(self):
        results = [
            QMDSearchResult(file="concepts/prediction-markets.md", score=0.85),
            QMDSearchResult(file="concepts/blockchain.md", score=0.7),
            QMDSearchResult(file="entries/foo.md", score=0.9),
        ]
        matches = _qmd_results_to_concept_matches(results, collection_filter="concepts")
        assert len(matches) == 2
        assert isinstance(matches[0], ConceptMatch)
        assert matches[0].concept == "prediction-markets"
        assert matches[0].score == pytest.approx(0.85)
        assert matches[1].concept == "blockchain"

    def test_empty_results(self):
        assert _qmd_results_to_concept_matches([]) == []

    def test_filters_out_wrong_collection(self):
        results = [
            QMDSearchResult(file="concepts/alpha.md", score=0.9, collection="concepts"),
            QMDSearchResult(file="entries/beta.md", score=0.8, collection="entries"),
        ]
        matches = _qmd_results_to_concept_matches(results, collection_filter="concepts")
        assert len(matches) == 1
        assert matches[0].concept == "alpha"


# ─── Real QMD server skip (integration level) ────────────────────────────

class TestRealQMDServer:
    """Skip if no QMD MCP server is running on the default port."""

    @pytest.fixture(scope="class")
    def _real_client(self):
        client = QMDMCPClient(base_url="http://localhost:8181", timeout=5)
        h = client.health()
        if h.get("status") != "ok":
            pytest.skip("QMD MCP server not running")
        return client

    def test_status(self, _real_client):
        status = _real_client.status()
        assert status.get("totalDocuments", 0) >= 0

    def test_lex_query(self, _real_client):
        results = _real_client.query("prediction", n_results=3, min_score=0.01)
        assert isinstance(results, list)
        for r in results:
            assert r.file
            assert 0 <= r.score <= 1
