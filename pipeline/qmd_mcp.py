"""QMD MCP HTTP client for semantic search.

Replaces Ollama-based embedding + cosine similarity with a long-lived
QMD MCP server running on http://localhost:8181.

The QMD server is configured with vault collections (concepts, entries, sources, mocs)
and handles all indexing, embedding, and search internally.

Session lifecycle:
  1. POST /mcp with initialize -> receive mcp-session-id
  2. Reuse session-id in subsequent requests
  3. Auto-reinitialize if session expires

Search priority (matches user requirement):
  1. query (hybrid / vec semantic)   — default
  2. lex (BM25 keyword fallback)      — fallback1
  3. keyword fallback (local Python)   — fallback2
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Default QMD MCP HTTP endpoint
_DEFAULT_QMD_MCP_URL = os.environ.get("QMD_MCP_URL", "http://localhost:8181")


@dataclass
class QMDSearchResult:
    """Single result from QMD search."""

    file: str
    score: float
    snippet: str = ""
    collection: str = ""


class QMDMCPClient:
    """MCP Streamable HTTP client for QMD.

    Handles session lifecycle automatically. Callers just call
    ``.query()``, ``.status()``, etc.
    """

    def __init__(self, base_url: str = _DEFAULT_QMD_MCP_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/mcp"
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._req_id = 0

    def _call(
        self, method: str, params: dict, timeout: int | None = None
    ) -> dict:
        """Raw JSON-RPC POST to /mcp."""
        self._req_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        t = timeout or self.timeout
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:
                hdrs = dict(resp.headers)
                if "mcp-session-id" in hdrs:
                    self._session_id = hdrs["mcp-session-id"]
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            return {"error": str(e)}

    def initialize(self) -> dict:
        """Initialize MCP session."""
        return self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "obsidian-llm-wiki", "version": "0.3.0"},
            },
        )

    def ensure_session(self) -> bool:
        """Return True if a valid MCP session exists, initializing if needed."""
        if self._session_id:
            return True
        self.initialize()
        return self._session_id is not None

    def health(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"error": str(e)}

    def _query_raw(
        self,
        searches: list[dict],
        n: int = 5,
        min_score: float = 0.2,
        intent: str = "",
        timeout: int | None = None,
    ) -> list[QMDSearchResult]:
        """Run a QMD query via MCP and parse structured results.

        Args:
            timeout: Override default timeout for this single call.
        """
        if not self.ensure_session():
            return []
        arguments: dict = {
            "searches": searches,
            "n": n,
            "minScore": min_score,
        }
        if intent:
            arguments["intent"] = intent

        call_timeout = timeout or self.timeout
        res = self._call(
            "tools/call", {"name": "query", "arguments": arguments}, timeout=call_timeout
        )
        if "error" in res:
            log.warning("QMD query failed: %s", res["error"])
            return []
        structured = res.get("result", {}).get("structuredContent", {})
        items = structured.get("results", [])
        results: list[QMDSearchResult] = []
        for item in items:
            results.append(
                QMDSearchResult(
                    file=item.get("file", ""),
                    score=item.get("score", 0.0),
                    snippet=item.get("snippet", ""),
                    collection=item.get("collection", ""),
                )
            )
        return results

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        min_score: float = 0.2,
        intent: str = "",
    ) -> list[QMDSearchResult]:
        """Semantic search via QMD.

        Strategy:
          1. Try ``type: vec`` (semantic). On CPU-only this may take 30–60 s
             on the very first query while the embedding model loads. After that,
             the model stays resident and subsequent vec queries are ~1 s.
          2. Fall back to ``type: lex`` (BM25 keyword) — always sub-second.
          3. Return empty list if QMD is unreachable (caller should use keyword
             fallback).

        The vector attempt uses a 60-second timeout so that a cold embedding
        context on CPU can finish loading without stalling the pipeline
        indefinitely.
        """
        # Strategy 1: vector semantic (longer timeout for cold CPU load)
        orig_timeout = self.timeout
        try:
            self.timeout = 60
            vec = self._query_raw(
                searches=[{"type": "vec", "query": query_text}],
                n=n_results,
                min_score=min_score,
                intent=intent,
            )
            if vec:
                return vec
        finally:
            self.timeout = orig_timeout

        # Strategy 2: BM25 keyword fallback
        return self._query_raw(
            searches=[{"type": "lex", "query": query_text}],
            n=n_results,
            min_score=min_score,
            intent=intent,
        )

    def status(self) -> dict:
        """QMD index status."""
        if not self.ensure_session():
            return {}
        res = self._call("tools/call", {"name": "status", "arguments": {}})
        if "error" in res:
            return {}
        return res.get("result", {}).get("structuredContent", {})


def _get_qmd_client(base_url: str = "") -> QMDMCPClient:
    """Return a QMDMCPClient, respecting env overrides."""
    url = base_url or _DEFAULT_QMD_MCP_URL
    return QMDMCPClient(base_url=url, timeout=60)


def _qmd_results_to_concept_matches(
    results: list[QMDSearchResult],
    collection_filter: str = "concepts",
) -> list:
    """Convert QMD results to pipeline ConceptMatch objects.

    Filters to a specific collection and extracts file stem as concept name.
    """
    from pipeline.models import ConceptMatch

    matches: list[ConceptMatch] = []
    for r in results:
        # Derive collection from file path when QMD doesn't populate it
        derived_collection = r.collection
        if not derived_collection:
            path_parts = r.file.split("/")
            if len(path_parts) > 1:
                derived_collection = path_parts[0]
        if collection_filter and derived_collection and derived_collection != collection_filter:
            continue
        # Extract stem from file path like "concepts/2025-03-01T00-00-00.md" -> "2025-03-01T00-00-00"
        stem = r.file.split("/")[-1].replace(".md", "")
        if not stem:
            continue
        matches.append(ConceptMatch(concept=stem, score=round(r.score, 3)))
    return matches
