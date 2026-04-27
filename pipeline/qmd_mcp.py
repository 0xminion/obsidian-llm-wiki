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
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from pipeline._version import __version__

log = logging.getLogger(__name__)

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
        self._lock = threading.Lock()

    def _call(
        self, method: str, params: dict, timeout: int | None = None
    ) -> dict:
        """Raw JSON-RPC POST to /mcp."""
        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            session_id = self._session_id
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["mcp-session-id"] = session_id
        req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        t = timeout or self.timeout
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:
                hdrs = resp.headers
                sid = hdrs.get("mcp-session-id") or hdrs.get("Mcp-Session-Id") or hdrs.get("MCP-SESSION-ID")
                if sid:
                    with self._lock:
                        self._session_id = sid
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
                "clientInfo": {"name": "obsidian-llm-wiki", "version": __version__},
            },
        )

    def ensure_session(self) -> bool:
        """Return True if a valid MCP session exists, initializing if needed."""
        with self._lock:
            if self._session_id:
                return True
        self.initialize()
        with self._lock:
            return self._session_id is not None

    def health(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def _query_raw(
        self,
        searches: list[dict],
        n: int = 5,
        min_score: float = 0.2,
        intent: str = "",
        timeout: int | None = None,
        collections: list[str] | None = None,
    ) -> list[QMDSearchResult]:
        """Run a QMD query via MCP and parse structured results.

        Args:
            timeout: Override default timeout for this single call.
            collections: QMD collection names to restrict search to (e.g. ["concepts"]).
        """
        if not self.ensure_session():
            return []
        arguments: dict = {
            "searches": searches,
            "limit": n,
            "minScore": min_score,
        }
        if intent:
            arguments["intent"] = intent
        if collections:
            arguments["collections"] = collections

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
        collections: list[str] | None = None,
        mode: str = "auto",
    ) -> list[QMDSearchResult]:
        """Search via QMD.

        Modes:
          - ``auto`` (default): try vector semantic first, then fall back to lex.
          - ``vec``: vector semantic only.
          - ``lex``: BM25 keyword only.

        The vector attempt uses a 60-second timeout so that a cold embedding
        context on CPU can finish loading without stalling the pipeline
        indefinitely.
        """
        mode = mode.lower().strip()
        if mode not in {"auto", "vec", "lex"}:
            raise ValueError(f"Unsupported QMD query mode: {mode}")

        if mode == "lex":
            return self._query_raw(
                searches=[{"type": "lex", "query": query_text}],
                n=n_results,
                min_score=min_score,
                intent=intent,
                collections=collections,
            )

        # Strategy 1: vector semantic (longer timeout for cold CPU load)
        orig_timeout = self.timeout
        try:
            self.timeout = 60
            vec = self._query_raw(
                searches=[{"type": "vec", "query": query_text}],
                n=n_results,
                min_score=min_score,
                intent=intent,
                collections=collections,
            )
            if vec or mode == "vec":
                return vec
        finally:
            self.timeout = orig_timeout

        # Strategy 2: BM25 keyword fallback
        return self._query_raw(
            searches=[{"type": "lex", "query": query_text}],
            n=n_results,
            min_score=min_score,
            intent=intent,
            collections=collections,
        )

    def status(self) -> dict:
        """QMD index status."""
        if not self.ensure_session():
            return {}
        res = self._call("tools/call", {"name": "status", "arguments": {}})
        if "error" in res:
            return {}
        return res.get("result", {}).get("structuredContent", {})

    def embed_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Embed a batch of texts via QMD MCP.

        Returns {text: embedding} for successes. Falls back to empty dict
        if the QMD server does not expose an embed tool."""
        if not texts:
            return {}
        if not self.ensure_session():
            return {}
        res = self._call(
            "tools/call",
            {"name": "embed", "arguments": {"texts": texts}},
            timeout=60,
        )
        if "error" in res:
            log.debug("QMD embed_batch failed: %s", res["error"])
            return {}
        structured = res.get("result", {}).get("structuredContent", {})
        embeddings = structured.get("embeddings", [])
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            return {}
        return {
            text: emb
            for text, emb in zip(texts, embeddings)
            if isinstance(emb, list) and len(emb) > 0
        }

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text via QMD MCP."""
        batch = self.embed_batch([text])
        return batch.get(text)


def _get_qmd_client(base_url: str = "") -> QMDMCPClient | None:
    """Return a QMDMCPClient if the server is reachable and healthy, else None."""
    url = base_url or _DEFAULT_QMD_MCP_URL
    client = QMDMCPClient(base_url=url, timeout=60)
    try:
        h = client.health()
        if h.get("status") != "ok":
            log.debug("QMD MCP server not healthy at %s", url)
            return None
    except Exception:
        log.debug("QMD MCP server unreachable at %s", url)
        return None
    if client.ensure_session():
        return client
    return None


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
