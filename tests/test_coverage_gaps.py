"""Close the identified coverage gaps to push 820+ tests toward 9+/10 quality.

Gaps covered:
1. embed_all key collision — two notes with identical title+preview
2. SQLite concurrent stress — multi-threaded writes
3. Camoufox/browser extraction — fallback chain structure
4. Compile agent subprocess fallback — direct-LLM fails → Hermes fallback
5. _semantic_concept_merge O(N²) boundary — >100 concepts
6. QMD health-check failure path — skip-to-heuristic when QMD unavailable
7. URL parenthesis stripping — URLs ending in )
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from pipeline.compile import (
    NoteIndex,
    _run_semantic_compile,
    _semantic_concept_merge,
)
from pipeline.compile.core import CompileResult
from pipeline.config import Config
from pipeline.extractors.web import extract_web
from pipeline.store import ContentStore


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path):
    for d in [
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "04-Wiki/sources",
        "06-Config",
        "Meta/Scripts",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Config(vault_path=tmp_path)


@pytest.fixture
def mock_client():
    """Mock LLM client."""
    client = MagicMock()
    client.generate.return_value = ""
    client.embed_batch.return_value = {}
    return client


@pytest.fixture
def store(tmp_path):
    return ContentStore(tmp_path / "test_store.db")


# ══════════════════════════════════════════════════════════════════════════════
# Gap 1 — embed_all key collision
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbedAllKeyCollision:
    def test_duplicate_title_preview_gets_same_embedding(self, cfg, mock_client, monkeypatch):
        """Two notes with identical title+preview should both receive the
        embedding returned by the batch client (dict-key collision handled)."""
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": None)
        text = "---\ntitle: Same Title\n---\n\nSame preview body.\n"
        (cfg.entries_dir / "note-a.md").write_text(text)
        (cfg.entries_dir / "note-b.md").write_text(text)

        index = NoteIndex()
        index.load(cfg)
        mock_client.embed_batch.return_value = {"Same Title\nSame preview body.\n": [0.1, 0.2, 0.3]}
        index.embed_all(mock_client)

        assert "note-a" in index.embeddings
        assert "note-b" in index.embeddings
        assert index.embeddings["note-a"] == [0.1, 0.2, 0.3]
        assert index.embeddings["note-b"] == [0.1, 0.2, 0.3]


# ══════════════════════════════════════════════════════════════════════════════
# Gap 2 — SQLite concurrent stress
# ══════════════════════════════════════════════════════════════════════════════

class TestSqliteConcurrentStress:
    def test_concurrent_register_url(self, tmp_path):
        """Many threads registering the same URL must not raise SQLite
        OperationalError or corrupt the store."""
        db = tmp_path / "stress.db"
        store = ContentStore(db)
        url = "https://example.com/concurrent"
        errors: list[Exception] = []
        latch = threading.Barrier(20)

        def _register():
            try:
                latch.wait(timeout=5)
                store.register_url(url, "web", status="ok")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_register) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent write errors: {errors[:3]}"
        # Exactly one row should exist
        stats = store.get_stats()
        assert stats["urls_total"] == 1
        store.close()

    def test_concurrent_dlq_add(self, tmp_path):
        """Parallel DLQ inserts from a ThreadPoolExecutor must not crash."""
        db = tmp_path / "stress_dlq.db"
        store = ContentStore(db)

        def _add(i: int):
            store.dlq_add(f"https://fail-{i}.com", "timeout", "err")

        with ThreadPoolExecutor(max_workers=10) as ex:
            list(ex.map(_add, range(50)))

        pending = store.dlq_get_pending()
        assert len(pending) == 50
        store.close()


# ══════════════════════════════════════════════════════════════════════════════
# Gap 3 — Camoufox / browser extraction fallback chain
# ══════════════════════════════════════════════════════════════════════════════

class TestCamoufoxFallbackChain:
    def test_extract_web_fallback_chain_order(self, cfg, monkeypatch):
        """When defuddle, curl, archive.org all fail, extract_web
        attempts Camoufox as final fallback."""
        calls = []

        def _fail_defuddle(url, timeout):
            calls.append("defuddle")
            return ""

        def _fail_curl(url, timeout, attempt=0):
            calls.append("curl")
            return ""

        def _fail_archive(url, timeout):
            calls.append("archive")
            return ""

        def _succeed_camoufox(url, timeout):
            calls.append("camoufox")
            return "Camoufox extracted content.\nBody.", "Camoufox Title"

        monkeypatch.setattr("pipeline.extractors.web._try_defuddle", _fail_defuddle)
        monkeypatch.setattr("pipeline.extractors.web._try_curl_extract", _fail_curl)
        monkeypatch.setattr("pipeline.extractors.web._try_archive_extract", _fail_archive)
        monkeypatch.setattr("pipeline.extractors.web._try_camoufox_with_title", _succeed_camoufox)

        result = extract_web("https://example.com/page", cfg)

        assert "camoufox" in calls
        assert result.content == "Camoufox extracted content.\nBody."
        # Verify the chain was traversed in order (archive before camoufox)
        assert calls.index("archive") < calls.index("camoufox")
        # Defuddle and curl are always attempted before archive/camoufox
        assert calls.index("defuddle") < calls.index("archive")
        assert calls.index("curl") < calls.index("archive")

    def test_extract_web_all_methods_fail_returns_stub(self, cfg, monkeypatch):
        """When every method fails, a stub with the URL is returned."""
        monkeypatch.setattr("pipeline.extractors.web._try_defuddle", lambda u, t: "")
        monkeypatch.setattr("pipeline.extractors.web._try_curl_extract", lambda u, t, attempt=0: "")
        monkeypatch.setattr("pipeline.extractors.web._try_archive_extract", lambda u, t: "")
        monkeypatch.setattr("pipeline.extractors.web._try_camoufox_with_title", lambda u, t: ("", ""))

        result = extract_web("https://example.com/total-fail", cfg)
        assert "URL: https://example.com/total-fail" in result.content
        assert "extraction failed" in result.content.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Gap 4 — Compile agent subprocess fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestCompileAgentFallback:
    def test_direct_llm_failure_falls_back_to_agent_compile(self, cfg, monkeypatch):
        """When _try_direct raises, _run_semantic_compile falls back
        to the Hermes agent subprocess."""
        result = CompileResult()

        call_count = {"direct": 0}

        def _blow_up(*args, **kwargs):
            call_count["direct"] += 1
            raise ConnectionError("simulated LLM failure")

        monkeypatch.setattr(
            "pipeline.compile.semantic._semantic_crosslink", _blow_up
        )

        # Fallback (agent) should succeed
        def _mock_agent_compile(cfg, result):
            result.agent_succeeded = True
            return True, "agent fallback output"

        monkeypatch.setattr(
            "pipeline.compile.core._run_agent_compile", _mock_agent_compile
        )

        ok, output = _run_semantic_compile(cfg, result)
        assert ok is True
        assert output == "agent fallback output"
        assert call_count["direct"] >= 1  # direct path was attempted

    def test_both_direct_and_agent_fail(self, cfg, monkeypatch):
        """If both direct LLM and Hermes fallback fail, the function
        returns False and sets result.error."""
        result = CompileResult()

        monkeypatch.setattr(
            "pipeline.compile.semantic._semantic_crosslink",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("boom")),
        )
        monkeypatch.setattr(
            "pipeline.compile.core._run_agent_compile",
            lambda cfg, result: (False, "agent also failed"),
        )

        ok, output = _run_semantic_compile(cfg, result)
        assert ok is False
        assert "both exhausted" in output.lower()
        assert result.agent_succeeded is False


# ══════════════════════════════════════════════════════════════════════════════
# Gap 5 — _semantic_concept_merge O(N²) boundary (>100 concepts)
# ══════════════════════════════════════════════════════════════════════════════

class TestConceptMergeScaleBoundary:
    def test_100_concepts_no_crash(self, cfg, mock_client):
        """With 100+ concepts and no embeddings, heuristic merge should
        run in <1s and return zero (no identical titles)."""
        for i in range(110):
            (cfg.concepts_dir / f"concept-{i:03d}.md").write_text(
                f"---\ntitle: Concept {i}\n---\n\n# Concept {i}\n\nBody {i}.\n"
            )

        index = NoteIndex()
        index.load(cfg)

        t0 = time.time()
        merged = _semantic_concept_merge(cfg, mock_client, index)
        elapsed = time.time() - t0

        assert merged == 0
        assert elapsed < 1.0, f"O(N²) merge took {elapsed:.2f}s for 110 concepts"

    def test_100_concepts_with_some_embeddings(self, cfg, mock_client):
        """With 100+ concepts and random embeddings, still finishes fast."""
        for i in range(110):
            (cfg.concepts_dir / f"concept-{i:03d}.md").write_text(
                f"---\ntitle: Concept {i}\n---\n\n# Concept {i}\n\nBody {i}.\n"
            )

        index = NoteIndex()
        index.load(cfg)
        # Inject a handful of embeddings to exercise the fast path
        for i in range(0, 110, 10):
            index.embeddings[f"concept-{i:03d}"] = [0.1] * 64

        mock_client.generate.return_value = "KEEP_BOTH concept-000 | concept-010 | scale-test"
        t0 = time.time()
        _semantic_concept_merge(cfg, mock_client, index)
        elapsed = time.time() - t0

        assert elapsed < 1.0, f"Merge with embeddings took {elapsed:.2f}s"


# ══════════════════════════════════════════════════════════════════════════════
# Gap 6 — QMD health-check failure / skip-to-heuristic path
# ══════════════════════════════════════════════════════════════════════════════

class TestQmdHealthCheckSkipPath:
    def test_qmd_available_skips_local_embed(self, cfg, mock_client, monkeypatch):
        """When QMD is healthy, embed_all should skip local batch and
        semantic ops should still function via heuristics."""
        qmd_mock = MagicMock()
        qmd_mock.health.return_value = {"status": "ok"}
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": qmd_mock)

        (cfg.entries_dir / "a.md").write_text("---\ntitle: A\ntags:\n  - ai\n---\n\n# A\n")
        (cfg.entries_dir / "b.md").write_text("---\ntitle: B\ntags:\n  - ai\n---\n\n# B\n")

        index = NoteIndex()
        index.load(cfg)
        index.embed_all(mock_client)

        # embed_batch on the local client should NOT have been called
        mock_client.embed_batch.assert_not_called()

    def test_qmd_unavailable_uses_local_embed(self, cfg, mock_client, monkeypatch):
        """When QMD health check fails, local embed_batch is used."""
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": None)

        (cfg.entries_dir / "a.md").write_text("---\ntitle: A\n---\n\nBody A\n")
        index = NoteIndex()
        index.load(cfg)
        mock_client.embed_batch.return_value = {"A\nBody A\n": [0.1]}
        index.embed_all(mock_client)

        mock_client.embed_batch.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# Gap 7 — URL parenthesis stripping
# ══════════════════════════════════════════════════════════════════════════════

class TestUrlParenthesisStripping:
    def test_normalize_url_strips_trailing_paren(self):
        """A URL copy-pasted with a trailing ')' should be normalized
        to the clean URL."""
        raw = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        norm_with_paren = ContentStore.normalize_url(raw + ")")
        norm_clean = ContentStore.normalize_url(raw)
        assert norm_with_paren == norm_clean

    def test_url_hash_consistent_with_trailing_paren(self):
        """Two semantically-identical URLs must hash identically."""
        clean = "https://example.com/page"
        dirty = "https://example.com/page)"
        assert ContentStore.url_hash(clean) == ContentStore.url_hash(dirty)
