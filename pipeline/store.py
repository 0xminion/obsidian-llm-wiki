"""SQLite-backed content store.

Replaces the flat-file ContentIndex with a proper database.
Provides: URL dedup, content dedup, extraction history, stats, and DLQ storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

log = logging.getLogger(__name__)


@dataclass
class UrlRecord:
    url_hash: str
    url: str
    canonical_url: str
    source_type: str
    extracted_at: float
    status: str
    content_hash: Optional[str] = None


@dataclass
class ContentRecord:
    content_hash: str
    title: str
    source_type: str
    word_count: int
    created_at: float
    vault_filename: Optional[str] = None


# Thread-safe proxy for sqlite3.Connection
class _LockedConnection:
    """Wraps a sqlite3.Connection so every execute/commit/executescript/close
    is guarded by the provided RLock."""

    def __init__(self, conn, lock):
        self._conn = conn
        self._lock = lock

    def execute(self, sql, parameters=None):
        with self._lock:
            if parameters is None:
                return self._conn.execute(sql)
            return self._conn.execute(sql, parameters)

    def executescript(self, sql_script):
        with self._lock:
            return self._conn.executescript(sql_script)

    def commit(self):
        with self._lock:
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


SCHEMA_VERSION = 2

_MIGRATIONS: list[str] = [
    # v0 → v1: initial schema (applied as baseline for new databases)
    "",
    # v1 → v2: add max_queue_size enforcement column
    "ALTER TABLE merge_queue ADD COLUMN priority REAL DEFAULT 0.0;",
]


class ContentStore:
    """SQLite-backed store (WAL mode) for URL/content dedup, extraction history, DLQ,
    pending reviews, compile state, pipeline runs, embeddings, merge queue, and vault
    cache (10 tables). Thread-safe via RLock-wrapped connection; usable as context manager.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.RLock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        raw_conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30,
        )
        raw_conn.row_factory = sqlite3.Row
        raw_conn.execute("PRAGMA journal_mode=WAL")
        raw_conn.execute("PRAGMA busy_timeout=5000")
        self._conn = _LockedConnection(raw_conn, self._lock)
        self._init_schema()
        self._run_migrations()

    def _get_schema_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return row["version"] if row else 0
        except sqlite3.OperationalError:
            return 0

    def _run_migrations(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        self._conn.commit()
        current = self._get_schema_version()
        if current >= SCHEMA_VERSION:
            return
        for version in range(current + 1, SCHEMA_VERSION + 1):
            sql = _MIGRATIONS[version - 1] if version - 1 < len(_MIGRATIONS) else ""
            if sql:
                for stmt in sql.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            self._conn.execute(stmt)
                        except sqlite3.OperationalError:
                            pass
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, time.time()),
            )
            self._conn.commit()
            log.info("Applied schema migration v%d", version)

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS urls (
                url_hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                source_type TEXT DEFAULT 'unknown',
                extracted_at REAL NOT NULL,
                status TEXT DEFAULT 'ok',
                content_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_urls_canonical ON urls(canonical_url);
            CREATE INDEX IF NOT EXISTS idx_urls_content ON urls(content_hash);
            CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);

            CREATE TABLE IF NOT EXISTS content (
                content_hash TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT DEFAULT 'unknown',
                word_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                vault_filename TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_content_title ON content(title);

            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                reason TEXT NOT NULL,
                attempts INTEGER DEFAULT 1,
                last_error TEXT,
                first_failed_at REAL NOT NULL,
                last_failed_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_dlq_status ON dead_letter_queue(status);
            CREATE INDEX IF NOT EXISTS idx_dlq_url ON dead_letter_queue(url);

            CREATE TABLE IF NOT EXISTS pending_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_hash TEXT NOT NULL,
                plan_data TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_content TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_status ON pending_reviews(status);

            CREATE TABLE IF NOT EXISTS vault_cache (
                cache_key TEXT PRIMARY KEY,
                cache_value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS compile_state (
                filename TEXT PRIMARY KEY,
                last_mtime REAL NOT NULL,
                last_hash TEXT,
                last_compile REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compile_state_last_compile ON compile_state(last_compile);

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                vault_path TEXT NOT NULL,
                total_sources INT DEFAULT 0,
                processed_sources INT DEFAULT 0,
                failed_sources INT DEFAULT 0,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status);
            CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs(started_at);

            CREATE TABLE IF NOT EXISTS pipeline_batches (
                run_id TEXT NOT NULL,
                batch_id TEXT PRIMARY KEY,
                batch_index INT NOT NULL,
                status TEXT NOT NULL,
                plan_hash TEXT,
                started_at REAL NOT NULL,
                completed_at REAL,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pipeline_batches_run_id ON pipeline_batches(run_id);
            CREATE INDEX IF NOT EXISTS idx_pipeline_batches_status ON pipeline_batches(status);

            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_updated ON embeddings(updated_at);

            CREATE TABLE IF NOT EXISTS merge_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                new_concept TEXT NOT NULL,
                existing_concept TEXT NOT NULL,
                similarity REAL NOT NULL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_merge_status ON merge_queue(status);
            CREATE INDEX IF NOT EXISTS idx_merge_existing ON merge_queue(existing_concept);
        """)
        self._conn.commit()

    # ─── URL Operations ───────────────────────────────────────────────────────

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for dedup comparison."""
        parsed = urlparse(url.rstrip(")"))
        skip_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_content",
            "utm_term", "ref", "source", "fbclid", "gclid",
        }
        params = parse_qs(parsed.query)
        filtered = {k: v for k, v in params.items() if k.lower() not in skip_params}
        clean_query = urlencode(sorted(filtered.items()), doseq=True)
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            parsed.params,
            clean_query,
            "",
        ))
        return normalized

    @staticmethod
    def url_hash(url: str) -> str:
        return hashlib.md5(ContentStore.normalize_url(url).encode(), usedforsecurity=False).hexdigest()[:12]

    @staticmethod
    def content_hash(content: str) -> str:
        from pipeline.utils import content_hash
        return content_hash(content)

    def is_url_extracted(self, url: str) -> bool:
        """Check if URL has been successfully extracted."""
        row = self._conn.execute(
            "SELECT 1 FROM urls WHERE url_hash = ? AND status = 'ok'",
            (self.url_hash(url),),
        ).fetchone()
        return row is not None

    def get_content_duplicate(self, content: str) -> Optional[str]:
        """Return vault filename of duplicate content, or None."""
        chash = self.content_hash(content)
        row = self._conn.execute(
            "SELECT vault_filename FROM content WHERE content_hash = ?",
            (chash,),
        ).fetchone()
        if not row:
            return None
        return row["vault_filename"] or "[unknown]"

    def register_url(
        self,
        url: str,
        source_type: str = "unknown",
        content_hash: Optional[str] = None,
        status: str = "ok",
    ) -> None:
        """Register an extracted URL."""
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO urls
               (url_hash, url, canonical_url, source_type, extracted_at, status, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                self.url_hash(url),
                url,
                self.normalize_url(url),
                source_type,
                now,
                status,
                content_hash,
            ),
        )
        self._conn.commit()

    def register_content(
        self,
        content: str,
        title: str,
        source_type: str = "unknown",
        vault_filename: str = "",
    ) -> str:
        """Register content and return its hash."""
        chash = self.content_hash(content)
        word_count = len(content.split())
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO content
               (content_hash, title, source_type, word_count, created_at, vault_filename)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chash, title, source_type, word_count, now, vault_filename or None),
        )
        self._conn.commit()
        return chash

    # ─── Dead Letter Queue ────────────────────────────────────────────────────

    def dlq_add(
        self,
        url: str,
        reason: str,
        error: str = "",
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a failed extraction to the dead letter queue."""
        now = time.time()
        # Check if URL already in DLQ — increment attempts
        existing = self._conn.execute(
            "SELECT id, attempts FROM dead_letter_queue WHERE url = ? AND status = 'pending'",
            (url,),
        ).fetchone()
        if existing:
            self._conn.execute(
                """UPDATE dead_letter_queue
                   SET attempts = attempts + 1, last_error = ?, last_failed_at = ?,
                       reason = ?, metadata = ?
                   WHERE id = ?""",
                (error, now, reason, json.dumps(metadata or {}), existing["id"]),
            )
            self._conn.commit()
            return existing["id"]
        else:
            cursor = self._conn.execute(
                """INSERT INTO dead_letter_queue
                   (url, reason, attempts, last_error, first_failed_at, last_failed_at, metadata)
                   VALUES (?, ?, 1, ?, ?, ?, ?)""",
                (url, reason, error, now, now, json.dumps(metadata or {})),
            )
            self._conn.commit()
            return cursor.lastrowid

    def dlq_get_pending(self, limit: int = 50) -> list[dict]:
        """Get pending failed extractions."""
        rows = self._conn.execute(
            """SELECT * FROM dead_letter_queue
               WHERE status = 'pending'
               ORDER BY last_failed_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def dlq_resolve(self, item_id: int) -> None:
        """Mark a DLQ item as resolved."""
        self._conn.execute(
            "UPDATE dead_letter_queue SET status = 'resolved' WHERE id = ?",
            (item_id,),
        )
        self._conn.commit()

    def dlq_clear(self, reason: Optional[str] = None) -> int:
        """Clear DLQ items. Returns count cleared."""
        if reason:
            cursor = self._conn.execute(
                "DELETE FROM dead_letter_queue WHERE status = 'pending' AND reason = ?",
                (reason,),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM dead_letter_queue WHERE status = 'pending'",
            )
        self._conn.commit()
        return cursor.rowcount

    # ─── Pending Reviews ──────────────────────────────────────────────────────

    def review_add(
        self,
        plan_hash: str,
        plan_data: dict,
        file_type: str,
        file_path: str,
        file_content: str,
    ) -> int:
        """Add a file to the pending review queue."""
        cursor = self._conn.execute(
            """INSERT INTO pending_reviews
               (plan_hash, plan_data, file_type, file_path, file_content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (plan_hash, json.dumps(plan_data), file_type, file_path,
             file_content, time.time()),
        )
        self._conn.commit()
        return cursor.lastrowid

    def review_get_pending(self) -> list[dict]:
        """Get all pending reviews."""
        rows = self._conn.execute(
            """SELECT * FROM pending_reviews
               WHERE status = 'pending'
               ORDER BY created_at ASC""",
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["plan_data"] = json.loads(d["plan_data"])
            results.append(d)
        return results

    def review_approve(self, review_id: int) -> None:
        """Approve a pending review."""
        self._conn.execute(
            "UPDATE pending_reviews SET status = 'approved' WHERE id = ?",
            (review_id,),
        )
        self._conn.commit()

    def review_reject(self, review_id: int) -> None:
        """Reject a pending review."""
        self._conn.execute(
            "UPDATE pending_reviews SET status = 'rejected' WHERE id = ?",
            (review_id,),
        )
        self._conn.commit()

    def review_clear(self) -> int:
        """Clear all pending reviews."""
        cursor = self._conn.execute(
            "DELETE FROM pending_reviews WHERE status = 'pending'",
        )
        self._conn.commit()
        return cursor.rowcount

    # ─── Vault Cache (for incremental lint/reindex) ───────────────────────────

    def cache_set(self, key: str, value: str) -> None:
        """Set a cache entry."""
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO vault_cache (cache_key, cache_value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, now),
        )
        self._conn.commit()

    def cache_get(self, key: str) -> Optional[str]:
        """Get a cache entry value, or None if not found."""
        row = self._conn.execute(
            "SELECT cache_value FROM vault_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return row["cache_value"] if row else None

    def cache_get_time(self, key: str) -> Optional[float]:
        """Get the last-updated timestamp for a cache entry."""
        row = self._conn.execute(
            "SELECT updated_at FROM vault_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return row["updated_at"] if row else None

    def cache_invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        self._conn.execute("DELETE FROM vault_cache WHERE cache_key = ?", (key,))
        self._conn.commit()

    def cache_invalidate_all(self, prefix: str = "") -> int:
        """Remove cache entries matching prefix. Returns count removed."""
        if prefix:
            cursor = self._conn.execute(
                "DELETE FROM vault_cache WHERE cache_key LIKE ?",
                (f"{prefix}%",),
            )
        else:
            cursor = self._conn.execute("DELETE FROM vault_cache")
        self._conn.commit()
        return cursor.rowcount

    def cache_get_file_index(self, directory: Path) -> dict[str, float]:
        """Get cached file mtime index for a directory.

        Returns {relative_path: mtime}. Returns empty dict if not cached.
        """
        cached = self.cache_get(f"file_index:{directory}")
        if not cached:
            return {}
        try:
            data = json.loads(cached)
            return {k: float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}

    def cache_set_file_index(self, directory: Path, index: dict[str, float]) -> None:
        """Store file mtime index for a directory."""
        self.cache_set(f"file_index:{directory}", json.dumps(index))

    def cache_is_directory_stale(self, directory: Path) -> bool:
        """Check if a directory has changed since last cache.

        Returns True if:
        - No cache exists
        - Any file was added/removed
        - Any file's mtime changed
        """
        if not directory.exists():
            return True

        cached_index = self.cache_get_file_index(directory)
        if not cached_index:
            return True

        current_index = {}
        for md in directory.glob("*.md"):
            try:
                current_index[md.name] = md.stat().st_mtime
            except OSError:
                continue

        # Different number of files
        if set(cached_index.keys()) != set(current_index.keys()):
            return True

        # Check mtimes
        for name, mtime in current_index.items():
            cached_mtime = cached_index.get(name, 0)
            if abs(mtime - cached_mtime) > 0.01:
                return True

        return False

    def cache_get_wikilinks(self, vault_path: Path) -> dict[str, set[str]]:
        """Get cached wikilink index: {note_name -> set of linked note names}."""
        cached = self.cache_get("wikilinks:index")
        if not cached:
            return {}
        try:
            data = json.loads(cached)
            return {k: set(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}

    def cache_set_wikilinks(self, links: dict[str, set[str]]) -> None:
        """Store wikilink index."""
        data = {k: list(v) for k, v in links.items()}
        self.cache_set("wikilinks:index", json.dumps(data))

    # ─── Compile State (Incremental Compile) ─────────────────────────────────

    def compile_state_get(self, filename: str) -> dict | None:
        """Get compile state for a single filename, or None."""
        row = self._conn.execute(
            "SELECT filename, last_mtime, last_hash, last_compile FROM compile_state WHERE filename = ?",
            (filename,),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def compile_state_set(
        self,
        filename: str,
        last_mtime: float,
        last_hash: str,
        last_compile: float,
    ) -> None:
        """Upsert compile state for a single filename."""
        self._conn.execute(
            """INSERT OR REPLACE INTO compile_state
               (filename, last_mtime, last_hash, last_compile)
               VALUES (?, ?, ?, ?)""",
            (filename, last_mtime, last_hash, last_compile),
        )
        self._conn.commit()

    def compile_state_get_all(self) -> dict[str, dict]:
        """Return {filename: {last_mtime, last_hash, last_compile}} for all rows."""
        rows = self._conn.execute(
            "SELECT filename, last_mtime, last_hash, last_compile FROM compile_state"
        ).fetchall()
        return {r["filename"]: dict(r) for r in rows}

    def compile_state_clear(self) -> int:
        """Clear the entire compile_state table. Returns rows deleted."""
        cursor = self._conn.execute("DELETE FROM compile_state")
        self._conn.commit()
        return cursor.rowcount

    # ─── Pipeline State Machine ────────────────────────────────────────────────

    def run_insert(
        self,
        run_id: str,
        status: str,
        started_at: float,
        vault_path: str,
        total_sources: int = 0,
        error: str | None = None,
    ) -> None:
        """Insert a new pipeline run record."""
        self._conn.execute(
            """INSERT INTO pipeline_runs
               (run_id, status, started_at, vault_path, total_sources, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, status, started_at, vault_path, total_sources, error),
        )
        self._conn.commit()

    def run_update_status(
        self,
        run_id: str,
        status: str,
        processed_sources: int | None = None,
        failed_sources: int | None = None,
        error: str | None = None,
        completed_at: float | None = None,
    ) -> None:
        """Update run status and optional counters."""
        parts = ["status = ?"]
        params: list = [status]
        if processed_sources is not None:
            parts.append("processed_sources = ?")
            params.append(processed_sources)
        if failed_sources is not None:
            parts.append("failed_sources = ?")
            params.append(failed_sources)
        if error is not None:
            parts.append("error = ?")
            params.append(error)
        if completed_at is not None:
            parts.append("completed_at = ?")
            params.append(completed_at)
        params.append(run_id)
        sql = "UPDATE pipeline_runs SET " + ", ".join(parts) + " WHERE run_id = ?"
        self._conn.execute(sql, tuple(params))
        self._conn.commit()

    def run_get_active(self) -> dict | None:
        """Return the most recent active (not completed) run, or None."""
        row = self._conn.execute(
            """SELECT * FROM pipeline_runs
               WHERE status IN ('running', 'paused')
               ORDER BY started_at DESC
               LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None

    def run_get_by_id(self, run_id: str) -> dict | None:
        """Get a run by ID."""
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def run_get_recent(self, limit: int = 10) -> list[dict]:
        """Return most recent runs ordered by started_at DESC."""
        rows = self._conn.execute(
            """SELECT * FROM pipeline_runs
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def batch_insert(
        self,
        run_id: str,
        batch_id: str,
        batch_index: int,
        status: str,
        started_at: float,
        plan_hash: str | None = None,
        error: str | None = None,
    ) -> None:
        """Insert a new batch record."""
        self._conn.execute(
            """INSERT INTO pipeline_batches
               (run_id, batch_id, batch_index, status, plan_hash, started_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, batch_id, batch_index, status, plan_hash, started_at, error),
        )
        self._conn.commit()

    def batch_update_status(
        self,
        batch_id: str,
        status: str,
        completed_at: float | None = None,
        error: str | None = None,
    ) -> None:
        """Update batch status."""
        parts = ["status = ?"]
        params: list = [status]
        if completed_at is not None:
            parts.append("completed_at = ?")
            params.append(completed_at)
        if error is not None:
            parts.append("error = ?")
            params.append(error)
        params.append(batch_id)
        sql = "UPDATE pipeline_batches SET " + ", ".join(parts) + " WHERE batch_id = ?"
        self._conn.execute(sql, tuple(params))
        self._conn.commit()

    def batch_get_by_run(self, run_id: str) -> list[dict]:
        """Get all batches for a run, ordered by batch_index."""
        rows = self._conn.execute(
            """SELECT * FROM pipeline_batches
               WHERE run_id = ?
               ORDER BY batch_index ASC""",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def batch_get_failed_or_skipped(self, run_id: str) -> list[dict]:
        """Get batches for a run that are failed or skipped (for resume)."""
        rows = self._conn.execute(
            """SELECT * FROM pipeline_batches
               WHERE run_id = ? AND status IN ('failed','skipped','pending')
               ORDER BY batch_index ASC""",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── Merge Queue ──────────────────────────────────────────────────────────

    def merge_queue_add(
        self,
        new_concept: str,
        existing_concept: str,
        similarity: float,
        max_size: int = 500,
    ) -> int:
        """Add a proposed concept merge to the queue. Returns the queue id.

        When the queue exceeds max_size, the lowest-similarity pending item is
        evicted to keep memory bounded.
        """
        now = time.time()
        cursor = self._conn.execute(
            """INSERT INTO merge_queue
               (new_concept, existing_concept, similarity, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (new_concept, existing_concept, similarity, now),
        )
        self._conn.commit()
        queue_id = cursor.lastrowid

        if not isinstance(max_size, int) or max_size <= 0:
            max_size = 500
        pending_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM merge_queue WHERE status = 'pending'"
        ).fetchone()["cnt"]
        if pending_count > max_size:
            overflow = pending_count - max_size
            self._conn.execute(
                """DELETE FROM merge_queue WHERE id IN (
                    SELECT id FROM merge_queue
                    WHERE status = 'pending'
                    ORDER BY similarity ASC
                    LIMIT ?
                )""",
                (overflow,),
            )
            self._conn.commit()
            log.warning("Merge queue exceeded %d; evicted %d low-similarity items", max_size, overflow)

        return queue_id

    def merge_queue_get_pending(self, limit: int = 50) -> list[dict]:
        """Get pending merge proposals."""
        rows = self._conn.execute(
            """SELECT * FROM merge_queue
               WHERE status = 'pending'
               ORDER BY similarity DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def merge_queue_approve(self, item_id: int) -> None:
        """Approve a merge proposal."""
        self._conn.execute(
            "UPDATE merge_queue SET status = 'approved' WHERE id = ?",
            (item_id,),
        )
        self._conn.commit()

    def merge_queue_reject(self, item_id: int) -> None:
        """Reject a merge proposal."""
        self._conn.execute(
            "UPDATE merge_queue SET status = 'rejected' WHERE id = ?",
            (item_id,),
        )
        self._conn.commit()

    def merge_queue_clear(self) -> int:
        """Clear all pending merges. Returns count cleared."""
        cursor = self._conn.execute(
            "DELETE FROM merge_queue WHERE status = 'pending'",
        )
        self._conn.commit()
        return cursor.rowcount

    # ─── Embeddings ───────────────────────────────────────────────────────────

    def embedding_set(self, content_hash: str, embedding: list[float]) -> None:
        """Store an embedding vector for a content hash."""
        import struct
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        now = time.time()
        self._conn.execute(
            """INSERT OR REPLACE INTO embeddings
               (content_hash, embedding, updated_at)
               VALUES (?, ?, ?)""",
            (content_hash, blob, now),
        )
        self._conn.commit()

    def embedding_get(self, content_hash: str) -> list[float] | None:
        """Retrieve an embedding vector, or None if not found."""
        import struct
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if not row:
            return None
        blob = row["embedding"]
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    def embedding_find_top_match(self, content_hash: str, min_similarity: float = 0.92) -> tuple[str, float] | None:
        """Return (content_hash, cosine_similarity) for the best matching existing embedding."""
        import math
        import struct
        target = self.embedding_get(content_hash)
        if target is None:
            return None
        rows = self._conn.execute(
            "SELECT content_hash, embedding FROM embeddings WHERE content_hash != ?",
            (content_hash,),
        ).fetchall()
        best_match: tuple[str, float] | None = None
        for row in rows:
            blob = row["embedding"]
            n = len(blob) // 4
            emb = list(struct.unpack(f"{n}f", blob))
            dot = sum(a * b for a, b in zip(target, emb))
            norm_target = math.sqrt(sum(a * a for a in target))
            norm_emb = math.sqrt(sum(b * b for b in emb))
            if norm_target == 0 or norm_emb == 0:
                continue
            sim = dot / (norm_target * norm_emb)
            if sim >= min_similarity:
                if best_match is None or sim > best_match[1]:
                    best_match = (row["content_hash"], sim)
        return best_match

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get content store statistics."""
        urls = self._conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) as ok, "
            "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed "
            "FROM urls"
        ).fetchone()
        content = self._conn.execute("SELECT COUNT(*) as total FROM content").fetchone()
        dlq = self._conn.execute(
            "SELECT COUNT(*) as total FROM dead_letter_queue WHERE status='pending'"
        ).fetchone()
        reviews = self._conn.execute(
            "SELECT COUNT(*) as total FROM pending_reviews WHERE status='pending'"
        ).fetchone()
        return {
            "urls_total": urls["total"] or 0,
            "urls_ok": urls["ok"] or 0,
            "urls_failed": urls["failed"] or 0,
            "content_total": content["total"] or 0,
            "dlq_pending": dlq["total"] or 0,
            "reviews_pending": reviews["total"] or 0,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @classmethod
    def open(cls, extract_dir: Path) -> ContentStore:
        """Open or create the content store in the extract directory."""
        return cls(extract_dir / "store.db")

    @classmethod
    def open_vault_cache(cls, vault_path: Path) -> ContentStore:
        """Open or create a persistent vault cache in Meta/Scripts/cache.db.

        This cache persists across pipeline runs and stores:
        - File mtime indices for incremental lint/reindex
        - Wikilink graph for fast orphan/link checks
        - Tag registry metadata
        """
        cache_dir = vault_path / "Meta" / "Scripts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cls(cache_dir / "cache.db")
