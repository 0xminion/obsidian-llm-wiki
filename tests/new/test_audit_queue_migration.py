"""Upgrade legacy queue databases and race real SQLite writers."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from obsidian_llm_wiki.core.registry import VaultRegistry

FUTURE = "2999-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"


@pytest.mark.parametrize("with_lease", [False, True])
def test_legacy_queue_migration_preserves_rows_with_concurrent_openers(
    tmp_path: Path, with_lease: bool
) -> None:
    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta VALUES ('schema_version', '5');
            CREATE TABLE dirty_queue (
                id INTEGER PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                claimed_by TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        if with_lease:
            connection.execute(
                "ALTER TABLE dirty_queue ADD COLUMN lease_expires_at TEXT NOT NULL DEFAULT ''"
            )
        for record_id, state, owner in (
            (41, "pending", ""),
            (42, "running", "pipeline:crashed"),
            (43, "running", "ingest:alive"),
            (44, "completed", ""),
        ):
            connection.execute(
                "INSERT INTO dirty_queue(id, entity_type, entity_id, reason, payload_json, "
                "state, attempts, next_attempt_at, claimed_by, error, created_at, updated_at) "
                "VALUES(?, 'source', ?, 'work', ?, ?, 7, ?, ?, 'diagnostic', ?, ?)",
                (record_id, str(record_id), json.dumps({"revision": 1}), state,
                 FUTURE if state == "pending" else PAST, owner, PAST, PAST),
            )
        if with_lease:
            connection.execute(
                "UPDATE dirty_queue SET lease_expires_at=? WHERE state='running'", (FUTURE,)
            )
        columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(dirty_queue)")]
        original = connection.execute("SELECT * FROM dirty_queue ORDER BY id").fetchall()

    barrier = Barrier(2)

    def open_registry() -> list[tuple]:
        barrier.wait(timeout=10)
        with VaultRegistry(tmp_path) as registry:
            return [tuple(row) for row in registry.connection.execute(
                f"SELECT {', '.join(columns)} FROM dirty_queue ORDER BY id"
            )]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_registry) for _ in range(2)]
        assert all(future.result(timeout=30) == original for future in futures)

    with VaultRegistry(tmp_path) as registry:
        assert registry.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "6"
        new_fields = registry.connection.execute(
            "SELECT pending_payload_json, pending_next_attempt_at, lease_token FROM dirty_queue"
        ).fetchall()
        assert all(tuple(row) == (None, "", "") for row in new_fields)
        registry.enqueue_dirty("source", "41", "work", {"revision": 2})
        delayed = registry.list_dirty()[0]
        assert (delayed.attempts, delayed.next_attempt_at, delayed.error) == (
            7, FUTURE, "diagnostic"
        )
        registry.enqueue_dirty("source", "42", "work", {"revision": 2})
        assert registry.release_abandoned_pipeline_workers() == 1
        assert [row.entity_id for row in registry.list_dirty(state="running")] == ["43"]
        claimed = registry.claim_dirty(limit=10, worker_id="pipeline:new")
        assert len(claimed) == 1
        assert (claimed[0].id, claimed[0].payload) == (42, {"revision": 2})


@pytest.mark.parametrize("retire", ["complete", "fail", "release"])
def test_concurrent_invalidation_and_retirement_never_drop_successor(
    tmp_path: Path, retire: str
) -> None:
    with VaultRegistry(tmp_path) as registry:
        record_id = registry.enqueue_dirty("source", "race.md", "work", {"revision": 1})
        assert registry.start_dirty(record_id, worker_id="pipeline:old")
    barrier = Barrier(2)

    def invalidate() -> int:
        with VaultRegistry(tmp_path) as registry:
            barrier.wait(timeout=10)
            return registry.enqueue_dirty("source", "race.md", "work", {"revision": 2})

    def finish() -> bool:
        with VaultRegistry(tmp_path) as registry:
            barrier.wait(timeout=10)
            if retire == "complete":
                return registry.complete_dirty(record_id, worker_id="pipeline:old")
            if retire == "fail":
                return registry.fail_dirty(record_id, "old failure", worker_id="pipeline:old")
            return registry.release_dirty_worker("pipeline:old") == 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        enqueued, finished = executor.submit(invalidate), executor.submit(finish)
        assert enqueued.result(timeout=30) >= record_id
        assert finished.result(timeout=30)
    with VaultRegistry(tmp_path) as registry:
        pending = registry.list_dirty()
        assert len(pending) == 1
        assert pending[0].payload == {"revision": 2}
        assert registry.list_dirty(state="running") == []
