"""Force retry and exclusive-compile-lock restart recovery contracts."""

import subprocess
import sys
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.registry import VaultRegistry

FUTURE = "2999-01-01T00:00:00+00:00"


def test_force_reset_is_explicit_and_only_resets_pending_retry(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path) as registry, VaultRegistry(tmp_path) as observer:
        record_id = registry.enqueue_dirty("source", "retry.md", "synthesis")
        assert registry.start_dirty(record_id, worker_id="pipeline:first")
        assert registry.fail_dirty(
            record_id, "old error", worker_id="pipeline:first", next_attempt_at=FUTURE
        )
        assert registry.enqueue_dirty(
            "source", "retry.md", "synthesis", {"revision": 2}, reset_retry=True
        ) == record_id
        reset = observer.list_dirty()[0]
        assert reset.attempts == 0
        assert reset.error == ""
        assert reset.next_attempt_at != FUTURE
        assert observer.start_dirty(record_id, worker_id="pipeline:second")
        assert observer.complete_dirty(record_id, worker_id="pipeline:second")


def test_force_cannot_mutate_or_steal_an_active_claim(tmp_path: Path) -> None:
    with VaultRegistry(tmp_path) as worker, VaultRegistry(tmp_path) as producer:
        record_id = worker.enqueue_dirty("source", "running.md", "synthesis", {"revision": 1})
        assert worker.start_dirty(record_id, worker_id="pipeline:old")
        first = worker.list_dirty(state="running")[0]
        producer.enqueue_dirty(
            "source", "running.md", "synthesis", {"revision": 2}, next_attempt_at=FUTURE
        )
        producer.enqueue_dirty(
            "source", "running.md", "synthesis", {"revision": 3}, reset_retry=True
        )
        assert worker.list_dirty(state="running")[0] == first
        assert not producer.start_dirty(record_id, worker_id="pipeline:forced")
        assert worker.complete_dirty(record_id, worker_id="pipeline:old")
        claimed = producer.claim_dirty(limit=1, worker_id="pipeline:forced")
        assert claimed[0].payload == {"revision": 3}
        assert claimed[0].attempts == 1


@pytest.mark.parametrize("successor", [False, True])
def test_abandoned_pipeline_release_only_matches_exact_prefix(
    tmp_path: Path, successor: bool
) -> None:
    owners = ["pipeline:old", "pipeline:other", "ingest:alive", "run-42",
              "pipeline", "Pipeline:alive", "xpipeline:alive", "pipeline-other"]
    with VaultRegistry(tmp_path) as registry:
        for owner in owners:
            record_id = registry.enqueue_dirty("source", owner, "work", {"revision": 1})
            assert registry.start_dirty(record_id, worker_id=owner)
            if successor:
                registry.enqueue_dirty("source", owner, "work", {"revision": 2})
        delayed_id = registry.enqueue_dirty("source", "backoff", "work")
        assert registry.start_dirty(delayed_id, worker_id="pipeline:failed")
        assert registry.fail_dirty(
            delayed_id, "backoff", worker_id="pipeline:failed", next_attempt_at=FUTURE
        )
        protected_before = [tuple(row) for row in registry.connection.execute(
            "SELECT * FROM dirty_queue WHERE claimed_by NOT GLOB 'pipeline:*' ORDER BY id"
        )]

    with VaultRegistry(tmp_path) as restarted, VaultRegistry(tmp_path) as observer:
        # The caller has acquired the exclusive compile lock before this API.
        assert restarted.release_abandoned_pipeline_workers() == 2
        assert restarted.release_abandoned_pipeline_workers() == 0
        protected_after = [tuple(row) for row in observer.connection.execute(
            "SELECT * FROM dirty_queue WHERE entity_id NOT IN ('pipeline:old', 'pipeline:other') "
            "ORDER BY id"
        )]
        assert protected_after == protected_before
        claimed = observer.claim_dirty(limit=100, worker_id="pipeline:new")
        assert {row.entity_id for row in claimed} == {"pipeline:old", "pipeline:other"}
        assert {row.attempts for row in claimed} == {1 if successor else 2}
        assert all(row.payload == {"revision": 2 if successor else 1} for row in claimed)
        for row in claimed:
            assert not restarted.complete_dirty(row.id, worker_id=row.entity_id)
            assert not restarted.fail_dirty(row.id, "late", worker_id=row.entity_id)
            assert restarted.release_dirty_worker(row.entity_id) == 0


def test_hard_exit_pipeline_claim_is_recoverable_without_waiting_for_ttl(tmp_path: Path) -> None:
    process = subprocess.run(
        [sys.executable, "-c", """
import os
import sys
from obsidian_llm_wiki.core.registry import VaultRegistry
registry = VaultRegistry(sys.argv[1])
record_id = registry.enqueue_dirty('source', 'crashed.md', 'work', {'revision': 1})
assert registry.start_dirty(record_id, worker_id='pipeline:killed')
os._exit(9)
""", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode == 9, process.stderr
    with VaultRegistry(tmp_path) as restarted:
        assert restarted.claim_dirty(limit=1, worker_id="pipeline:new") == []
        restarted.enqueue_dirty("source", "crashed.md", "work", {"revision": 2})
        assert restarted.release_abandoned_pipeline_workers() == 1
        claimed = restarted.claim_dirty(limit=1, worker_id="pipeline:new")
        assert claimed[0].payload == {"revision": 2}
        assert restarted.complete_dirty(claimed[0].id, worker_id="pipeline:new")
