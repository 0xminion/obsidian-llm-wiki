"""Dirty queue contracts exercised against independent durable SQLite connections."""

from pathlib import Path

import pytest

from obsidian_llm_wiki.core.registry import VaultRegistry

FUTURE = "2999-01-01T00:00:00+00:00"
PAST = "2000-01-01T00:00:00+00:00"


def test_ordinary_reenqueue_preserves_retry_metadata_after_reopen(tmp_path: Path) -> None:
    directory = tmp_path / ".llmwiki"
    with VaultRegistry(directory) as worker:
        record_id = worker.enqueue_dirty("source", "retry.md", "synthesis", {"revision": 1})
        assert worker.start_dirty(record_id, worker_id="pipeline:first")
        assert worker.fail_dirty(
            record_id, "provider unavailable", worker_id="pipeline:first", next_attempt_at=FUTURE
        )
        original = worker.list_dirty()[0]

    with VaultRegistry(directory) as producer, VaultRegistry(directory) as worker:
        assert producer.enqueue_dirty(
            "source", "retry.md", "synthesis", {"revision": 2}, next_attempt_at=PAST
        ) == record_id
        pending = worker.list_dirty()[0]
        assert pending.next_attempt_at == original.next_attempt_at
        assert pending.attempts == original.attempts == 1
        assert pending.error == original.error == "provider unavailable"
        assert pending.payload == {"revision": 2}
        assert worker.claim_dirty(limit=10, worker_id="pipeline:second") == []
        assert not worker.start_dirty(record_id, worker_id="pipeline:second")


def test_completion_of_old_generation_keeps_latest_invalidation(tmp_path: Path) -> None:
    directory = tmp_path / ".llmwiki"
    with VaultRegistry(directory) as worker, VaultRegistry(directory) as producer:
        record_id = producer.enqueue_dirty("source", "changed.md", "synthesis", {"revision": 1})
        first = worker.claim_dirty(limit=1, worker_id="pipeline:first")[0]
        assert producer.enqueue_dirty(
            "source", "changed.md", "synthesis", {"revision": 2}
        ) == record_id
        assert worker.list_dirty(state="running")[0].payload == first.payload == {"revision": 1}
        producer.enqueue_dirty("source", "changed.md", "synthesis", {"revision": 3})
        assert producer.claim_dirty(limit=1, worker_id="pipeline:competing") == []
        assert worker.complete_dirty(record_id, worker_id="pipeline:first")
        assert not worker.complete_dirty(record_id, worker_id="pipeline:first")

    with VaultRegistry(directory) as restarted:
        pending = restarted.list_dirty()
        assert len(pending) == 1
        assert pending[0].payload == {"revision": 3}
        assert pending[0].attempts == 0
        assert pending[0].error == ""
        assert restarted.list_dirty(state="completed") == []
        claimed = restarted.claim_dirty(limit=1, worker_id="pipeline:second")
        assert claimed[0].payload == {"revision": 3}
        assert claimed[0].attempts == 1
        assert not restarted.fail_dirty(record_id, "late", worker_id="pipeline:first")
        assert not restarted.complete_dirty(record_id, worker_id="pipeline:first")
        assert restarted.complete_dirty(record_id, worker_id="pipeline:second")
        assert restarted.list_dirty() == []


@pytest.mark.parametrize("retire", ["fail", "release", "expired-claim", "expired-start"])
def test_interrupted_generation_promotes_newest_payload(tmp_path: Path, retire: str) -> None:
    directory = tmp_path / ".llmwiki"
    with VaultRegistry(directory) as old, VaultRegistry(directory) as producer:
        record_id = old.enqueue_dirty("source", "changed.md", "synthesis", {"revision": 1})
        assert old.start_dirty(record_id, worker_id="pipeline:old")
        producer.enqueue_dirty("source", "changed.md", "synthesis", {"revision": 2})
        if retire == "fail":
            assert old.fail_dirty(record_id, "old failed", worker_id="pipeline:old")
        elif retire == "release":
            assert old.release_dirty_worker("pipeline:old") == 1
        else:
            with producer.connection:
                producer.connection.execute(
                    "UPDATE dirty_queue SET lease_expires_at=? WHERE id=?", (PAST, record_id)
                )

    with VaultRegistry(directory) as restarted:
        if retire == "expired-start":
            assert restarted.start_dirty(record_id, worker_id="pipeline:new")
            claimed = restarted.list_dirty(state="running")
        else:
            claimed = restarted.claim_dirty(limit=1, worker_id="pipeline:new")
        assert len(claimed) == 1
        assert claimed[0].payload == {"revision": 2}
        assert claimed[0].attempts == 1
        assert claimed[0].error == ""
        assert not restarted.complete_dirty(record_id, worker_id="pipeline:old")
        assert not restarted.fail_dirty(record_id, "stale", worker_id="pipeline:old")
        assert restarted.release_dirty_worker("pipeline:old") == 0
        assert restarted.complete_dirty(record_id, worker_id="pipeline:new")
        assert restarted.list_dirty() == []


@pytest.mark.parametrize("finish", ["complete", "fail", "fail-explicit"])
def test_expired_owner_is_fenced_even_before_reclaim(tmp_path: Path, finish: str) -> None:
    with VaultRegistry(tmp_path) as old, VaultRegistry(tmp_path) as other:
        record_id = old.enqueue_dirty("source", "expired.md", "synthesis")
        assert old.start_dirty(record_id, worker_id="pipeline:old")
        other.enqueue_dirty("source", "expired.md", "synthesis", {"revision": 2})
        with other.connection:
            other.connection.execute(
                "UPDATE dirty_queue SET lease_expires_at=? WHERE id=?", (PAST, record_id)
            )
        if finish == "complete":
            assert not old.complete_dirty(record_id, worker_id="pipeline:old")
        else:
            assert not old.fail_dirty(
                record_id, "too late", worker_id="pipeline:old",
                next_attempt_at=FUTURE if finish == "fail-explicit" else None,
            )
        claimed = other.claim_dirty(limit=1, worker_id="pipeline:new")
        assert claimed[0].payload == {"revision": 2}


@pytest.mark.parametrize("claim", ["start", "batch"])
def test_expired_lease_respects_successor_due_time(tmp_path: Path, claim: str) -> None:
    with VaultRegistry(tmp_path) as old, VaultRegistry(tmp_path) as new:
        record_id = old.enqueue_dirty("source", "scheduled.md", "synthesis")
        assert old.start_dirty(record_id, worker_id="pipeline:old")
        new.enqueue_dirty(
            "source", "scheduled.md", "synthesis", {"revision": 2}, next_attempt_at=FUTURE
        )
        with new.connection:
            new.connection.execute(
                "UPDATE dirty_queue SET lease_expires_at=? WHERE id=?", (PAST, record_id)
            )
        if claim == "start":
            assert not new.start_dirty(record_id, worker_id="pipeline:new")
        else:
            assert new.claim_dirty(limit=1, worker_id="pipeline:new") == []
        assert old.release_dirty_worker("pipeline:old") == 1
        assert new.list_dirty()[0].payload == {"revision": 2}
        assert new.list_dirty()[0].next_attempt_at == FUTURE


@pytest.mark.parametrize("claim", ["start", "batch"])
def test_lease_token_fences_reused_worker_identity(tmp_path: Path, claim: str) -> None:
    with VaultRegistry(tmp_path) as old, VaultRegistry(tmp_path) as new:
        record_id = old.enqueue_dirty("source", "same-worker.md", "synthesis")
        if claim == "start":
            assert old.start_dirty(record_id, worker_id="worker", lease_token="first")
        else:
            assert old.claim_dirty(limit=1, worker_id="worker", lease_token="first")
        new.enqueue_dirty("source", "same-worker.md", "synthesis", {"revision": 2})
        with new.connection:
            new.connection.execute(
                "UPDATE dirty_queue SET lease_expires_at=? WHERE id=?", (PAST, record_id)
            )
        if claim == "start":
            assert new.start_dirty(record_id, worker_id="worker", lease_token="second")
        else:
            assert new.claim_dirty(limit=1, worker_id="worker", lease_token="second")
        assert not old.complete_dirty(record_id, worker_id="worker", lease_token="first")
        assert not old.fail_dirty(record_id, "late", worker_id="worker", lease_token="first")
        # Omitting a token must not bypass a token-protected claim.
        assert not old.complete_dirty(record_id, worker_id="worker")
        assert not old.fail_dirty(record_id, "late", worker_id="worker", next_attempt_at=FUTURE)
        assert new.complete_dirty(record_id, worker_id="worker", lease_token="second")
        assert new.list_dirty() == []
