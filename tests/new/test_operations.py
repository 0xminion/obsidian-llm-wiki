"""Durable source-operation records and resumable cancellation markers."""

from __future__ import annotations

import json
import multiprocessing

from obsidian_llm_wiki.core.operations import (
    OperationRecord,
    OperationStatus,
    OperationStore,
)
from obsidian_llm_wiki.core.registry import VaultRegistry


def _save_many(llmwiki_dir: str, worker: int, start: multiprocessing.synchronize.Event) -> None:
    """Worker used by the interprocess lock regression test."""
    store = OperationStore(llmwiki_dir)
    start.wait(timeout=10)
    for item in range(20):
        store.save(OperationRecord.create(run_id=f"run-{worker}", source=f"source-{worker}-{item}"))


def _remove_resume_source(
    llmwiki_dir: str, source: str, start: multiprocessing.synchronize.Event
) -> None:
    store = OperationStore(llmwiki_dir)
    start.wait(timeout=10)
    store.remove_resume_source(source, "url")


def test_operation_store_persists_source_lifecycle_and_resume_marker(tmp_path) -> None:
    store = OperationStore(tmp_path / ".llmwiki")
    record = OperationRecord.create(run_id="run-1", source="https://example.test/a")

    store.save(record)
    store.transition(record, OperationStatus.RUNNING)
    store.transition(record, OperationStatus.CANCELLED, error="interrupted")
    store.write_resume_marker(
        "run-1",
        [
            {"source": record.source, "source_kind": "url", "status": "cancelled"},
            {"source": "clip.md", "source_kind": "clipping", "status": "cancelled"},
        ],
    )

    persisted = json.loads((tmp_path / ".llmwiki" / "operations.json").read_text(encoding="utf-8"))
    assert persisted["operations"][0]["source"] == "https://example.test/a"
    assert persisted["operations"][0]["status"] == "cancelled"
    assert persisted["operations"][0]["attempt"] == 1
    assert store.read_resume_marker() == [
        {"source": "https://example.test/a", "source_kind": "url", "status": "cancelled"},
        {"source": "clip.md", "source_kind": "clipping", "status": "cancelled"},
    ]


def test_transition_to_retrying_increments_attempt_and_preserves_history(tmp_path) -> None:
    store = OperationStore(tmp_path / ".llmwiki")
    record = OperationRecord.create(run_id="run-1", source="https://example.test/a")
    store.save(record)
    store.transition(record, OperationStatus.FAILED, error="timeout")

    store.transition(record, OperationStatus.RETRYING)

    assert record.status is OperationStatus.RETRYING
    assert record.attempt == 2
    assert record.error == ""
    assert [event["status"] for event in record.history] == [
        "planned",
        "failed",
        "retrying",
    ]


def test_operation_store_serializes_concurrent_process_saves(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [
        context.Process(target=_save_many, args=(str(tmp_path / ".llmwiki"), worker, start))
        for worker in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=20)

    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    payload = json.loads((tmp_path / ".llmwiki" / "operations.json").read_text(encoding="utf-8"))
    assert len(payload["operations"]) == 80


def test_corrupt_legacy_operations_file_is_marked_as_inspected(tmp_path) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    llmwiki_dir.mkdir()
    operations_path = llmwiki_dir / "operations.json"
    operations_path.write_text("{not-json", encoding="utf-8")

    with VaultRegistry(llmwiki_dir) as registry:
        marker = registry.connection.execute(
            "SELECT value FROM meta WHERE key='legacy_operations_migration_complete'"
        ).fetchone()
    assert marker is not None
    assert tuple(marker) == ("invalid-json",)

    operations_path.write_text(
        json.dumps(
            {
                "operations": [
                    OperationRecord.create(
                        run_id="late-valid-file", source="https://example.test/late"
                    ).to_dict()
                ]
            }
        ),
        encoding="utf-8",
    )
    with VaultRegistry(llmwiki_dir) as registry:
        assert registry.latest_operation_for_source("https://example.test/late") is None


def test_resume_marker_removal_is_interprocess_safe_and_preserves_run_id(tmp_path) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    store = OperationStore(llmwiki_dir)
    sources = [
        {"source": f"https://example.test/{item}", "source_kind": "url", "status": "cancelled"}
        for item in range(8)
    ]
    store.write_resume_marker("run-1", sources)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    workers = [
        context.Process(
            target=_remove_resume_source,
            args=(str(llmwiki_dir), source["source"], start),
        )
        for source in sources[:-1]
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=20)

    assert [worker.exitcode for worker in workers] == [0] * len(workers)
    marker = json.loads((llmwiki_dir / "ingest-resume.json").read_text(encoding="utf-8"))
    assert marker["run_id"] == "run-1"
    assert marker["sources"] == [sources[-1]]


def test_legacy_operations_migrate_once_and_remain_queryable(tmp_path) -> None:
    llmwiki_dir = tmp_path / ".llmwiki"
    llmwiki_dir.mkdir()
    record = OperationRecord.create(run_id="legacy-run", source="legacy-source")
    (llmwiki_dir / "operations.json").write_text(
        json.dumps({"operations": [record.to_dict()]}),
        encoding="utf-8",
    )

    with VaultRegistry(llmwiki_dir) as registry:
        assert registry.latest_operation_for_source("legacy-source") is not None

    second = OperationRecord.create(run_id="late-run", source="late-json-only")
    (llmwiki_dir / "operations.json").write_text(
        json.dumps({"operations": [record.to_dict(), second.to_dict()]}),
        encoding="utf-8",
    )
    with VaultRegistry(llmwiki_dir) as registry:
        assert registry.latest_operation_for_source("legacy-source") is not None
        assert registry.latest_operation_for_source("late-json-only") is None


def test_operation_rows_archive_instead_of_disappearing(
    tmp_path,
    monkeypatch,
) -> None:
    import obsidian_llm_wiki.core.registry as registry_module

    monkeypatch.setattr(registry_module, "_LIVE_OPERATION_HISTORY", 2)
    store = OperationStore(tmp_path / ".llmwiki")
    records = [
        OperationRecord.create(run_id=f"run-{index}", source=f"source-{index}")
        for index in range(3)
    ]
    for index, record in enumerate(records):
        record.created_at = f"2026-01-01T00:00:0{index}+00:00"
        record.updated_at = record.created_at
        store.save(record)

    with VaultRegistry(tmp_path / ".llmwiki") as registry:
        live = registry.connection.execute("SELECT count(*) FROM operation").fetchone()[0]
        archived = registry.connection.execute(
            "SELECT count(*) FROM operation_archive"
        ).fetchone()[0]
        assert (live, archived) == (2, 1)
        assert registry.latest_operation_for_source("source-0") is not None
