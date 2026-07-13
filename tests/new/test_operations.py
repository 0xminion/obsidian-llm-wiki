"""Durable source-operation records and resumable cancellation markers."""

from __future__ import annotations

import json
import multiprocessing

from obsidian_llm_wiki.core.operations import (
    OperationRecord,
    OperationStatus,
    OperationStore,
)


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
