"""Tests for durable contradiction records and guarded status transitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.core.contradictions import (
    ContradictionRecord,
    ContradictionStatus,
    ContradictionStore,
    InvalidStatusTransition,
    SourceRevision,
)


def _record() -> ContradictionRecord:
    return ContradictionRecord(
        id="conflict-1",
        summary="Sources disagree about the date.",
        sources=(
            SourceRevision("sources/a.md", "a@1", "aaa"),
            SourceRevision("sources/b.md", "b@2", "bbb"),
        ),
        evidence=("A says 2020", "B says 2021"),
    )


def test_store_persists_records_and_source_revisions_atomically(tmp_path: Path) -> None:
    state_path = tmp_path / ".llmwiki" / "contradictions.json"
    store = ContradictionStore(state_path)
    record = _record()

    store.add(record)

    assert state_path.is_file()
    assert not list(state_path.parent.glob("*.tmp"))
    restored = ContradictionStore(state_path)
    assert restored.get("conflict-1") == record
    assert restored.source_revisions() == list(record.sources)


def test_store_allows_guarded_human_status_transitions(tmp_path: Path) -> None:
    store = ContradictionStore(tmp_path / "contradictions.json")
    store.add(_record())

    pending = store.transition("conflict-1", ContradictionStatus.PENDING_FIX)
    resolved = store.transition("conflict-1", "resolved")

    assert pending.status is ContradictionStatus.PENDING_FIX
    assert resolved.status is ContradictionStatus.RESOLVED
    assert ContradictionStore(tmp_path / "contradictions.json").get("conflict-1") == resolved


def test_store_rejects_reopening_terminal_status_without_new_detection(tmp_path: Path) -> None:
    store = ContradictionStore(tmp_path / "contradictions.json")
    store.add(_record())
    store.transition("conflict-1", ContradictionStatus.SUPPRESSED)

    with pytest.raises(InvalidStatusTransition, match="suppressed.*detected"):
        store.transition("conflict-1", ContradictionStatus.DETECTED)


def test_record_rejects_unknown_status_and_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="status"):
        ContradictionRecord(id="bad", summary="Bad status", status="unknown")  # type: ignore[arg-type]

    store = ContradictionStore(tmp_path / "contradictions.json")
    store.add(_record())
    with pytest.raises(ValueError, match="already exists"):
        store.add(_record())
