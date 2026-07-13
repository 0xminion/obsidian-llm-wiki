"""Durable contradiction records with explicit human-governed state changes.

The store records evidence and source revisions, but it deliberately contains
no conflict-resolution logic.  A contradiction can only move through the
allowed workflow; terminal decisions are never silently reopened.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "ContradictionRecord",
    "ContradictionStatus",
    "ContradictionStore",
    "InvalidStatusTransition",
    "SourceRevision",
]


class ContradictionStatus(StrEnum):
    """The complete lifecycle for a detected factual contradiction."""

    DETECTED = "detected"
    REVIEW_OK = "review_ok"
    PENDING_FIX = "pending_fix"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class InvalidStatusTransitionError(ValueError):
    """Raised when an attempted transition bypasses the review workflow."""


InvalidStatusTransition = InvalidStatusTransitionError

@dataclass(frozen=True, slots=True)
class SourceRevision:
    """An immutable source revision used as contradiction evidence."""

    source_path: str
    revision: str
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class ContradictionRecord:
    """Evidence for a factual conflict; resolution remains a human decision."""

    id: str
    summary: str
    sources: tuple[SourceRevision, ...] = ()
    evidence: tuple[str, ...] = ()
    status: ContradictionStatus = ContradictionStatus.DETECTED

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("contradiction id must not be empty")
        if not self.summary.strip():
            raise ValueError("contradiction summary must not be empty")
        try:
            status = ContradictionStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"invalid contradiction status: {self.status!r}") from exc
        sources = tuple(self.sources)
        if not all(isinstance(source, SourceRevision) for source in sources):
            raise TypeError("sources must contain SourceRevision values")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "evidence", tuple(self.evidence))


_ALLOWED_TRANSITIONS: dict[ContradictionStatus, frozenset[ContradictionStatus]] = {
    ContradictionStatus.DETECTED: frozenset({
        ContradictionStatus.REVIEW_OK,
        ContradictionStatus.PENDING_FIX,
        ContradictionStatus.RESOLVED,
        ContradictionStatus.SUPPRESSED,
    }),
    ContradictionStatus.REVIEW_OK: frozenset({
        ContradictionStatus.PENDING_FIX,
        ContradictionStatus.RESOLVED,
        ContradictionStatus.SUPPRESSED,
    }),
    ContradictionStatus.PENDING_FIX: frozenset({
        ContradictionStatus.REVIEW_OK,
        ContradictionStatus.RESOLVED,
        ContradictionStatus.SUPPRESSED,
    }),
    ContradictionStatus.RESOLVED: frozenset(),
    ContradictionStatus.SUPPRESSED: frozenset(),
}


class ContradictionStore:
    """JSON-backed storage for records and their immutable source revisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, ContradictionRecord] = {}
        self._source_revisions: dict[tuple[str, str], SourceRevision] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to load contradiction store: {self.path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("contradiction store must be a JSON object")
        for data in raw.get("source_revisions", []):
            revision = _source_revision_from_dict(data)
            self._source_revisions[(revision.source_path, revision.revision)] = revision
        for data in raw.get("records", []):
            record = _record_from_dict(data)
            if record.id in self._records:
                raise ValueError(f"duplicate contradiction id in store: {record.id}")
            self._records[record.id] = record

    def _persist(self) -> None:
        payload = {
            "records": [_record_to_dict(record) for record in self.records()],
            "source_revisions": [
                asdict(revision) for revision in self.source_revisions()
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def records(self, status: ContradictionStatus | str | None = None) -> list[ContradictionRecord]:
        """Return records in a deterministic identifier order, optionally filtered."""
        selected_status = ContradictionStatus(status) if status is not None else None
        return [
            record
            for _, record in sorted(self._records.items())
            if selected_status is None or record.status is selected_status
        ]

    def source_revisions(self, source_path: str | None = None) -> list[SourceRevision]:
        """Return stored source revisions in deterministic source/revision order."""
        return [
            revision
            for _, revision in sorted(self._source_revisions.items())
            if source_path is None or revision.source_path == source_path
        ]

    def get(self, record_id: str) -> ContradictionRecord:
        """Return one record or raise ``KeyError`` for an unknown identifier."""
        return self._records[record_id]

    def add_source_revision(self, revision: SourceRevision) -> SourceRevision:
        """Persist a source revision even when it produces no contradiction."""
        key = (revision.source_path, revision.revision)
        existing = self._source_revisions.get(key)
        if existing is not None:
            if existing != revision:
                raise ValueError(f"source revision already exists with different evidence: {key}")
            return existing
        self._source_revisions[key] = revision
        self._persist()
        return revision

    def add(self, record: ContradictionRecord) -> ContradictionRecord:
        """Persist a newly detected record and all source revisions it cites."""
        if record.id in self._records:
            raise ValueError(f"contradiction record already exists: {record.id}")
        self._records[record.id] = record
        for revision in record.sources:
            self._source_revisions[(revision.source_path, revision.revision)] = revision
        self._persist()
        return record

    def transition(
        self,
        record_id: str,
        new_status: ContradictionStatus | str,
    ) -> ContradictionRecord:
        """Persist an allowed status transition without resolving any content."""
        record = self.get(record_id)
        try:
            target = ContradictionStatus(new_status)
        except ValueError as exc:
            raise ValueError(f"invalid contradiction status: {new_status!r}") from exc
        if target is record.status:
            return record
        if target not in _ALLOWED_TRANSITIONS[record.status]:
            raise InvalidStatusTransition(
                f"invalid contradiction transition: {record.status} -> {target}"
            )
        updated = replace(record, status=target)
        self._records[record_id] = updated
        self._persist()
        return updated

    add_record = add
    transition_status = transition


def _source_revision_from_dict(data: Any) -> SourceRevision:
    if not isinstance(data, dict):
        raise ValueError("source revision must be an object")
    return SourceRevision(
        source_path=str(data.get("source_path", "")),
        revision=str(data.get("revision", "")),
        content_hash=str(data.get("content_hash", "")),
    )


def _record_from_dict(data: Any) -> ContradictionRecord:
    if not isinstance(data, dict):
        raise ValueError("contradiction record must be an object")
    sources = tuple(_source_revision_from_dict(item) for item in data.get("sources", []))
    evidence = tuple(str(item) for item in data.get("evidence", []))
    return ContradictionRecord(
        id=str(data.get("id", "")),
        summary=str(data.get("summary", "")),
        sources=sources,
        evidence=evidence,
        status=data.get("status", ContradictionStatus.DETECTED),
    )


def _record_to_dict(record: ContradictionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "summary": record.summary,
        "sources": [asdict(source) for source in record.sources],
        "evidence": list(record.evidence),
        "status": record.status.value,
    }
