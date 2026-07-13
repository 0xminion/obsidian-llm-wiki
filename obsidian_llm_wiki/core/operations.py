"""Durable, local operation history for resumable CLI work.

The store deliberately lives under a vault's ``.llmwiki`` directory.  It is
not telemetry: it only records source identifiers, lifecycle state, bounded
errors, and timestamps needed to resume interrupted ingest work.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = ["OperationRecord", "OperationStatus", "OperationStore"]

_MAX_OPERATIONS = 500
_MAX_HISTORY = 20
_MAX_ERROR_CHARS = 1_000
ResumeSource = dict[str, str]


class OperationStatus(StrEnum):
    """Lifecycle states for one source operation."""

    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class OperationRecord:
    """A durable lifecycle record for one URL or clipping source."""

    id: str
    run_id: str
    source: str
    source_kind: str = "url"
    mode: str = "ingest"
    status: OperationStatus = OperationStatus.PLANNED
    attempt: int = 1
    created_at: str = ""
    updated_at: str = ""
    title: str = ""
    bytes_extracted: int = 0
    output_file: str = ""
    error: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        source: str,
        source_kind: str = "url",
        mode: str = "ingest",
    ) -> OperationRecord:
        now = _now()
        return cls(
            id=f"op-{uuid.uuid4().hex}",
            run_id=run_id,
            source=source,
            source_kind=source_kind,
            mode=mode,
            created_at=now,
            updated_at=now,
            history=[{"at": now, "status": OperationStatus.PLANNED.value}],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationRecord:
        status = OperationStatus(data.get("status", OperationStatus.PLANNED.value))
        history = data.get("history", [])
        return cls(
            id=str(data.get("id", "")),
            run_id=str(data.get("run_id", "")),
            source=str(data.get("source", "")),
            source_kind=str(data.get("source_kind", "url")),
            mode=str(data.get("mode", "ingest")),
            status=status,
            attempt=max(1, int(data.get("attempt", 1))),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            title=str(data.get("title", "")),
            bytes_extracted=max(0, int(data.get("bytes_extracted", 0))),
            output_file=str(data.get("output_file", "")),
            error=str(data.get("error", ""))[:_MAX_ERROR_CHARS],
            history=[event for event in history if isinstance(event, dict)][-_MAX_HISTORY:],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


class OperationStore:
    """Append/update source operation records and manage one resume marker.

    Every read-modify-write operation is protected by a vault-local advisory
    lock. Atomic replacement keeps readers from observing partial JSON while
    the lock prevents independent ingest processes from losing each other's
    history updates.
    """

    def __init__(self, llmwiki_dir: str | Path) -> None:
        self.llmwiki_dir = Path(llmwiki_dir)
        self.path = self.llmwiki_dir / "operations.json"
        self.resume_path = self.llmwiki_dir / "ingest-resume.json"
        self.lock_path = self.llmwiki_dir / "operations.lock"

    def save(self, record: OperationRecord) -> None:
        """Persist a record atomically, retaining a bounded local history."""
        with self._locked(write=True):
            records = self._read_records()
            records = [existing for existing in records if existing.id != record.id]
            records.append(record)
            records = records[-_MAX_OPERATIONS:]
            self._write_json(self.path, {"operations": [item.to_dict() for item in records]})

    def transition(
        self,
        record: OperationRecord,
        status: OperationStatus,
        *,
        error: str = "",
        title: str | None = None,
        bytes_extracted: int | None = None,
        output_file: str | None = None,
    ) -> None:
        """Advance a record and persist it immediately for crash recovery."""
        if status is OperationStatus.RETRYING:
            record.attempt += 1
        record.status = status
        record.updated_at = _now()
        record.error = error[:_MAX_ERROR_CHARS]
        if title is not None:
            record.title = title
        if bytes_extracted is not None:
            record.bytes_extracted = max(0, bytes_extracted)
        if output_file is not None:
            record.output_file = output_file
        record.history.append({"at": record.updated_at, "status": status.value})
        record.history = record.history[-_MAX_HISTORY:]
        self.save(record)

    def latest_for_source(self, source: str) -> OperationRecord | None:
        """Return the most recently updated record for an exact source identifier."""
        with self._locked(write=False):
            candidates = [record for record in self._read_records() if record.source == source]
        if not candidates:
            return None
        return max(candidates, key=lambda record: (record.updated_at, record.created_at, record.id))

    def write_resume_marker(
        self, run_id: str, sources: Sequence[Mapping[str, object] | str]
    ) -> None:
        """Record structured remaining sources after cooperative cancellation.

        String entries are accepted only to migrate legacy markers; all writes
        use structured ``source``, ``source_kind``, and ``status`` entries.
        """
        marker_sources = _normalise_resume_sources(sources)
        with self._locked(write=True):
            self._write_json(
                self.resume_path,
                {"run_id": run_id, "created_at": _now(), "sources": marker_sources},
            )

    def read_resume_marker(self) -> list[ResumeSource]:
        """Return structured resumable sources, treating malformed state as absent."""
        with self._locked(write=False):
            return self._read_resume_marker()

    def remove_resume_source(self, source: str, source_kind: str) -> None:
        """Remove one successfully resumed source without dropping pending peers."""
        with self._locked(write=True):
            run_id, marker_sources = self._read_resume_marker_payload()
            remaining = [
                item
                for item in marker_sources
                if (item["source"], item["source_kind"]) != (source, source_kind)
            ]
            if remaining:
                marker = {"created_at": _now(), "sources": remaining}
                if run_id:
                    marker["run_id"] = run_id
                self._write_json(self.resume_path, marker)
            else:
                with suppress(FileNotFoundError):
                    self.resume_path.unlink()

    def clear_resume_marker(self) -> None:
        with self._locked(write=True), suppress(FileNotFoundError):
            self.resume_path.unlink()

    @contextmanager
    def _locked(self, *, write: bool) -> Iterator[None]:
        """Lock vault history without making read-only CLI modes write state."""
        lock_file = None
        if write:
            self.llmwiki_dir.mkdir(parents=True, exist_ok=True)
            lock_file = self.lock_path.open("a+", encoding="utf-8")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif self.lock_path.exists():
            lock_file = self.lock_path.open(encoding="utf-8")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            if lock_file is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

    def _read_records(self) -> list[OperationRecord]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        raw_records = data.get("operations", []) if isinstance(data, dict) else []
        records: list[OperationRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            try:
                records.append(OperationRecord.from_dict(raw))
            except (TypeError, ValueError):
                continue
        return records

    def _read_resume_marker(self) -> list[ResumeSource]:
        return self._read_resume_marker_payload()[1]

    def _read_resume_marker_payload(self) -> tuple[str, list[ResumeSource]]:
        try:
            data = json.loads(self.resume_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "", []
        if not isinstance(data, dict):
            return "", []
        run_id = data.get("run_id", "")
        sources = data.get("sources", [])
        return (
            run_id if isinstance(run_id, str) else "",
            _normalise_resume_sources(sources) if isinstance(sources, list) else [],
        )

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _normalise_resume_sources(sources: Sequence[Mapping[str, object] | str]) -> list[ResumeSource]:
    """Validate marker entries and migrate legacy string-only entries."""
    normalised: list[ResumeSource] = []
    seen: set[tuple[str, str]] = set()
    for item in sources:
        if isinstance(item, str):
            source, source_kind, status = item, "url", OperationStatus.CANCELLED.value
        elif isinstance(item, Mapping):
            source = item.get("source", "")
            source_kind = item.get("source_kind", "url")
            status = item.get("status", OperationStatus.CANCELLED.value)
        else:
            continue
        if not isinstance(source, str) or not source:
            continue
        if not isinstance(source_kind, str) or not source_kind:
            source_kind = "url"
        if not isinstance(status, str) or not status:
            status = OperationStatus.CANCELLED.value
        key = (source, source_kind)
        if key not in seen:
            normalised.append({"source": source, "source_kind": source_kind, "status": status})
            seen.add(key)
    return normalised
