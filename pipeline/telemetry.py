"""Structured pipeline telemetry helpers.

The pipeline already has human-readable logs. This module adds low-friction JSONL
stage events so automation can answer: what ran, how long did it take, and did it
fail? It is deliberately dependency-free and safe to disable by not passing a log
path.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass
class StageEvent:
    """A single machine-readable stage telemetry event."""

    stage: str
    status: str
    duration_s: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "stage": self.stage,
                "status": self.status,
                "duration_s": round(self.duration_s, 3),
                "details": self.details,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class TelemetrySink:
    """Append-only JSONL telemetry sink."""

    def __init__(self, path: Path | None):
        self.path = path

    def emit(self, event: StageEvent) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
        except OSError:
            log.debug("Failed to write pipeline telemetry", exc_info=True)


@contextmanager
def record_stage(
    sink: TelemetrySink,
    stage: str,
    **details: object,
) -> Iterator[dict[str, object]]:
    """Record a stage duration and final status.

    The yielded dict can be updated by the caller with counters before the stage
    exits. Exceptions are recorded and re-raised.
    """
    started = time.monotonic()
    stage_details = dict(details)
    try:
        yield stage_details
    except Exception as exc:
        stage_details.setdefault("error_type", type(exc).__name__)
        stage_details.setdefault("error", str(exc)[:500])
        sink.emit(StageEvent(stage=stage, status="error", duration_s=time.monotonic() - started, details=stage_details))
        raise
    else:
        status = str(stage_details.pop("status", "ok"))
        sink.emit(StageEvent(stage=stage, status=status, duration_s=time.monotonic() - started, details=stage_details))
