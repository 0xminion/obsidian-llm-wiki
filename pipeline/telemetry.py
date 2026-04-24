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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

log = logging.getLogger(__name__)

_SENSITIVE_QUERY_KEYS = {
    "token", "access_token", "auth", "authorization", "signature", "sig",
    "key", "api_key", "apikey", "password", "secret", "x-amz-signature",
}


def redact_url(url: str) -> str:
    """Redact common sensitive query parameters while preserving correlation value."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "[invalid-url]"
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _SENSITIVE_QUERY_KEYS or "token" in key.lower() or "secret" in key.lower():
            query.append((key, "[REDACTED]"))
        else:
            query.append((key, value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), ""))


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


def read_recent_events(path: Path, limit: int = 20) -> list[dict[str, object]]:
    """Read the most recent valid telemetry events from a JSONL file."""
    if limit < 1 or not path.exists():
        return []
    events: deque[dict[str, object]] = deque(maxlen=limit)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return list(events)
