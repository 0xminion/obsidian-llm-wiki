"""Tests for structured telemetry."""

import json

import pytest

from pipeline.telemetry import StageEvent, TelemetrySink, record_stage


def test_record_stage_writes_ok_event(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = TelemetrySink(path)

    with record_stage(sink, "stage.one", count=1) as event:
        event["result"] = "created"

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["stage"] == "stage.one"
    assert payload["status"] == "ok"
    assert payload["details"] == {"count": 1, "result": "created"}
    assert payload["duration_s"] >= 0


def test_record_stage_writes_error_event_and_reraises(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = TelemetrySink(path)

    with pytest.raises(RuntimeError, match="boom"):
        with record_stage(sink, "stage.fail"):
            raise RuntimeError("boom")

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["stage"] == "stage.fail"
    assert payload["status"] == "error"
    assert payload["details"]["error_type"] == "RuntimeError"
    assert payload["details"]["error"] == "boom"


def test_telemetry_write_failure_is_best_effort(monkeypatch, tmp_path):
    sink = TelemetrySink(tmp_path / "events.jsonl")

    def fail_open(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(type(sink.path), "open", fail_open)

    sink.emit(StageEvent(stage="stage.best_effort", status="ok", duration_s=0.1))


def test_record_stage_preserves_original_exception_when_error_telemetry_fails(monkeypatch, tmp_path):
    sink = TelemetrySink(tmp_path / "events.jsonl")

    def fail_open(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(type(sink.path), "open", fail_open)

    with pytest.raises(RuntimeError, match="real failure"):
        with record_stage(sink, "stage.fail"):
            raise RuntimeError("real failure")
