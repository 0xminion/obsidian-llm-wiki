"""Regression coverage for post-review quality recommendations."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.compile import _build_edges
from pipeline.config import Config
from pipeline.doctor import run_doctor
from pipeline.fixtures import create_example_vault
from pipeline.release import check_release_hygiene
from pipeline.stats import collect_stats, generate_dashboard
from pipeline.telemetry import read_recent_events, redact_url

runner = CliRunner()


def test_fixture_generator_creates_deterministic_valid_vault(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    summary1 = create_example_vault(first)
    summary2 = create_example_vault(second)

    assert summary1["entries"] == 1
    assert summary1 == summary2
    rels1 = sorted(str(p.relative_to(first)) for p in first.rglob("*") if p.is_file())
    rels2 = sorted(str(p.relative_to(second)) for p in second.rglob("*") if p.is_file())
    assert rels1 == rels2
    for rel in rels1:
        assert (first / rel).read_text(encoding="utf-8") == (second / rel).read_text(encoding="utf-8")


def test_doctor_reports_config_and_structure_without_leaking_secrets(tmp_path: Path):
    create_example_vault(tmp_path)
    cfg = Config(vault_path=tmp_path)
    cfg.llm_api_key = "super-secret-token"
    cfg.transcript_api_key = "transcript-secret"
    cfg.supadata_api_key = "supadata-secret"
    cfg.assemblyai_api_key = "assembly-secret"

    report = run_doctor(cfg)

    assert report["ok"] is True
    assert report["checks"]
    rendered = json.dumps(report)
    for secret in ["super-secret-token", "transcript-secret", "supadata-secret", "assembly-secret"]:
        assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_stats_json_collects_graph_semantics_and_counts(tmp_path: Path):
    create_example_vault(tmp_path)
    cfg = Config(vault_path=tmp_path)

    stats = collect_stats(cfg)
    dashboard = generate_dashboard(cfg)

    assert stats["total"] >= 4
    assert stats["graph"]["edge_types"]
    assert stats["graph"]["node_types"] == ["concept", "entry", "moc", "source"]
    _build_edges(cfg)
    edges = cfg.edges_file.read_text(encoding="utf-8")
    assert "research-builder-signals\tresearch-builder-signals-source\trelates_to" in edges
    assert "Graph Semantics" in dashboard
    assert "source notes are first-class nodes" in dashboard


def test_json_cli_modes_for_stats_doctor_and_review_status(tmp_path: Path):
    create_example_vault(tmp_path)

    stats = runner.invoke(app, ["stats", str(tmp_path), "--json"])
    assert stats.exit_code == 0, stats.stdout
    assert json.loads(stats.stdout)["entries"] == 1

    doctor = runner.invoke(app, ["doctor", str(tmp_path), "--json"])
    assert doctor.exit_code == 0, doctor.stdout
    assert json.loads(doctor.stdout)["ok"] is True

    config_doctor = runner.invoke(app, ["config-doctor", str(tmp_path), "--json"])
    assert config_doctor.exit_code == 0, config_doctor.stdout
    assert json.loads(config_doctor.stdout)["config"]

    reviews = runner.invoke(app, ["review-status", str(tmp_path), "--json"])
    assert reviews.exit_code == 0, reviews.stdout
    assert json.loads(reviews.stdout)["pending"] == 0


def test_telemetry_recent_events_and_extraction_attempts_are_structured(tmp_path: Path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        '{"timestamp":"2026-01-01T00:00:00+00:00","stage":"extract","status":"ok","duration_s":0.1,"details":{"url":"https://example.com","attempt":1,"source_type":"web"}}\n',
        encoding="utf-8",
    )

    events = read_recent_events(telemetry, limit=1)

    assert events == [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "stage": "extract",
            "status": "ok",
            "duration_s": 0.1,
            "details": {"url": "https://example.com", "attempt": 1, "source_type": "web"},
        }
    ]
    assert redact_url("https://example.com/a?token=secret&ok=1") == "https://example.com/a?token=%5BREDACTED%5D&ok=1"


def test_release_hygiene_detects_version_and_docs_alignment():
    report = check_release_hygiene(Path.cwd())

    assert report["ok"] is True
    assert report["version"]
    assert any(check["name"] == "docs_index" for check in report["checks"])
