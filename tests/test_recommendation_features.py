from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.compile import _build_edges
from pipeline.config import Config
from pipeline.doctor import run_doctor
from pipeline.fixtures import create_example_vault
from pipeline.models import ExtractedSource
from pipeline.plan import (
    _keyword_dedup_fallback,
    _process_concept_merge_queue,
    _semantic_dedup,
)
from pipeline.release import check_release_hygiene
from pipeline.stats import collect_stats, generate_dashboard
from pipeline.store import ContentStore
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


# ─── Rec 3: Semantic Near-Duplicate Detection ────────────────────────────────



def test_semantic_dedup_skips_qmd_duplicate(tmp_path: Path):
    cfg = MagicMock()
    cfg.sources_dir = tmp_path / "sources"
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)

    store = ContentStore(tmp_path / "store.db")
    # Seed an embedding for existing content
    store.embedding_set("existing-hash", [1.0, 0.0, 0.0])

    src = ExtractedSource(
        url="https://example.com/a",
        title="Example A",
        content="x" * 1200,
        source_file="a.url",
    )
    # Mock QMD to return embedding identical to stored one
    with patch("pipeline.plan.batch_embed") as mock_batch:
        mock_batch.return_value = {src.content[:1000]: [1.0, 0.0, 0.0]}
        result = _semantic_dedup([src], cfg, store)
    assert len(result) == 0
    assert src.semantic_similarity == 1.0


def test_semantic_dedup_fallback_to_jaccard(tmp_path: Path):
    cfg = MagicMock()
    cfg.sources_dir = tmp_path / "sources"
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)
    # Write an existing source with very similar text
    (cfg.sources_dir / "old.md").write_text("hello world foo bar baz qux")

    store = ContentStore(tmp_path / "store2.db")
    src = ExtractedSource(
        url="https://example.com/b",
        title="old",
        content="hello world foo bar baz qux xyz",
        source_file="b.url",
    )
    with patch("pipeline.plan.batch_embed", return_value={}):
        result = _semantic_dedup([src], cfg, store)
    assert len(result) == 0


def test_keyword_dedup_fallback(tmp_path: Path):
    cfg = MagicMock()
    cfg.sources_dir = tmp_path / "sources"
    cfg.sources_dir.mkdir(parents=True, exist_ok=True)
    (cfg.sources_dir / "target.md").write_text("# Target")

    src = ExtractedSource(
        url="https://example.com/c",
        title="target",
        content="some content here",
        source_file="c.url",
    )
    result = _keyword_dedup_fallback([src], cfg)
    assert len(result) == 0


# ─── Rec 8: Concept Merge Queue ──────────────────────────────────────────────



def test_merge_queue_adds_similar_concept(tmp_path: Path):
    cfg = MagicMock()
    cfg.concepts_dir = tmp_path / "concepts"
    cfg.concepts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.concepts_dir / "blockchain.md").write_text("# Blockchain")

    store = ContentStore(tmp_path / "store3.db")
    plan = MagicMock()
    plan.concept_new = ["blockchain"]

    plans = MagicMock()
    plans.plans = [plan]

    with patch("pipeline.plan.batch_embed") as mock_emb:
        mock_emb.return_value = {
            "blockchain": [1.0, 0.0, 0.0],
        }
        _process_concept_merge_queue(plans, cfg, store)

    pending = store.merge_queue_get_pending()
    assert len(pending) == 1
    assert pending[0]["new_concept"] == "blockchain"
    assert pending[0]["existing_concept"] == "blockchain"
    assert pending[0]["similarity"] > 0.5
    # concept_new should be emptied because it matched existing
    assert plan.concept_new == []


def test_merge_queue_review_and_approve(tmp_path: Path):
    store = ContentStore(tmp_path / "store4.db")
    store.merge_queue_add("new-a", "old-a", 0.92)
    pending = store.merge_queue_get_pending()
    assert len(pending) == 1
    store.merge_queue_approve(pending[0]["id"])
    assert store.merge_queue_get_pending() == []


def test_merge_queue_review_and_reject(tmp_path: Path):
    store = ContentStore(tmp_path / "store4b.db")
    store.merge_queue_add("new-b", "old-b", 0.85)
    pending = store.merge_queue_get_pending()
    assert len(pending) == 1
    store.merge_queue_reject(pending[0]["id"])
    assert store.merge_queue_get_pending() == []



