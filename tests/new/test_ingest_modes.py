"""CLI acceptance tests for extract-only ingest modes."""

from __future__ import annotations

import json
from hashlib import sha256

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app

runner = CliRunner()


def _events(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_ingest_plan_is_no_network_no_write_and_emits_ndjson_source_events(
    tmp_path, monkeypatch
) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))

    def unexpected_extract(_url: str):
        raise AssertionError("--plan must not extract or use the network")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", unexpected_extract)

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", "https://example.test/a", "--plan", "--json"],
    )

    assert result.exit_code == 0, result.output
    events = _events(result.output)
    assert any(event["type"] == "source" and event["status"] == "planned" for event in events)
    assert not (config.llmwiki_dir / "operations.json").exists()
    assert not config.sources_dir.exists()


def test_legacy_dry_run_is_no_network_no_write_plan(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))

    def unexpected_extract(_url: str):
        raise AssertionError("legacy --dry-run must not extract or use the network")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", unexpected_extract)

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", "https://example.test/a", "--dry-run", "--json"],
    )

    assert result.exit_code == 0, result.output
    events = _events(result.output)
    assert any(event["type"] == "source" and event["status"] == "planned" for event in events)
    assert not config.llmwiki_dir.exists()
    assert not config.sources_dir.exists()


def test_ingest_preview_extracts_for_real_but_never_writes(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.models import SourceDoc

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    calls: list[str] = []

    def fake_extract(url: str) -> SourceDoc:
        calls.append(url)
        return SourceDoc(title="Preview article", content="real extracted body", url=url)

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", fake_extract)

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", "https://example.test/a", "--preview", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["https://example.test/a"]
    source_events = [event for event in _events(result.output) if event["type"] == "source"]
    assert source_events == [
        {
            "bytes": len(b"real extracted body"),
            "preview": True,
            "run_id": source_events[0]["run_id"],
            "source": "https://example.test/a",
            "source_kind": "url",
            "status": "succeeded",
            "title": "Preview article",
            "type": "source",
        }
    ]
    assert not config.sources_dir.exists()
    assert not config.llmwiki_dir.exists()


def test_skip_synthesis_enforces_source_byte_limit_before_writing(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
    from obsidian_llm_wiki.render.obsidian import parse_frontmatter

    config = Config(vault_path=str(tmp_path), max_source_chars=5)
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    original_content = "ééééé"
    original_hash = sha256(original_content.encode("utf-8")).hexdigest()
    extracted = SourceDoc(
        title="Long",
        content=original_content,
        url="https://example.test/long",
        provenance=SourceProvenance(content_sha256=original_hash, diagnostics=("extract ok",)),
    )
    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", lambda _url: extracted)

    result = runner.invoke(
        app,
        [
            "ingest",
            str(tmp_path),
            "--url",
            "https://example.test/long",
            "--skip-synthesis",
            "--parallel",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    source_page = (config.sources_dir / "long.md").read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(source_page)
    rendered_content = body.strip().splitlines()[-1]
    assert rendered_content == "éé"
    assert len(rendered_content.encode("utf-8")) <= config.max_source_chars
    assert extracted.content == original_content
    assert extracted.provenance.content_sha256 == original_hash
    assert metadata["provenance"]["content_sha256"] == sha256("éé".encode()).hexdigest()
    assert "truncated" in metadata["provenance"]["diagnostics"][-1]


def test_ingest_cancellation_persists_source_state_and_resume_marker(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda _url: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            str(tmp_path),
            "--url",
            "https://example.test/a",
            "--url",
            "https://example.test/b",
            "--skip-synthesis",
            "--parallel",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 130, result.output
    events = _events(result.output)
    assert any(
        event.get("type") == "run"
        and event.get("event") == "cancelled"
        and event.get("remaining") == 2
        for event in events
    )
    operations_path = config.llmwiki_dir / "operations.json"
    operations = json.loads(operations_path.read_text(encoding="utf-8"))
    assert operations["operations"][0]["status"] == "cancelled"
    resume_path = config.llmwiki_dir / "ingest-resume.json"
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    assert resume["sources"] == [
        {"source": "https://example.test/a", "source_kind": "url", "status": "cancelled"},
        {"source": "https://example.test/b", "source_kind": "url", "status": "cancelled"},
    ]


def test_resume_retries_cancelled_source_and_clears_marker(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.operations import OperationRecord, OperationStatus, OperationStore

    config = Config(vault_path=str(tmp_path))
    store = OperationStore(config.llmwiki_dir)
    record = OperationRecord.create(run_id="old-run", source="https://example.test/a")
    store.save(record)
    store.transition(record, OperationStatus.CANCELLED, error="interrupted")
    store.write_resume_marker(
        "old-run", [{"source": record.source, "source_kind": "url", "status": "cancelled"}]
    )
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc(title="Retry", content="retry body", url=url),
    )

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--resume", "--skip-synthesis", "--parallel", "0", "--json"],
    )

    assert result.exit_code == 0, result.output
    operations_path = config.llmwiki_dir / "operations.json"
    persisted = json.loads(operations_path.read_text(encoding="utf-8"))["operations"]
    assert persisted[0]["status"] == "succeeded"
    assert persisted[0]["attempt"] == 2
    assert not (config.llmwiki_dir / "ingest-resume.json").exists()


def test_preview_cancellation_never_creates_history_or_resume_files(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config

    config = Config(vault_path=str(tmp_path))
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda _url: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--url", "https://example.test/a", "--preview", "--json"],
    )

    assert result.exit_code == 130, result.output
    assert not config.llmwiki_dir.exists()
    assert not config.sources_dir.exists()


def test_failed_resume_keeps_marker_for_another_retry(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.operations import OperationRecord, OperationStatus, OperationStore

    config = Config(vault_path=str(tmp_path))
    store = OperationStore(config.llmwiki_dir)
    record = OperationRecord.create(run_id="old-run", source="https://example.test/a")
    store.save(record)
    store.transition(record, OperationStatus.CANCELLED, error="interrupted")
    store.write_resume_marker(
        "old-run", [{"source": record.source, "source_kind": "url", "status": "cancelled"}]
    )
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda _url: (_ for _ in ()).throw(RuntimeError("still unavailable")),
    )

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--resume", "--skip-synthesis", "--parallel", "0", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert store.read_resume_marker() == [
        {"source": "https://example.test/a", "source_kind": "url", "status": "cancelled"}
    ]


def test_resume_preserves_clipping_kind_and_removes_each_success(tmp_path, monkeypatch) -> None:
    import obsidian_llm_wiki.cli.ingest as ingest_module
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.operations import OperationStore

    config = Config(vault_path=str(tmp_path))
    clipping_path = config.clippings_dir / "saved.md"
    clipping = SourceDoc(title="Saved clipping", content="clipped text")
    store = OperationStore(config.llmwiki_dir)
    store.write_resume_marker(
        "old-run",
        [
            {"source": str(clipping_path), "source_kind": "clipping", "status": "cancelled"},
            {"source": "https://example.test/retry", "source_kind": "url", "status": "cancelled"},
        ],
    )
    monkeypatch.setattr(ingest_module, "resolve_vault", lambda _vault: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.clippings.collect_clippings",
        lambda _config: [(clipping_path, clipping)],
    )
    attempts: list[str] = []

    def extract(url: str) -> SourceDoc:
        attempts.append(url)
        raise RuntimeError("still unavailable")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)

    result = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--resume", "--skip-synthesis", "--parallel", "0"],
    )

    assert result.exit_code == 0, result.output
    assert attempts == ["https://example.test/retry"]
    assert (config.sources_dir / "saved.md").exists()
    assert store.read_resume_marker() == [
        {"source": "https://example.test/retry", "source_kind": "url", "status": "cancelled"}
    ]
