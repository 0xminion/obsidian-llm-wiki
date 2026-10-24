"""Offline CLI regressions: honest terminal outcomes and write-free planning."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc


@pytest.fixture
def vault(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, retry_base_ms=0)
    for name in ("build", "ingest", "ops"):
        module = importlib.import_module(f"obsidian_llm_wiki.cli.{name}")
        monkeypatch.setattr(module, "resolve_vault", lambda _vault: (tmp_path, config))
    return config


def _events(result):
    # All stdout in --json mode must be NDJSON, including failures.
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def _snapshot(directory: Path):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        if path.is_file()
        else None
        for path in directory.rglob("*")
    }


@pytest.mark.parametrize("mode", ["--skip-synthesis", "--preview"])
@pytest.mark.parametrize("partial", [False, True])
def test_extraction_failures_exit_one_with_honest_terminal_event(vault, monkeypatch, mode, partial):
    def extract(url):
        if url.endswith("/bad"):
            raise RuntimeError("fixture extraction failed")
        return SourceDoc("Good", "Successful evidence. " * 80, url=url)

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    before = _snapshot(vault.vault)
    urls = ["--url", "https://example.test/bad"]
    if partial:
        urls += ["--url", "https://example.test/good"]
    result = CliRunner().invoke(
        app, ["ingest", str(vault.vault), "--parallel", "0", "--json", mode, *urls]
    )
    assert result.exit_code == 1, result.output
    events = _events(result)
    terminal = events[-1]
    assert terminal["type"] == "run"
    assert terminal["event"] == ("partial_failure" if partial else "failed")
    assert terminal["extraction_failures"] == terminal["failure_count"] == 1
    assert terminal["synthesis_failures"] == terminal["inventory_failures"] == 0
    assert any("fixture extraction failed" in message for message in terminal["errors"])
    assert not any(e.get("event") == "completed" for e in events)
    if mode == "--preview":
        assert _snapshot(vault.vault) == before
    elif partial:
        assert "Successful evidence." in (vault.sources_dir / "good.md").read_text()


def test_failed_clipping_is_not_a_successful_noop(vault, monkeypatch):
    # A resumed clipping may disappear before it can be extracted.
    from obsidian_llm_wiki.core.operations import OperationStore

    store = OperationStore(vault.llmwiki_dir)
    store.write_resume_marker(
        "old",
        [
            {
                "source": str(vault.clippings_dir / "missing.md"),
                "source_kind": "clipping",
                "status": "cancelled",
            }
        ],
    )
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--parallel",
            "0",
            "--resume",
            "--skip-synthesis",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    terminal = _events(result)[-1]
    assert terminal["event"] == "failed"
    assert terminal["extraction_failures"] == 1
    assert terminal["errors"] and terminal["errors"][0]


@pytest.mark.parametrize("command", ["ingest", "build", "recompile"])
def test_provider_synthesis_failure_is_not_cli_success(vault, monkeypatch, command):
    from unittest.mock import AsyncMock

    vault.sources_dir.mkdir(parents=True)
    (vault.sources_dir / "bad.md").write_text("Substantial source evidence. " * 80)
    provider = AsyncMock(return_value="not valid synthesis JSON")
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("New", "New source evidence. " * 80, url=url),
    )
    args = [command, str(vault.vault)]
    if command == "ingest":
        args += ["--url", "https://example.test/new", "--parallel", "0", "--json"]
    elif command == "recompile":
        args += ["bad.md"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert provider.await_count > 0
    assert "✅ Complete" not in result.output and "✅ Recompiled" not in result.output
    if command == "ingest":
        terminal = _events(result)[-1]
        assert terminal["event"] == "partial_failure"
        assert terminal["synthesis_failures"] == terminal["failure_count"] == 2
        assert len(terminal["errors"]) == 2
        assert (vault.sources_dir / "new.md").exists()


@pytest.mark.parametrize("damage", ["empty", "denied", "unstable"])
def test_ingest_inventory_error_is_structured_without_pipeline(vault, monkeypatch, damage):
    from unittest.mock import AsyncMock

    vault.sources_dir.mkdir(parents=True)
    broken = vault.sources_dir / "broken.md"
    broken.write_text("" if damage == "empty" else "Evidence. " * 80)
    if damage == "denied":
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == broken:
                raise PermissionError("fixture source denied")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)
    elif damage == "unstable":

        def unstable(*_args):
            raise OSError("Source changed while it was being read: broken.md")

        monkeypatch.setattr(
            "obsidian_llm_wiki.ingest.sources._load_source_with_stable_stat", unstable
        )
    pipeline = AsyncMock(side_effect=AssertionError("inventory failure must abort synthesis"))
    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.run_pipeline", pipeline)
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("New", "New source evidence. " * 80, url=url),
    )
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--url",
            "https://example.test/new",
            "--parallel",
            "0",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    pipeline.assert_not_called()
    terminal = _events(result)[-1]
    # Backfill read errors now abort before extracting any new body; failures
    # found only by the later loader retain the successfully extracted source.
    assert terminal["type"] == "run"
    assert terminal["event"] == ("failed" if damage == "denied" else "partial_failure")
    assert terminal["inventory_failures"] == terminal["failure_count"] == 1
    assert "Source inventory failed" in terminal["errors"][0]
    assert (vault.sources_dir / "new.md").exists() == (damage != "denied")


def test_ingest_collection_oserror_is_structured_without_extraction(vault, monkeypatch):
    from unittest.mock import Mock

    def denied(_config):
        raise PermissionError("fixture clippings denied")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.clippings.collect_clippings", denied)
    extract = Mock(side_effect=AssertionError("must not extract"))
    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    before = _snapshot(vault.vault)
    result = CliRunner().invoke(app, ["ingest", str(vault.vault), "--plan", "--json"])
    assert result.exit_code == 1, result.output
    terminal = _events(result)[-1]
    assert terminal["event"] == "failed" and terminal["inventory_failures"] == 1
    assert "fixture clippings denied" in terminal["errors"][0]
    extract.assert_not_called()
    assert _snapshot(vault.vault) == before


@pytest.mark.parametrize("command", ["ingest", "build", "recompile"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_pipeline_exception_has_explicit_failure_or_cancellation(
    vault,
    monkeypatch,
    command,
    cancelled,
):
    from unittest.mock import AsyncMock

    failure = KeyboardInterrupt() if cancelled else RuntimeError("fixture pipeline crashed")
    target = "recompile_single_source" if command == "recompile" else "run_pipeline"
    monkeypatch.setattr(f"obsidian_llm_wiki.core.pipeline.{target}", AsyncMock(side_effect=failure))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("Good", "Source evidence. " * 80, url=url),
    )
    args = [command, str(vault.vault)]
    if command == "ingest":
        args += ["--url", "https://example.test/good", "--parallel", "0", "--json"]
    elif command == "recompile":
        args += ["good.md"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (130 if cancelled else 1), result.output
    if command == "ingest":
        terminal = _events(result)[-1]
        assert terminal["type"] == "run"
        assert terminal["event"] == ("cancelled" if cancelled else "partial_failure")
        if not cancelled:
            assert terminal["synthesis_failures"] == terminal["failure_count"] == 1
            assert "fixture pipeline crashed" in terminal["errors"][0]
    else:
        assert ("cancelled" if cancelled else "fixture pipeline crashed") in result.output.lower()
    assert "✅ Complete" not in result.output


@pytest.mark.parametrize("existing_db", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_build_dry_run_reports_changes_without_opening_sqlite(
    vault, monkeypatch, existing_db, force
):
    import sqlite3

    from obsidian_llm_wiki.core.state import hash_content

    vault.sources_dir.mkdir(parents=True)
    vault.llmwiki_dir.mkdir()
    (vault.sources_dir / "new.md").write_text("New source.")
    (vault.sources_dir / "changed.md").write_text("Changed source.")
    (vault.sources_dir / "same.md").write_text("---\ntitle: Same\n---\nSame source.\n")
    (vault.sources_dir / "index-ba1-000.md").write_text("# Generated index")
    vault.state_file.write_text(
        json.dumps(
            {
                "sources": {
                    "changed.md": {"hash": hash_content("Old source.")},
                    "same.md": {"hash": hash_content("Same source.")},
                    "deleted.md": {"hash": hash_content("Deleted source.")},
                }
            }
        )
    )
    if existing_db:
        # An old/incomplete database must not be migrated, not even opened read-only
        # (SQLite WAL can create sidecars on an ostensibly read-only connection).
        with sqlite3.connect(vault.llmwiki_dir / "registry.db") as connection:
            connection.execute("CREATE TABLE legacy_marker (id INTEGER)")
    before = _snapshot(vault.vault)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not open SQLite or invoke synthesis")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr("obsidian_llm_wiki.core.pipeline.run_pipeline", forbidden)
    result = CliRunner().invoke(
        app,
        [
            "build",
            str(vault.vault),
            "--dry-run",
            *(["--force"] if force else []),
        ],
    )
    assert result.exit_code == 0, result.output
    for category, name in (
        ("new", "new"),
        ("changed", "changed"),
        ("unchanged", "same"),
        ("deleted", "deleted"),
    ):
        assert f"{category}: 1" in result.output.lower()
        assert f"{name}.md" in result.output
    assert "index-ba1-000.md" not in result.output
    assert f"Would synthesise (content-only): {3 if force else 2}" in result.output
    assert ("--force: all present sources" if force else "--force: off") in result.output
    assert "model/schema compatibility" in result.output.lower() and "not checked" in result.output
    assert _snapshot(vault.vault) == before


def test_build_dry_run_on_absent_vault_creates_nothing(vault):
    before = _snapshot(vault.vault)
    result = CliRunner().invoke(app, ["build", str(vault.vault), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "new: 0" in result.output.lower()
    assert "state.json absent" in result.output
    assert _snapshot(vault.vault) == before


@pytest.mark.parametrize("damage", ["empty_source", "missing_sources", "corrupt_state"])
def test_build_dry_run_fails_closed_without_any_writes(vault, damage):
    from obsidian_llm_wiki.core.state import hash_content

    vault.llmwiki_dir.mkdir(parents=True)
    vault.state_file.write_text(
        json.dumps({"sources": {"known.md": {"hash": hash_content("body")}}})
    )
    if damage == "empty_source":
        vault.sources_dir.mkdir()
        (vault.sources_dir / "known.md").write_text("")
    elif damage == "corrupt_state":
        vault.state_file.write_text("{broken")
    before = _snapshot(vault.vault)
    result = CliRunner().invoke(app, ["build", str(vault.vault), "--dry-run"])
    assert result.exit_code == 1, result.output
    assert "failed" in result.output.lower()
    assert _snapshot(vault.vault) == before


@pytest.mark.parametrize("command", ["build", "ingest"])
def test_model_override_reaches_ingest_provider_without_global_mutation(
    vault, monkeypatch, command
):
    import os

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    vault.llm.model = "configured-default"
    vault.llm.ingest_model = "configured-ingest"
    requests = []

    class FixtureClient:
        def chat(self, _system, _user, **kwargs):
            requests.append(kwargs)
            return FAKE_SYNTH_A

    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.create_llm_client", lambda _: FixtureClient()
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("Good", "Source evidence. " * 80, url=url),
    )
    vault.sources_dir.mkdir(parents=True)
    (vault.sources_dir / "alpha.md").write_text("Distinct existing source evidence. " * 80)
    before = dict(os.environ)
    args = [command, str(vault.vault), "--model", "cli-override"]
    if command == "ingest":
        # Default parallelism used to reload Config and leak a process-global env override.
        args += ["--url", "https://example.test/good"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert requests and all(request["model"] == "cli-override" for request in requests)
    assert dict(os.environ) == before
    assert vault.llm.model == "configured-default"
    assert vault.llm.ingest_model == "configured-ingest"


@pytest.mark.parametrize("json_output", [False, True])
def test_extraction_failure_does_not_erase_successful_synthesis(vault, monkeypatch, json_output):
    from unittest.mock import AsyncMock

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    def extract(url):
        if url.endswith("/bad"):
            raise RuntimeError("fixture failed extraction")
        return SourceDoc("Good", "Source evidence. " * 80, url=url)

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    provider = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--parallel",
            "0",
            "--url",
            "https://example.test/bad",
            "--url",
            "https://example.test/good",
            *(["--json"] if json_output else []),
        ],
    )
    assert result.exit_code == 1, result.output
    assert provider.await_count == 1
    assert "good.md" in json.loads(vault.state_file.read_text())["sources"]
    assert (vault.concepts_dir / "concept-a1.md").is_file()
    assert "✅ Complete" not in result.output
    if json_output:
        terminal = _events(result)[-1]
        assert terminal["event"] == "partial_failure"
        assert terminal["compiled"] == 1
        assert terminal["extraction_failures"] == terminal["failure_count"] == 1
        assert terminal["synthesis_failures"] == 0


def test_failed_extraction_with_existing_compiled_corpus_is_partial_failure(vault, monkeypatch):
    from unittest.mock import AsyncMock

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    vault.sources_dir.mkdir(parents=True)
    (vault.sources_dir / "existing.md").write_text("Source evidence. " * 80)

    def failed(_url):
        raise RuntimeError("fixture extraction failed")

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", failed)
    monkeypatch.setattr(
        "obsidian_llm_wiki.providers.llm.acall_llm", AsyncMock(return_value=FAKE_SYNTH_A)
    )
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--parallel",
            "0",
            "--url",
            "https://example.test/bad",
            "--json",
        ],
    )
    assert result.exit_code == 1, result.output
    terminal = _events(result)[-1]
    assert terminal["event"] == "partial_failure"
    assert terminal["compiled"] == 1
    assert "existing.md" in json.loads(vault.state_file.read_text())["sources"]


def test_ingest_noop_and_build_noop_and_recompile_success_keep_zero_status(vault, monkeypatch):
    from unittest.mock import AsyncMock

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    provider = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    result = CliRunner().invoke(app, ["ingest", str(vault.vault), "--json"])
    assert result.exit_code == 0, result.output
    assert _events(result)[-1]["event"] == "completed"
    provider.assert_not_called()
    vault.sources_dir.mkdir(parents=True)
    (vault.sources_dir / "alpha.md").write_text("Source evidence. " * 80)
    first = CliRunner().invoke(app, ["build", str(vault.vault)])
    assert first.exit_code == 0, first.output
    noop = CliRunner().invoke(app, ["build", str(vault.vault)])
    assert noop.exit_code == 0, noop.output
    assert "Already up-to-date" in noop.output
    assert provider.await_count == 1
    retried = CliRunner().invoke(app, ["recompile", str(vault.vault), "alpha.md"])
    assert retried.exit_code == 0, retried.output
    assert "Recompiled" in retried.output
    assert provider.await_count == 2


def test_known_source_read_oserror_is_a_structured_inventory_failure(vault, monkeypatch):
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("Good", "Source evidence. " * 80, url=url),
    )
    args = [
        "ingest",
        str(vault.vault),
        "--parallel",
        "0",
        "--url",
        "https://example.test/good",
        "--skip-synthesis",
        "--json",
    ]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == vault.sources_dir / "good.md":
            raise PermissionError("fixture indexed source denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    terminal = _events(result)[-1]
    assert terminal["event"] == "failed" and terminal["inventory_failures"] == 1
    assert "fixture indexed source denied" in terminal["errors"][0]


def test_sigint_during_last_source_does_not_report_success(vault, monkeypatch):
    module = importlib.import_module("obsidian_llm_wiki.cli.ingest")
    token = module._CancellationToken()
    monkeypatch.setattr(module, "_CancellationToken", lambda: token)

    def extract(_self, url):
        token.request(2, None)
        return SourceDoc("Good", "Source evidence. " * 80, url=url)

    monkeypatch.setattr(module._BoundedURLExtractor, "extract", extract)
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc("Good", "Source evidence. " * 80, url=url),
    )
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--parallel",
            "0",
            "--url",
            "https://example.test/good",
            "--skip-synthesis",
            "--json",
        ],
    )
    assert result.exit_code == 130, result.output
    assert _events(result)[-1]["event"] == "cancelled"
    assert (vault.sources_dir / "good.md").exists()


@pytest.mark.parametrize("mode", ["--plan", "--preview", "--skip-synthesis"])
def test_inbox_read_oserror_never_becomes_a_successful_partial_inventory(vault, monkeypatch, mode):
    from unittest.mock import Mock

    inbox = vault.vault / "00-Inbox"
    inbox.mkdir()
    unreadable = inbox / "blocked.url"
    unreadable.write_text("https://example.test/blocked")
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("fixture inbox denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    extract = Mock(side_effect=AssertionError("inventory failure must precede extraction"))
    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    result = CliRunner().invoke(app, ["ingest", str(vault.vault), mode, "--json"])
    assert result.exit_code == 1, result.output
    terminal = _events(result)[-1]
    assert terminal["event"] == "failed" and terminal["inventory_failures"] == 1
    assert "fixture inbox denied" in terminal["errors"][0]
    extract.assert_not_called()


def test_refresh_title_only_is_not_discarded_as_body_duplicate(vault, monkeypatch):
    from obsidian_llm_wiki.ingest.sources import load_source_file

    title = "Original"
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc(title, "Stable evidence. " * 80, url=url),
    )
    args = ["ingest", str(vault.vault), "--url", "https://example.test/refresh", "--skip-synthesis"]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    path = vault.sources_dir / "original.md"
    before = path.read_bytes()
    title = "Revised title"
    refreshed = CliRunner().invoke(app, [*args, "--refresh"])
    assert refreshed.exit_code == 0, refreshed.output
    source = load_source_file(path)
    assert source is not None and source.title == "Revised title"
    assert list(vault.sources_dir.glob("*.md")) == [path]
    assert any(p.read_bytes() == before for p in (vault.llmwiki_dir / "backups").rglob("*.bak"))


def test_reingest_recovers_failed_synthesis_after_backoff(vault, monkeypatch):
    from datetime import datetime, timedelta
    from unittest.mock import AsyncMock, Mock

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    extract = Mock(return_value=SourceDoc("Good", "Source evidence. " * 80))
    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    provider = AsyncMock(return_value="invalid JSON")
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    args = ["ingest", str(vault.vault), "--url", "https://example.test/retry", "--json"]
    failed = CliRunner().invoke(app, args)
    assert failed.exit_code == 1, failed.output
    provider.return_value = FAKE_SYNTH_A

    class DueClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=120)

    monkeypatch.setattr("obsidian_llm_wiki.core.registry.datetime", DueClock)
    recovered = CliRunner().invoke(app, args)
    assert recovered.exit_code == 0, recovered.output
    assert _events(recovered)[-1]["compiled"] == 1
    assert (vault.wiki_dir / "entries" / "good.md").exists()
    assert extract.call_count == 1


@pytest.mark.parametrize("title", ["Index", "Index-bab-000"])
def test_reserved_source_titles_are_allocated_loadable_filenames(vault, monkeypatch, title):
    from unittest.mock import AsyncMock

    from tests.new.test_incremental_cache import FAKE_SYNTH_A

    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc(title, "Source evidence. " * 80, url=url),
    )
    provider = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            str(vault.vault),
            "--url",
            "https://example.test/index",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    terminal = _events(result)[-1]
    assert terminal.get("compiled") == 1
    assert provider.await_count == 1
    filename = next(e["output_file"] for e in _events(result) if e["type"] == "source")
    assert filename not in {"index.md", "index-bab-000.md"}
    assert (vault.wiki_dir / "entries" / filename).is_file()
