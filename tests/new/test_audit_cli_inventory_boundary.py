"""Public CLI regressions for finalA's pre-loader inventory boundary."""

from __future__ import annotations

import importlib
import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.registry import VaultRegistry


@pytest.fixture
def vault(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path / "vault"), embeddings_enabled=False)
    config.vault.mkdir()
    for command in ("ingest", "build"):
        module = importlib.import_module(f"obsidian_llm_wiki.cli.{command}")
        monkeypatch.setattr(module, "resolve_vault", lambda _: (config.vault, config))
    return config


def _invoke(config, *options):
    return CliRunner().invoke(app, ["ingest", str(config.vault), "--json", *options])


def _terminal_failure(result, *, event="failed", succeeded=0):
    assert result.exit_code == 1, result.output
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    terminal = [e for e in events if e["type"] == "run" and e["event"] != "started"]
    assert len(terminal) == 1, events
    assert events[-1] == terminal[0]
    failed = terminal[0]
    assert failed["event"] == event
    assert failed["inventory_failures"] == failed["failure_count"] == 1
    assert failed["extraction_failures"] == failed["synthesis_failures"] == 0
    assert failed["succeeded"] == succeeded
    assert len(failed["errors"]) == 1
    assert "Source inventory failed" in failed["errors"][0]
    assert "Traceback" not in result.output
    return failed


def _escape(config):
    config.sources_dir.mkdir(parents=True, exist_ok=True)
    outside = config.vault.parent / "outside.md"
    outside.write_text(
        "---\ntitle: Outside\nurl: https://outside.test/private\n---\nSECRET EVIDENCE"
    )
    link = config.sources_dir / "zzz-escape.md"
    link.symlink_to(outside)
    return link, outside


def _audit_body_reads(monkeypatch, config, outside):
    opened = []
    original = io.open

    def observed(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file)
            if path.resolve() == outside or path.parent == config.sources_dir:
                opened.append(str(path))
        return original(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", observed)
    return opened


def _ingest_rows(config):
    db = config.llmwiki_dir / "registry.db"
    if not db.exists():
        return []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return conn.execute(
            "SELECT normalized_url, content_hash, output_file "
            "FROM ingest_hash ORDER BY normalized_url"
        ).fetchall()


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("mode", [[], ["--skip-synthesis"], ["--plan"], ["--preview"]])
def test_static_escape_precedes_all_bodies_and_ingest_index_writes(
    vault, monkeypatch, indexed, mode
):
    url = "https://example.test/known"
    extract = Mock(return_value=SourceDoc("Known", "Safe evidence. " * 80, url=url))
    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    if indexed:
        first = _invoke(vault, "--url", url, "--skip-synthesis")
        assert first.exit_code == 0, first.output
        extract.reset_mock()
    else:
        vault.sources_dir.mkdir(parents=True)
        (vault.sources_dir / "aaa-legacy.md").write_text(
            "---\nurl: https://example.test/legacy\n---\nSafe legacy evidence"
        )
    before = _ingest_rows(vault)
    _, outside = _escape(vault)
    reads = _audit_body_reads(monkeypatch, vault, outside)
    result = _invoke(vault, "--url", url, *mode)
    assert reads == [], f"source bodies read before validating the complete inventory: {reads}"
    assert _ingest_rows(vault) == before
    failed = _terminal_failure(result)
    assert "Source path escapes" in failed["errors"][0]
    assert failed["sources"] == 0
    extract.assert_not_called()


@pytest.mark.parametrize("reader", ["backfill", "legacy_dedupe"])
def test_legacy_readers_validate_whole_inventory_before_any_body(vault, monkeypatch, reader):
    from obsidian_llm_wiki.cli.ingest import _backfill_ingest_index, _content_hash_matches_existing

    _, outside = _escape(vault)
    (vault.sources_dir / "aaa-safe.md").write_text("Safe evidence")
    reads = _audit_body_reads(monkeypatch, vault, outside)
    with VaultRegistry(vault.llmwiki_dir) as registry:
        with pytest.raises(ValueError, match="Source path escapes"):
            if reader == "backfill":
                _backfill_ingest_index(registry, vault.sources_dir)
            else:
                _content_hash_matches_existing(vault.sources_dir, "Safe evidence")
        assert not registry.ingest_backfill_complete()
        assert registry.find_ingest_duplicate(requested_url="https://outside.test/private") is None
    assert reads == []


def test_containment_failure_after_extraction_has_one_partial_terminal(vault, monkeypatch):
    def extract(url):
        _escape(vault)
        return SourceDoc("Good", "Safe evidence. " * 80, url=url)

    monkeypatch.setattr("obsidian_llm_wiki.ingest.extractors.extract", extract)
    # Introduce the link after preflight to reach the real loader's ValueError.
    result = _invoke(vault, "--url", "https://example.test/new")
    failed = _terminal_failure(result, event="partial_failure", succeeded=1)
    assert failed["sources"] == 1
    assert "Source path escapes" in failed["errors"][0]
    assert (vault.sources_dir / "good.md").is_file()
    assert all(row[0] != "https://outside.test/private" for row in _ingest_rows(vault))


@pytest.mark.parametrize("dry_run", [False, True])
def test_build_containment_error_is_a_clear_diagnostic(vault, monkeypatch, dry_run):
    _, outside = _escape(vault)
    reads = _audit_body_reads(monkeypatch, vault, outside)
    args = ["build", str(vault.vault), *(["--dry-run"] if dry_run else [])]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    diagnostic = (
        "Build plan failed; no files written"
        if dry_run
        else "Source inventory failed; no build performed"
    )
    assert diagnostic in result.stderr
    assert "Source path escapes" in result.stderr
    assert "Traceback" not in result.stderr
    assert reads == []
    if dry_run:
        assert not vault.llmwiki_dir.exists()


@pytest.mark.parametrize("reader", ["backfill", "legacy_dedupe"])
def test_early_body_read_errors_are_not_suppressed(vault, monkeypatch, reader):
    from obsidian_llm_wiki.cli.ingest import _backfill_ingest_index, _content_hash_matches_existing

    vault.sources_dir.mkdir(parents=True)
    blocked = vault.sources_dir / "denied.md"
    blocked.write_text("Evidence")
    original = Path.read_text

    def denied(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("fixture backfill denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with VaultRegistry(vault.llmwiki_dir) as registry:
        with pytest.raises(PermissionError, match="fixture backfill denied"):
            if reader == "backfill":
                _backfill_ingest_index(registry, vault.sources_dir)
            else:
                _content_hash_matches_existing(vault.sources_dir, "Evidence")
        assert not registry.ingest_backfill_complete()
    result = _invoke(vault, "--skip-synthesis")
    failed = _terminal_failure(result)
    assert "fixture backfill denied" in failed["errors"][0]


@pytest.mark.parametrize("indexed", [False, True])
def test_real_process_cli_reports_static_escape_as_ndjson(tmp_path, indexed):
    import os
    import subprocess
    import sys

    config = Config(vault_path=str(tmp_path / "vault"))
    config.clippings_dir.mkdir(parents=True)
    (config.vault / ".env").write_text(
        "EMBEDDINGS_ENABLED=false\nCLIPPING_MIN_BODY_CHARS=10\nRETRY_COUNT=1\n"
    )
    (config.clippings_dir / "safe.md").write_text(
        "---\ntitle: Safe\nurl: https://example.test/safe\n---\n" + "Safe clipped evidence. " * 80
    )
    guard = tmp_path / "guard"
    guard.mkdir()
    # No live requests. Observe outside opens in the actual CLI process.
    (guard / "sitecustomize.py").write_text(
        "import os, socket, sys\n"
        "def no_network(*args, **kwargs):\n"
        "    raise AssertionError('Live network forbidden')\n"
        "socket.socket.connect = no_network\n"
        "socket.socket.connect_ex = no_network\n"
        "def audit(event, args):\n"
        "    if (event == 'open' and isinstance(args[0], str)\n"
        "            and os.path.realpath(args[0]) == os.environ['OUTSIDE_PATH']):\n"
        "        with open(os.environ['READ_MARKER'], 'w') as f: f.write(args[0])\n"
        "sys.addaudithook(audit)\n"
    )
    repo = Path(__file__).resolve().parents[2]
    outside = tmp_path / "outside.md"
    marker = tmp_path / "outside-read"
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{guard}{os.pathsep}{repo}",
        "OUTSIDE_PATH": str(outside),
        "READ_MARKER": str(marker),
    }
    command = [
        str(Path(sys.executable).with_name("olw")),
        "ingest",
        str(config.vault),
        "--json",
        "--parallel",
        "1",
    ]

    def invoke(*options):
        return subprocess.run(
            [*command, *options], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )

    if indexed:
        first = invoke("--skip-synthesis")
        assert first.returncode == 0, first.stderr
        duplicate = invoke("--url", "https://example.test/safe", "--skip-synthesis")
        assert duplicate.returncode == 0, duplicate.stderr
        assert any(
            e.get("duplicate_of") == "safe.md"
            for e in map(json.loads, duplicate.stdout.splitlines())
        )
    before = _ingest_rows(config)
    _escape(config)
    result = invoke(*(["--url", "https://example.test/safe"] if indexed else []))
    assert result.returncode == 1, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    terminal = [e for e in events if e["type"] == "run" and e["event"] != "started"]
    assert len(terminal) == 1, (events, result.stderr)
    assert terminal[0]["event"] == "failed"
    assert terminal[0]["inventory_failures"] == terminal[0]["failure_count"] == 1
    assert "Source path escapes" in terminal[0]["errors"][0]
    assert "Traceback" not in result.stderr
    assert not marker.exists()
    assert _ingest_rows(config) == before
