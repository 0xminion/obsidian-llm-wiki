"""Positive/no-op controls and the actual-read integrity boundary."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.identity_migration import preview_identity_migration
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
from obsidian_llm_wiki.query.context import build_context
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.query.retrieval import retrieve
from tests.new.quotation_context_helpers import QUOTE
from tests.new.test_immutable_publication import human_state
from tests.new.test_rendered_query_admission import ALTERED, _alter, _build, _reseal


def test_healthy_render_preview_query_and_noop_preserve_human_notes(tmp_path, monkeypatch):
    config, provider, protected = _build(tmp_path, monkeypatch)
    before = {p: human_state(p) for p in tmp_path.rglob('*') if p.is_file()}
    graph = build_graph_from_vault(config.wiki_dir)
    page = next(p for p in graph.pages.values() if p.path.endswith('/concepts/willow-pump.md'))
    assert page.current_eligible and QUOTE in page.body
    preview = preview_identity_migration(tmp_path)
    assert preview['rendered_pages']['certified'] is True
    assert preview['rendered_pages']['unverified'] == []
    assert preview['bindings'][0]['evidence']['rendered_page_certified'] is True
    cli = CliRunner().invoke(app, ['identity-migration', str(tmp_path), '--dry-run', '--json'])
    assert cli.exit_code == 0, cli.stdout
    calls = len(provider.requests)
    noop = asyncio.run(run_pipeline(tmp_path, load_sources_from_dir(config.sources_dir), config))
    assert noop.compiled == 0 and noop.skipped == 1 and not noop.errors
    assert len(provider.requests) == calls
    assert {p: human_state(p) for p in before} == before
    assert {p: human_state(p) for p in protected} == protected
    answer = Mock(return_value=f'Idle during calibration. [[{page.path}]]')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', answer)
    cli = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'Evidence', '--json'])
    assert cli.exit_code == 0, cli.stdout
    assert answer.call_count == 1 and QUOTE in answer.call_args.args[0]


def test_admission_binds_consumed_bytes_not_a_later_reread(tmp_path, monkeypatch):
    config, _provider, _protected = _build(tmp_path, monkeypatch)
    root, _manifest = current_publication(config.wiki_dir)
    note = root / 'concepts/willow-pump.md'
    original = note.read_bytes()
    _alter(config)
    pinned = current_publication(config.wiki_dir)
    read_bytes = Path.read_bytes
    consumed = []

    def restore_after_read(path):
        raw = read_bytes(path)
        if path == note and not consumed:
            consumed.append(raw)
            note.write_bytes(original)
        return raw

    monkeypatch.setattr(Path, 'read_bytes', restore_after_read)
    graph = build_graph_from_vault(config.wiki_dir, committed=pinned)
    page = next(p for p in graph.pages.values() if p.path.endswith('/concepts/willow-pump.md'))
    assert ALTERED.encode() in consumed[0] and note.read_bytes() == original
    assert ALTERED in page.body and not page.current_eligible
    assert page.path not in {p.path for p in build_context(
        graph, retrieve('RENDER_DRIFT', graph).candidates, 'RENDER_DRIFT')}


def test_missing_render_hash_never_adopts_observed_bytes(tmp_path, monkeypatch):
    config, _provider, _protected = _build(tmp_path, monkeypatch)
    root, manifest = current_publication(config.wiki_dir)
    inventory = root / '.llmwiki/render-page-hashes.json'
    hashes = json.loads(inventory.read_bytes())
    del hashes['concepts/willow-pump.md']
    inventory.write_text(json.dumps(hashes))
    _reseal(config, root, manifest)
    before = {p: human_state(p) for p in root.rglob('*') if p.is_file()}
    graph = build_graph_from_vault(config.wiki_dir)
    page = next(p for p in graph.pages.values() if p.path.endswith('/concepts/willow-pump.md'))
    assert not page.current_eligible
    assert any('no valid render hash' in reason for reason in page.status_reasons)
    report = preview_identity_migration(tmp_path)
    assert not report['rendered_pages']['certified']
    assert not report['bindings'][0]['evidence']['rendered_page_certified']
    assert {p: human_state(p) for p in before} == before
