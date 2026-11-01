"""Outer resealing cannot make altered generated quotations current evidence.

Only inference is scripted; publication, receipt admission, query and preview
readers are real. This is a local integrity counterexample, not a model test.
"""

import asyncio
import hashlib
import json
from unittest.mock import Mock

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.immutable_publication import current_publication, publication_root
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.query_freshness import eligible_graph, page_diagnostics
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
from obsidian_llm_wiki.query.context import build_context
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.query.retrieval import retrieve
from tests.new.quotation_context_helpers import INTRO, QUOTE, QuotationProvider
from tests.new.test_immutable_publication import human_state

ALTERED = '> RENDER_DRIFT: the pump can run during calibration.'


def _build(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), retry_count=1, retry_base_ms=1)
    source = config.sources_dir / 'willow.md'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes((INTRO + '\r\n\r\n' + QUOTE).encode())
    human = config.concepts_dir / 'willow-pump.md'
    human.parent.mkdir(parents=True, exist_ok=True)
    human.write_text('# Human canonical note\nNot generated evidence.\n')
    annotation = config.annotations_dir / 'human.md'
    annotation.parent.mkdir(parents=True, exist_ok=True)
    annotation.write_text('# 人类注释\nPreserve exact notes.\n')
    protected = {p: human_state(p) for p in (source, human, annotation)}
    provider = QuotationProvider()
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.acall_llm', provider)
    result = asyncio.run(run_pipeline(tmp_path, load_sources_from_dir(config.sources_dir), config))
    assert result.compiled == 1 and not result.errors
    assert {p: human_state(p) for p in protected} == protected
    return config, provider, protected


def _reseal(config, root, manifest):
    manifest['files'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in root.rglob('*') if p.is_file()}
    (publication_root(config.wiki_dir) / 'current.json').write_text(json.dumps(manifest))
    assert current_publication(config.wiki_dir)[0] == root


def _alter(config):
    root, manifest = current_publication(config.wiki_dir)
    note = root / 'concepts/willow-pump.md'
    raw = note.read_bytes()
    assert QUOTE.encode() in raw
    note.write_bytes(raw.replace(QUOTE.encode(), ALTERED.encode()))
    _reseal(config, root, manifest)
    return root, note


def test_resealed_quote_excluded_from_default_graph_context_and_cli(tmp_path, monkeypatch):
    config, _provider, protected = _build(tmp_path, monkeypatch)
    root, note = _alter(config)
    before = {p: human_state(p) for p in root.rglob('*') if p.is_file()}
    for kwargs in ({}, {'config': config}):
        graph = build_graph_from_vault(config.wiki_dir, **kwargs)
        page = next(p for p in graph.pages.values() if p.path.endswith('/concepts/willow-pump.md'))
        assert ALTERED in page.body  # Historical inspection is retained, not rewritten.
        assert not page.current_eligible
        assert any('render' in r.lower() and 'mismatch' in r.lower() for r in page.status_reasons)
        candidates = retrieve('RENDER_DRIFT', eligible_graph(graph)).candidates
        assert page.path not in {p.path for p in candidates}
        # Context is independently guarded even with unfiltered retrieval.
        context = build_context(graph, retrieve('RENDER_DRIFT', graph).candidates, 'RENDER_DRIFT')
        assert page.path not in {p.path for p in context}
        assert any(p.current_eligible and p.path.endswith('/sources/willow.md')
                   for p in graph.pages.values())
        assert any(not p['eligible'] and p['path'] == page.path for p in page_diagnostics(graph))
    answer = Mock(return_value='Only the original source remains available.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', answer)
    cli = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'RENDER_DRIFT', '--json'])
    assert cli.exit_code == 1, cli.stdout
    answer.assert_not_called()
    report = json.loads(cli.stdout)
    assert report['freshness']['status'] == 'no_eligible_evidence'
    diagnostic = next(p for p in report['retrieval_trace']['pages'] if p['path'] == page.path)
    assert not diagnostic['eligible'] and diagnostic['reasons']
    healthy = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'Willow', '--json'])
    assert healthy.exit_code == 0, healthy.stdout
    assert answer.call_count == 1
    assert ALTERED not in answer.call_args.args[0]
    assert page.path not in answer.call_args.args[0]
    assert {p: human_state(p) for p in before} == before
    assert {p: human_state(p) for p in protected} == protected
    assert ALTERED.encode() in note.read_bytes()  # Never repair/reseal the captured generation.


def test_preview_separates_identity_proof_from_render_certification(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.identity_migration import preview_identity_migration

    config, _provider, _protected = _build(tmp_path, monkeypatch)
    _alter(config)
    before = {p: human_state(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = preview_identity_migration(tmp_path)
    binding = report['bindings'][0]
    assert binding['status'] == 'resolved'  # The historical identity proof remains valid.
    assert binding['evidence']['historical_admission_integrity'] is True
    assert report['rendered_pages']['certified'] is False
    assert binding['evidence']['rendered_page_certified'] is False
    assert any(b['kind'] == 'rendered_page_unverified' and
               b['path'] == 'concepts/willow-pump.md' for b in report['blockers'])
    for flags in (['--json'], []):
        result = CliRunner().invoke(app, ['identity-migration', str(tmp_path), '--dry-run', *flags])
        assert result.exit_code == 3, result.stdout
        assert 'render' in result.stdout.lower() and 'mismatch' in result.stdout.lower()
    assert {p: human_state(p) for p in before} == before
