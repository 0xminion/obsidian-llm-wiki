"""Real build/query lifecycle controls, no network or writable registry readers."""
import json
import sqlite3
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from tests.new.grounding_doubles import legacy_fixture_llm
from tests.new.test_incremental_cache import FAKE_SYNTH_B
from tests.new.test_query_currentness import _build, _protected, _query


def test_failed_review_and_backoff_never_query_retained_facts(tmp_path, monkeypatch):
    config, paths, _llm = _build(tmp_path, monkeypatch)
    paths[0].write_text(paths[0].read_text() + 'Changed failed revision.')
    from tests.new.test_incremental_cache import FAKE_SYNTH_A
    reject = legacy_fixture_llm(FAKE_SYNTH_A, verdicts={'A1': 'insufficient'})
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.acall_llm', reject)
    for _ in range(2):
        built = CliRunner().invoke(app, ['build', str(tmp_path)])
        assert built.exit_code == 1, built.output
        before = _protected(config)
        llm = Mock(return_value='A1 detail one.')
        monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
        result = _query(tmp_path, '--max-results', '1', '--session', 'failed')
        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload['freshness']['status'] == 'no_eligible_evidence'
        assert any('stale' in str(p) for p in payload['retrieval_trace']['pages'])
        llm.assert_not_called()
        assert _protected(config) == before
        if _ == 0:
            count = len(reject.calls)
        else:
            assert len(reject.calls) == count


def test_source_deletion_orphans_history_and_refuses_query_in_json_and_human(tmp_path, monkeypatch):
    config, paths, _llm = _build(tmp_path, monkeypatch)
    old, _ = current_publication(config.wiki_dir)
    old_page = (old / 'concepts/concept-a1.md').read_bytes()
    paths[0].unlink()
    built = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert built.exit_code == 0, built.output
    before = _protected(config)
    llm = Mock(return_value='A1 detail one.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
    result = _query(tmp_path, '--max-results', '1')
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert 'orphan' in str(payload).lower() and payload['answer'] == ''
    assert any(p['orphaned'] for p in payload['retrieval_trace']['pages'])
    human = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'A1', '-n', '1'])
    assert human.exit_code == 1 and 'orphan' in human.output.lower()
    llm.assert_not_called()
    assert _protected(config) == before
    assert (old / 'concepts/concept-a1.md').read_bytes() == old_page


def test_unchecked_generation_excluded_even_when_review_off(tmp_path, monkeypatch):
    config, _paths, _llm = _build(tmp_path, monkeypatch)
    (tmp_path / '.env').write_text('GROUNDING_REVIEW=off\nEMBEDDINGS_ENABLED=false\n')
    result = CliRunner().invoke(app, ['build', str(tmp_path), '--force'])
    assert result.exit_code == 0, result.output
    llm = Mock(return_value='A1 detail one.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
    result = _query(tmp_path)
    assert result.exit_code == 1, result.output
    assert 'unchecked' in result.output
    llm.assert_not_called()


def test_coowned_mixed_page_excluded_but_healthy_source_queries_work(tmp_path, monkeypatch):
    config, paths, _llm = _build(tmp_path, monkeypatch)
    (config.sources_dir / 'beta.md').write_text('Beta observations ' * 30)
    provider = legacy_fixture_llm(
        FAKE_SYNTH_B, identity_verdicts={('Concept Shared', 'Concept Shared'): 'same'},
    )
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.acall_llm', provider)
    result = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert result.exit_code == 0, result.output
    paths[0].write_text('Failed Alpha revision. ' * 30)

    async def fail(*args, **kwargs):
        raise RuntimeError('offline fixture')

    monkeypatch.setattr('obsidian_llm_wiki.core.pipeline._synthesize_with_retry', fail)
    result = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert result.exit_code == 1, result.output
    graph = build_graph_from_vault(config.wiki_dir)
    shared = next(p for p in graph.pages.values() if p.path.endswith('/concept-shared.md'))
    assert dict(shared.source_statuses) == {'alpha.md': 'stale', 'beta.md': 'eligible'}
    assert not shared.current_eligible
    seen = []

    def query(system, messages, *args, **kwargs):
        seen.append(system)
        return 'B1 detail one.'

    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', query)
    result = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'B1', '-n', '1', '--json'])
    assert result.exit_code == 0, result.output
    assert 'Shared from A' not in seen[0] and 'A1 detail' not in seen[0]
    assert 'captured' in json.loads(result.output)['answer'].lower()
    assert 'Excluded' in json.loads(result.output)['answer']


def test_new_generation_during_provider_refuses_without_query_writes(tmp_path, monkeypatch):
    config, paths, _llm = _build(tmp_path, monkeypatch)

    def query(*args, **kwargs):
        paths[0].write_text(paths[0].read_text() + ' New observation.')
        result = CliRunner().invoke(app, ['build', str(tmp_path)])
        assert result.exit_code == 0, result.output
        return 'Old A1 detail one.'

    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', query)
    result = _query(tmp_path, '--save-answer', 'new-generation.md')
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload['freshness']['status'] == 'generation_changed'
    assert not (config.queries_dir / 'new-generation.md').exists()
    assert not (config.wiki_dir / 'OLW-Publications/query-state/query-sessions.json').exists()


@pytest.mark.parametrize('damage', ['schema', 'page'])
def test_unknown_schema_or_corruption_refuses_without_provider_or_mutation(
    tmp_path, monkeypatch, damage,
):
    import hashlib

    config, _paths, _llm = _build(tmp_path, monkeypatch)
    root, manifest = current_publication(config.wiki_dir)
    if damage == 'schema':
        database = root / '.llmwiki/registry.db'
        connection = sqlite3.connect(database)
        connection.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        connection.commit()
        connection.close()
        sha = hashlib.sha256(database.read_bytes()).hexdigest()
        manifest['files']['.llmwiki/registry.db'] = sha
        (root.parent.parent / 'current.json').write_text(json.dumps(manifest))
    else:
        (root / 'index.md').write_text('Corruption')
    before = _protected(config)
    llm = Mock(return_value='A1 detail one.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
    result = _query(tmp_path)
    assert result.exit_code == 1, result.output
    assert 'schema' in result.output or 'corrupt' in result.output
    llm.assert_not_called()
    assert _protected(config) == before
