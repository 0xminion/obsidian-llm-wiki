"""Default queries must retain history without advertising it as current."""
import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.query.context import build_context
from obsidian_llm_wiki.query.graph import build_graph
from obsidian_llm_wiki.query.retrieval import retrieve
from tests.new.test_immutable_publication import fixture, human_state


def _build(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    built = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert built.exit_code == 0, built.output
    return config, paths, llm


def _query(tmp_path, *args):
    return CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'Concept A1',
                                  '--json', *args])


def _protected(config):
    return {str(p): human_state(p) for p in config.vault.rglob('*') if p.is_file()}


def test_current_query_succeeds_and_noop_remains_zero_call(tmp_path, monkeypatch):
    config, paths, llm = _build(tmp_path, monkeypatch)
    root, manifest = current_publication(config.wiki_dir)
    calls = llm.call_count
    noop = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert noop.exit_code == 0 and llm.call_count == calls
    relative = (root / 'concepts/concept-a1.md').relative_to(config.wiki_dir).as_posix()
    citation = '04-Wiki/' + relative
    query_llm = Mock(return_value=f'A1 detail one. [[{citation}]]')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', query_llm)
    result = _query(tmp_path, '--max-results', '1')
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload['freshness']['status'] == 'observed_current'
    assert payload['freshness']['generation'] == manifest['generation']
    assert 'captured' in payload['answer'].lower()
    assert query_llm.call_count == 1


def test_source_edit_without_build_refuses_before_provider_and_writes(tmp_path, monkeypatch):
    config, paths, _llm = _build(tmp_path, monkeypatch)
    paths[0].write_text('Replacement observation contradicts A1. ' * 20)
    before = _protected(config)
    query_llm = Mock(return_value='A1 detail one.')  # deliberately ignores freshness
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', query_llm)
    result = _query(tmp_path, '--session', 'rejected', '--save-answer', 'rejected.md')
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload['answer'] == '' and payload['citations'] == []
    assert payload['freshness']['status'] == 'drifted'
    assert any('source' in e.lower() or 'input' in e.lower() for e in payload['errors'])
    query_llm.assert_not_called()
    assert _protected(config) == before


def test_orphan_lifecycle_survives_parsing_and_single_result_snippet():
    graph = build_graph({'concepts/boreal.md': '''---
title: Boreal Archive
orphaned: true
orphaned_reason: 'Source owners deleted: boreal.md'
---
# Boreal Archive

## Tablets
Boreal Archive preserves 23 cedar tablets.
'''})
    page = graph.pages['concepts/boreal.md']
    assert page.orphaned is True
    assert page.orphaned_reason == 'Source owners deleted: boreal.md'
    assert not page.current_eligible
    candidates = retrieve('Tablets', graph, max_results=1).candidates
    assert candidates
    assert build_context(graph, candidates, 'Tablets', max_chars_per_page=40) == ()


@pytest.mark.parametrize('seam', ['provider', 'grounding'])
@pytest.mark.parametrize('edit', ['source', 'policy'])
def test_late_input_edit_cannot_create_session_or_saved_answer(
    tmp_path, monkeypatch, seam, edit,
):
    import obsidian_llm_wiki.cli.query as query

    config, paths, _llm = _build(tmp_path, monkeypatch)
    target = paths[0] if edit == 'source' else paths[-2]
    real_ground = query._ground_answer
    calls = []

    def mutate():
        target.write_text('domain: history\n' if edit == 'policy' else 'Changed source. ' * 20)

    def provider(*_args, **_kwargs):
        calls.append(True)
        if seam == 'provider':
            mutate()
        return 'A1 detail one.'

    def grounding(*args):
        mutate()
        return real_ground(*args)

    monkeypatch.setattr(query, 'call_llm', provider)
    if seam == 'grounding':
        monkeypatch.setattr(query, '_ground_answer', grounding)
    result = _query(tmp_path, '--session', 'late', '--save-answer', 'late.md')
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload['freshness']['status'] == 'drifted'
    assert payload['retrieval_trace']['freshness']['status'] == 'drifted'
    assert payload['answer'] == '' and len(calls) == 1
    assert not (config.queries_dir / 'late.md').exists()
    assert not (config.wiki_dir / 'OLW-Publications/query-state/query-sessions.json').exists()


def test_old_generation_session_answer_is_not_provider_evidence(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query

    config, paths, _llm = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(query, 'call_llm', lambda *a, **kw: 'OLD_SESSION_SENTINEL')
    first = _query(tmp_path, '--session', 'followup')
    assert first.exit_code == 0, first.output
    paths[0].write_text(paths[0].read_text() + ' New source observation.')
    built = CliRunner().invoke(app, ['build', str(tmp_path)])
    assert built.exit_code == 0, built.output
    seen = []

    def provider(system, messages, *args, **kwargs):
        seen.append((system, messages))
        return 'New answer.'

    monkeypatch.setattr(query, 'call_llm', provider)
    result = _query(tmp_path, '--session', 'followup')
    assert result.exit_code == 0, result.output
    assert 'OLD_SESSION_SENTINEL' not in str(seen)
    assert json.loads(result.output)['retrieval_trace']['previous_session'] == 'excluded'


def test_context_keeps_status_outside_bounded_snippet(tmp_path, monkeypatch):
    from obsidian_llm_wiki.query.graph import build_graph_from_vault

    config, _paths, _llm = _build(tmp_path, monkeypatch)
    graph = build_graph_from_vault(config.wiki_dir)
    sections = build_context(graph, retrieve('A1', graph, max_results=1).candidates,
                             'A1', max_chars_per_page=5)
    assert len(sections) == 1
    assert sections[0].publication_status == 'eligible'
    assert sections[0].source_statuses == (('alpha.md', 'eligible'),)


@pytest.mark.parametrize('change', ['delete', 'add', 'policy'])
def test_inventory_or_policy_edit_without_build_refuses(tmp_path, monkeypatch, change):
    config, paths, _llm = _build(tmp_path, monkeypatch)
    if change == 'delete':
        paths[0].unlink()
    elif change == 'add':
        (config.sources_dir / 'new.md').write_text('New source. ' * 30)
    else:
        paths[-2].write_text('domain: history\n')
    before = _protected(config)
    llm = Mock(return_value='Stale A1.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
    result = _query(tmp_path)
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)['freshness']['status'] == 'drifted'
    llm.assert_not_called()
    assert _protected(config) == before


def test_absent_shared_input_inspector_fails_closed(tmp_path, monkeypatch):
    import sys

    config, _paths, _llm = _build(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, 'obsidian_llm_wiki.core.publication_inputs', None)
    before = _protected(config)
    llm = Mock(return_value='A1')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', llm)
    result = _query(tmp_path)
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)['freshness']['status'] == 'unverifiable'
    llm.assert_not_called()
    assert _protected(config) == before


def test_answer_install_collision_does_not_save_rejected_session(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.query as query

    config, _paths, _llm = _build(tmp_path, monkeypatch)
    monkeypatch.setattr(query, 'call_llm', lambda *a, **kw: 'A1')
    real_save = query._save_answer

    def collide(path, answer):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Concurrent human winner')
        real_save(path, answer)

    monkeypatch.setattr(query, '_save_answer', collide)
    result = _query(tmp_path, '--save-answer', 'race.md', '--session', 'rejected')
    assert result.exit_code == 1, result.output
    assert (config.queries_dir / 'race.md').read_text() == 'Concurrent human winner'
    assert not (config.wiki_dir / 'OLW-Publications/query-state/query-sessions.json').exists()
