"""Read-only default APIs must check policy, not just historical integrity."""
import asyncio
import json
import os
import shutil
from dataclasses import replace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.query.context import build_context
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.query.retrieval import retrieve
from obsidian_llm_wiki.synth import review
from tests.new.test_compiler_reader_compatibility import tree_state
from tests.new.test_immutable_publication import fixture


def test_old_policy_default_graph_retains_history_but_excludes_current_context(
    tmp_path, monkeypatch,
):
    config, sources, _paths, llm = fixture(tmp_path, monkeypatch)
    # Publish normally under older in-memory policy, not fabricated receipts.
    with monkeypatch.context() as old:
        old.setattr(review, 'REVIEW_PROMPT_VERSION', review.REVIEW_PROMPT_VERSION + '-older-policy')
        assert not asyncio.run(run_pipeline(tmp_path, sources, config)).errors
    calls = llm.call_count
    before = tree_state(tmp_path)
    environment = dict(os.environ)
    graphs = [build_graph_from_vault(config.wiki_dir, config=config),
              build_graph_from_vault(config.wiki_dir)]
    for graph in graphs:
        assert graph.page_count > 0
        assert any('A1 detail one' in page.body for page in graph.pages.values())
        results = retrieve('Concept A1', graph, max_results=5)
        assert results.candidates, 'inspection/history remains available'
        assert not any(page.current_eligible for page in graph.pages.values())
        assert build_context(graph, results.candidates, 'Concept A1') == ()
        assert any('reviewer configuration' in str(page.status_reasons)
                   for page in graph.pages.values())
    answer = Mock(side_effect=AssertionError('old-policy answer provider reached'))
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', answer)
    result = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'Concept A1', '--json',
                                     '--session', 'rejected', '--save-answer', 'rejected.md'])
    assert result.exit_code == 1, result.output
    assert json.loads(result.output)['errors']
    answer.assert_not_called()
    assert llm.call_count == calls
    assert dict(os.environ) == environment
    assert tree_state(tmp_path) == before


def test_low_level_classifier_without_independent_policy_is_history(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.immutable_publication import current_publication
    from obsidian_llm_wiki.core.query_freshness import classify_pages

    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not asyncio.run(run_pipeline(tmp_path, sources, config)).errors
    directory, manifest = current_publication(config.wiki_dir)
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert any(p.current_eligible for p in graph.pages.values())
    before = tree_state(tmp_path)
    classified = classify_pages(graph, directory, manifest)
    assert not any(p.current_eligible for p in classified.pages.values())
    assert all(p.publication_status == 'unverifiable' for p in classified.pages.values())
    assert build_context(
        classified, retrieve('Concept A1', classified).candidates, 'Concept A1',
    ) == ()
    assert tree_state(tmp_path) == before


def forbid_reader_side_effects(monkeypatch):
    """Readers must never instantiate writer config/registry or contact providers."""
    def forbidden(*_args, **_kwargs):
        raise AssertionError('read-only graph touched a mutable dependency')

    monkeypatch.setattr('obsidian_llm_wiki.config.Config.__post_init__', forbidden)
    monkeypatch.setattr('obsidian_llm_wiki.core.registry.VaultRegistry.__init__', forbidden)
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.call_llm', forbidden)
    monkeypatch.setattr('obsidian_llm_wiki.providers.llm.acall_llm', forbidden)


@pytest.mark.parametrize('problem', ['invalid', 'off', 'unreadable'])
def test_unavailable_default_policy_is_unverified_history(tmp_path, monkeypatch, problem):
    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not asyncio.run(run_pipeline(tmp_path, sources, config)).errors
    if problem == 'unreadable':
        def fail(*_args, **_kwargs):
            raise PermissionError('synthetic unavailable policy input')
        monkeypatch.setattr('obsidian_llm_wiki.config.load_review_policy', fail)
    else:
        (tmp_path / '.env').write_text('GROUNDING_REVIEW=' + problem + '\n')
    before = tree_state(tmp_path)
    forbid_reader_side_effects(monkeypatch)
    graph = build_graph_from_vault(config.wiki_dir)
    owned = [p for p in graph.pages.values() if p.source_statuses]
    assert owned
    assert all(p.publication_status == 'unverifiable' for p in owned)
    assert all(dict(p.source_statuses) == {'alpha.md': 'unverifiable'} for p in owned)
    assert all('review policy' in str(p.status_reasons) for p in owned)
    assert not any(p.current_eligible for p in graph.pages.values())
    candidates = retrieve('Concept A1', graph).candidates
    assert candidates and build_context(graph, candidates, 'Concept A1') == ()
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize('copied', [False, True])
@pytest.mark.parametrize('old_policy', [False, True])
def test_healthy_and_old_policy_readers_on_copied_roots(
    tmp_path, monkeypatch, copied, old_policy,
):
    origin = tmp_path / 'origin'
    config, sources, _paths, _llm = fixture(origin, monkeypatch)
    (origin / '.env').write_text('LLM_MODEL=independent-vault-model\n')
    config = load_config(str(origin / '.env'), VAULT_PATH=str(origin))
    with monkeypatch.context() as policy:
        if old_policy:
            policy.setattr(review, 'REVIEW_PROMPT_VERSION', review.REVIEW_PROMPT_VERSION + '-old')
        assert not asyncio.run(run_pipeline(origin, sources, config)).errors
    if copied:
        destination = tmp_path / 'relocated'
        shutil.copytree(origin, destination)
        (origin / '.env').write_text('LLM_MODEL=wrong-origin\nGROUNDING_REVIEW=off\n')
        config = replace(config, vault_path=str(destination), _vault=None)
    before = tree_state(tmp_path)
    environment = dict(os.environ)
    forbid_reader_side_effects(monkeypatch)
    from obsidian_llm_wiki.core.immutable_publication import current_publication
    committed = current_publication(config.wiki_dir)
    for kwargs in ({}, {'config': config}, {'committed': committed}):
        graph = build_graph_from_vault(config.wiki_dir, **kwargs)
        eligible = [p for p in graph.pages.values() if p.current_eligible]
        candidates = retrieve('Concept A1', graph).candidates
        context = build_context(graph, candidates, 'Concept A1')
        assert candidates and bool(eligible) == (not old_policy)
        assert bool(context) == (not old_policy)
        if context:
            assert all(c.publication_status == 'eligible' for c in context)
    assert dict(os.environ) == environment
    assert tree_state(tmp_path) == before


@pytest.mark.parametrize('vault_env', [False, True])
@pytest.mark.parametrize('process_env', [False, True])
def test_default_review_policy_uses_cli_precedence(tmp_path, monkeypatch, vault_env, process_env):
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    (cwd / '.env').write_text('LLM_MODEL=cwd-model\nLLM_INGEST_MODEL=cwd-ingest\n')
    monkeypatch.chdir(cwd)
    vault = tmp_path / 'vault'
    _config, sources, _paths, _llm = fixture(vault, monkeypatch)
    if vault_env:
        (vault / '.env').write_text('LLM_MODEL=vault-model\nLLM_INGEST_MODEL=vault-ingest\n')
    if process_env:
        monkeypatch.setenv('LLM_MODEL', 'process-model')
        monkeypatch.setenv('INGEST_MODEL', 'process-ingest')
    # Explicit vault .env wins; without it process wins over implicit CWD.
    env = str(vault / '.env') if vault_env else None
    config = load_config(env, VAULT_PATH=str(vault))
    expected = 'vault-ingest' if vault_env else 'process-ingest' if process_env else 'cwd-ingest'
    assert config.llm.ingest_model == expected
    assert not asyncio.run(run_pipeline(vault, sources, config)).errors
    before = tree_state(tmp_path)
    environment = dict(os.environ)
    forbid_reader_side_effects(monkeypatch)
    for kwargs in ({}, {'config': config}):
        graph = build_graph_from_vault(config.wiki_dir, **kwargs)
        assert any(p.current_eligible for p in graph.pages.values())
        assert build_context(graph, retrieve('Concept A1', graph).candidates, 'Concept A1')
    assert tree_state(tmp_path) == before
    assert dict(os.environ) == environment


def test_explicit_policy_is_not_overridden_by_invalid_local_config(tmp_path, monkeypatch):
    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    assert not asyncio.run(run_pipeline(tmp_path, sources, config)).errors
    (tmp_path / '.env').write_text('GROUNDING_REVIEW=invalid\nLLM_MODEL=not-the-model\n')
    before = tree_state(tmp_path)
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert any(p.current_eligible for p in graph.pages.values())
    assert build_context(graph, retrieve('Concept A1', graph).candidates, 'Concept A1')
    assert tree_state(tmp_path) == before
