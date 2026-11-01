"""Characterize default-policy status precedence without modifying the app."""
import asyncio
import json
import os

from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.query_freshness import classify_pages
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from tests.new.test_compiler_reader_compatibility import tree_state
from tests.new.test_default_reader_policy import forbid_reader_side_effects
from tests.new.test_immutable_publication import fixture


def test_unknown_policy_preserves_unverifiable_status_and_exclusion(tmp_path, monkeypatch):
    vault = tmp_path / 'vault'
    config, sources, _paths, llm = fixture(vault, monkeypatch)
    assert not asyncio.run(run_pipeline(vault, sources, config)).errors
    directory, manifest = current_publication(config.wiki_dir)
    graph = build_graph_from_vault(config.wiki_dir, config=config)
    assert any(page.current_eligible for page in graph.pages.values())
    calls = llm.call_count
    before = tree_state(vault)
    environment = dict(os.environ)
    forbid_reader_side_effects(monkeypatch)
    classified = classify_pages(graph, directory, manifest)
    observed = [{"path": page.path, "status": page.publication_status,
                 "reasons": list(page.status_reasons),
                 "source_statuses": list(page.source_statuses)}
                for page in classified.pages.values()]
    assert not any(page.current_eligible for page in classified.pages.values())
    assert tree_state(vault) == before
    assert dict(os.environ) == environment
    assert llm.call_count == calls
    assert all(page.publication_status == 'unverifiable'
               for page in classified.pages.values()), json.dumps(observed, indent=2)
