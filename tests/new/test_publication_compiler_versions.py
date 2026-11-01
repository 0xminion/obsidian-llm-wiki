"""Outer publication no-op must honor inner compiler compatibility contracts."""
import importlib
import json

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.immutable_publication import (
    _compiler_versions,
    _fingerprint_matches,
    current_publication,
)
from obsidian_llm_wiki.core.pipeline import run_pipeline
from tests.new.test_immutable_publication import fixture, human_state


@pytest.mark.parametrize(('module', 'name'), [
    ('core.pipeline', '_PROMPT_VERSION'),
    ('core.pipeline', '_SYNTHESIS_SCHEMA_VERSION'),
    ('core.span_catalog', 'CATALOG_VERSION'),
    ('core.identity_plan', 'PLAN_VERSION'),
    ('synth.review', 'REVIEW_PROMPT_VERSION'),
    ('synth.review', 'REVIEW_RECEIPT_VERSION'),
    ('synth.review', 'UNIT_IDENTITY_VERSION'),
    ('synth.identity', 'IDENTITY_PROMPT_VERSION'),
    ('synth.identity', 'IDENTITY_RECORD_VERSION'),
])
def test_semantic_version_change_blocks_outer_noop(tmp_path, monkeypatch, module, name):
    config = Config(vault_path=str(tmp_path))
    manifest = {'source_inputs_format': 'vault-relative-v1', 'input_fingerprint': 'same',
                'compiler_versions': _compiler_versions(config)}
    assert _fingerprint_matches(config, {}, (), 'same', manifest)
    target = importlib.import_module('obsidian_llm_wiki.' + module)
    monkeypatch.setattr(target, name, str(getattr(target, name)) + '-changed')
    assert not _fingerprint_matches(config, {}, (), 'same', manifest)


async def test_unstamped_generation_revalidates_once_without_rewriting_history(
    tmp_path, monkeypatch,
):
    config, sources, human_paths, provider = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    generation, manifest = current_publication(config.wiki_dir)
    pointer = config.wiki_dir / 'OLW-Publications/current.json'
    manifest.pop('compiler_versions')  # Explicit pre-stamp compatibility fixture.
    pointer.write_text(json.dumps(manifest))
    original = {p: human_state(p) for p in human_paths}
    history = {p: human_state(p) for p in generation.rglob('*') if p.is_file()}
    calls = provider.call_count
    result = await run_pipeline(tmp_path, sources, config)
    assert not result.errors and result.compiled == 0 and result.skipped == 1
    assert provider.call_count == calls
    upgraded, recorded = current_publication(config.wiki_dir)
    assert upgraded != generation and recorded['compiler_versions'] == _compiler_versions(config)
    assert history == {p: human_state(p) for p in history}
    assert original == {p: human_state(p) for p in original}
    stable = human_state(pointer)
    again = await run_pipeline(tmp_path, sources, config)
    assert not again.errors and again.skipped == 1 and provider.call_count == calls
    assert human_state(pointer) == stable
