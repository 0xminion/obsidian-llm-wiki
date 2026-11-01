"""Integration controls using the actual shared helper, never its test adapter."""
from types import SimpleNamespace

from obsidian_llm_wiki.core.immutable_publication import _inputs
from obsidian_llm_wiki.core.publication_inputs import INPUTS_FORMAT, input_freshness


def test_capture_records_empty_discovered_source(tmp_path):
    root = tmp_path / '04-Wiki/sources'
    root.mkdir(parents=True)
    (root / 'empty.md').write_bytes(b'')
    config = SimpleNamespace(vault=tmp_path, sources_dir=root,
                             llmwiki_dir=tmp_path / '04-Wiki/.llmwiki')
    manifest = {'format': 'immutable-generation-v1', 'status': 'committed',
                'source_inputs_format': INPUTS_FORMAT, 'source_inputs': _inputs(config, {})}
    assert input_freshness(tmp_path, manifest)['status'] == 'current'
    (root / 'empty.md').write_text('New substantive evidence.')
    result = input_freshness(tmp_path, manifest)
    assert result['status'] == 'drifted'
    assert result['changed_paths'] == ['04-Wiki/sources/empty.md']


async def test_api_only_query_is_explicitly_captured_only(tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock

    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from tests.new.test_immutable_publication import fixture

    config, sources, paths, _llm = fixture(tmp_path, monkeypatch)
    paths[0].unlink()  # API supplies a body, not an observable human source file.
    built = await run_pipeline(tmp_path, sources, config)
    assert not built.errors
    answer = Mock(return_value='A1 detail one.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', answer)
    result = CliRunner().invoke(app, ['query', str(tmp_path), '--ask', 'Concept A1', '--json'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['freshness']['status'] == 'captured_only'
    assert data['freshness']['unobserved_sources'] == ['04-Wiki/sources/alpha.md']
    assert 'No live source file' in data['answer']
    assert answer.call_count == 1


def test_new_unsafe_inventory_entry_refuses_before_any_content_read(tmp_path, monkeypatch):
    from pathlib import Path

    vault = tmp_path / 'vault'
    root = vault / '04-Wiki/sources'
    root.mkdir(parents=True)
    (root / 'alpha.md').write_text('Existing source.')
    config = SimpleNamespace(vault=vault, sources_dir=root,
                             llmwiki_dir=vault / '04-Wiki/.llmwiki')
    manifest = {'format': 'immutable-generation-v1', 'status': 'committed',
                'source_inputs_format': INPUTS_FORMAT, 'source_inputs': _inputs(config, {})}
    outside = tmp_path / 'outside.md'
    outside.write_text('Must not be read.')
    (root / 'added.md').symlink_to(outside)
    reads = []
    original = Path.read_bytes

    def tracked(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', tracked)
    result = input_freshness(vault, manifest)
    assert result['status'] == 'unverifiable'
    assert reads == []
