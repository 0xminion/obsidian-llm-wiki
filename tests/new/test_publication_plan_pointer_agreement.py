"""All immutable readers must select the same committed identity plan."""
import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.immutable_publication import _files, current_publication
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from tests.new.test_immutable_publication import fixture, human_state


@pytest.mark.parametrize('archived_alias', [False, True], ids=['missing-plan', 'archived-plan'])
async def test_outer_plan_must_match_committed_marker_before_readers(
    tmp_path, monkeypatch, archived_alias,
):
    config, sources, paths, _ = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    root, manifest = current_publication(config.wiki_dir)
    runner = CliRunner()
    healthy = runner.invoke(app, ['identity-migration', str(tmp_path), '--dry-run', '--json'])
    assert healthy.exit_code == 0, healthy.output
    plan_id = manifest['identity_plan_id']
    other = ('f' if set(plan_id) != {'f'} else 'e') * len(plan_id)
    archive = root / '.llmwiki/publications'
    if archived_alias:
        (archive / f'{other}.json').write_bytes((archive / f'{plan_id}.json').read_bytes())
    manifest['identity_plan_id'] = other
    manifest['files'] = _files(root)  # Reseal outer bytes only; inner marker stays intact.
    pointer = config.wiki_dir / 'OLW-Publications/current.json'
    pointer.write_text(json.dumps(manifest))
    before = {p: human_state(p) for p in config.vault.rglob('*') if p.is_file()}
    answer = Mock(return_value='A1 detail one.')
    monkeypatch.setattr('obsidian_llm_wiki.cli.query.call_llm', answer)
    with pytest.raises(ValueError, match='identity plan pointer'):
        current_publication(config.wiki_dir)
    for cfg in (None, config):
        with pytest.raises(ValueError, match='identity plan pointer'):
            build_graph_from_vault(config.wiki_dir, config=cfg)
    for command in (
        ['current', str(tmp_path)],
        ['query', str(tmp_path), '--ask', 'Concept A1', '--json'],
        ['identity-migration', str(tmp_path), '--dry-run', '--json'],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code != 0, result.output
        assert 'identity plan pointer' in result.output.lower(), result.output
    assert answer.call_count == 0
    assert before == {p: human_state(p) for p in before}
