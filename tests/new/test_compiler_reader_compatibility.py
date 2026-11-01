"""Normal compiler evolution must not be hidden by the outer publication no-op."""
import importlib
import os
from pathlib import Path

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import identity_context
from obsidian_llm_wiki.core.immutable_publication import _compiler_versions, current_publication
from obsidian_llm_wiki.core.pipeline import (
    _artifact_versions_for_config,
    _pipeline_checkpoint,
    run_pipeline,
)
from tests.new.test_immutable_publication import fixture, human_state


def tree_state(root):
    """Include real bytes/inodes/modes and directory metadata; exclude read atime."""
    state = {}
    for path in [root, *sorted(root.rglob('*'))]:
        stat = path.lstat()
        state[str(path.relative_to(root))] = (
            path.read_bytes() if path.is_file() else None,
            stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            stat.st_uid, stat.st_gid,
            {n: os.getxattr(path, n) for n in os.listxattr(path)},
        )
    return state


async def test_public_pipeline_context_evolution_revalidates_then_noops(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    paths[-1].chmod(0o640)
    os.setxattr(paths[-1], 'user.compatibility', b'\x00retained\xff')
    protected = {p: human_state(p) for p in paths}
    first = await run_pipeline(tmp_path, sources, config)
    assert first.compiled == 1 and not first.errors
    previous, manifest = current_publication(config.wiki_dir)
    old_state = tree_state(previous)
    old_stamp = _compiler_versions(config)
    checkpoint = _pipeline_checkpoint(config, _artifact_versions_for_config(config))
    calls = llm.call_count
    # Normal in-memory future-version simulation, no manifest/source tampering.
    monkeypatch.setattr(identity_context, 'CONTEXT_VERSION',
                        identity_context.CONTEXT_VERSION + '-compatibility-next')
    assert checkpoint != _pipeline_checkpoint(config, _artifact_versions_for_config(config))
    result = await run_pipeline(tmp_path, sources, config)
    assert not result.errors
    directory, updated = current_publication(config.wiki_dir)
    assert directory != previous, (
        'public run_pipeline bypassed changed inner identity-context checkpoint', result,
    )
    assert updated['compiler_versions'] != old_stamp
    assert updated['compiler_versions']['identity_context'] == identity_context.CONTEXT_VERSION
    assert set(manifest['compiler_versions']) <= set(updated['compiler_versions'])
    assert tree_state(previous) == old_state
    assert protected == {p: human_state(p) for p in paths}
    assert llm.call_count == calls
    after = tree_state(Path(tmp_path))
    repeat = await run_pipeline(tmp_path, sources, config)
    assert not repeat.errors and repeat.compiled == 0 and repeat.skipped == 1
    assert llm.call_count == calls
    assert tree_state(Path(tmp_path)) == after


# Finite version surface consumed by artifact versions, inner checkpoint,
# outer stamp, or the earlier immutable-registry reader gate. The conditional
# two-pass and review-off cases matter: not every version affects every mode.
VERSION_PROBES = (
    ('core.pipeline', '_EXTRACTOR_VERSION'),
    ('core.pipeline', '_PROMPT_VERSION'),
    ('core.pipeline', '_SYNTHESIS_SCHEMA_VERSION'),
    ('core.pipeline', '_FEEDBACK_CONTRACT'),
    ('core.pipeline', '_TWO_PASS_GROUNDED_CONTRACT'),
    ('core.span_catalog', 'CATALOG_VERSION'),
    ('synth.grounding', 'CITATION_INTERFACE_VERSION'),
    ('synth.review', 'REVIEW_PROMPT_VERSION'),
    ('synth.review', 'REVIEW_RECEIPT_VERSION'),
    ('synth.review', 'UNIT_IDENTITY_VERSION'),
    ('synth.identity', 'IDENTITY_PROMPT_VERSION'),
    ('synth.identity', 'IDENTITY_RECORD_VERSION'),
    ('core.identity_plan', 'PLAN_VERSION'),
    ('core.identity_context', 'CONTEXT_VERSION'),
    ('core.registry', '_SCHEMA_VERSION'),
)


@pytest.mark.parametrize('mode', ['required', 'off'])
@pytest.mark.parametrize('synthesis', ['single', 'two_pass'])
@pytest.mark.parametrize('module,name', VERSION_PROBES)
def test_finite_version_matrix_matches_inner_semantics(monkeypatch, mode, synthesis, module, name):
    config = Config(grounding_review=mode, synthesis_mode=synthesis)
    before = _compiler_versions(config)
    checkpoint = _pipeline_checkpoint(config, _artifact_versions_for_config(config))
    mod = importlib.import_module('obsidian_llm_wiki.' + module)
    value = getattr(mod, name)
    monkeypatch.setattr(mod, name, value + 1 if isinstance(value, int) else value + '-next')
    after = _compiler_versions(config)
    inner_changed = checkpoint != _pipeline_checkpoint(
        config, _artifact_versions_for_config(config),
    )
    inactive = (name == '_SCHEMA_VERSION'
                or name == 'REVIEW_PROMPT_VERSION' and mode == 'off'
                or name == '_TWO_PASS_GROUNDED_CONTRACT'
                and (mode == 'off' or synthesis == 'single'))
    assert (after != before) == (not inactive)
    assert not inner_changed or after != before, 'outer bypasses an inner compatibility change'
    assert set(before) == {'artifacts', 'identity_assessor', 'identity_plan',
                           'review_receipt', 'identity_context'}


async def test_registry_version_refuses_before_noop_without_mutation(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core import registry

    config, sources, _paths, llm = fixture(tmp_path, monkeypatch)
    assert not (await run_pipeline(tmp_path, sources, config)).errors
    before = tree_state(tmp_path)
    calls = llm.call_count
    monkeypatch.setattr(registry, '_SCHEMA_VERSION', registry._SCHEMA_VERSION + 1)
    with pytest.raises(RuntimeError, match='unsupported generation registry schema'):
        await run_pipeline(tmp_path, sources, config)
    assert tree_state(tmp_path) == before
    assert llm.call_count == calls
