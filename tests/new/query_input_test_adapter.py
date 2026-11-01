"""TEMPORARY integration adapter, never imported by application code.

Enable explicitly with pytest -p tests.new.query_input_test_adapter until the
publication_inputs lane is integrated. Only accepts original absolute paths
inside this exact test vault; there is deliberately no relocation/rebasing.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def input_freshness(vault_root, manifest):
    root = Path(vault_root).resolve()
    changed, reasons = [], []
    observed = manifest.get('source_inputs')
    if not isinstance(observed, list):
        return {'status': 'unverifiable', 'changed_paths': [], 'reasons': ['missing capture']}
    captured = set()
    for name, expected in observed:
        path = Path(name)
        if not path.is_absolute() or not path.is_relative_to(root):
            reasons.append('Test adapter cannot map publication input')
            continue
        relative = path.relative_to(root).as_posix()
        captured.add(relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            changed.append(relative)
        if relative.startswith('04-Wiki/sources/') and expected is None:
            reasons.append('API-only captured source has no observable human input')
    for path in (root / '04-Wiki/sources').glob('*.md'):
        relative = path.relative_to(root).as_posix()
        if relative not in captured:
            changed.append(relative)
    return {'status': 'unverifiable' if reasons else 'drifted' if changed else 'current',
            'changed_paths': sorted(set(changed)), 'reasons': reasons}


def pytest_configure(config):
    name = 'obsidian_llm_wiki.core.publication_inputs'
    if importlib.util.find_spec(name) is None:
        module = ModuleType(name)
        module.input_freshness = input_freshness
        sys.modules[name] = module
        config._query_input_adapter = True


def pytest_unconfigure(config):
    if getattr(config, '_query_input_adapter', False):
        sys.modules.pop('obsidian_llm_wiki.core.publication_inputs', None)
