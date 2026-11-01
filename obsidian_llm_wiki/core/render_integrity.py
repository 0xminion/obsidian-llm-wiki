"""Read-only binding of consumed Markdown bytes to the renderer's inventory.

Container checks and admission receipts alone do not certify a rendered body.
Callers must hash the bytes they actually parse, not reread the path afterward.
This checks a captured projection; it is not semantic review or race immunity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from obsidian_llm_wiki.core.generated_paths import checked_generated_path


def load_render_hashes(wiki: Path) -> dict[str, str]:
    """Read positive renderer ownership without adopting observed page bytes."""
    path = checked_generated_path(wiki / '.llmwiki/render-page-hashes.json', wiki)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {key: sha for key, sha in value.items()
            if isinstance(key, str) and isinstance(sha, str)}


def rendered_page_problem(relative: str, observed_sha256: str,
                          hashes: Mapping[str, str]) -> str:
    """Return an explicit exclusion reason, or empty for a matching projection."""
    expected = hashes.get(relative, '')
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        return 'Rendered page has no valid render hash (unknown ownership)'
    if expected != observed_sha256:
        return 'Rendered page render hash mismatch (edited after generation)'
    return ''
