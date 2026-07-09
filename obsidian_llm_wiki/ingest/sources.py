"""Source loading helpers — read source markdown from ``sources/`` directory.

Both ``olw ingest`` and ``olw build`` need to load all source files from the
vault's ``sources/`` directory.  This module centralises that logic so the
pipeline always receives the full corpus.
"""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

__all__ = ["load_source_file", "load_sources_from_dir"]


def load_source_file(path: Path) -> SourceDoc | None:
    """Load a single source markdown file into a SourceDoc.

    Returns ``None`` if the file is empty or unreadable.
    """
    raw = safe_read_file(path)
    if not raw.strip():
        return None
    meta, body = parse_frontmatter(raw)
    title = meta.get("title", path.stem)
    url = meta.get("url") or meta.get("source_url") or None
    return SourceDoc(title=title, content=body, url=url, source_file=path.name)


def load_sources_from_dir(sources_dir: Path) -> dict[str, SourceDoc]:
    """Load all ``*.md`` files from ``sources_dir`` into a dict keyed by filename.

    Skips empty/unreadable files.
    """
    if not sources_dir.is_dir():
        return {}
    result: dict[str, SourceDoc] = {}
    for f in sorted(sources_dir.glob("*.md")):
        doc = load_source_file(f)
        if doc is not None:
            result[f.name] = doc
    return result
