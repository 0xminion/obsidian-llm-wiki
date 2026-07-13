"""Source loading helpers — read source markdown from ``sources/`` directory.

Both ``olw ingest`` and ``olw build`` need to load all source files from the
vault's ``sources/`` directory.  This module centralises that logic so the
pipeline always receives the full corpus.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
from obsidian_llm_wiki.render.frontmatter import sanitize_tag
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

__all__ = ["load_source_file", "load_sources_from_dir"]

_MAX_SOURCE_METADATA_ITEMS = 32
_MAX_ALIAS_CHARS = 160
_MAX_SOURCE_TYPE_CHARS = 64


def _metadata_strings(
    value: object,
    *,
    max_chars: int,
    transform: Callable[[str], str] | None = None,
) -> list[str]:
    """Return a small, de-duplicated list from list-valued source frontmatter."""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            continue
        cleaned = " ".join(str(item).replace("\x00", "").split())
        if not cleaned or len(cleaned) > max_chars:
            continue
        if transform is not None:
            cleaned = transform(cleaned)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) == _MAX_SOURCE_METADATA_ITEMS:
            break
    return result


def _source_type(value: object) -> str:
    """Normalize the optional frontmatter type used by granularity selection."""
    if not isinstance(value, str):
        return ""
    value = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return value if len(value) <= _MAX_SOURCE_TYPE_CHARS else ""


def _safe_string(value: object, fallback: str = "") -> str:
    """Read scalar frontmatter without allowing a YAML collection as metadata."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return fallback
    return " ".join(str(value).replace("\x00", "").split()) or fallback


def _provenance_strings(value: object) -> tuple[str, ...]:
    """Convert provenance list fields without treating a scalar as characters."""
    return tuple(_metadata_strings(value, max_chars=512))


def load_source_file(path: Path) -> SourceDoc | None:
    """Load a single source markdown file into a SourceDoc.

    Returns ``None`` if the file is empty or unreadable.
    """
    raw = safe_read_file(path)
    if not raw.strip():
        return None
    meta, body = parse_frontmatter(raw)
    title = _safe_string(meta.get("title"), path.stem)
    url_value = meta.get("url") or meta.get("source_url")
    url = _safe_string(url_value) or None
    aliases = _metadata_strings(meta.get("aliases"), max_chars=_MAX_ALIAS_CHARS)
    tags = _metadata_strings(
        meta.get("tags"),
        max_chars=_MAX_SOURCE_TYPE_CHARS,
        transform=lambda tag: sanitize_tag(tag).casefold(),
    )
    source_type = _source_type(
        meta.get("source_type", meta.get("document_type", meta.get("content_type", "")))
    )
    provenance_data = meta.get("provenance")
    if not isinstance(provenance_data, dict):
        provenance_data = {}
    provenance = SourceProvenance(
        requested_url=_safe_string(provenance_data.get("requested_url")),
        resolved_url=_safe_string(provenance_data.get("resolved_url")),
        extracted_url=_safe_string(provenance_data.get("extracted_url")),
        extractor_chain=_provenance_strings(provenance_data.get("extractor_chain")),
        content_type=_safe_string(provenance_data.get("content_type")),
        document_format=_safe_string(provenance_data.get("document_format")),
        retrieved_at=_safe_string(provenance_data.get("retrieved_at")),
        content_sha256=_safe_string(provenance_data.get("content_sha256")),
        diagnostics=_provenance_strings(provenance_data.get("diagnostics")),
    )
    return SourceDoc(
        title=title,
        content=body,
        url=url,
        source_file=path.name,
        provenance=provenance,
        aliases=aliases,
        tags=tags,
        source_type=source_type,
    )


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
