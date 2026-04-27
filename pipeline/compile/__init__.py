"""Compile package — deterministic + semantic compile passes for the wiki vault.

Re-exports all public symbols for backward compatibility with ``from pipeline.compile import …``.
"""

from __future__ import annotations

# --- core ---
from pipeline.compile.core import (
    CompileResult,
    IncrementalCompiler,
    VaultSnapshot,
    _archive_duplicate,
    _compiling,
    _do_merge,
    _parse_agent_metrics,
    _process_merge_queue,
    _run_agent,
    _run_agent_compile,
    run_compile,
)

# --- semantic ---
from pipeline.compile.semantic import (
    NoteIndex,
    _add_wikilink,
    _merge_concepts,
    _replace_wikilink_in_dir,
    _run_semantic_compile,
    _semantic_concept_merge,
    _semantic_crosslink,
    _semantic_moc_rebuild,
)

# --- structural ---
from pipeline.compile.structural import (
    _build_edges,
    _detect_duplicates,
    _rebuild_wiki_index,
)

# --- watch ---
from pipeline.compile.watch import watch_compile

# Re-export _load_prompt so monkeypatching via ``pipeline.compile._load_prompt`` works in tests.
from pipeline.utils import load_prompt as _load_prompt

__all__ = [
    "CompileResult",
    "IncrementalCompiler",
    "VaultSnapshot",
    "_archive_duplicate",
    "_compiling",
    "_do_merge",
    "_parse_agent_metrics",
    "_process_merge_queue",
    "_run_agent",
    "_run_agent_compile",
    "run_compile",
    "NoteIndex",
    "_add_wikilink",
    "_merge_concepts",
    "_replace_wikilink_in_dir",
    "_run_semantic_compile",
    "_semantic_concept_merge",
    "_semantic_crosslink",
    "_semantic_moc_rebuild",
    "_build_edges",
    "_detect_duplicates",
    "_rebuild_wiki_index",
    "watch_compile",
    "_load_prompt",
]
