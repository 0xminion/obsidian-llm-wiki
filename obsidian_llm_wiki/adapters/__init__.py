"""Adapters — optional tools that work on top of the rendered vault.

These are NOT part of the core pipeline.  They operate on the vault after
it has been rendered:

  * migrate — migrate a legacy Obsidian vault to the new format
  * visualizer — generate an interactive HTML graph
  * bundle_io — export/import the vault as a tarball
  * enrich — web-crawl enrichment agent

The adapters import from the legacy ``pipeline`` package which still
exists for backward compatibility.  Over time these will be rewritten to
use the new ``obsidian_llm_wiki`` package directly.
"""

from __future__ import annotations

__all__ = ["migrate", "visualizer", "bundle_io", "enrich"]
