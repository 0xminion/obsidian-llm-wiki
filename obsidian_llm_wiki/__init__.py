"""obsidian-llm-wiki — LLM-powered knowledge compiler for Obsidian vaults.

Ingests web sources and clippings, runs a single LLM synthesis call that
produces a structured JSON bundle (summaries, tags, concepts, relationships,
citations, MOCs), then renders an Obsidian vault with wikilinks and
frontmatter.  All rendering is deterministic — the LLM only produces the
intermediate synthesis; markdown generation is pure functions.
"""

__version__ = "3.0.0"
