"""Tests for semantic compile operations in pipeline/compile.py.

These tests cover the recently-added semantic cross-linking, concept merging,
and MoC rebuild functionality that uses direct LLM calls instead of Hermes subprocess.
"""

from unittest.mock import MagicMock

import pytest

from pipeline.compile import (
    NoteIndex,
    _semantic_crosslink,
    _semantic_concept_merge,
    _semantic_moc_rebuild,
    _merge_concepts,
    _replace_wikilink_in_dir,
    _add_wikilink,
)
from pipeline.config import Config


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path):
    for d in ["04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources", "06-Config", "Meta/Scripts"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Config(vault_path=tmp_path)


@pytest.fixture
def mock_client():
    """Mock LLM client that returns predictable responses."""
    client = MagicMock()
    client.generate.return_value = ""
    client.embed_batch.return_value = {}
    return client


# ─── NoteIndex ──────────────────────────────────────────────────────────────

class TestNoteIndex:
    def test_load_empty_vault(self, cfg):
        index = NoteIndex()
        index.load(cfg)
        assert index.notes == {}

    def test_load_with_entries(self, cfg):
        (cfg.entries_dir / "test-entry.md").write_text(
            "---\ntitle: Test Entry\ntags:\n  - ai\n---\n\n# Test Entry\n\nContent here.\n"
        )
        index = NoteIndex()
        index.load(cfg)
        assert "test-entry" in index.notes
        assert index.notes["test-entry"]["type"] == "entry"
        assert index.notes["test-entry"]["tags"] == {"ai"}

    def test_similarity_without_embeddings(self, cfg):
        index = NoteIndex()
        index.load(cfg)
        assert index.similarity("a", "b") == 0.0

    def test_similarity_with_embeddings(self, cfg):
        index = NoteIndex()
        index.embeddings["a"] = [1.0, 0.0]
        index.embeddings["b"] = [1.0, 0.0]
        assert index.similarity("a", "b") == pytest.approx(1.0)

        index.embeddings["c"] = [0.0, 1.0]
        assert index.similarity("a", "c") == pytest.approx(0.0)

    def test_embed_all_delegates_to_client(self, cfg, mock_client, monkeypatch):
        """When QMD is not available, NoteIndex falls back to client.embed_batch."""
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": None)
        (cfg.entries_dir / "entry.md").write_text("---\ntitle: T\n---\n\nBody\n")
        index = NoteIndex()
        index.load(cfg)
        # preview strips frontmatter, leaving "Body\n"; text = "T\nBody\n"
        mock_client.embed_batch.return_value = {"T\nBody\n": [0.1, 0.2]}
        index.embed_all(mock_client)
        assert "entry" in index.embeddings


# ─── _add_wikilink ──────────────────────────────────────────────────────────

class TestAddWikilink:
    def test_adds_link_to_linked_concepts_section(self, cfg):
        (cfg.entries_dir / "source.md").write_text(
            "---\ntitle: Source\n---\n\n# Source\n\n## Linked concepts\n\n- [[existing]]\n"
        )
        assert _add_wikilink(cfg, "source", "target", "reason") is True
        content = (cfg.entries_dir / "source.md").read_text()
        assert "[[target]]" in content
        assert "reason" in content

    def test_skips_if_link_already_exists(self, cfg):
        (cfg.entries_dir / "source.md").write_text(
            "---\ntitle: Source\n---\n\n# Source\n\n## Linked concepts\n\n- [[target]]\n"
        )
        assert _add_wikilink(cfg, "source", "target", "reason") is False

    def test_missing_file_returns_false(self, cfg):
        assert _add_wikilink(cfg, "nonexistent", "target", "reason") is False


# ─── _replace_wikilink_in_dir ───────────────────────────────────────────────

class TestReplaceWikilinkInDir:
    def test_replaces_wikilink(self, cfg):
        (cfg.entries_dir / "a.md").write_text("---\ntitle: A\n---\n\nSee [[old-name]] for more.")
        _replace_wikilink_in_dir(cfg.entries_dir, "old-name", "new-name")
        assert "[[new-name]]" in (cfg.entries_dir / "a.md").read_text()
        assert "[[old-name]]" not in (cfg.entries_dir / "a.md").read_text()

    def test_noop_on_missing_dir(self, cfg):
        _replace_wikilink_in_dir(cfg.vault_path / "nonexistent", "old", "new")


# ─── _merge_concepts ────────────────────────────────────────────────────────

class TestMergeConcepts:
    def test_merge_moves_content_and_updates_refs(self, cfg):
        # Setup: canonical concept, duplicate concept, entry linking to duplicate
        (cfg.concepts_dir / "canonical.md").write_text(
            "---\ntitle: Canonical\n---\n\n# Canonical\n\nOriginal body.\n"
        )
        (cfg.concepts_dir / "duplicate.md").write_text(
            "---\ntitle: Duplicate\n---\n\n# Duplicate\n\nDuplicate body.\n"
        )
        (cfg.entries_dir / "entry.md").write_text(
            "---\ntitle: Entry\n---\n\n# Entry\n\nSee [[duplicate]] for context.\n"
        )

        index = NoteIndex()
        index.load(cfg)

        result = _merge_concepts(cfg, "canonical", "duplicate", index)
        assert result is True

        # Duplicate file deleted
        assert not (cfg.concepts_dir / "duplicate.md").exists()

        # Canonical file has merged content
        canonical_content = (cfg.concepts_dir / "canonical.md").read_text()
        assert "Merged from duplicate" in canonical_content
        assert "Duplicate body" in canonical_content

        # Entry updated
        entry_content = (cfg.entries_dir / "entry.md").read_text()
        assert "[[canonical]]" in entry_content
        assert "[[duplicate]]" not in entry_content

    def test_merge_updates_moc_and_concept_refs(self, cfg):
        (cfg.concepts_dir / "canonical.md").write_text("---\ntitle: C\n---\n\n# C\n")
        (cfg.concepts_dir / "dup.md").write_text("---\ntitle: D\n---\n\n# D\n")
        (cfg.mocs_dir / "moc.md").write_text("---\ntitle: M\n---\n\n# M\n\n- [[dup]]\n")
        (cfg.concepts_dir / "other.md").write_text("---\ntitle: O\n---\n\n# O\n\nRelated: [[dup]]\n")

        index = NoteIndex()
        index.load(cfg)

        result = _merge_concepts(cfg, "canonical", "dup", index)
        assert result is True

        assert "[[canonical]]" in (cfg.mocs_dir / "moc.md").read_text()
        assert "[[canonical]]" in (cfg.concepts_dir / "other.md").read_text()

    def test_embed_all_delegates_to_client(self, cfg, mock_client, monkeypatch):
        """When QMD is not available, NoteIndex falls back to client.embed_batch."""
        monkeypatch.setattr("pipeline.qmd._get_client", lambda base_url="": None)
        (cfg.entries_dir / "entry.md").write_text("---\ntitle: T\n---\n\nBody\n")
        index = NoteIndex()
        index.load(cfg)
        # The text key is "T\nBody\n" (title + "\n" + preview)
        mock_client.embed_batch.return_value = {"T\nBody\n": [0.1, 0.2]}
        index.embed_all(mock_client)
        assert "entry" in index.embeddings

    def test_adds_links_from_llm_output(self, cfg, mock_client):
        (cfg.entries_dir / "a.md").write_text(
            "---\ntitle: A\ntags:\n  - ai\n  - ml\n---\n\n# A\n\n## Linked concepts\n\n"
        )
        (cfg.entries_dir / "b.md").write_text(
            "---\ntitle: B\ntags:\n  - ai\n  - ml\n---\n\n# B\n\n## Linked concepts\n\n"
        )

        mock_client.generate.return_value = 'LINK "a" | "b" | shared topic'

        index = NoteIndex()
        index.load(cfg)
        count = _semantic_crosslink(cfg, mock_client, index)
        assert count >= 1

    def test_no_candidates_returns_zero(self, cfg, mock_client):
        index = NoteIndex()
        index.load(cfg)
        assert _semantic_crosslink(cfg, mock_client, index) == 0


# ─── _semantic_concept_merge ────────────────────────────────────────────────

class TestSemanticConceptMerge:
    def test_merges_from_llm_output(self, cfg, mock_client):
        (cfg.concepts_dir / "a.md").write_text("---\ntitle: AI Safety\n---\n\n# AI Safety\n\nAbout AI safety.\n")
        (cfg.concepts_dir / "b.md").write_text("---\ntitle: AI Safety Research\n---\n\n# AI Safety Research\n\nSame thing.\n")

        mock_client.generate.return_value = 'MERGE a | b | same concept'

        index = NoteIndex()
        index.load(cfg)
        merged = _semantic_concept_merge(cfg, mock_client, index)
        assert merged == 1
        assert not (cfg.concepts_dir / "b.md").exists()

    def test_no_candidates_returns_zero(self, cfg, mock_client):
        index = NoteIndex()
        index.load(cfg)
        assert _semantic_concept_merge(cfg, mock_client, index) == 0


# ─── _semantic_moc_rebuild ──────────────────────────────────────────────────

class TestSemanticMocRebuild:
    def test_rebuilds_moc_from_llm_output(self, cfg, mock_client):
        (cfg.mocs_dir / "topic.md").write_text(
            "---\ntitle: Topic\n---\n\n# Topic\n\n## Overview\n\nOld overview.\n"
        )
        (cfg.entries_dir / "entry.md").write_text(
            "---\ntitle: Entry\ntags:\n  - topic\n---\n\n# Entry\n\nAbout topic.\n"
        )

        mock_client.generate.return_value = "## Overview\n\nNew overview.\n\n## Notes\n\n- [[entry]] — about topic\n"

        index = NoteIndex()
        index.load(cfg)
        updated = _semantic_moc_rebuild(cfg, mock_client, index)
        assert updated == 1
        content = (cfg.mocs_dir / "topic.md").read_text()
        assert "New overview" in content

    def test_no_mocs_returns_zero(self, cfg, mock_client):
        index = NoteIndex()
        index.load(cfg)
        assert _semantic_moc_rebuild(cfg, mock_client, index) == 0
