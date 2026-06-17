"""Tests for pipeline.state — read_state, write_state, remove_source_state."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.models import SourceState, WikiState
from pipeline.state import read_state, remove_source_state, write_state


class TestReadState:
    """read_state edge cases."""

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        """read_state on non-existent file should return empty WikiState."""
        state = read_state(tmp_path / "state.json")
        assert isinstance(state, WikiState)
        assert state.sources == {}

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        """read_state on malformed JSON should return empty WikiState."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{ this is not valid json {{{", encoding="utf-8")
        state = read_state(state_file)
        assert isinstance(state, WikiState)
        assert state.sources == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """read_state on empty file should return empty WikiState."""
        state_file = tmp_path / "state.json"
        state_file.write_text("", encoding="utf-8")
        state = read_state(state_file)
        assert isinstance(state, WikiState)
        assert state.sources == {}

    def test_valid_state(self, tmp_path: Path) -> None:
        """read_state on valid JSON should return correct WikiState."""
        state_file = tmp_path / "state.json"
        data = {
            "sources": {
                "file1.md": {
                    "hash": "abc123",
                    "concepts": ["concept1", "concept2"],
                    "compiled_at": "2024-01-01T00:00:00Z",
                },
                "file2.md": {
                    "hash": "def456",
                    "concepts": [],
                    "compiled_at": None,
                },
            }
        }
        state_file.write_text(json.dumps(data), encoding="utf-8")
        state = read_state(state_file)
        assert isinstance(state, WikiState)
        assert set(state.sources.keys()) == {"file1.md", "file2.md"}

        s1 = state.sources["file1.md"]
        assert s1.hash == "abc123"
        assert s1.concepts == ["concept1", "concept2"]
        assert s1.compiled_at == "2024-01-01T00:00:00Z"

        s2 = state.sources["file2.md"]
        assert s2.hash == "def456"
        assert s2.concepts == []
        assert s2.compiled_at is None

    def test_valid_state_no_sources_key(self, tmp_path: Path) -> None:
        """read_state on JSON without 'sources' key should return empty WikiState."""
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"other_key": "value"}), encoding="utf-8")
        state = read_state(state_file)
        assert isinstance(state, WikiState)
        assert state.sources == {}

    def test_compiled_at_snake_case(self, tmp_path: Path) -> None:
        """read_state should handle compiled_at written as snake_case."""
        state_file = tmp_path / "state.json"
        data = {
            "sources": {
                "f.md": {
                    "hash": "h1",
                    "concepts": [],
                    "compiled_at": "2024-06-01T12:00:00Z",
                }
            }
        }
        state_file.write_text(json.dumps(data), encoding="utf-8")
        state = read_state(state_file)
        assert state.sources["f.md"].compiled_at == "2024-06-01T12:00:00Z"


class TestWriteState:
    """write_state and roundtrip."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """write_state then read_state should preserve all data."""
        state_file = tmp_path / "state.json"
        original = WikiState(
            sources={
                "alpha.md": SourceState(
                    hash="hash_aaa",
                    concepts=["concept_x", "concept_y"],
                    compiled_at="2024-03-15T10:30:00Z",
                ),
                "beta.md": SourceState(
                    hash="hash_bbb",
                    concepts=[],
                    compiled_at=None,
                ),
            }
        )
        write_state(state_file, original)
        assert state_file.exists()

        loaded = read_state(state_file)
        assert set(loaded.sources.keys()) == {"alpha.md", "beta.md"}

        a = loaded.sources["alpha.md"]
        assert a.hash == "hash_aaa"
        assert a.concepts == ["concept_x", "concept_y"]
        assert a.compiled_at == "2024-03-15T10:30:00Z"

        b = loaded.sources["beta.md"]
        assert b.hash == "hash_bbb"
        assert b.concepts == []
        assert b.compiled_at is None

    def test_write_empty_state(self, tmp_path: Path) -> None:
        """Writing an empty WikiState should produce valid JSON with empty sources."""
        state_file = tmp_path / "state.json"
        write_state(state_file, WikiState(sources={}))
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data == {"sources": {}}

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """write_state should create parent directories."""
        state_file = tmp_path / "subdir" / "nested" / "state.json"
        write_state(state_file, WikiState(sources={}))
        assert state_file.exists()

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """write_state should atomically overwrite an existing state file."""
        state_file = tmp_path / "state.json"
        write_state(state_file, WikiState(sources={
            "old.md": SourceState(hash="old", concepts=[], compiled_at="2024-01-01")
        }))
        write_state(state_file, WikiState(sources={
            "new.md": SourceState(hash="new", concepts=["c1"], compiled_at="2024-06-01")
        }))
        loaded = read_state(state_file)
        assert "old.md" not in loaded.sources
        assert "new.md" in loaded.sources
        assert loaded.sources["new.md"].hash == "new"


class TestRemoveSourceState:
    """remove_source_state tests."""

    def test_remove_existing_entry(self) -> None:
        """remove_source_state should remove an existing entry."""
        state = WikiState(
            sources={
                "a.md": SourceState(hash="h1", concepts=[]),
                "b.md": SourceState(hash="h2", concepts=["c1"]),
            }
        )
        remove_source_state(state, "a.md")
        assert "a.md" not in state.sources
        assert "b.md" in state.sources

    def test_remove_nonexistent_entry_no_crash(self) -> None:
        """remove_source_state on a non-existent entry should not crash."""
        state = WikiState(sources={"a.md": SourceState(hash="h1", concepts=[])})
        # Should not raise KeyError
        remove_source_state(state, "nonexistent.md")
        assert "a.md" in state.sources

    def test_remove_from_empty_state(self) -> None:
        """remove_source_state on empty state should not crash."""
        state = WikiState(sources={})
        remove_source_state(state, "anything.md")
        assert state.sources == {}

    def test_remove_all_entries(self) -> None:
        """Removing all entries should leave an empty sources dict."""
        state = WikiState(
            sources={
                "a.md": SourceState(hash="h1", concepts=[]),
                "b.md": SourceState(hash="h2", concepts=[]),
            }
        )
        remove_source_state(state, "a.md")
        remove_source_state(state, "b.md")
        assert state.sources == {}