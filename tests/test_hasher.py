"""Tests for pipeline.hasher — hash_file stability and detect_changes."""

from __future__ import annotations

from pathlib import Path

from pipeline.hasher import detect_changes, hash_content, hash_file
from pipeline.models import SourceState, WikiState


class TestHashFile:
    """hash_file basic properties."""

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Identical content should produce identical hashes."""
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("hello world", encoding="utf-8")
        f2.write_text("hello world", encoding="utf-8")
        assert hash_file(f1) == hash_file(f2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content should produce different hashes."""
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("hello world", encoding="utf-8")
        f2.write_text("hello earth", encoding="utf-8")
        assert hash_file(f1) != hash_file(f2)

    def test_hash_is_hex_sha256(self, tmp_path: Path) -> None:
        """Hash should be a 64-character hex string (SHA-256)."""
        f = tmp_path / "a.md"
        f.write_text("some content", encoding="utf-8")
        result = hash_file(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_content_matches_hash_file(self, tmp_path: Path) -> None:
        """hash_content of a string should match hash_file of a file with same content."""
        f = tmp_path / "a.md"
        f.write_text("test content", encoding="utf-8")
        assert hash_file(f) == hash_content("test content")

    def test_binary_non_utf8_file_not_crash(self, tmp_path: Path) -> None:
        """hash_file reads with encoding='utf-8' — binary data that's valid UTF-8 should hash fine.
        Truly invalid UTF-8 bytes would raise UnicodeDecodeError from read_text.
        Here we test valid UTF-8 with high bytes to confirm no crash on non-ASCII content."""
        f = tmp_path / "binary.md"
        # Write bytes that are valid UTF-8 (emoji + accented chars)
        f.write_bytes("héllo wörld 🌍".encode())
        result = hash_file(f)
        assert len(result) == 64


class TestDetectChanges:
    """detect_changes against a WikiState."""

    def test_new_files(self, tmp_path: Path) -> None:
        """Files not in prev_state should be NEW."""
        (tmp_path / "a.md").write_text("content a", encoding="utf-8")
        (tmp_path / "b.md").write_text("content b", encoding="utf-8")

        changes = detect_changes(tmp_path, WikiState(sources={}))
        statuses = {c.file: c.status for c in changes}
        assert statuses == {"a.md": "new", "b.md": "new"}

    def test_unchanged_files(self, tmp_path: Path) -> None:
        """Files with matching hash should be UNCHANGED."""
        f = tmp_path / "a.md"
        content = "unchanged content"
        f.write_text(content, encoding="utf-8")

        prev_hash = hash_content(content)
        prev_state = WikiState(sources={"a.md": SourceState(hash=prev_hash, concepts=[])})

        changes = detect_changes(tmp_path, prev_state)
        statuses = {c.file: c.status for c in changes}
        assert statuses == {"a.md": "unchanged"}

    def test_changed_files(self, tmp_path: Path) -> None:
        """Files with different hash than prev_state should be CHANGED."""
        f = tmp_path / "a.md"
        f.write_text("new content", encoding="utf-8")

        prev_state = WikiState(
            sources={"a.md": SourceState(hash="old_hash_value", concepts=[])}
        )

        changes = detect_changes(tmp_path, prev_state)
        statuses = {c.file: c.status for c in changes}
        assert statuses == {"a.md": "changed"}

    def test_deleted_files(self, tmp_path: Path) -> None:
        """Files in prev_state but not on disk should be DELETED."""
        prev_state = WikiState(
            sources={"gone.md": SourceState(hash="some_hash", concepts=[])}
        )

        # Sources dir exists but is empty
        changes = detect_changes(tmp_path, prev_state)
        statuses = {c.file: c.status for c in changes}
        assert statuses == {"gone.md": "deleted"}

    def test_mixed_changes(self, tmp_path: Path) -> None:
        """Mix of new, changed, unchanged, and deleted files."""
        # On disk: a.md (unchanged), b.md (changed), c.md (new)
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        c = tmp_path / "c.md"
        a.write_text("content a", encoding="utf-8")
        b.write_text("content b modified", encoding="utf-8")
        c.write_text("content c", encoding="utf-8")

        prev_state = WikiState(
            sources={
                "a.md": SourceState(hash=hash_content("content a"), concepts=[]),
                "b.md": SourceState(hash=hash_content("content b original"), concepts=[]),
                "gone.md": SourceState(hash="some_hash", concepts=[]),
            }
        )

        changes = detect_changes(tmp_path, prev_state)
        statuses = {c.file: c.status for c in changes}
        assert statuses == {
            "a.md": "unchanged",
            "b.md": "changed",
            "c.md": "new",
            "gone.md": "deleted",
        }

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty sources dir with empty state should return no changes."""
        changes = detect_changes(tmp_path, WikiState(sources={}))
        assert changes == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Non-existent sources dir should report all prev_state files as deleted."""
        missing_dir = tmp_path / "nonexistent"
        prev_state = WikiState(
            sources={"x.md": SourceState(hash="h1", concepts=[])}
        )
        changes = detect_changes(missing_dir, prev_state)
        statuses = {c.file: c.status for c in changes}
        assert statuses == {"x.md": "deleted"}

    def test_only_md_files_scanned(self, tmp_path: Path) -> None:
        """Non-.md files should be ignored by detect_changes."""
        (tmp_path / "a.md").write_text("md content", encoding="utf-8")
        (tmp_path / "b.txt").write_text("txt content", encoding="utf-8")
        (tmp_path / "c.json").write_text("{}", encoding="utf-8")

        changes = detect_changes(tmp_path, WikiState(sources={}))
        files = {c.file for c in changes}
        assert files == {"a.md"}
