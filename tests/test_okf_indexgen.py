"""Tests for pipeline.okf_indexgen."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.okf_indexgen import (
    append_log_entry,
    generate_bundle_index,
    generate_directory_index,
    generate_log,
)
from pipeline.okf_models import LogEntry

# ── generate_directory_index ──────────────────────────────────────────


def _write_concept(path: Path, title: str, description: str,
                    concept_type: str = "Concept") -> None:
    """Write a minimal OKF concept .md file with frontmatter."""
    content = (
        "---\n"
        f"type: {concept_type}\n"
        f"title: {title}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {title}\n\n{description}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestGenerateDirectoryIndex:
    """generate_directory_index with multiple concept files."""

    def test_multiple_concept_files(self, tmp_path: Path):
        directory = tmp_path / "concepts"
        directory.mkdir()

        files: list[Path] = []
        _write_concept(directory / "alpha.md", "Alpha", "First concept")
        _write_concept(directory / "beta.md", "Beta", "Second concept")
        _write_concept(directory / "gamma.md", "Gamma", "Third concept")
        files = [directory / "alpha.md", directory / "beta.md",
                directory / "gamma.md"]

        out = generate_directory_index(directory, files)

        assert out.startswith("# concepts\n")
        # No frontmatter
        assert not out.startswith("---")
        # Each concept present with title + description
        assert "- [Alpha](/concepts/alpha.md) - First concept" in out
        assert "- [Beta](/concepts/beta.md) - Second concept" in out
        assert "- [Gamma](/concepts/gamma.md) - Third concept" in out

    def test_skips_index_and_log(self, tmp_path: Path):
        directory = tmp_path / "concepts"
        directory.mkdir()

        _write_concept(directory / "alpha.md", "Alpha", "First concept")
        # Create index.md and log.md (should be skipped)
        (directory / "index.md").write_text("# concepts\n", encoding="utf-8")
        (directory / "log.md").write_text("# Change Log\n", encoding="utf-8")

        files = sorted(directory.glob("*.md"))
        out = generate_directory_index(directory, files)

        # Only alpha should appear as a concept bullet
        assert "[Alpha](/concepts/alpha.md)" in out
        # index.md and log.md should not be listed
        assert "[concepts](/concepts/index.md)" not in out
        assert "log.md" not in out

    def test_no_description_omits_dash(self, tmp_path: Path):
        directory = tmp_path / "concepts"
        directory.mkdir()

        # Write a concept with no description in frontmatter
        content = (
            "---\n"
            "type: Concept\n"
            "title: NoDesc\n"
            "---\n\n"
            "Body only\n"
        )
        (directory / "nodesc.md").write_text(content, encoding="utf-8")

        out = generate_directory_index(directory, [directory / "nodesc.md"])
        # Line should end with the link, no trailing " - "
        line = "- [NoDesc](/concepts/nodesc.md)"
        assert line in out
        assert " - \n" not in out

    def test_empty_directory(self, tmp_path: Path):
        directory = tmp_path / "empty"
        out = generate_directory_index(directory, [])
        assert out == f"# {directory.name}\n\n"


# ── generate_bundle_index ──────────────────────────────────────────────


class TestGenerateBundleIndex:
    """generate_bundle_index with subdirectories."""

    def test_with_subdirectories(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        # Create subdirectories with different concept counts
        concepts_dir = bundle / "concepts"
        concepts_dir.mkdir()
        _write_concept(concepts_dir / "a.md", "A", "desc a")
        _write_concept(concepts_dir / "b.md", "B", "desc b")
        _write_concept(concepts_dir / "c.md", "C", "desc c")

        notes_dir = bundle / "notes"
        notes_dir.mkdir()
        _write_concept(notes_dir / "x.md", "X", "desc x")

        out = generate_bundle_index(bundle)

        # Frontmatter with okf_version
        assert out.startswith("---\n")
        assert "okf_version: '0.1'" in out
        # Header
        assert "# Knowledge Bundle" in out
        # Subdirectory listings with counts (excludes index.md/log.md)
        assert "- [concepts/](/concepts/index.md) (3 concepts)" in out
        assert "- [notes/](/notes/index.md) (1 concepts)" in out

    def test_custom_okf_version(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        out = generate_bundle_index(bundle, okf_version="0.2")
        assert "okf_version: '0.2'" in out

    def test_bundle_excludes_index_and_log_from_count(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        concepts_dir = bundle / "concepts"
        concepts_dir.mkdir()
        _write_concept(concepts_dir / "real.md", "Real", "real desc")
        # index.md and log.md in the subdir should not count
        (concepts_dir / "index.md").write_text("# concepts\n", encoding="utf-8")
        (concepts_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

        out = generate_bundle_index(bundle)
        assert "(1 concepts)" in out

    def test_bundle_with_no_subdirectories(self, tmp_path: Path):
        bundle = tmp_path / "empty_bundle"
        bundle.mkdir()
        out = generate_bundle_index(bundle)
        assert out.startswith("---\n")
        assert "# Knowledge Bundle" in out
        # No subdirectory bullets
        assert "concepts" not in out


# ── generate_log ────────────────────────────────────────────────────────


class TestGenerateLog:
    """generate_log with multiple entries on different dates."""

    def test_multiple_entries_different_dates(self):
        entries = [
            LogEntry(date="2025-01-01", action="created",
                     concept_id="alpha", description="Created alpha"),
            LogEntry(date="2025-06-17", action="updated",
                     concept_id="beta", description="Updated beta"),
            LogEntry(date="2025-03-10", action="deleted",
                     concept_id="gamma", description="Deleted gamma"),
        ]
        out = generate_log(entries)

        assert out.startswith("# Change Log\n")
        # Newest date first
        assert out.index("## 2025-06-17") < out.index("## 2025-03-10")
        assert out.index("## 2025-03-10") < out.index("## 2025-01-01")
        # Entry content
        assert "- **updated** [beta](/beta.md) - Updated beta" in out
        assert "- **created** [alpha](/alpha.md) - Created alpha" in out
        assert "- **deleted** [gamma](/gamma.md) - Deleted gamma" in out

    def test_entries_grouped_by_date(self):
        entries = [
            LogEntry(date="2025-01-01", action="created",
                     concept_id="a", description="first"),
            LogEntry(date="2025-01-01", action="updated",
                     concept_id="b", description="second"),
            LogEntry(date="2025-06-17", action="created",
                     concept_id="c", description="third"),
        ]
        out = generate_log(entries)

        # Two date sections
        assert out.count("## ") == 2
        # Both entries on the same date appear under that section
        idx_jan = out.index("## 2025-01-01")
        idx_jun = out.index("## 2025-06-17")
        idx_a = out.index("[a](/a.md)")
        idx_b = out.index("[b](/b.md)")
        # a and b are under the Jan section, which comes *after* Jun (newest first)
        assert idx_jun < idx_jan
        assert idx_jan < idx_a < idx_b

    def test_empty_list(self):
        out = generate_log([])
        assert out.startswith("# Change Log\n")
        # No date sections
        assert "## " not in out

    def test_log_entry_format(self):
        entry = LogEntry(date="2025-01-01", action="created",
                         concept_id="foo", description="Made foo")
        out = generate_log([entry])
        assert "- **created** [foo](/foo.md) - Made foo" in out


# ── append_log_entry ───────────────────────────────────────────────────


class TestAppendLogEntry:
    """append_log_entry to existing log."""

    def test_append_to_existing_date(self):
        existing = (
            "# Change Log\n\n"
            "## 2025-01-01\n\n"
            "- **created** [alpha](/alpha.md) - First\n\n"
        )
        entry = LogEntry(date="2025-01-01", action="updated",
                         concept_id="beta", description="Second")
        out = append_log_entry(existing, entry)

        # Header preserved
        assert out.startswith("# Change Log\n")
        # Only one section for that date
        assert out.count("## 2025-01-01") == 1
        # New entry appended after the existing one
        assert "- **created** [alpha](/alpha.md) - First" in out
        assert "- **updated** [beta](/beta.md) - Second" in out
        assert out.index("[alpha](/alpha.md)") < out.index("[beta](/beta.md)")

    def test_append_creates_new_date_section(self):
        existing = (
            "# Change Log\n\n"
            "## 2025-01-01\n\n"
            "- **created** [alpha](/alpha.md) - First\n\n"
        )
        # New entry on a newer date
        entry = LogEntry(date="2025-06-17", action="created",
                         concept_id="beta", description="Newer")
        out = append_log_entry(existing, entry)

        # Two date sections now
        assert out.count("## ") == 2
        # Newer date section appears first (newest first ordering)
        assert out.index("## 2025-06-17") < out.index("## 2025-01-01")
        assert "- **created** [beta](/beta.md) - Newer" in out
        # Existing content preserved
        assert "- **created** [alpha](/alpha.md) - First" in out

    def test_append_to_empty_log(self):
        # If the existing log has no proper header, build from scratch.
        existing = ""
        entry = LogEntry(date="2025-01-01", action="created",
                         concept_id="foo", description="bar")
        out = append_log_entry(existing, entry)
        assert out.startswith("# Change Log\n")
        assert "## 2025-01-01" in out
        assert "- **created** [foo](/foo.md) - bar" in out

    def test_append_multiple_dates_preserves_order(self):
        existing = (
            "# Change Log\n\n"
            "## 2025-06-17\n\n"
            "- **created** [a](/a.md) - first\n\n"
            "## 2025-01-01\n\n"
            "- **created** [b](/b.md) - second\n\n"
        )
        # Append to the older section
        entry = LogEntry(date="2025-01-01", action="updated",
                         concept_id="c", description="third")
        out = append_log_entry(existing, entry)

        # Order preserved: Jun still first
        assert out.index("## 2025-06-17") < out.index("## 2025-01-01")
        # New entry appended within Jan section
        assert "- **updated** [c](/c.md) - third" in out
        idx_jan = out.index("## 2025-01-01")
        idx_b = out.index("[b](/b.md)")
        idx_c = out.index("[c](/c.md)")
        assert idx_jan < idx_b < idx_c


# ── pytest entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
