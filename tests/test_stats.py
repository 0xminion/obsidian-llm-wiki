"""Tests for pipeline.stats module."""

from pathlib import Path

import pytest

from pipeline.config import Config
from pipeline.stats import generate_dashboard, run_stats


@pytest.fixture
def cfg(tmp_path):
    """Create a minimal vault structure."""
    for d in ["04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources", "06-Config"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    (tmp_path / "06-Config" / "edges.tsv").write_text("source\ttarget\ttype\tdescription\n")
    (tmp_path / "06-Config" / "log.md").write_text("# Wiki Activity Log\n\n---\n\n")
    (tmp_path / "06-Config" / "url-index.tsv").write_text("")

    return Config(vault_path=tmp_path)


class TestGenerateDashboard:
    def test_generates_valid_markdown(self, cfg):
        content = generate_dashboard(cfg)
        assert "# Wiki Dashboard" in content
        assert "## Vault Size" in content
        assert "## Growth" in content
        assert "## Review Status" in content
        assert "## Health" in content
        assert "## Recent Activity" in content

    def test_counts_empty_vault(self, cfg):
        content = generate_dashboard(cfg)
        assert "| Entries | 0 |" in content
        assert "| Concepts | 0 |" in content
        assert "| Sources | 0 |" in content
        assert "| MoCs | 0 |" in content

    def test_counts_with_notes(self, cfg):
        # Add some notes
        (cfg.entries_dir / "entry1.md").write_text("# Entry 1\n")
        (cfg.entries_dir / "entry2.md").write_text("# Entry 2\n")
        (cfg.concepts_dir / "concept1.md").write_text("# Concept 1\n")
        (cfg.sources_dir / "source1.md").write_text("# Source 1\n")

        content = generate_dashboard(cfg)
        assert "| Entries | 2 |" in content
        assert "| Concepts | 1 |" in content
        assert "| Sources | 1 |" in content

    def test_review_status_counts(self, cfg):
        (cfg.entries_dir / "reviewed.md").write_text(
            "---\nreviewed: \"2026-04-19\"\ndate_entry: \"2026-04-19\"\n---\n# Reviewed\n"
        )
        (cfg.entries_dir / "unreviewed.md").write_text(
            "---\nreviewed: \"\"\ndate_entry: \"2026-04-19\"\n---\n# Unreviewed\n"
        )

        content = generate_dashboard(cfg)
        assert "| Reviewed | 1 |" in content
        assert "| Unreviewed | 1 |" in content


class TestRunStats:
    def test_returns_summary(self, cfg):
        summary = run_stats(cfg)
        assert "total" in summary
        assert "entries" in summary
        assert "concepts" in summary
        assert "dashboard_path" in summary

    def test_writes_dashboard_file(self, cfg):
        run_stats(cfg)
        assert (cfg.config_dir / "dashboard.md").exists()

    def test_counts_match(self, cfg):
        (cfg.entries_dir / "a.md").write_text("# A\n")
        (cfg.entries_dir / "b.md").write_text("# B\n")
        summary = run_stats(cfg)
        assert summary["entries"] == 2
        assert summary["total"] == 2
