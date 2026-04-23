"""Integration tests for the pipeline — stages working together.

These tests exercise the pipeline stages as a unit, mocking all external
calls (curl, hermes, defuddle). No live network or real agents.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.config import Config, load_config
from pipeline.extract import extract_all, extract_url
from pipeline.plan import plan_sources
from pipeline.create import create_all
from pipeline.models import (
    ExtractedSource, Manifest, Plan, Plans, Language, Template, SourceType,
)
from pipeline.vault import reindex, archive_inbox


runner = CliRunner()


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _make_url_file(inbox: Path, url: str, name: str = "article.url") -> Path:
    """Create a .url file in the inbox."""
    p = inbox / name
    p.write_text(f"[InternetShortcut]\nURL={url}\n")
    return p


def _make_extracted_source(url: str, title: str = "Test Article") -> ExtractedSource:
    return ExtractedSource(
        url=url,
        title=title,
        content=f"# {title}\n\nThis is the full content of the article. " * 20,
        type=SourceType.WEB,
        author="Test Author",
    )


def _make_plan(hash_val: str, title: str = "Test Article") -> Plan:
    return Plan(
        hash=hash_val,
        title=title,
        language=Language.EN,
        template=Template.STANDARD,
        tags=["test", "article"],
        concept_updates=[],
        concept_new=["Test Concept"],
        moc_targets=["General"],
    )


def _mock_hermes_create_side_effect(cfg: Config, sources: list[ExtractedSource]):
    """Side effect for hermes agent during create stage — writes vault files directly."""
    for src in sources:
        # Write source
        cfg.sources_dir.mkdir(parents=True, exist_ok=True)
        fname = src.title.lower().replace(" ", "-")[:120]
        (cfg.sources_dir / f"{fname}.md").write_text(
            f"---\ntitle: {src.title}\nsource_url: {src.url}\nsource_type: web\n"
            f"author: {src.author}\ndate_captured: 2026-01-01\ntags: []\nstatus: raw\n---\n"
            f"# {src.title}\n\n{src.content}\n"
        )

        # Write entry
        cfg.entries_dir.mkdir(parents=True, exist_ok=True)
        (cfg.entries_dir / f"{fname}.md").write_text(
            f"---\ntitle: {src.title}\nsource: \"[[{fname}]]\"\n"
            f"date_entry: 2026-01-01\nstatus: draft\ntemplate: standard\ntags:\n  - test\n---\n"
            f"# {src.title}\n\n## Summary\n\nA summary of the article.\n\n"
            f"## Core insights\n\n- Insight 1\n- Insight 2\n\n"
            f"## Linked concepts\n\n- [[Test Concept]]\n"
        )

        # Write concept
        cfg.concepts_dir.mkdir(parents=True, exist_ok=True)
        (cfg.concepts_dir / "test-concept.md").write_text(
            f"---\ntitle: Test Concept\ntype: concept\nstatus: draft\nsources:\n  - {fname}\ntags: []\n---\n"
            f"# Test Concept\n\n## Core concept\n\nA test concept.\n\n"
            f"## Context\n\nContext here.\n\n## Links\n\n- [[{fname}]]\n"
        )


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a minimal vault directory structure."""
    for d in ["01-Raw", "04-Wiki/sources", "04-Wiki/entries",
              "04-Wiki/concepts", "04-Wiki/mocs", "06-Config",
              "08-Archive-Raw", "Meta/Scripts", "Meta/prompts",
              "Meta/Templates"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def cfg(vault: Path) -> Config:
    """Config pointing to the test vault."""
    extract_dir = vault / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    return Config(vault_path=vault, extract_dir=extract_dir)


# ─── test_full_pipeline_ingest ─────────────────────────────────────────────────

class TestFullPipelineIngest:
    """Test the full ingest flow via CLI with mocked external calls."""

    def test_ingest_dry_run(self, vault: Path):
        """--dry-run should not write any vault files."""
        _make_url_file(vault / "01-Raw", "https://example.com/article", "article.url")

        result = runner.invoke(app, [
            "ingest", str(vault), "--dry-run"
        ])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        # No files should be created in entries/concepts
        entries = list((vault / "04-Wiki" / "entries").glob("*.md"))
        assert len(entries) == 0

    def test_ingest_review_mode(self, vault: Path):
        """--review should stage files for approval and exit before Stage 3."""
        _make_url_file(vault / "01-Raw", "https://example.com/article", "article.url")

        # Mock extract_all, plan_sources, and stage_for_review
        url = "https://example.com/article"
        h = _url_hash(url)
        source = _make_extracted_source(url)

        with patch("pipeline.cli.extract_all") as mock_extract, \
             patch("pipeline.cli.plan_sources") as mock_plan, \
             patch("pipeline.review.stage_for_review") as mock_stage:
            mock_extract.return_value = Manifest(entries=[source])
            plan = _make_plan(h)
            mock_plan.return_value = Plans(plans=[plan])
            mock_stage.return_value = {"staged": 1, "failed": 0}

            result = runner.invoke(app, [
                "ingest", str(vault), "--review"
            ])

        assert result.exit_code == 0
        assert "review mode" in result.output.lower()
        assert "staged" in result.output.lower()
        assert "approve" in result.output.lower()

    def test_ingest_resume_mode(self, vault: Path):
        """--resume should skip Stages 1+2 and use saved manifest + plans."""
        extract_dir = Config(vault_path=vault).resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)

        url = "https://example.com/article"
        h = _url_hash(url)
        source = _make_extracted_source(url)

        # Pre-save manifest and plans
        Manifest(entries=[source]).save(extract_dir)
        plan = _make_plan(h)
        Plans(plans=[plan]).save(extract_dir)

        with patch("pipeline.cli.create_all") as mock_create:
            mock_create.return_value = {"created": 1, "failed": 0, "sources": 1, "entries": 1}

            result = runner.invoke(app, [
                "ingest", str(vault), "--resume"
            ])

        assert result.exit_code == 0
        assert "SKIPPED (--resume)" in result.output
        mock_create.assert_called_once()

    def test_ingest_no_urls(self, vault: Path):
        """Empty inbox should exit cleanly."""
        result = runner.invoke(app, ["ingest", str(vault)])
        assert result.exit_code == 0
        assert "No .url files" in result.output


# ─── test_extract_to_plan_flow ─────────────────────────────────────────────────

class TestExtractToPlanFlow:
    """Test Stage 1 → Stage 2 handoff."""

    def test_manifest_passes_to_plan(self, cfg: Config):
        """Extract output (Manifest) should be readable by plan_sources."""
        url = "https://example.com/test-article"
        source = _make_extracted_source(url, "Test Article")
        manifest = Manifest(entries=[source])
        manifest.save(cfg.resolved_extract_dir)

        # Verify plan_sources can load it
        loaded = Manifest.load(cfg.resolved_extract_dir)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].url == url
        assert loaded.entries[0].title == "Test Article"

    def test_multiple_sources_handoff(self, cfg: Config):
        """Multiple sources should round-trip through manifest."""
        sources = [
            _make_extracted_source(f"https://example.com/a{i}", f"Article {i}")
            for i in range(4)
        ]
        manifest = Manifest(entries=sources)
        manifest.save(cfg.resolved_extract_dir)

        loaded = Manifest.load(cfg.resolved_extract_dir)
        assert len(loaded.entries) == 4
        hashes = {e.hash for e in loaded.entries}
        assert len(hashes) == 4  # all unique


# ─── test_plan_to_create_flow ──────────────────────────────────────────────────

class TestPlanToCreateFlow:
    """Test Stage 2 → Stage 3 handoff."""

    def test_plans_pass_to_create(self, cfg: Config):
        """Plans should be serializable and loadable by create stage."""
        plans = Plans(plans=[
            _make_plan(_url_hash("https://example.com/a"), "Article A"),
            _make_plan(_url_hash("https://example.com/b"), "Article B"),
        ])
        plans.save(cfg.resolved_extract_dir)

        loaded = Plans.load(cfg.resolved_extract_dir)
        assert len(loaded.plans) == 2
        assert loaded.plans[0].title == "Article A"

    def test_batch_split(self):
        """Plans.split_batches should correctly split into N batches."""
        plans = Plans(plans=[
            Plan(hash=f"hash{i}", title=f"Plan {i}")
            for i in range(7)
        ])
        batches = plans.split_batches(3)
        total = sum(len(b) for b in batches)
        assert total == 7
        assert len(batches) == 3


# ─── test_lint_command ─────────────────────────────────────────────────────────

class TestLintCommand:
    """Test the lint CLI command."""

    def test_lint_clean_vault(self, vault: Path):
        """A healthy vault should pass lint."""
        # Create well-formed vault contents
        (vault / "06-Config" / "wiki-index.md").write_text(
            "# Wiki Index\n\n---\n\n## Entries\n\n- [[test-entry]]: Test entry (entry)\n\n## Concepts\n\n- [[test-concept]]: Test concept (concept)\n\n## Maps of Content\n\n- [[test-moc]]: Overview of test MoC (moc)\n\n---\n\n*Reindexed*\n"
        )
        (vault / "06-Config" / "edges.tsv").write_text("source\ttarget\ttype\tdescription\n")

        (vault / "04-Wiki" / "entries" / "test-entry.md").write_text(
            '---\ntitle: "Test Entry"\nsource: "[[test-source]]"\ndate_entry: "2026-04-19"\nstatus: "draft"\ntemplate: "standard"\ntags:\n  - test\nreviewed: "2026-04-19"\n---\n# Test Entry\n\n## Summary\n\nThis is a well-formed test entry with enough content to pass all checks.\n\n## Core insights\n\nFirst insight from the source material that adds value.\n\n## Other takeaways\n\nA key takeaway that provides additional context and understanding.\n\n## Diagrams\n\nn/a\n\n## Open questions\n\nWhat should we explore next about this topic?\n\n## Linked concepts\n\n- [[test-concept]]\n\n## Maps\n\n- [[test-moc]]\n'
        )
        (vault / "04-Wiki" / "concepts" / "test-concept.md").write_text(
            '---\ntitle: "Test Concept"\ntype: "concept"\nlanguage: "en"\nstatus: "draft"\nsources: []\ntags: []\n---\n# Test Concept\n\n## Core concept\n\nA test concept for the lint check that provides foundational understanding.\n\n## Context\n\nThe context of this concept is testing. It verifies that the lint module works correctly on well-formed content with sufficient depth and detail.\n\n## Links\n\n- [[test-entry]]\n'
        )
        (vault / "04-Wiki" / "sources" / "test-source.md").write_text(
            '---\ntitle: "Test Source"\nsource_url: "https://example.com/article"\nsource_type: "blog"\nauthor: "Test Author"\ndate_captured: "2026-04-19"\ntags:\n  - test\nstatus: "raw"\n---\n# Test Source\n\nThis is a detailed source note with enough body content to pass the empty note check. The source material covers interesting topics that will be processed into entries and concepts.\n'
        )
        (vault / "04-Wiki" / "mocs" / "test-moc.md").write_text(
            '---\ntitle: "Test MoC"\ntype: "moc"\nstatus: "draft"\ntags: []\n---\n# Test MoC\n\n## Overview / 概述\n\nOverview of test MoC that provides a comprehensive map of the topics covered.\n\n## Topic / 主题\n\n- [[test-entry]]: A test entry that covers key topics\n\n'
        )
        result = runner.invoke(app, ["lint", str(vault)])
        assert result.exit_code == 0
        assert "passed" in result.output

    def test_lint_catches_stubs(self, vault: Path):
        """Lint should catch stub content."""
        entries = vault / "04-Wiki" / "entries"
        entries.mkdir(parents=True, exist_ok=True)
        (entries / "stub-entry.md").write_text(
            "---\ntitle: Stub\ntags: []\n---\n# Stub\n\n> TODO: fill this in\n"
        )
        (vault / "06-Config" / "wiki-index.md").write_text("# Wiki Index\n")
        (vault / "06-Config" / "edges.tsv").write_text("source\ttarget\ttype\tdescription\n")
        result = runner.invoke(app, ["lint", str(vault)])
        assert result.exit_code == 1
        assert "Stubs" in result.output or "stubs" in result.output.lower()


# ─── test_reindex_command ──────────────────────────────────────────────────────

class TestReindexCommand:
    """Test the reindex CLI command."""

    def test_reindex_creates_index(self, vault: Path):
        """reindex should create wiki-index.md."""
        entries = vault / "04-Wiki" / "entries"
        entries.mkdir(parents=True, exist_ok=True)
        (entries / "test.md").write_text(
            "---\ntitle: Test\n---\n# Test\n## Summary\nA summary.\n"
        )
        result = runner.invoke(app, ["reindex", str(vault)])
        assert result.exit_code == 0
        assert "Rebuilt" in result.output
        assert (vault / "06-Config" / "wiki-index.md").exists()


# ─── test_stats_command ────────────────────────────────────────────────────────

class TestStatsCommand:
    """Test the stats CLI command."""

    def test_stats_shows_counts(self, vault: Path):
        """Stats should show correct counts."""
        entries = vault / "04-Wiki" / "entries"
        entries.mkdir(parents=True, exist_ok=True)
        (entries / "a.md").write_text("test")
        (entries / "b.md").write_text("test")

        result = runner.invoke(app, ["stats", str(vault)])
        assert result.exit_code == 0
        assert "Entries:  2" in result.output

    def test_stats_empty_vault(self, vault: Path):
        """Stats should handle empty vault."""
        result = runner.invoke(app, ["stats", str(vault)])
        assert result.exit_code == 0
        assert "Entries:  0" in result.output


# ─── test_validate_command ─────────────────────────────────────────────────────

class TestValidateCommand:
    """Test the validate CLI command."""

    def test_validate_empty_vault(self, vault: Path):
        """validate on empty vault should pass."""
        result = runner.invoke(app, ["validate", str(vault)])
        assert result.exit_code == 0
        assert "passed" in result.output

    def test_validate_catches_issues(self, vault: Path):
        """validate should catch entry violations."""
        entries = vault / "04-Wiki" / "entries"
        entries.mkdir(parents=True, exist_ok=True)

        # Create a bad entry (missing required sections, no H1)
        (entries / "bad-entry.md").write_text(
            '---\ntitle: "Bad Entry"\n---\nJust prose, no sections.\n'
        )
        result = runner.invoke(app, ["validate", str(vault)])
        assert result.exit_code == 1
        assert result.output
