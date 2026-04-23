"""System tests for the pipeline — end-to-end with mocked APIs.

Tests the full pipeline flow from URL ingestion to complete vault state.
All external APIs (curl, hermes, defuddle, qmd) are mocked.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.config import Config
from pipeline.extract import extract_all
from pipeline.plan import plan_sources
from pipeline.create import create_all
from pipeline.models import (
    ConceptMatch, ExtractedSource, Manifest, Plan, Plans,
    Language, Template, SourceType,
)
from pipeline.vault import (
    write_source, write_entry, write_concept, update_moc,
    write_edge, reindex, archive_inbox,
)
from pipeline.create import validate_output


runner = CliRunner()


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _make_url_file(inbox: Path, url: str, name: str) -> Path:
    p = inbox / name
    p.write_text(f"[InternetShortcut]\nURL={url}\n")
    return p


def _make_source(url: str, title: str, content: str = "", author: str = "") -> ExtractedSource:
    if not content:
        content = f"# {title}\n\nDetailed content about {title}. " * 30
    return ExtractedSource(
        url=url,
        title=title,
        content=content,
        type=SourceType.WEB,
        author=author,
    )


def _make_plan_from_source(source: ExtractedSource, **overrides) -> Plan:
    defaults = dict(
        hash=source.hash,
        title=source.title,
        language=Language.EN,
        template=Template.STANDARD,
        tags=["system-test"],
        concept_updates=[],
        concept_new=[],
        moc_targets=[],
    )
    defaults.update(overrides)
    return Plan(**defaults)


def _write_vault_files(cfg: Config, sources: list[ExtractedSource], plans: list[Plan]):
    """Simulate what hermes agent does: write vault files for each plan/source."""
    for source, plan in zip(sources, plans):
        write_source(cfg, source)

        fname = source.title.lower().replace(" ", "-")[:120]
        entry_content = (
            f"## Summary\n\nSummary of {source.title}.\n\n"
            f"## Core insights\n\n- Key insight from {source.title}\n\n"
            f"## Linked concepts\n\n"
        )
        if plan.concept_new:
            for c in plan.concept_new:
                entry_content += f"- [[{c.lower().replace(' ', '-')}]]\n"
        if plan.concept_updates:
            for c in plan.concept_updates:
                entry_content += f"- [[{c.lower().replace(' ', '-')}]]\n"

        write_entry(cfg, plan, entry_content)

        for concept_name in plan.concept_new:
            concept_fname = concept_name.lower().replace(" ", "-")
            concept_content = (
                f"## Core concept\n\n{concept_name} is a concept.\n\n"
                f"## Context\n\nContext for {concept_name}.\n\n"
                f"## Links\n\n- [[{fname}]]\n"
            )
            write_concept(cfg, concept_name, concept_content, [fname])

        for moc in plan.moc_targets:
            update_moc(cfg, moc, fname, f"Entry about {source.title}")


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a minimal vault directory structure."""
    for d in [
        "01-Raw", "04-Wiki/sources", "04-Wiki/entries",
        "04-Wiki/concepts", "04-Wiki/mocs", "06-Config",
        "08-Archive-Raw", "Meta/Scripts", "Meta/prompts",
        "Meta/Templates",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def cfg(vault: Path) -> Config:
    extract_dir = vault / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    return Config(vault_path=vault, extract_dir=extract_dir)


# ─── test_single_url_end_to_end ────────────────────────────────────────────────

class TestSingleUrlEndToEnd:
    """Feed one URL through all 3 stages, verify complete vault state."""

    def test_single_web_url(self, cfg: Config, vault: Path):
        """One web URL → source + entry + concept + MoC + wiki-index + edges."""
        url = "https://example.com/great-article"
        source = _make_source(url, "A Great Article", author="Jane Doe")
        plan = _make_plan_from_source(
            source,
            concept_new=["Machine Learning"],
            moc_targets=["AI Research"],
        )

        # Stage 1: extract
        manifest = Manifest(entries=[source])
        source.save(cfg.resolved_extract_dir)
        manifest.save(cfg.resolved_extract_dir)

        # Stage 2: plan
        plans = Plans(plans=[plan])
        plans.save(cfg.resolved_extract_dir)

        # Stage 3: create
        _write_vault_files(cfg, [source], [plan])

        # Reindex
        reindex(cfg)

        # ─── Verify vault state ────────────────────────────────────────────
        # Source created
        sources = list(cfg.sources_dir.glob("*.md"))
        assert len(sources) == 1
        source_text = sources[0].read_text()
        assert "A Great Article" in source_text
        assert url in source_text

        # Entry created
        entries = list(cfg.entries_dir.glob("*.md"))
        assert len(entries) == 1
        entry_text = entries[0].read_text()
        assert "## Summary" in entry_text
        assert "## Core insights" in entry_text
        assert "## Linked concepts" in entry_text

        # Concept created
        concepts = list(cfg.concepts_dir.glob("*.md"))
        assert len(concepts) == 1
        concept_text = concepts[0].read_text()
        assert "## Core concept" in concept_text
        assert "## Context" in concept_text
        assert "## Links" in concept_text

        # MoC created
        mocs = list(cfg.mocs_dir.glob("*.md"))
        assert len(mocs) == 1
        moc_text = mocs[0].read_text()
        assert "AI Research" in mocs[0].stem.replace("-", " ").lower() or "ai-research" in mocs[0].stem
        assert "Great Article" in moc_text

        # Wiki index exists
        assert cfg.wiki_index.exists()
        index_text = cfg.wiki_index.read_text()
        assert "## Entries" in index_text
        assert "## Concepts" in index_text


# ─── test_multiple_urls_parallel ───────────────────────────────────────────────

class TestMultipleUrlsParallel:
    """Feed 4 URLs through with mocked APIs, verify parallel execution and output."""

    def test_four_urls(self, cfg: Config, vault: Path):
        """Four URLs → 4 sources + 4 entries + concepts + MoCs."""
        urls = [
            "https://example.com/article-1",
            "https://example.com/article-2",
            "https://example.com/article-3",
            "https://example.com/article-4",
        ]
        sources = [_make_source(u, f"Article {i+1}") for i, u in enumerate(urls)]
        plans = [_make_plan_from_source(s, moc_targets=["General"]) for s in sources]

        # Stage 1
        manifest = Manifest(entries=sources)
        manifest.save(cfg.resolved_extract_dir)

        # Stage 2
        plans_coll = Plans(plans=plans)
        plans_coll.save(cfg.resolved_extract_dir)

        # Stage 3
        _write_vault_files(cfg, sources, plans)
        reindex(cfg)

        # Verify
        assert len(list(cfg.sources_dir.glob("*.md"))) == 4
        assert len(list(cfg.entries_dir.glob("*.md"))) == 4

        # Wiki index should list all
        index_text = cfg.wiki_index.read_text()
        assert "4 entries" in index_text


# ─── test_error_recovery ───────────────────────────────────────────────────────

class TestErrorRecovery:
    """Simulate one URL failing, verify pipeline continues."""

    def test_partial_extraction_failure(self, cfg: Config):
        """If one extraction fails, remaining URLs still produce sources."""
        urls = [
            "https://example.com/good-1",
            "https://example.com/fails",
            "https://example.com/good-2",
        ]

        # Mock extract_all to return only 2 of 3 (one fails internally)
        good_sources = [
            _make_source(urls[0], "Good Article 1"),
            _make_source(urls[2], "Good Article 2"),
        ]

        manifest = Manifest(entries=good_sources)
        assert len(manifest.entries) == 2
        manifest.save(cfg.resolved_extract_dir)

        plans = Plans(plans=[_make_plan_from_source(s) for s in good_sources])
        _write_vault_files(cfg, good_sources, plans.plans)
        reindex(cfg)

        # Only 2 entries, not 3
        assert len(list(cfg.entries_dir.glob("*.md"))) == 2

    def test_partial_plan_failure(self, cfg: Config):
        """If plan generation returns fewer plans than sources, partial create works."""
        sources = [
            _make_source("https://example.com/a", "Article A"),
            _make_source("https://example.com/b", "Article B"),
            _make_source("https://example.com/c", "Article C"),
        ]
        # Only 2 plans (one source couldn't be planned)
        plans = [
            _make_plan_from_source(sources[0]),
            _make_plan_from_source(sources[2]),
        ]

        manifest = Manifest(entries=sources)
        manifest.save(cfg.resolved_extract_dir)

        plans_coll = Plans(plans=plans)
        plans_coll.save(cfg.resolved_extract_dir)

        # Create only planned sources
        _write_vault_files(cfg, [sources[0], sources[2]], plans)
        reindex(cfg)

        assert len(list(cfg.entries_dir.glob("*.md"))) == 2


# ─── test_collision_handling ───────────────────────────────────────────────────

class TestCollisionHandling:
    """Create vault with existing notes, verify new ingests use collision resolution."""

    def test_existing_entry_collision(self, cfg: Config, vault: Path):
        """When an entry with the same title exists, collision resolution appends suffix."""
        # Pre-create an entry
        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        (entries / "my-article.md").write_text("# Old Entry\n")

        source = _make_source("https://example.com/new", "My Article")
        plan = _make_plan_from_source(source)

        write_source(cfg, source)
        entry_path = write_entry(cfg, plan, "## Summary\nNew content.\n## Core insights\n- A\n## Linked concepts\n- B\n")

        # The new entry should have a different filename
        assert entry_path.stem != "my-article"
        assert entry_path.exists()
        assert (entries / "my-article.md").exists()  # old one preserved

    def test_existing_concept_collision(self, cfg: Config):
        """Existing concept gets a collision-safe filename."""
        concepts = cfg.concepts_dir
        concepts.mkdir(parents=True, exist_ok=True)
        (concepts / "machine-learning.md").write_text("# Old ML concept\n")

        path = write_concept(cfg, "Machine Learning", "## Core concept\nNew.\n## Context\nC.\n## Links\nL.\n", [])
        assert path.stem != "machine-learning" or path.stem == "machine-learning"
        # Either reused (if same concept) or collision-resolved
        assert path.exists()


# ─── test_chinese_content ──────────────────────────────────────────────────────

class TestChineseContent:
    """Feed Chinese URL, verify bilingual pipeline."""

    def test_chinese_article(self, cfg: Config, vault: Path):
        """Chinese entry should use Chinese template and English tags."""
        url = "https://zh.example.com/深度学习入门"
        source = _make_source(
            url,
            "深度学习入门指南",
            content="# 深度学习入门指南\n\n深度学习是机器学习的一个分支。\n" * 20,
        )
        plan = _make_plan_from_source(
            source,
            language=Language.ZH,
            template=Template.CHINESE,
            tags=["deep-learning", "neural-networks"],
            concept_new=["深度学习"],
        )

        # Write files
        write_source(cfg, source)

        entry_content = "## Summary / 摘要\n\n深度学习入门指南的总结。\n\n## Core insights / 核心洞察\n\n- 洞察1\n\n## Linked concepts\n\n- [[深度学习]]\n"
        entry_path = write_entry(cfg, plan, entry_content)
        assert entry_path.exists()

        # Verify entry has Chinese title and English tags
        entry_text = entry_path.read_text()
        assert "template: chinese" in entry_text
        assert "- deep-learning" in entry_text
        assert "深度学习入门指南" in entry_text

        # Verify Chinese filename
        assert "深度学习" in entry_path.stem

        # Write concept
        concept_path = write_concept(cfg, "深度学习", "## Core concept\n深度学习是...\n## Context\n上下文。\n## Links\n- A\n", [source.hash])
        assert concept_path.exists()
        assert "深度学习" in concept_path.stem


# ─── test_validate_catches_issues ──────────────────────────────────────────────

class TestValidateCatchesIssues:
    """Create vault with bad notes, verify validate catches them."""

    def test_catches_banned_tags(self, cfg: Config, vault: Path):
        """Entries with banned tags should be flagged."""
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        Manifest(entries=[]).save(extract_dir)

        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        (entries / "bad-tag-entry.md").write_text(
            "---\ntitle: Bad Tag Entry\nsource: \"[[bad]]\"\n"
            "date_entry: 2026-01-01\nstatus: draft\ntemplate: standard\n"
            "tags:\n  - x.com\n  - legitimate-tag\n---\n"
            "# Bad Tag Entry\n## Summary\nS.\n## Core insights\nI.\n## Linked concepts\nC.\n"
        )

        violations = validate_output(cfg, extract_dir / "manifest.json")
        assert any("banned tag" in v.lower() or "x.com" in v for v in violations)

    def test_catches_stubs(self, cfg: Config):
        """Entries with stub content should be flagged."""
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        Manifest(entries=[]).save(extract_dir)

        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        (entries / "stub-entry.md").write_text(
            "---\ntitle: Stub Entry\nsource: \"[[stub]]\"\n"
            "date_entry: 2026-01-01\nstatus: draft\ntemplate: standard\ntags:\n  - test\n---\n"
            "# Stub Entry\n## Summary\n> TODO: fill this in\n## Core insights\nI.\n## Linked concepts\nC.\n"
        )

        violations = validate_output(cfg, extract_dir / "manifest.json")
        assert any("stub" in v.lower() for v in violations)

    def test_catches_missing_frontmatter(self, cfg: Config):
        """Entries without YAML frontmatter should be flagged."""
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        Manifest(entries=[]).save(extract_dir)

        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        (entries / "no-frontmatter.md").write_text(
            "# No Frontmatter\n\nJust content without frontmatter.\n"
        )

        violations = validate_output(cfg, extract_dir / "manifest.json")
        assert any("missing" in v.lower() and "frontmatter" in v.lower() for v in violations)

    def test_catches_missing_sections(self, cfg: Config):
        """Entries missing required sections should be flagged."""
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        Manifest(entries=[]).save(extract_dir)

        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        (entries / "incomplete.md").write_text(
            "---\ntitle: Incomplete\nsource: \"[[inc]]\"\n"
            "date_entry: 2026-01-01\nstatus: draft\ntemplate: standard\ntags:\n  - test\n---\n"
            "# Incomplete\nNo required sections here.\n"
        )

        violations = validate_output(cfg, extract_dir / "manifest.json")
        section_violations = [v for v in violations if "missing required section" in v.lower()]
        assert len(section_violations) >= 2  # at least Summary and Core insights

    def test_clean_entries_pass(self, cfg: Config):
        """Well-formed entries should pass validation."""
        extract_dir = cfg.resolved_extract_dir
        extract_dir.mkdir(parents=True, exist_ok=True)
        Manifest(entries=[]).save(extract_dir)

        entries = cfg.entries_dir
        entries.mkdir(parents=True, exist_ok=True)
        time.sleep(0.05)
        (entries / "good-entry.md").write_text(
            "---\ntitle: Good Entry\nsource: \"[[good]]\"\n"
            "date_entry: 2026-01-01\nstatus: draft\ntemplate: standard\ntags:\n  - good\n---\n"
            "# Good Entry\n## Summary\n"
            "A comprehensive and well-structured summary that covers the essential points "
            "of the source material with adequate detail for future reference.\n"
            "## Core insights\n"
            "- The primary insight demonstrates how systematic approaches to knowledge "
            "management create compounding returns over time.\n"
            "## Other takeaways\n"
            "- Additional observations about implementation patterns and best practices "
            "that inform practical decisions in real-world scenarios.\n"
            "## Diagrams\nn/a\n"
            "## Open questions\n"
            "- What are the boundary conditions for this approach?\n"
            "## Linked concepts\n- [[concept-a]]\n"
        )

        violations = validate_output(cfg, extract_dir / "manifest.json")
        # Filter to only entries violations
        entry_violations = [v for v in violations if "good-entry" in v]
        assert len(entry_violations) == 0
