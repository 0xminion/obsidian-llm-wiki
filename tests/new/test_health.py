"""Tests for CLI health command and health report generation."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.cli.health import _generate_health_report
from obsidian_llm_wiki.render.obsidian import atomic_write, render_concept_page, render_moc_page


def _make_minimal_vault(bundle_dir: Path) -> None:
    """Create a minimal vault with some content for health checks."""
    # Create directories
    (bundle_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "mocs").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "entries").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "sources").mkdir(parents=True, exist_ok=True)

    # Good concept (in MoC, high confidence, enough body)
    good_concept = _make_concept_md(
        title="Good Concept", slug="good-concept", summary="A good concept",
        tags=["good"], confidence=0.9,
        body="# Good Concept\n\nThis is a sufficiently long concept body. " * 20,
    )
    atomic_write(bundle_dir / "concepts" / "good-concept.md", good_concept)

    # Stub concept (< 500 chars)
    stub_concept = _make_concept_md(
        title="Stub", slug="stub", summary="Short",
        tags=["stub"], confidence=0.8, body="# Stub\n\nShort.",
    )
    atomic_write(bundle_dir / "concepts" / "stub.md", stub_concept)

    # Low confidence concept
    low_conf = _make_concept_md(
        title="Low Conf", slug="low-conf", summary="Low confidence",
        tags=["low"], confidence=0.3,
        body="# Low Conf\n\nThis is a concept with low confidence. " * 20,
    )
    atomic_write(bundle_dir / "concepts" / "low-conf.md", low_conf)

    # Orphan concept (not in any MoC, enough body)
    orphan = _make_concept_md(
        title="Orphan", slug="orphan", summary="No MoC",
        tags=["orphan"], confidence=0.8,
        body="# Orphan\n\nThis concept is not in any MoC. " * 20,
    )
    atomic_write(bundle_dir / "concepts" / "orphan.md", orphan)

    # MoC with good concept (2+ concepts is OK)
    moc_content = _make_moc_md(
        title="Good MOC", slug="good-moc", summary="A good MOC",
        concept_slugs=["good-concept"],
    )
    atomic_write(bundle_dir / "mocs" / "good-moc.md", moc_content)

    # MoC with only 1 concept (small MoC)
    small_moc = _make_moc_md(
        title="Small MOC", slug="small-moc", summary="Too small",
        concept_slugs=["stub"],
    )
    atomic_write(bundle_dir / "mocs" / "small-moc.md", small_moc)

    # Concept with broken wikilink
    broken = _make_concept_md(
        title="Broken Links", slug="broken-links", summary="Has broken links",
        tags=["test"], confidence=0.8,
        body="# Broken Links\n\nThis links to [[nonexistent-concept]] which doesn't exist. " * 10,
    )
    atomic_write(bundle_dir / "concepts" / "broken-links.md", broken)

    # Index file (reserved, should be skipped)
    atomic_write(bundle_dir / "concepts" / "index.md", "# Concepts\n\n")


def _make_concept_md(
    title: str, slug: str, summary: str, tags: list[str],
    confidence: float, body: str,
) -> str:
    """Build a minimal concept markdown file."""
    from obsidian_llm_wiki.core.models import ConceptNote
    concept = ConceptNote(
        title=title, slug=slug, summary=summary, tags=tags,
        confidence=confidence,
    )
    # Use the real renderer but override body
    page = render_concept_page(concept, "2026-01-01T00:00:00Z")
    # Append extra body content
    return page + "\n\n" + body


def _make_moc_md(
    title: str, slug: str, summary: str, concept_slugs: list[str],
) -> str:
    """Build a minimal MoC markdown file."""
    from obsidian_llm_wiki.core.models import MapOfContent
    moc = MapOfContent(
        title=title, slug=slug, summary=summary,
        concept_slugs=concept_slugs,
    )
    return render_moc_page(moc, "2026-01-01T00:00:00Z")


# ── Tests ────────────────────────────────────────────────────────────────


def test_health_report_generates_markdown(tmp_path: Path):
    """Health report should be generated as markdown."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "# Vault Health Report" in report
    assert "## Summary" in report


def test_health_report_finds_orphan_concepts(tmp_path: Path):
    """Orphan concepts not in any MoC should be detected."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "Orphan Concepts" in report
    assert "orphan" in report


def test_health_report_finds_stub_entries(tmp_path: Path):
    """Stub entries (<500 chars) should be detected."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "Stub Entries" in report
    assert "stub" in report


def test_health_report_finds_low_confidence(tmp_path: Path):
    """Low-confidence concepts (<0.5) should be detected."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "Low-Confidence" in report
    assert "low-conf" in report


def test_health_report_finds_broken_wikilinks(tmp_path: Path):
    """Broken wikilinks should be detected."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "Broken Wikilinks" in report
    assert "nonexistent-concept" in report


def test_health_report_finds_small_mocs(tmp_path: Path):
    """MoCs with <2 concepts should be detected."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "MoCs with <2 Concepts" in report


def test_health_report_summary_table_has_counts(tmp_path: Path):
    """Summary table should have counts for all checks."""
    bundle_dir = tmp_path / "04-Wiki"
    _make_minimal_vault(bundle_dir)

    report = _generate_health_report(bundle_dir)

    assert "Broken wikilinks" in report
    assert "Orphan concepts" in report
    assert "Stub entries" in report
    assert "Low-confidence" in report
    assert "Missing source links" in report
    assert "Tag violations" in report
    assert "MoCs with" in report


def test_health_report_clean_vault(tmp_path: Path):
    """A vault with no issues should show clean checks."""
    bundle_dir = tmp_path / "04-Wiki"
    (bundle_dir / "concepts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "mocs").mkdir(parents=True, exist_ok=True)

    report = _generate_health_report(bundle_dir)

    # All sections should show "No issues found"
    assert "No issues found" in report


def test_health_report_tag_violations(tmp_path: Path):
    """Tags with spaces should be flagged."""
    bundle_dir = tmp_path / "04-Wiki"
    (bundle_dir / "concepts").mkdir(parents=True, exist_ok=True)

    # Write a file with space-containing tag
    bad_tag_content = """---
type: Concept
title: Bad Tag
tags:
  - "has space"
confidence: 0.9
---

# Bad Tag

This is a concept with a bad tag. """ * 20

    atomic_write(bundle_dir / "concepts" / "bad-tag.md", bad_tag_content)

    report = _generate_health_report(bundle_dir)

    assert "Tag Violations" in report
    assert "has space" in report
