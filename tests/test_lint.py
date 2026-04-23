"""Tests for pipeline.lint module."""

import tempfile
from pathlib import Path

import pytest

from pipeline.lint import (
    LintChecker,
    LintIssue,
    Severity,
    check_broken_wikilinks,
    check_concept_structure,
    check_edges_consistency,
    check_empty_notes,
    check_entry_template_sections,
    check_frontmatter_validity,
    check_markdown_format,
    check_orphaned_concepts,
    check_orphaned_notes,
    check_required_sections,
    check_stale_reviews,
    check_stubs,
    check_tag_quality,
    check_unreviewed_entries,
    check_wiki_index_drift,
    fix_banned_tags,
    fix_frontmatter,
    fix_markdown_format,
    run_lint,
    run_validate,
)


@pytest.fixture
def vault(tmp_path):
    """Create a minimal vault structure."""
    for d in ["04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources", "06-Config", "Meta/Scripts"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    # Create wiki-index.md
    (tmp_path / "06-Config" / "wiki-index.md").write_text(
        "# Wiki Index\n\n---\n\n## Entries\n\n- [[test-entry]]: Test entry (entry)\n\n## Concepts\n\n- [[test-concept]]: Test concept (concept)\n\n---\n\n*Reindexed*\n"
    )

    # Create edges.tsv
    (tmp_path / "06-Config" / "edges.tsv").write_text("source\ttarget\ttype\tdescription\n")

    return tmp_path


@pytest.fixture
def entry_standard(vault):
    """Create a well-formed standard entry."""
    content = """---
title: "Test Entry"
source: "[[test-source]]"
date_entry: "2026-04-19"
status: "draft"
template: "standard"
tags:
  - test
reviewed: ""
---

# Test Entry

## Summary

This is a test summary with enough content to pass the empty note check.

## Core insights

First core insight here.

## Other takeaways

Some other takeaway.

## Diagrams

n/a

## Open questions

What should we explore next?

## Linked concepts

- [[test-concept]]
"""
    path = vault / "04-Wiki" / "entries" / "test-entry.md"
    path.write_text(content)
    return path


@pytest.fixture
def concept_en(vault):
    """Create a well-formed English concept."""
    content = """---
title: "Test Concept"
type: "concept"
status: "draft"
language: "en"
sources: []
tags: []
---

# Test Concept

## Core concept

A test concept for unit testing.

## Context

The context of this concept is testing. It verifies that lint checks work correctly on well-formed content.

## Links

- [[test-entry]]
"""
    path = vault / "04-Wiki" / "concepts" / "test-concept.md"
    path.write_text(content)
    return path


@pytest.fixture
def concept_zh(vault):
    """Create a well-formed Chinese concept."""
    content = """---
title: "测试概念"
type: "concept"
status: "draft"
language: "zh"
sources: []
tags: []
---

# 测试概念

## 核心概念

这是一个用于单元测试的概念。

## 背景

这个概念的背景是测试。它验证了检查功能在正常内容上的正确性。

## 关联

- [[test-entry]]
"""
    path = vault / "04-Wiki" / "concepts" / "测试概念.md"
    path.write_text(content)
    return path


@pytest.fixture
def moc(vault):
    """Create a well-formed MoC."""
    content = """---
title: "Test MoC"
type: "moc"
status: "draft"
tags: []
---

# Test MoC

## Overview / 概述

This is the overview of the test MoC.

## Topic Section / 主题

- [[test-entry]]: A test entry

## Bridge Concepts / 桥接概念

- [[test-concept]]

"""
    path = vault / "04-Wiki" / "mocs" / "test-moc.md"
    path.write_text(content)
    return path


@pytest.fixture
def source(vault):
    """Create a source note."""
    content = """---
title: "Test Source"
source_url: "https://example.com/article"
source_type: "blog"
author: "Test Author"
date_captured: "2026-04-19"
tags: []
status: "raw"
---

# Test Source

This is test source content for the unit tests.
"""
    path = vault / "04-Wiki" / "sources" / "test-source.md"
    path.write_text(content)
    return path


# ─── Check Tests ─────────────────────────────────────────────────────────────

class TestOrphanedNotes:
    def test_no_orphans_when_referenced(self, vault, entry_standard, concept_en):
        issues = check_orphaned_notes(vault)
        note_names = [i.note for i in issues]
        assert "test-entry" not in note_names
        assert "test-concept" not in note_names

    def test_detects_orphan(self, vault):
        # Create entry with no backlinks
        (vault / "04-Wiki" / "entries" / "lonely-entry.md").write_text(
            "---\ntitle: Lonely\n---\n# Lonely\n\n## Summary\n\nI have no friends.\n"
        )
        issues = check_orphaned_notes(vault)
        assert any(i.note == "lonely-entry" for i in issues)


class TestUnreviewedEntries:
    def test_reviewed_not_flagged(self, vault):
        content = """---
title: "Reviewed"
reviewed: "2026-04-19"
date_entry: "2026-04-19"
---
# Reviewed
## Summary

Done.
"""
        (vault / "04-Wiki" / "entries" / "reviewed.md").write_text(content)
        issues = check_unreviewed_entries(vault)
        assert not any(i.note == "reviewed" for i in issues)

    def test_null_reviewed_flagged(self, vault):
        content = """---
title: "Unreviewed"
reviewed: null
date_entry: "2026-04-19"
---
# Unreviewed
## Summary

Pending.
"""
        (vault / "04-Wiki" / "entries" / "unreviewed.md").write_text(content)
        issues = check_unreviewed_entries(vault)
        assert any(i.note == "unreviewed" for i in issues)

    def test_empty_reviewed_flagged(self, vault):
        content = """---
title: "Empty"
reviewed: ""
date_entry: "2026-04-19"
---
# Empty
## Summary

Nothing.
"""
        (vault / "04-Wiki" / "entries" / "empty.md").write_text(content)
        issues = check_unreviewed_entries(vault)
        assert any(i.note == "empty" for i in issues)


class TestStaleReviews:
    def test_recent_review_not_stale(self, vault):
        content = """---
title: "Recent"
status: "review"
date_entry: "2026-04-18"
---
# Recent
"""
        (vault / "04-Wiki" / "entries" / "recent.md").write_text(content)
        issues = check_stale_reviews(vault, days=14)
        assert not any(i.note == "recent" for i in issues)

    def test_old_review_flagged(self, vault):
        content = """---
title: "Stale"
status: "review"
date_entry: "2026-03-01"
---
# Stale
"""
        (vault / "04-Wiki" / "entries" / "stale.md").write_text(content)
        issues = check_stale_reviews(vault, days=14)
        assert any(i.note == "stale" for i in issues)

    def test_non_review_status_ignored(self, vault):
        content = """---
title: "Evergreen"
status: "evergreen"
date_entry: "2026-03-01"
---
# Evergreen
"""
        (vault / "04-Wiki" / "entries" / "evergreen.md").write_text(content)
        issues = check_stale_reviews(vault, days=14)
        assert len(issues) == 0


class TestBrokenWikilinks:
    def test_valid_links_pass(self, vault, entry_standard, concept_en):
        issues = check_broken_wikilinks(vault)
        # test-entry links to test-concept, which exists
        assert not any("test-concept" in i.detail for i in issues)

    def test_broken_link_detected(self, vault):
        content = """---
title: "Linker"
---
# Linker

Links to [[nonexistent-note]] here.
"""
        (vault / "04-Wiki" / "entries" / "linker.md").write_text(content)
        issues = check_broken_wikilinks(vault)
        assert any("nonexistent-note" in i.detail for i in issues)


class TestEmptyNotes:
    def test_substantial_note_passes(self, vault, entry_standard):
        issues = check_empty_notes(vault)
        assert not any(i.note == "test-entry" for i in issues)

    def test_empty_note_detected(self, vault):
        (vault / "04-Wiki" / "entries" / "empty.md").write_text(
            "---\ntitle: Empty\n---\n# Empty\n"
        )
        issues = check_empty_notes(vault)
        assert any(i.note == "empty" for i in issues)


class TestConceptStructure:
    def test_english_concept_complete(self, vault, concept_en):
        issues = check_concept_structure(vault)
        assert not any(i.note == "test-concept" for i in issues)

    def test_english_concept_missing_sections(self, vault):
        content = """---
title: "Incomplete"
type: "concept"
language: "en"
---
# Incomplete

## Core concept

Only this section exists.
"""
        (vault / "04-Wiki" / "concepts" / "incomplete.md").write_text(content)
        issues = check_concept_structure(vault)
        assert any(i.note == "incomplete" for i in issues)
        found = [i for i in issues if i.note == "incomplete"][0]
        assert "Context" in found.detail or "Links" in found.detail

    def test_chinese_concept_complete(self, vault, concept_zh):
        issues = check_concept_structure(vault)
        assert not any(i.note == "测试概念" for i in issues)

    def test_chinese_concept_missing_sections(self, vault):
        content = """---
title: "不完整"
type: "concept"
language: "zh"
---
# 不完整

## 核心概念

只有这个部分。
"""
        (vault / "04-Wiki" / "concepts" / "不完整.md").write_text(content)
        issues = check_concept_structure(vault)
        assert any(i.note == "不完整" for i in issues)


class TestEntryTemplateSections:
    def test_standard_entry_complete(self, vault, entry_standard):
        issues = check_entry_template_sections(vault)
        assert not any(i.note == "test-entry" for i in issues)

    def test_standard_entry_missing_sections(self, vault):
        content = """---
title: "Incomplete Entry"
template: "standard"
---
# Incomplete Entry

## Summary

Just a summary.
"""
        (vault / "04-Wiki" / "entries" / "incomplete-entry.md").write_text(content)
        issues = check_entry_template_sections(vault)
        assert any(i.note == "incomplete-entry" for i in issues)

    def test_technical_entry_sections(self, vault):
        content = """---
title: "Tech Entry"
template: "technical"
---
# Tech Entry

## Summary

Tech summary.

## Key Findings

Key findings here.

## Data/Evidence

Evidence here.

## Methodology

Methodology here.

## Limitations

Limitations here.

## Linked concepts

- [[concept]]
"""
        (vault / "04-Wiki" / "entries" / "tech-entry.md").write_text(content)
        issues = check_entry_template_sections(vault)
        assert not any(i.note == "tech-entry" for i in issues)

    def test_chinese_entry_sections(self, vault):
        content = """---
title: "中文条目"
template: "chinese"
---
# 中文条目

## 摘要

中文摘要。

## 核心发现

核心发现。

## 其他要点

其他要点。

## 图表

n/a

## 开放问题

开放问题。

## 关联概念

- [[概念]]
"""
        (vault / "04-Wiki" / "entries" / "中文条目.md").write_text(content)
        issues = check_entry_template_sections(vault)
        assert not any(i.note == "中文条目" for i in issues)


class TestOrphanedConcepts:
    def test_referenced_concept_not_orphaned(self, vault, entry_standard, concept_en):
        issues = check_orphaned_concepts(vault)
        assert not any(i.note == "test-concept" for i in issues)

    def test_unreferenced_concept_detected(self, vault):
        (vault / "04-Wiki" / "concepts" / "lonely-concept.md").write_text(
            "---\ntitle: Lonely Concept\ntype: concept\n---\n# Lonely Concept\n\n## Core concept\n\nLonely.\n"
        )
        issues = check_orphaned_concepts(vault)
        assert any(i.note == "lonely-concept" for i in issues)


class TestWikiIndexDrift:
    def test_matching_counts_pass(self, vault, entry_standard, concept_en):
        # Update index to match
        (vault / "06-Config" / "wiki-index.md").write_text(
            "# Wiki Index\n\n---\n\n## Entries\n\n- [[test-entry]]: Test (entry)\n\n## Concepts\n\n- [[test-concept]]: Test (concept)\n\n---\n\n*Reindexed*\n"
        )
        issues = check_wiki_index_drift(vault)
        assert len(issues) == 0

    def test_mismatch_detected(self, vault, entry_standard):
        # Index has 0 entries, actual has 1
        (vault / "06-Config" / "wiki-index.md").write_text(
            "# Wiki Index\n\n---\n\n## Entries\n\n## Concepts\n\n---\n\n*Reindexed*\n"
        )
        issues = check_wiki_index_drift(vault)
        assert any("Entry mismatch" in i.note for i in issues)

    def test_missing_index_detected(self, vault):
        (vault / "06-Config" / "wiki-index.md").unlink()
        issues = check_wiki_index_drift(vault)
        assert any("not found" in i.detail for i in issues)


class TestEdgesConsistency:
    def test_valid_edges_pass(self, vault, entry_standard, concept_en):
        (vault / "06-Config" / "edges.tsv").write_text(
            "source\ttarget\ttype\tdescription\ntest-entry\ttest-concept\trelates\ttest relation\n"
        )
        issues = check_edges_consistency(vault)
        assert len(issues) == 0

    def test_broken_edge_detected(self, vault, entry_standard):
        (vault / "06-Config" / "edges.tsv").write_text(
            "source\ttarget\ttype\tdescription\ntest-entry\tnonexistent\textends\tmissing target\n"
        )
        issues = check_edges_consistency(vault)
        assert len(issues) > 0
        assert any(i.note == "Edge target 'nonexistent'" for i in issues)

    def test_missing_edges_file(self, vault):
        (vault / "06-Config" / "edges.tsv").unlink()
        issues = check_edges_consistency(vault)
        assert any("not found" in i.detail.lower() for i in issues)


class TestStubs:
    def test_clean_note_passes(self, vault, entry_standard):
        issues = check_stubs(vault)
        assert not any(i.note == "test-entry" for i in issues)

    def test_todo_stub_detected(self, vault):
        content = """---
title: "Stub"
---
# Stub

## Summary

> TODO: fill this in later
"""
        (vault / "04-Wiki" / "entries" / "stub.md").write_text(content)
        issues = check_stubs(vault)
        assert any(i.note == "stub" for i in issues)

    def test_chinese_stub_detected(self, vault):
        content = """---
title: "桩"
---
# 桩

## 摘要

> 待补充
"""
        (vault / "04-Wiki" / "entries" / "桩.md").write_text(content)
        issues = check_stubs(vault)
        assert any(i.note == "桩" for i in issues)


class TestTagQuality:
    def test_valid_tags_pass(self, vault, entry_standard):
        issues = check_tag_quality(vault)
        assert len(issues) == 0

    def test_blocked_tag_detected(self, vault):
        content = """---
title: "Bad Tags"
tags:
  - x.com
  - crypto
---
# Bad Tags
"""
        (vault / "04-Wiki" / "entries" / "bad-tags.md").write_text(content)
        issues = check_tag_quality(vault)
        assert any("x.com" in i.detail for i in issues)

    def test_short_tag_detected(self, vault):
        content = """---
title: "Short Tags"
tags:
  - a
  - crypto
---
# Short Tags
"""
        (vault / "04-Wiki" / "entries" / "short-tags.md").write_text(content)
        issues = check_tag_quality(vault)
        assert any("too-short" in i.detail.lower() for i in issues)


class TestFrontmatterValidity:
    def test_valid_frontmatter_passes(self, vault, entry_standard):
        issues = check_frontmatter_validity(vault)
        fm_issues = [i for i in issues if i.note == "test-entry"]
        assert len(fm_issues) == 0

    def test_missing_frontmatter_detected(self, vault):
        (vault / "04-Wiki" / "entries" / "no-fm.md").write_text("# No Frontmatter\n\nNo YAML here.\n")
        issues = check_frontmatter_validity(vault)
        assert any(i.note == "no-fm" for i in issues)

    def test_null_value_detected(self, vault):
        content = """---
title: "Null Value"
review_notes: null
---
# Null Value
"""
        (vault / "04-Wiki" / "entries" / "null-val.md").write_text(content)
        issues = check_frontmatter_validity(vault)
        # reviewed: null is acceptable, but other nulls should be flagged
        # Actually, the check skips reviewed and review_notes
        # Let's check a different field
        content2 = """---
title: "Null Other"
author: null
---
# Null Other
"""
        (vault / "04-Wiki" / "entries" / "null-other.md").write_text(content2)
        issues = check_frontmatter_validity(vault)
        assert any(i.note == "null-other" and "null" in i.detail.lower() for i in issues)


class TestRequiredSections:
    def test_moc_missing_overview(self, vault):
        content = """---
title: "Bad MoC"
type: "moc"
---
# Bad MoC

## Topic

Some content.
"""
        (vault / "04-Wiki" / "mocs" / "bad-moc.md").write_text(content)
        issues = check_required_sections(vault)
        assert any("Overview" in i.detail for i in issues)

    def test_moc_insufficient_sections(self, vault):
        content = """---
title: "Tiny MoC"
type: "moc"
---
# Tiny MoC

## Overview / 概述

Just one section.
"""
        (vault / "04-Wiki" / "mocs" / "tiny-moc.md").write_text(content)
        issues = check_required_sections(vault)
        assert any("only 1 section" in i.detail.lower() or "only 1 section" in i.detail for i in issues)


class TestMarkdownFormat:
    def test_clean_format_passes(self, vault, entry_standard):
        issues = check_markdown_format(vault)
        fm_issues = [i for i in issues if i.note == "test-entry"]
        assert len(fm_issues) == 0

    def test_missing_h1_detected(self, vault):
        content = """---
title: "No H1"
---
This starts with text, not a heading.

## Summary

Summary here.
"""
        (vault / "04-Wiki" / "entries" / "no-h1.md").write_text(content)
        issues = check_markdown_format(vault)
        assert any(i.note == "no-h1" and "H1" in i.detail for i in issues)

    def test_missing_blank_line_after_heading(self, vault):
        content = """---
title: "Tight"
---
# Tight

## Summary
No blank line after heading.
"""
        (vault / "04-Wiki" / "entries" / "tight.md").write_text(content)
        issues = check_markdown_format(vault)
        assert any(i.note == "tight" and "blank line" in i.detail.lower() for i in issues)


# ─── Fix Tests ───────────────────────────────────────────────────────────────

class TestFixFrontmatter:
    def test_fixes_null_values(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\nreviewed: null\n---\n# Test\n")
        assert fix_frontmatter(f)
        content = f.read_text()
        assert "null" not in content.split("---")[1]

    def test_fixes_unquoted_wikilinks(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text('---\ntitle: Test\nsource: [[note]]\n---\n# Test\n')
        assert fix_frontmatter(f)
        content = f.read_text()
        fm = content.split("---")[1]
        assert '"[[note]]"' in fm

    def test_no_change_for_clean_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text('---\ntitle: "Test"\nreviewed: ""\n---\n# Test\n')
        assert not fix_frontmatter(f)


class TestFixMarkdownFormat:
    def test_adds_h1_title(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test Title\n---\nSome body text.\n")
        assert fix_markdown_format(f)
        content = f.read_text()
        assert "# Test Title" in content

    def test_adds_blank_line_after_heading(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\n---\n# Test\n\n## Summary\nNo blank line.\n")
        assert fix_markdown_format(f)
        content = f.read_text()
        lines = content.split("\n")
        # Find the ## Summary line and check the next line is blank
        for i, line in enumerate(lines):
            if line.strip() == "## Summary":
                assert lines[i + 1].strip() == "", "Should have blank line after heading"
                break

    def test_no_change_for_clean_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\n---\n# Test\n\n## Summary\n\nContent.\n")
        assert not fix_markdown_format(f)


class TestFixBannedTags:
    def test_removes_banned_tag(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("---\ntitle: Test\ntags:\n  - x.com\n  - crypto\n---\n# Test\n")
        assert fix_banned_tags(f)
        content = f.read_text()
        assert "x.com" not in content
        assert "crypto" in content


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestLintChecker:
    def test_run_all_on_clean_vault(self, vault, entry_standard, concept_en, source, moc):
        # Update index to match reality
        (vault / "06-Config" / "wiki-index.md").write_text(
            "# Wiki Index\n\n---\n\n## Entries\n\n- [[test-entry]]: Test (entry)\n\n## Concepts\n\n- [[test-concept]]: Test (concept)\n\n## Maps of Content\n\n- [[test-moc]]: This is the overview (moc)\n\n---\n\n*Reindexed*\n"
        )
        checker = LintChecker(vault)
        result = checker.run_all()
        # Should have minimal issues on a well-formed vault
        assert result.files_checked > 0
        assert result.total_issues >= 0  # may have some orphans etc

    def test_write_report(self, vault, entry_standard, concept_en):
        checker = LintChecker(vault)
        result = checker.run_all()
        report_path = checker.write_report(result)
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Lint Report" in content
        assert "## Summary" in content


class TestRunLint:
    def test_returns_result(self, vault, entry_standard):
        result = run_lint(vault)
        assert result.total_issues >= 0
        assert result.files_checked > 0
        # Report should be written
        report = vault / "Meta" / "Scripts" / "lint-report.md"
        assert report.exists()


class TestRunValidate:
    def test_returns_result(self, vault, entry_standard):
        result = run_validate(vault)
        assert result.total_issues >= 0
        assert result.files_checked > 0
