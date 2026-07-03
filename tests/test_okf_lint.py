"""Tests for pipeline.okf_lint."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.okf_lint import LintIssue, LintReport, lint_bundle

# ── Helpers ─────────────────────────────────────────────────────────────


def _write(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


_CONFORMANT = """---
type: Concept
title: Foo
tags:
- alpha
- beta
timestamp: 2025-01-02T10:30:00
---

# Foo

See [Bar](bar.md) for more.
"""


def _rules(report: LintReport) -> set[str]:
    """Return the set of rule ids present in ``report``."""
    return {issue.rule for issue in report.issues}


def _issues_for(rule: str, report: LintReport) -> list[LintIssue]:
    return [i for i in report.issues if i.rule == rule]


# ── Conformance ─────────────────────────────────────────────────────────


def test_conformant_bundle_passes(tmp_path: Path):
    """A bundle where every concept file has a valid type passes cleanly."""
    _write(tmp_path / "concepts" / "foo.md", _CONFORMANT)
    _write(tmp_path / "concepts" / "bar.md", _CONFORMANT.replace("Foo", "Bar")
           .replace("bar.md", "foo.md"))
    # Reserved files present so they don't count as missing concepts.
    _write(tmp_path / "index.md", "# Knowledge Bundle\n")
    _write(tmp_path / "log.md", "# Change Log\n")

    report = lint_bundle(tmp_path)
    assert report.passed is True
    assert report.errors == 0
    assert report.files_checked == 4
    # No concept-rule issues expected (OKF-001..005).
    assert not {"OKF-001", "OKF-002", "OKF-003", "OKF-004", "OKF-005"} & _rules(report)


def test_empty_bundle(tmp_path: Path):
    """An empty directory yields files_checked=0 and passes."""
    report = lint_bundle(tmp_path)
    assert report.files_checked == 0
    assert report.issues == []
    assert report.passed is True


# ── OKF-001: missing frontmatter ────────────────────────────────────────


def test_missing_frontmatter_okf001(tmp_path: Path):
    _write(tmp_path / "concepts" / "bare.md", "# No frontmatter\n\nBody.\n")
    report = lint_bundle(tmp_path)
    assert report.passed is False
    assert report.errors >= 1
    assert "OKF-001" in _rules(report)
    issue = _issues_for("OKF-001", report)[0]
    assert issue.severity == "error"
    assert issue.file == "concepts/bare.md"
    assert issue.line == 1


# ── OKF-002: missing/empty type ─────────────────────────────────────────


def test_empty_type_okf002(tmp_path: Path):
    _write(tmp_path / "concepts" / "notype.md",
           "---\ntitle: No Type\ntags: []\ntimestamp: 2025-01-02\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-002" in _rules(report)
    issue = _issues_for("OKF-002", report)[0]
    assert issue.severity == "error"
    assert issue.file == "concepts/notype.md"
    assert report.passed is False


def test_blank_type_okf002(tmp_path: Path):
    _write(tmp_path / "concepts" / "blank.md",
           "---\ntype: '   '\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-002" in _rules(report)
    assert report.errors >= 1


# ── OKF-003: invalid timestamp ──────────────────────────────────────────


def test_invalid_timestamp_okf003(tmp_path: Path):
    _write(tmp_path / "concepts" / "ts.md",
           "---\ntype: Concept\ntimestamp: 'yesterday'\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-003" in _rules(report)
    issue = _issues_for("OKF-003", report)[0]
    assert issue.severity == "warning"
    assert issue.file == "concepts/ts.md"
    # Timestamp error is a warning, so passed is still True (no errors).
    assert report.passed is True


def test_valid_timestamp_no_okf003(tmp_path: Path):
    _write(tmp_path / "concepts" / "ts.md",
           "---\ntype: Concept\ntimestamp: 2025-01-02T10:30:00Z\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-003" not in _rules(report)


# ── OKF-004: tags not a list ─────────────────────────────────────────────


def test_tags_not_list_okf004(tmp_path: Path):
    _write(tmp_path / "concepts" / "tags.md",
           "---\ntype: Concept\ntags: 'alpha, beta'\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-004" in _rules(report)
    issue = _issues_for("OKF-004", report)[0]
    assert issue.severity == "warning"
    assert issue.file == "concepts/tags.md"


def test_tags_list_ok_no_okf004(tmp_path: Path):
    _write(tmp_path / "concepts" / "tags.md",
           "---\ntype: Concept\ntags:\n- alpha\n---\nBody.\n")
    report = lint_bundle(tmp_path)
    assert "OKF-004" not in _rules(report)


# ── OKF-005: broken cross-link ───────────────────────────────────────────


def test_broken_link_okf005(tmp_path: Path):
    _write(tmp_path / "concepts" / "linker.md",
           "---\ntype: Concept\n---\nSee [Missing](ghost.md).\n")
    report = lint_bundle(tmp_path)
    assert "OKF-005" in _rules(report)
    issue = _issues_for("OKF-005", report)[0]
    assert issue.severity == "info"
    assert issue.file == "concepts/linker.md"
    # Info-level: does not affect passed.
    assert report.passed is True


def test_valid_link_no_okf005(tmp_path: Path):
    _write(tmp_path / "concepts" / "a.md",
           "---\ntype: Concept\n---\nSee [B](b.md).\n")
    _write(tmp_path / "concepts" / "b.md",
           "---\ntype: Concept\n---\n# B\n")
    report = lint_bundle(tmp_path)
    assert "OKF-005" not in _rules(report)


def test_external_link_not_checked_okf005(tmp_path: Path):
    _write(tmp_path / "concepts" / "ext.md",
           "---\ntype: Concept\n---\n[site](https://example.com)\n")
    report = lint_bundle(tmp_path)
    assert "OKF-005" not in _rules(report)


# ── OKF-006: index.md frontmatter ───────────────────────────────────────


def test_subdir_index_with_frontmatter_okf006(tmp_path: Path):
    """A sub-directory index.md carrying frontmatter triggers OKF-006."""
    _write(tmp_path / "concepts" / "index.md", "---\ntitle: Bad\n---\n# Concepts\n")
    report = lint_bundle(tmp_path)
    assert "OKF-006" in _rules(report)
    issue = _issues_for("OKF-006", report)[0]
    assert issue.severity == "warning"
    assert issue.file == "concepts/index.md"


def test_bundle_root_index_with_okf_version_no_okf006(tmp_path: Path):
    """Bundle-root index.md with okf_version is allowed (no OKF-006)."""
    _write(tmp_path / "index.md", "---\nokf_version: '0.1'\n---\n# Knowledge Bundle\n")
    report = lint_bundle(tmp_path)
    assert "OKF-006" not in _rules(report)


def test_bundle_root_index_without_okf_version_okf006(tmp_path: Path):
    """Bundle-root index.md with frontmatter but no okf_version triggers OKF-006."""
    _write(tmp_path / "index.md", "---\ntitle: Bad\n---\n# Knowledge Bundle\n")
    report = lint_bundle(tmp_path)
    assert "OKF-006" in _rules(report)


def test_subdir_index_no_frontmatter_no_okf006(tmp_path: Path):
    """A clean sub-directory index.md does not trigger OKF-006."""
    _write(tmp_path / "concepts" / "index.md", "# Concepts\n")
    report = lint_bundle(tmp_path)
    assert "OKF-006" not in _rules(report)


# ── OKF-007: log.md date headings ───────────────────────────────────────


def test_log_invalid_date_heading_okf007(tmp_path: Path):
    _write(tmp_path / "log.md",
           "# Change Log\n\n## 2025-13-99\n\n- did a thing\n")
    report = lint_bundle(tmp_path)
    assert "OKF-007" in _rules(report)
    issue = _issues_for("OKF-007", report)[0]
    assert issue.severity == "warning"
    assert issue.file == "log.md"
    assert issue.line is not None and issue.line >= 1


def test_log_valid_date_heading_no_okf007(tmp_path: Path):
    _write(tmp_path / "log.md",
           "# Change Log\n\n## 2025-01-02\n\n- did a thing\n")
    report = lint_bundle(tmp_path)
    assert "OKF-007" not in _rules(report)


# ── LintReport unit tests ────────────────────────────────────────────────


def test_report_add_updates_counters():
    report = LintReport()
    report.add(LintIssue("error", "a.md", 1, "boom", "OKF-001"))
    report.add(LintIssue("warning", "b.md", 2, "meh", "OKF-003"))
    report.add(LintIssue("info", "c.md", 3, "fyi", "OKF-005"))
    assert report.errors == 1
    assert report.warnings == 1
    assert len(report.issues) == 3
    assert report.passed is False


def test_report_passed_when_no_errors():
    report = LintReport()
    assert report.passed is True
    report.add(LintIssue("warning", "a.md", 1, "meh", "OKF-003"))
    assert report.passed is True


# ── pytest entry ────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
