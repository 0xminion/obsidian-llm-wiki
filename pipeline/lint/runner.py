"""Lint runner — LintChecker class and CLI entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.lint.checks import (
    _find_md_files,
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
    check_staleness,
    check_stubs,
    check_tag_quality,
    check_unreviewed_entries,
    check_weak_links,
    check_wiki_index_drift,
)
from pipeline.lint.fixes import (
    fix_banned_tags,
    fix_frontmatter,
    fix_markdown_format,
)
from pipeline.lint.models import LintIssue, LintResult, Severity


class LintChecker:
    """Main lint checker — runs all checks on a vault.

    Uses vault cache for incremental processing — only re-scans files
    that changed since the last lint run.
    """

    def __init__(self, vault_path: Path):
        self.vault = vault_path
        self._cache = None
        try:
            from pipeline.store import ContentStore
            self._cache = ContentStore.open_vault_cache(vault_path)
        except (ImportError, OSError):
            pass

    def close(self) -> None:
        if self._cache:
            try:
                self._cache.close()
            except OSError:
                pass
            self._cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()

    def run_all(self, fix: bool = False) -> LintResult:
        """Run all lint checks. Optionally fix issues first."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = LintResult(report_date=today)

        # Collect all .md files for counting
        all_files = _find_md_files(
            self.vault,
            "04-Wiki/entries",
            "04-Wiki/concepts",
            "04-Wiki/mocs",
            "04-Wiki/sources",
        )
        result.files_checked = len(all_files)

        # Fix mode: apply fixes before checks
        if fix:
            for md in all_files:
                if fix_frontmatter(md):
                    result.fixes_applied += 1
                if fix_markdown_format(md):
                    result.fixes_applied += 1
                if fix_banned_tags(md):
                    result.fixes_applied += 1

        # Rebuild link graph from disk for correctness. The cache is retained
        # for future optimization, but lint must not report stale graph issues
        # after a create/review operation has just modified files.
        cache_enabled: set[str] = set()

        # Run all checks
        checks = [
            ("1. Orphaned Notes", check_orphaned_notes, "orphaned_notes"),
            ("2. Unreviewed Entries", check_unreviewed_entries, "unreviewed_entries"),
            ("3. Stale Reviews", check_stale_reviews, "stale_reviews"),
            ("4. Broken Wikilinks", check_broken_wikilinks, "broken_wikilinks"),
            ("5. Empty Notes", check_empty_notes, "empty_notes"),
            ("6. Concept Structure", check_concept_structure, "concept_structure"),
            ("7. Entry Template Sections", check_entry_template_sections, "entry_template_sections"),
            ("8. Orphaned Concepts", check_orphaned_concepts, "orphaned_concepts"),
            ("9. Wiki Index Drift", check_wiki_index_drift, "wiki_index_drift"),
            ("10. Edges Consistency", check_edges_consistency, "edges_consistency"),
            ("10b. Weak Links", check_weak_links, "weak_links"),
            ("11. Stubs/Placeholders", check_stubs, "stubs"),
            ("12. Tag Quality", check_tag_quality, "tag_quality"),
            ("13. Frontmatter Validity", check_frontmatter_validity, "frontmatter_validity"),
            ("14. Required Sections", check_required_sections, "required_sections"),
            ("15. Markdown Format", check_markdown_format, "markdown_format"),
            ("16. Staleness", check_staleness, "staleness"),
        ]

        for name, check_fn, check_id in checks:
            try:
                if check_id in cache_enabled and self._cache:
                    issues = check_fn(self.vault, _cache=self._cache)
                else:
                    issues = check_fn(self.vault)
            except Exception as e:
                issues = [LintIssue(
                    check=name,
                    severity=Severity.ERROR,
                    note="(check failed)",
                    detail=str(e),
                )]
            result.issues.extend(issues)
            result.issues_by_check[name] = len(issues)

        result.total_issues = len(result.issues)
        return result

    def write_report(self, result: LintResult, report_path: Optional[Path] = None) -> Path:
        """Write lint report to Meta/Scripts/lint-report.md."""
        if report_path is None:
            report_path = self.vault / "Meta" / "Scripts" / "lint-report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Lint Report — {result.report_date} (v2.2.0)",
            "",
            "> Karpathy-style linting: catches what the LLM misses.",
            "",
        ]

        for check_name, count in sorted(result.issues_by_check.items()):
            lines.append(f"## {check_name}")
            lines.append("")
            check_issues = [i for i in result.issues if _issue_matches_check(i, check_name)]

            if not check_issues:
                lines.append("All clear.")
            else:
                for issue in check_issues:
                    prefix = {"error": "🔴", "warning": "🟡", "info": "🔵"}[issue.severity.value]
                    lines.append(f"- {prefix} **{issue.note}**: {issue.detail}")
                lines.append("")
                lines.append(f"**Total: {count} issue(s)**")
            lines.append("")

        # Summary table
        lines.extend([
            "---",
            "",
            "## Summary",
            "",
            "| Check | Issues |",
            "|-------|--------|",
        ])
        for check_name, count in sorted(result.issues_by_check.items()):
            lines.append(f"| {check_name} | {count} |")
        lines.append(f"| **TOTAL** | **{result.total_issues}** |")
        lines.append("")
        if result.fixes_applied:
            lines.append(f"*Fixes applied: {result.fixes_applied}*")
            lines.append("")
        lines.append(f"*Files checked: {result.files_checked}*")
        lines.append("")
        lines.append("*Run `pipeline lint` to regenerate this report.*")
        lines.append("")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


def _issue_matches_check(issue: LintIssue, check_name: str) -> bool:
    """Match issue check field to check name used in report."""
    # Normalize: "1. Orphaned Notes" → "orphaned_notes"
    clean = check_name.split(". ", 1)[-1].lower().replace(" ", "_").replace("/", "_")
    return issue.check == clean or issue.check in clean


# ─── Standalone Functions for CLI ────────────────────────────────────────────

def run_lint(vault_path: Path, fix: bool = False) -> LintResult:
    """Run lint checks and write report. Returns result."""
    with LintChecker(vault_path) as checker:
        result = checker.run_all(fix=fix)
        checker.write_report(result)
        return result


def run_validate(vault_path: Path, fix: bool = False) -> LintResult:
    """Run validation checks (subset focused on post-write quality)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = LintResult(report_date=today)

    all_files = _find_md_files(
        vault_path,
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "04-Wiki/sources",
    )
    result.files_checked = len(all_files)

    if fix:
        for md in all_files:
            if fix_frontmatter(md):
                result.fixes_applied += 1
            if fix_markdown_format(md):
                result.fixes_applied += 1
            if fix_banned_tags(md):
                result.fixes_applied += 1

    checks = [
        ("Frontmatter Validity", check_frontmatter_validity),
        ("Required Sections", check_required_sections),
        ("Entry Template Sections", check_entry_template_sections),
        ("Concept Structure", check_concept_structure),
        ("Stubs/Placeholders", check_stubs),
        ("Tag Quality", check_tag_quality),
        ("Markdown Format", check_markdown_format),
    ]

    for name, check_fn in checks:
        try:
            issues = check_fn(vault_path)
        except Exception as e:
            issues = [LintIssue(check=name, severity=Severity.ERROR, note="(check failed)", detail=str(e))]
        result.issues.extend(issues)
        result.issues_by_check[name] = len(issues)

    result.total_issues = len(result.issues)
    return result
