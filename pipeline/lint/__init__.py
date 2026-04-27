"""Vault lint module — comprehensive health checks for the Obsidian wiki.

Consolidates all checks from lint-vault.sh (12 checks) and validate-output.sh
(6 checks + fix mode) into a single Python module.

Supports incremental checking via vault cache — only re-scans files that changed
since the last lint run.

Usage:
    from pipeline.lint import run_lint
    result = run_lint(vault_path, fix=False)

Or via CLI:
    pipeline lint ~/MyVault
    pipeline lint ~/MyVault --fix
"""

# Re-export all public symbols for backward compatibility.
# Every name that was importable from the old pipeline.lint monolith
# must remain importable from pipeline.lint after the split.

from pipeline.lint.models import (  # noqa: F401
    LintIssue,
    LintResult,
    Severity,
)

from pipeline.lint.checks import (  # noqa: F401
    _BLOCKED_TAGS,
    _STALENESS_THRESHOLDS,
    _STUB_PATTERNS,
    _VOLATILITY_DEFAULT_DAYS,
    _VOLATILITY_MAP,
    _WIKI_DIRS,
    _build_wikilink_index,
    _compute_staleness,
    _find_md_files,
    _parse_note_date,
    _volatility_rank,
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

from pipeline.lint.fixes import (  # noqa: F401
    fix_banned_tags,
    fix_frontmatter,
    fix_markdown_format,
)

from pipeline.lint.runner import (  # noqa: F401
    LintChecker,
    _issue_matches_check,
    run_lint,
    run_validate,
)

# Re-export _parse_frontmatter so `from pipeline.lint import _parse_frontmatter` works.
# This was available in the old monolith because it was imported at module level.
from pipeline.utils import parse_frontmatter as _parse_frontmatter  # noqa: F401
