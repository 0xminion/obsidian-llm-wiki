"""OKF v0.1 bundle linter.

Scans an OKF bundle directory tree and produces a :class:`LintReport` of
issues found, each tagged with a stable rule id (``OKF-NNN``).

Rules
-----
Non-reserved files (anything that is not ``index.md`` or ``log.md``):

* **OKF-001** (error): missing YAML frontmatter entirely.
* **OKF-002** (error): frontmatter has no ``type`` field, or it is empty.
* **OKF-003** (warning): ``timestamp`` present but not a valid ISO 8601 value.
* **OKF-004** (warning): ``tags`` present but not a YAML list.
* **OKF-005** (info): a markdown cross-link target does not resolve to an
  existing file in the bundle.

Reserved files:

* **OKF-006** (warning): a non-bundle-root ``index.md`` carries frontmatter.
  The bundle-root ``index.md`` is permitted to carry frontmatter *only* when
  it contains an ``okf_version`` key.
* **OKF-007** (warning): ``log.md`` date headings (``## YYYY-MM-DD``) that are
  not valid ISO 8601 dates.

The linter is read-only: it never modifies files on disk.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.okf_markdown import extract_links, parse_frontmatter, safe_read_file

__all__ = [
    "LintIssue",
    "LintReport",
    "lint_bundle",
]

# Files that receive the special reserved-file lint pass.
RESERVED_FILES: frozenset[str] = frozenset({"index.md", "log.md"})

# Heading marker for log.md date sections, e.g. "## 2025-01-02".
_DATE_HEADING_RE = re.compile(r"^##\s+(?P<date>\S+)\s*$", re.MULTILINE)


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class LintIssue:
    """A single lint finding for one file.

    Attributes
    ----------
    severity:
        One of ``"error"``, ``"warning"``, or ``"info"``.
    file:
        Path of the offending file, relative to the bundle root (POSIX style).
    line:
        1-indexed line number of the issue, or ``None`` when not applicable.
    message:
        Human-readable description of the problem.
    rule:
        Stable rule identifier such as ``"OKF-001"``.
    """

    severity: str
    file: str
    line: int | None
    message: str
    rule: str


@dataclass
class LintReport:
    """Aggregated result of linting a bundle.

    ``issues`` is kept in scan order (sorted by file path then by rule). The
    convenience counters (``errors``, ``warnings``) and the ``passed``
    property are derived from the issue list so the report stays internally
    consistent.
    """

    issues: list[LintIssue] = field(default_factory=list)
    files_checked: int = 0
    errors: int = 0
    warnings: int = 0

    @property
    def passed(self) -> bool:
        """True when the report has no error-severity issues."""
        return self.errors == 0

    def add(self, issue: LintIssue) -> None:
        """Append ``issue`` and update the severity counters."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.errors += 1
        elif issue.severity == "warning":
            self.warnings += 1


# ── Helpers ─────────────────────────────────────────────────────────────


def _is_valid_iso_timestamp(value: object) -> bool:
    """Return True if ``value`` represents a valid ISO 8601 datetime/date.

    PyYAML's ``safe_load`` may already convert an ISO 8601 scalar into a
    :class:`datetime.datetime` / :class:`datetime.date` object — those are
    accepted directly. String scalars are parsed with
    :func:`datetime.datetime.fromisoformat` (3.11+) which accepts trailing
    ``Z`` and offsets, and fall back to :func:`datetime.date.fromisoformat`
    for bare dates. Booleans, ``None`` and other non-string types are
    rejected.
    """
    # Already-parsed datetime/date objects are valid by construction.
    if isinstance(value, (datetime.datetime, datetime.date)):
        return True
    if not isinstance(value, str) or not value:
        return False
    candidate = value
    # ``datetime.fromisoformat`` (3.11+) accepts trailing ``Z`` and offsets.
    try:
        datetime.datetime.fromisoformat(candidate)
        return True
    except ValueError:
        pass
    # Try as a bare date.
    try:
        datetime.date.fromisoformat(candidate)
        return True
    except ValueError:
        return False


def _is_valid_iso_date(value: str) -> bool:
    """Return True if ``value`` is a valid ISO 8601 *date* (``YYYY-MM-DD``)."""
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _resolve_link_target(src_file: Path, url: str) -> Path | None:
    """Resolve a markdown link URL relative to ``src_file``.

    Absolute-style OKF links (``/concepts/foo.md``) are resolved against the
    bundle root (the parent directory of ``src_file``). Relative links are
    resolved against the directory containing ``src_file``. External links
    (``http://``, ``https://``, ``mailto:``, anchors ``#``) return ``None``
    meaning "not checked".
    """
    if not url:
        return None
    lowered = url.lower()
    if lowered.startswith(("http://", "https://", "mailto:", "ftp://")):
        return None
    if url.startswith("#"):
        return None
    # Strip any in-file anchor or query.
    path_part = url.split("#", 1)[0].split("?", 1)[0]
    if not path_part:
        return None
    bundle_root = src_file.parent
    if path_part.startswith("/"):
        return bundle_root.joinpath(path_part.lstrip("/"))
    return src_file.parent.joinpath(path_part)


# ── Main entry point ─────────────────────────────────────────────────────


def lint_bundle(bundle_dir: str | Path) -> LintReport:
    """Lint an OKF bundle rooted at ``bundle_dir``.

    Scans every ``.md`` file under ``bundle_dir`` (recursively) and returns a
    :class:`LintReport`. Reserved files (``index.md``, ``log.md``) are
    delegated to :func:`_lint_reserved`; all other files are checked with the
    non-reserved rule set (OKF-001 through OKF-005).
    """
    root = Path(bundle_dir)
    report = LintReport()
    if not root.is_dir():
        return report

    md_files = sorted(root.rglob("*.md"), key=lambda p: str(p.relative_to(root)))
    report.files_checked = len(md_files)

    # Cache of every .md path in the bundle, used for cross-link validation.
    all_paths: set[Path] = set(md_files)

    for md_file in md_files:
        rel = md_file.relative_to(root).as_posix()
        raw = safe_read_file(md_file)
        if not raw:
            # Empty/unreadable file: nothing to lint here.
            continue
        fm, body = parse_frontmatter(raw)
        has_frontmatter = raw.startswith("---\n")

        if md_file.name in RESERVED_FILES:
            _lint_reserved(md_file, rel, fm, body, has_frontmatter, report)
        else:
            _lint_concept(md_file, rel, fm, body, has_frontmatter,
                          report, all_paths)

    return report


# ── Non-reserved files ───────────────────────────────────────────────────


def _lint_concept(
    md_file: Path,
    rel: str,
    fm: dict,
    body: str,
    has_frontmatter: bool,
    report: LintReport,
    all_paths: set[Path],
) -> None:
    """Apply OKF-001..005 to a regular concept ``.md`` file."""
    # OKF-001: Missing YAML frontmatter.
    if not has_frontmatter:
        report.add(LintIssue(
            severity="error", file=rel, line=1,
            message="Missing YAML frontmatter block (no leading '---' fence).",
            rule="OKF-001",
        ))
        # Without frontmatter there is no type/timestamp/tags to check, but we
        # still check cross-links in the body.
    else:
        # OKF-002: Missing or empty type field.
        type_val = fm.get("type")
        if not isinstance(type_val, str) or type_val.strip() == "":
            report.add(LintIssue(
                severity="error", file=rel, line=None,
                message="Missing or empty 'type' field in frontmatter.",
                rule="OKF-002",
            ))

        # OKF-003: Invalid ISO 8601 timestamp (only when a timestamp is set).
        ts_val = fm.get("timestamp")
        if ts_val is not None and not _is_valid_iso_timestamp(ts_val):
            report.add(LintIssue(
                severity="warning", file=rel, line=None,
                message="timestamp is not a valid ISO 8601 value: "
                        f"{ts_val!r}",
                rule="OKF-003",
            ))

        # OKF-004: tags not a YAML list.
        tags_val = fm.get("tags")
        if tags_val is not None and not isinstance(tags_val, list):
            report.add(LintIssue(
                severity="warning", file=rel, line=None,
                message="'tags' should be a YAML list, got "
                        f"{type(tags_val).__name__}.",
                rule="OKF-004",
            ))

    # OKF-005: Broken cross-link. Checked for every file regardless of
    # frontmatter presence, because links live in the body.
    _check_cross_links(md_file, rel, body, report, all_paths)


def _check_cross_links(
    md_file: Path,
    rel: str,
    body: str,
    report: LintReport,
    all_paths: set[Path],
) -> None:
    """Emit OKF-005 for each markdown link whose target does not exist."""
    # Normalise candidate paths once for case-insensitive comparison.
    known: set[str] = {str(p).lower() for p in all_paths}
    for _text, url in extract_links(body):
        target = _resolve_link_target(md_file, url)
        if target is None:
            # External/anchor link — not checked.
            continue
        if str(target).lower() in known:
            continue
        report.add(LintIssue(
            severity="info", file=rel, line=None,
            message=f"Broken cross-link target does not exist: {url}",
            rule="OKF-005",
        ))


# ── Reserved files ───────────────────────────────────────────────────────


def _lint_reserved(
    md_file: Path,
    rel: str,
    fm: dict,
    body: str,
    has_frontmatter: bool,
    report: LintReport,
) -> None:
    """Lint ``index.md`` and ``log.md`` specifically (OKF-006, OKF-007)."""
    name = md_file.name

    if name == "index.md":
        # OKF-006: a non-bundle-root index.md should not carry frontmatter.
        # The bundle-root index.md may carry frontmatter *only* if it has an
        # ``okf_version`` key. ``rel == "index.md"`` means it lives at the
        # bundle root (the top of the scanned tree).
        is_bundle_root = rel == "index.md"
        if has_frontmatter:
            if is_bundle_root and "okf_version" in fm:
                # Allowed: bundle-root index.md with okf_version.
                pass
            else:
                detail = ("bundle-root index.md lacks 'okf_version'"
                          if is_bundle_root
                          else "sub-directory index.md should have no frontmatter")
                report.add(LintIssue(
                    severity="warning", file=rel, line=1,
                    message=f"index.md carries unexpected frontmatter: {detail}.",
                    rule="OKF-006",
                ))

    elif name == "log.md":
        # OKF-007: date headings (## YYYY-MM-DD) should be ISO 8601 dates.
        for match in _DATE_HEADING_RE.finditer(body):
            date_str = match.group("date")
            if not _is_valid_iso_date(date_str):
                line_no = body.count("\n", 0, match.start()) + 1
                report.add(LintIssue(
                    severity="warning", file=rel, line=line_no,
                    message=f"log.md date heading is not a valid ISO 8601 "
                            f"date: {date_str!r}",
                    rule="OKF-007",
                ))
