"""Lint data models — Severity, LintIssue, LintResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class LintIssue:
    check: str
    severity: Severity
    note: str
    detail: str
    section: str = ""


@dataclass
class LintResult:
    report_date: str
    total_issues: int = 0
    issues_by_check: dict[str, int] = field(default_factory=dict)
    issues: list[LintIssue] = field(default_factory=list)
    files_checked: int = 0
    fixes_applied: int = 0
