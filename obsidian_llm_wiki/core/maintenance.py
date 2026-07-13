"""Typed maintenance findings and deterministic, non-executing fix plans.

Scanning and filesystem mutation intentionally live outside this module.  The
core planner converts already-detected findings into a serializable proposal
that a CLI can display, dry-run, or explicitly apply after backups are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "FindingKind",
    "FindingSeverity",
    "FixKind",
    "MaintenanceFinding",
    "PlannedFix",
    "plan_deterministic_fixes",
    "plan_fixes",
]


class FindingSeverity(StrEnum):
    """User-visible severity for a maintenance finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingKind(StrEnum):
    """Supported machine-readable maintenance finding categories."""

    BROKEN_RELATION = "broken_relation"
    ORPHAN_CONCEPT = "orphan_concept"
    MOC_ASSIGNMENT_CANDIDATE = "moc_assignment_candidate"
    TAG_NORMALIZATION = "tag_normalization"
    ALIAS_CANDIDATE = "alias_candidate"
    EMPTY_GENERATED_STUB = "empty_generated_stub"


class FixKind(StrEnum):
    """Deterministic actions a caller may choose to execute later."""

    REMOVE_BROKEN_RELATION = "remove_broken_relation"
    PROPOSE_MOC_ASSIGNMENT = "propose_moc_assignment"
    NORMALIZE_TAG = "normalize_tag"
    PROPOSE_ALIAS = "propose_alias"
    REMOVE_EMPTY_GENERATED_STUB = "remove_empty_generated_stub"


@dataclass(frozen=True, slots=True)
class MaintenanceFinding:
    """A typed fact emitted by a health scan, without any mutation behavior."""

    kind: FindingKind
    path: str
    message: str
    severity: FindingSeverity = FindingSeverity.WARNING
    details: dict[str, Any] = field(default_factory=dict)
    reviewed: bool = False


@dataclass(frozen=True, slots=True)
class PlannedFix:
    """A deterministic proposal; this data object never performs a repair."""

    kind: FixKind
    path: str
    payload: dict[str, Any] = field(default_factory=dict)
    requires_review: bool = False


_FIX_BY_FINDING: dict[FindingKind, tuple[FixKind, bool]] = {
    FindingKind.BROKEN_RELATION: (FixKind.REMOVE_BROKEN_RELATION, False),
    FindingKind.ORPHAN_CONCEPT: (FixKind.PROPOSE_MOC_ASSIGNMENT, True),
    FindingKind.MOC_ASSIGNMENT_CANDIDATE: (FixKind.PROPOSE_MOC_ASSIGNMENT, True),
    FindingKind.TAG_NORMALIZATION: (FixKind.NORMALIZE_TAG, False),
    FindingKind.ALIAS_CANDIDATE: (FixKind.PROPOSE_ALIAS, True),
    FindingKind.EMPTY_GENERATED_STUB: (FixKind.REMOVE_EMPTY_GENERATED_STUB, False),
}


def plan_fixes(findings: list[MaintenanceFinding]) -> list[PlannedFix]:
    """Build a stable, non-executing plan for safe maintenance actions.

    Every finding for a reviewed page is omitted: callers must not silently
    alter curated content.  Assignment and alias suggestions remain visible as
    plans but are marked for human review instead of being auto-applied.
    """
    plan: list[PlannedFix] = []
    for finding in sorted(findings, key=lambda item: (item.path, item.kind.value, item.message)):
        if finding.reviewed:
            continue
        fix_kind, requires_review = _FIX_BY_FINDING[finding.kind]
        plan.append(
            PlannedFix(
                kind=fix_kind,
                path=finding.path,
                payload=dict(finding.details),
                requires_review=requires_review,
            )
        )
    return plan


plan_deterministic_fixes = plan_fixes
