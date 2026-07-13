"""Tests for typed maintenance findings and non-executing fix plans."""

from __future__ import annotations

from obsidian_llm_wiki.core.maintenance import (
    FindingKind,
    FindingSeverity,
    FixKind,
    MaintenanceFinding,
    plan_fixes,
)


def test_plan_fixes_maps_deterministic_broken_relation_repair() -> None:
    finding = MaintenanceFinding(
        kind=FindingKind.BROKEN_RELATION,
        path="concepts/alpha.md",
        message="Relation target does not exist",
        severity=FindingSeverity.ERROR,
        details={"target": "missing-page"},
    )

    plan = plan_fixes([finding])

    assert len(plan) == 1
    assert plan[0].kind is FixKind.REMOVE_BROKEN_RELATION
    assert plan[0].path == "concepts/alpha.md"
    assert plan[0].payload == {"target": "missing-page"}
    assert plan[0].requires_review is False


def test_plan_fixes_skips_any_change_to_reviewed_content() -> None:
    finding = MaintenanceFinding(
        kind=FindingKind.TAG_NORMALIZATION,
        path="concepts/curated.md",
        message="Tag needs normalization",
        details={"tag": "has space", "normalized_tag": "has-space"},
        reviewed=True,
    )

    assert plan_fixes([finding]) == []


def test_plan_fixes_marks_assignment_candidates_for_human_review() -> None:
    finding = MaintenanceFinding(
        kind=FindingKind.MOC_ASSIGNMENT_CANDIDATE,
        path="concepts/orphan.md",
        message="Likely MoC candidate",
        details={"moc": "topic"},
    )

    plan = plan_fixes([finding])

    assert len(plan) == 1
    assert plan[0].kind is FixKind.PROPOSE_MOC_ASSIGNMENT
    assert plan[0].requires_review is True


def test_plan_fixes_is_deterministic_and_never_executes_actions() -> None:
    later = MaintenanceFinding(
        kind=FindingKind.EMPTY_GENERATED_STUB,
        path="concepts/zeta.md",
        message="Empty generated stub",
    )
    earlier = MaintenanceFinding(
        kind=FindingKind.TAG_NORMALIZATION,
        path="concepts/alpha.md",
        message="Normalize tag",
        details={"tag": "has space", "normalized_tag": "has-space"},
    )

    plan = plan_fixes([later, earlier])

    assert [item.path for item in plan] == ["concepts/alpha.md", "concepts/zeta.md"]
    assert [item.kind for item in plan] == [
        FixKind.NORMALIZE_TAG,
        FixKind.REMOVE_EMPTY_GENERATED_STUB,
    ]
    assert all(not hasattr(item, "apply") for item in plan)
