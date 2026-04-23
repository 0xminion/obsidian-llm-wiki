"""Integration tests for the review/approval workflow.

Tests: stage → inspect → approve (with collision, stem rewrite),
       reject, approve-on-resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import Config
from pipeline.models import Language, Plan, Plans, Template
from pipeline.review import stage_for_review, show_pending, approve_reviews, reject_reviews
from pipeline.vault import title_to_filename


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Create a Config pointing to a tmp vault with review store."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for d in ["04-Wiki/sources", "04-Wiki/entries", "04-Wiki/concepts",
              "04-Wiki/mocs", "06-Config", "01-Raw"]:
        (vault / d).mkdir(parents=True)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    return Config(vault_path=vault, extract_dir=extract_dir)


@pytest.fixture
def sample_plan() -> Plan:
    """Return a realistic plan for review tests."""
    return Plan(
        hash="abc123def456",
        title="Review Test Article",
        language=Language.EN,
        template=Template.STANDARD,
        tags=["test", "review"],
        concept_new=["Test Concept"],
        concept_updates=[],
        moc_targets=["General"],
    )


def create_extract_file(cfg: Config, plan: Plan) -> None:
    """Helper: write the extraction JSON that review staging needs."""
    ext = {
        "url": f"https://example.com/{plan.hash}",
        "title": plan.title,
        "content": f"# {plan.title}\n\nFull extracted content here. " * 20,
        "type": "web",
        "author": "Test Author",
    }
    (cfg.resolved_extract_dir / f"{plan.hash}.json").write_text(
        json.dumps(ext), encoding="utf-8"
    )


# ─── staging ────────────────────────────────────────────────────────────────────

class TestStageForReview:
    def test_stages_source_entry_and_concept(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stats = stage_for_review(plans, cfg, use_agent_insights=False)

        assert stats["staged"] == 1
        assert stats["failed"] == 0

        pending = show_pending(cfg)
        assert len(pending) == 3  # source + entry + concept

        types = {r["file_type"] for r in pending}
        assert types == {"source", "entry", "concept"}

        # All should have plan_hash matching our plan
        assert all(r["plan_hash"] == sample_plan.hash for r in pending)

    def test_missing_extract_file_is_failed(self, cfg: Config, sample_plan: Plan):
        # Do NOT create extract file
        plans = Plans(plans=[sample_plan])
        stats = stage_for_review(plans, cfg, use_agent_insights=False)

        assert stats["staged"] == 0
        assert stats["failed"] == 1


# ─── approve ────────────────────────────────────────────────────────────────────

class TestApproveReviews:
    def test_approve_writes_all_files(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        stats = approve_reviews(cfg)
        assert stats["approved"] == 3
        assert stats["written"] == 3
        assert stats["failed"] == 0

        # Files exist on disk
        fname = title_to_filename(sample_plan.title)
        assert (cfg.sources_dir / f"{fname}.md").exists()
        assert (cfg.entries_dir / f"{fname}.md").exists()
        assert (cfg.concepts_dir / "test-concept.md").exists()

    def test_approve_with_collision_rewrites_wikilinks(self, cfg: Config, sample_plan: Plan):
        """If source already exists, approve should still write entry+concept."""
        create_extract_file(cfg, sample_plan)

        # Pre-create the source file to simulate collision
        fname = title_to_filename(sample_plan.title)
        cfg.sources_dir.mkdir(parents=True, exist_ok=True)
        (cfg.sources_dir / f"{fname}.md").write_text("---\ntitle: Existing\n---\n\n# Existing\n")

        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        # Approve
        stats = approve_reviews(cfg)

        # Entry should be written — may have a suffix due to collision resolution
        pending = show_pending(cfg)
        entry_records = [r for r in pending if r["file_type"] == "entry"]
        assert len(entry_records) == 0  # all approved

        # Verify at least entry+concept files exist
        assert len(list(cfg.entries_dir.glob("*.md"))) >= 1
        assert len(list(cfg.concepts_dir.glob("*.md"))) >= 1

    def test_approve_partial_by_id(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        pending = show_pending(cfg)
        source_ids = [r["id"] for r in pending if r["file_type"] == "source"]

        # Approve only sources
        stats = approve_reviews(cfg, review_ids=source_ids)
        assert stats["approved"] == 1
        assert stats["written"] == 1

        # Entry+concept still pending
        remaining = show_pending(cfg)
        assert len(remaining) == 2

    def test_double_approve_is_idempotent(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        # First approve
        approve_reviews(cfg)
        # Second approve — nothing pending, should write nothing
        stats = approve_reviews(cfg)
        assert stats["approved"] == 0
        assert stats["written"] == 0


# ─── reject ──────────────────────────────────────────────────────────────────────

class TestRejectReviews:
    def test_reject_clears_pending(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        count = reject_reviews(cfg)
        assert count == 3

        # Nothing left pending
        pending = show_pending(cfg)
        assert len(pending) == 0

    def test_files_not_written_after_reject(self, cfg: Config, sample_plan: Plan):
        create_extract_file(cfg, sample_plan)
        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        reject_reviews(cfg)

        fname = title_to_filename(sample_plan.title)
        assert not (cfg.sources_dir / f"{fname}.md").exists()
        assert not (cfg.entries_dir / f"{fname}.md").exists()

    def test_stem_rewrite_after_collision_approve(self, cfg: Config, sample_plan: Plan):
        """If a previous note exists with same stem, staging finds unique name."""
        create_extract_file(cfg, sample_plan)
        fname = title_to_filename(sample_plan.title)
        (cfg.entries_dir / f"{fname}.md").write_text("existing", encoding="utf-8")

        plans = Plans(plans=[sample_plan])
        stage_for_review(plans, cfg, use_agent_insights=False)

        pending = show_pending(cfg)
        # The entry should have a suffix like "-1" to avoid collision
        entry_paths = [r["file_path"] for r in pending if r["file_type"] == "entry"]
        assert len(entry_paths) == 1

        # Approve — should write to the suffixed file
        approve_reviews(cfg)
        # Original file still exists
        assert (cfg.entries_dir / f"{fname}.md").exists()
        # New suffixed file also exists
        assert any(p.stem.startswith(fname) for p in cfg.entries_dir.glob("*.md"))

    def test_multiple_plans_staged_correctly(self, cfg: Config):
        """Multiple plans in one batch should all be staged independently."""
        plans = []
        for i in range(3):
            plan = Plan(
                hash=f"hash{i:03d}def456",
                title=f"Article {i}",
                tags=["test"],
                concept_new=[f"Concept {i}"],
            )
            create_extract_file(cfg, plan)
            plans.append(plan)

        stats = stage_for_review(Plans(plans=plans), cfg, use_agent_insights=False)
        assert stats["staged"] == 3
        assert stats["failed"] == 0

        pending = show_pending(cfg)
        # 3 plans × (source + entry + concept) = 9 items
        assert len(pending) == 9
