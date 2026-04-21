"""Review/approval workflow for vault writes.

Instead of writing directly to the vault, files are staged for review.
Users can approve, reject, or inspect pending files before they're written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pipeline.config import Config
from pipeline.models import Plan, Plans
from pipeline.store import ContentStore

log = logging.getLogger(__name__)


def stage_for_review(
    plans: Plans,
    cfg: Config,
    use_agent_insights: bool = True,
) -> dict:
    """Generate file content and stage for review instead of writing to vault.

    Uses the template-based creator to generate all files, then stores them
    in the pending_reviews table for approval.
    """
    from pipeline.create import (
        generate_source_content,
        generate_entry_content,
        generate_entry_insights,
        _generate_concept_template,
    )
    from pipeline.vault import title_to_filename

    store = ContentStore.open(cfg.resolved_extract_dir)
    extract_dir = cfg.resolved_extract_dir
    stats = {"staged": 0, "failed": 0}

    for plan in plans.plans:
        try:
            extract_file = extract_dir / f"{plan.hash}.json"
            if not extract_file.exists():
                log.warning("Extract file missing for %s", plan.hash)
                stats["failed"] += 1
                continue

            extracted = json.loads(extract_file.read_text(encoding="utf-8"))
            filename = title_to_filename(plan.title)

            # 1. Stage Source
            source_content = generate_source_content(plan, extracted)
            source_path = str(cfg.sources_dir / f"{filename}.md")
            store.review_add(
                plan_hash=plan.hash,
                plan_data=plan.to_dict(),
                file_type="source",
                file_path=source_path,
                file_content=source_content,
            )

            # 2. Generate insights + stage Entry
            insights = ""
            if use_agent_insights:
                insights = generate_entry_insights(plan, extracted, cfg)

            entry_content = generate_entry_content(plan, extracted, filename, insights)
            entry_path = str(cfg.entries_dir / f"{filename}.md")
            store.review_add(
                plan_hash=plan.hash,
                plan_data=plan.to_dict(),
                file_type="entry",
                file_path=entry_path,
                file_content=entry_content,
            )

            # 3. Stage Concepts (if new)
            for concept_name in plan.concept_new:
                concept_content = _generate_concept_template(concept_name, plan)
                concept_filename = title_to_filename(concept_name)
                concept_path = str(cfg.concepts_dir / f"{concept_filename}.md")
                store.review_add(
                    plan_hash=plan.hash,
                    plan_data=plan.to_dict(),
                    file_type="concept",
                    file_path=concept_path,
                    file_content=concept_content,
                )

            stats["staged"] += 1

        except Exception as e:
            log.error("Failed to stage %s: %s", plan.title, e)
            stats["failed"] += 1

    store.close()
    return stats


def show_pending(cfg: Config) -> list[dict]:
    """Show all pending reviews."""
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        return store.review_get_pending()
    finally:
        store.close()


def approve_reviews(cfg: Config, review_ids: Optional[list[int]] = None) -> dict:
    """Approve and write pending reviews to the vault.

    If review_ids is None, approves all pending reviews.
    After writing, runs reindex and archives inbox.
    """
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        pending = store.review_get_pending()

        if review_ids:
            pending = [r for r in pending if r["id"] in review_ids]

        stats = {"approved": 0, "written": 0, "failed": 0}
        approved_hashes: set[str] = set()

        for review in pending:
            try:
                # Write file to vault
                file_path = Path(review["file_path"])
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(review["file_content"], encoding="utf-8")

                # Mark as approved
                store.review_approve(review["id"])
                stats["approved"] += 1
                stats["written"] += 1
                approved_hashes.add(review["plan_hash"])
                log.info("Approved and wrote: %s", file_path.name)

            except Exception as e:
                log.error("Failed to write %s: %s", review["file_path"], e)
                stats["failed"] += 1
    finally:
        store.close()

    # Post-processing: reindex + archive
    if stats["written"] > 0:
        try:
            from pipeline.vault import reindex as vault_reindex, archive_inbox
            vault_reindex(cfg)
            archive_inbox(cfg, approved_hashes)
        except Exception as e:
            log.warning("Post-approve reindex/archive failed: %s", e)

    return stats


def reject_reviews(cfg: Config, review_ids: Optional[list[int]] = None) -> int:
    """Reject pending reviews.

    If review_ids is None, rejects all pending reviews.
    Returns count rejected.
    """
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        if review_ids:
            count = 0
            for rid in review_ids:
                store.review_reject(rid)
                count += 1
        else:
            pending = store.review_get_pending()
            count = len(pending)
            for r in pending:
                store.review_reject(r["id"])
        return count
    finally:
        store.close()
