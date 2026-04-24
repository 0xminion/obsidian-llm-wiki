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
from pipeline.models import Plans
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
    reserved_paths = {Path(item["file_path"]) for item in store.review_get_pending()}

    def _reserve_note_filename(base_filename: str) -> str:
        candidate = base_filename
        suffix = 0
        while True:
            source_path = cfg.sources_dir / f"{candidate}.md"
            entry_path = cfg.entries_dir / f"{candidate}.md"
            if (
                not source_path.exists()
                and not entry_path.exists()
                and source_path not in reserved_paths
                and entry_path not in reserved_paths
            ):
                reserved_paths.add(source_path)
                reserved_paths.add(entry_path)
                return candidate
            suffix += 1
            candidate = f"{base_filename}-{suffix}"

    def _reserve_path(directory: Path, base_filename: str) -> tuple[str, Path]:
        candidate = base_filename
        suffix = 0
        while True:
            path = directory / f"{candidate}.md"
            if not path.exists() and path not in reserved_paths:
                reserved_paths.add(path)
                return candidate, path
            suffix += 1
            candidate = f"{base_filename}-{suffix}"

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
            note_filename = _reserve_note_filename(filename)
            source_filename = note_filename
            entry_filename = note_filename
            note_suffix = note_filename[len(filename):] if note_filename.startswith(filename) else ""
            entry_link_name = f"{plan.title}{note_suffix}"
            source_content = generate_source_content(
                plan,
                extracted,
                note_title=entry_link_name,
            )
            source_path = cfg.sources_dir / f"{source_filename}.md"
            store.review_add(
                plan_hash=plan.hash,
                plan_data=plan.to_dict(),
                file_type="source",
                file_path=str(source_path),
                file_content=source_content,
            )

            # 2. Generate insights + stage Entry
            insights = ""
            if use_agent_insights:
                insights = generate_entry_insights(plan, extracted, cfg)

            entry_path = cfg.entries_dir / f"{entry_filename}.md"
            entry_content = generate_entry_content(
                plan,
                extracted,
                source_filename,
                insights,
                note_title=entry_link_name,
            )
            store.review_add(
                plan_hash=plan.hash,
                plan_data=plan.to_dict(),
                file_type="entry",
                file_path=str(entry_path),
                file_content=entry_content,
            )

            # 3. Stage Concepts (if new)
            for concept_name in plan.concept_new:
                concept_content = _generate_concept_template(
                    concept_name,
                    plan,
                    source_note_name=entry_filename,
                    source_display_title=entry_link_name,
                )
                concept_filename = title_to_filename(concept_name)
                _, concept_path = _reserve_path(cfg.concepts_dir, concept_filename)
                store.review_add(
                    plan_hash=plan.hash,
                    plan_data=plan.to_dict(),
                    file_type="concept",
                    file_path=str(concept_path),
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


def _replace_wikilink_target(content: str, old_stem: str, new_stem: str) -> str:
    """Replace wikilink targets while preserving aliases/anchors."""
    import re

    pattern = re.compile(rf"\[\[{re.escape(old_stem)}(?P<suffix>[|#][^\]]*)?\]\]")
    return pattern.sub(lambda m: f"[[{new_stem}{m.group('suffix') or ''}]]", content)


def _review_content_is_valid(content: str) -> bool:
    """Cheap pre-approval validation for staged generated notes."""
    return "TODO" not in content


def _rewrite_review_content(review: dict, plan_targets: dict[str, dict[str, str]], stem_map: dict[str, str]) -> str:
    """Rewrite staged content to match any collision-resolved filenames.

    Replaces every wikilink [[old]] with [[new]] across the entire file
    (frontmatter + body) using a global stem_map.
    """
    content = review["file_content"]

    targets = plan_targets.get(review["plan_hash"], {})
    replacements: dict[str, str] = {}
    if review["file_type"] == "entry":
        # The source frontmatter in an entry points to the source note. Do not
        # rewrite it to the entry's collision-resolved stem just because the
        # source and entry shared an original basename.
        old = targets.get("source_old")
        new = targets.get("source_new")
        if old and new:
            replacements[old] = new
    elif review["file_type"] == "concept":
        old = targets.get("entry_old")
        new = targets.get("entry_new")
        if old and new:
            replacements[old] = new
    else:
        old = targets.get("entry_old")
        new = targets.get("entry_new")
        if old and new and old != targets.get("source_old"):
            replacements[old] = new

    for old_stem, new_stem in replacements.items():
        if old_stem != new_stem:
            content = _replace_wikilink_target(content, old_stem, new_stem)

    return content


def approve_reviews(cfg: Config, review_ids: Optional[list[int]] = None) -> dict:
    """Approve and write pending reviews to the vault.

    If review_ids is None, approves all pending reviews.
    After writing, runs reindex and archives inbox.
    """
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        all_pending = store.review_get_pending()
        pending = all_pending

        if review_ids:
            pending = [r for r in pending if r["id"] in review_ids]

        stats = {"approved": 0, "written": 0, "failed": 0}
        plan_outcomes: dict[str, dict[str, int]] = {}
        plan_targets: dict[str, dict[str, str]] = {}
        resolved_reviews: list[dict] = []

        reserved_paths: set[Path] = set()

        for review in pending:
            file_path = Path(review["file_path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path = file_path
            if resolved_path.exists() or resolved_path in reserved_paths:
                idx = 1
                while True:
                    candidate = resolved_path.parent / f"{file_path.stem}-{idx}.md"
                    if not candidate.exists() and candidate not in reserved_paths:
                        resolved_path = candidate
                        break
                    idx += 1
                log.warning("Collision resolved: wrote %s instead", resolved_path.name)
            reserved_paths.add(resolved_path)

            targets = plan_targets.setdefault(review["plan_hash"], {})
            if review["file_type"] == "source":
                targets.setdefault("source_old", file_path.stem)
                targets["source_new"] = resolved_path.stem
            elif review["file_type"] == "entry":
                targets.setdefault("entry_old", file_path.stem)
                targets["entry_new"] = resolved_path.stem

            resolved_reviews.append({**review, "resolved_path": resolved_path})

        # ─── Build global stem_map from all collision resolutions ───────────
        stem_map: dict[str, str] = {}
        for targets in plan_targets.values():
            for key in ("source_old", "entry_old"):
                old = targets.get(key)
                new_key = key.replace("_old", "_new")
                new = targets.get(new_key)
                if old and new and old != new:
                    stem_map[old] = new

        for review in resolved_reviews:
            plan_hash = review["plan_hash"]
            plan_outcomes.setdefault(plan_hash, {"written": 0, "failed": 0})
            try:
                file_path = review["resolved_path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_content = _rewrite_review_content(review, plan_targets, stem_map)
                if not _review_content_is_valid(file_content):
                    raise ValueError("staged content failed validation")
                file_path.write_text(file_content, encoding="utf-8")

                # Mark as approved
                store.review_approve(review["id"])
                stats["approved"] += 1
                stats["written"] += 1
                plan_outcomes[plan_hash]["written"] += 1
                log.info("Approved and wrote: %s", file_path.name)

            except Exception as e:
                log.error("Failed to write %s: %s", review["file_path"], e)
                stats["failed"] += 1
                plan_outcomes[plan_hash]["failed"] += 1
    finally:
        store.close()

    # Post-processing: reindex + archive
    if stats["written"] > 0:
        selected_pending_ids = {r["id"] for r in pending}
        successful_hashes = {
            plan_hash
            for plan_hash, outcome in plan_outcomes.items()
            if outcome["written"] > 0
            and outcome["failed"] == 0
            and all(r["id"] in selected_pending_ids for r in all_pending if r["plan_hash"] == plan_hash)
        }
        try:
            from pipeline.vault import reindex as vault_reindex, archive_inbox
            vault_reindex(cfg)
            archive_inbox(cfg, successful_hashes)
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
