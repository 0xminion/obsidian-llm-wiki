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
from pipeline.utils import assert_path_within, safe_note_path, safe_note_stem

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
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        return _stage_for_review_impl(store, plans, cfg, use_agent_insights)
    finally:
        store.close()


def _stage_for_review_impl(
    store: ContentStore,
    plans: Plans,
    cfg: Config,
    use_agent_insights: bool,
) -> dict:
    from pipeline.create import (
        _generate_concept_template,
        generate_entry_content,
        generate_entry_insights,
        generate_source_content,
    )
    from pipeline.vault import title_to_filename

    extract_dir = cfg.resolved_extract_dir
    stats = {"staged": 0, "failed": 0}
    reserved_paths = {Path(item["file_path"]) for item in store.review_get_pending()}

    def _reserve_note_filenames(base_filename: str) -> tuple[str, str]:
        """Reserve distinct (source, entry) stems for graph-unambiguous notes."""
        safe_base = safe_note_stem(base_filename)
        suffix = 0
        max_attempts = 1000
        while True:
            entry_candidate = safe_base if suffix == 0 else f"{safe_base}-{suffix}"
            source_candidate = f"{entry_candidate}-source"
            source_path = safe_note_path(cfg.sources_dir, source_candidate)
            entry_path = safe_note_path(cfg.entries_dir, entry_candidate)
            if (
                not source_path.exists()
                and not entry_path.exists()
                and source_path not in reserved_paths
                and entry_path not in reserved_paths
            ):
                reserved_paths.add(source_path)
                reserved_paths.add(entry_path)
                return source_candidate, entry_candidate
            suffix += 1
            if suffix > max_attempts:
                raise RuntimeError("Too many filename collisions")

    def _reserve_path(directory: Path, base_filename: str) -> tuple[str, Path]:
        candidate = safe_note_stem(base_filename)
        suffix = 0
        max_attempts = 1000
        while True:
            path = safe_note_path(directory, candidate)
            if not path.exists() and path not in reserved_paths:
                reserved_paths.add(path)
                return candidate, path
            suffix += 1
            candidate = f"{base_filename}-{suffix}"
            if suffix > max_attempts:
                raise RuntimeError("Too many filename collisions")

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
            source_filename, entry_filename = _reserve_note_filenames(filename)
            note_suffix = entry_filename[len(filename):] if entry_filename.startswith(filename) else ""
            entry_link_name = f"{plan.title}{note_suffix}"
            source_content = generate_source_content(
                plan,
                extracted,
                note_title=entry_link_name,
            )
            source_path = safe_note_path(cfg.sources_dir, source_filename)
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

            entry_path = safe_note_path(cfg.entries_dir, entry_filename)
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

        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            log.error("Failed to stage %s: %s", plan.title, e)
            stats["failed"] += 1

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


def _review_base_dir(cfg: Config, file_type: str) -> Path:
    mapping = {
        "source": cfg.sources_dir,
        "entry": cfg.entries_dir,
        "concept": cfg.concepts_dir,
        "moc": cfg.mocs_dir,
    }
    if file_type not in mapping:
        raise ValueError(f"unsupported review file_type: {file_type}")
    return mapping[file_type]


def _safe_review_target(cfg: Config, review: dict) -> Path:
    """Resolve a staged review path and require it to stay under its type dir."""
    base_dir = _review_base_dir(cfg, review["file_type"])
    original = Path(review["file_path"])
    stem = safe_note_stem(original.stem)
    target = safe_note_path(base_dir, stem)
    # Accept existing absolute/relative paths only when they resolve to this exact
    # safe target or another path under the same allowed type directory. This
    # preserves old staged rows while rejecting arbitrary absolute paths.
    if original.is_absolute():
        assert_path_within(base_dir, original)
        target = original.resolve()
    return target


def approve_reviews(cfg: Config, review_ids: Optional[list[int]] = None) -> dict:
    """Approve and atomically write pending reviews to the vault.

    Review rows are grouped by plan_hash. A plan writes only if every selected
    row for that plan has a safe path and valid content. If one row fails, all
    sibling rows stay pending and no vault file is written for that plan.
    """
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        all_pending = store.review_get_pending()
        pending = all_pending
        if review_ids:
            pending = [r for r in pending if r["id"] in review_ids]

        stats = {"approved": 0, "written": 0, "failed": 0, "written_paths": []}
        if not pending:
            return stats

        plan_targets: dict[str, dict[str, str]] = {}
        resolved_reviews: list[dict] = []
        reserved_paths: set[Path] = set()
        failed_plan_hashes: set[str] = set()

        # Preflight: resolve paths, collisions, and cross-file stem rewrites.
        for review in pending:
            plan_hash = review["plan_hash"]
            try:
                file_path = _safe_review_target(cfg, review)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                resolved_path = file_path
                if resolved_path.exists() or resolved_path in reserved_paths:
                    idx = 1
                    max_attempts = 1000
                    while True:
                        candidate = safe_note_path(resolved_path.parent, f"{resolved_path.stem}-{idx}")
                        if not candidate.exists() and candidate not in reserved_paths:
                            resolved_path = candidate
                            break
                        idx += 1
                        if idx > max_attempts:
                            raise RuntimeError("Too many filename collisions")
                    log.warning("Collision resolved: wrote %s instead", resolved_path.name)
                reserved_paths.add(resolved_path)

                targets = plan_targets.setdefault(plan_hash, {})
                original_stem = safe_note_stem(Path(review["file_path"]).stem)
                if review["file_type"] == "source":
                    targets.setdefault("source_old", original_stem)
                    targets["source_new"] = resolved_path.stem
                elif review["file_type"] == "entry":
                    targets.setdefault("entry_old", original_stem)
                    targets["entry_new"] = resolved_path.stem

                resolved_reviews.append({**review, "resolved_path": resolved_path})
            except (OSError, ValueError, RuntimeError) as e:
                log.error("Failed to preflight review %s: %s", review.get("id"), e)
                failed_plan_hashes.add(plan_hash)
                stats["failed"] += 1

        # Build global stem map from all collision resolutions.
        stem_map: dict[str, str] = {}
        for targets in plan_targets.values():
            for key in ("source_old", "entry_old"):
                old = targets.get(key)
                new = targets.get(key.replace("_old", "_new"))
                if old and new and old != new:
                    stem_map[old] = new

        by_plan: dict[str, list[dict]] = {}
        for review in resolved_reviews:
            by_plan.setdefault(review["plan_hash"], []).append(review)

        successful_hashes: set[str] = set()
        for plan_hash, reviews in by_plan.items():
            if plan_hash in failed_plan_hashes:
                stats["failed"] += len(reviews)
                continue
            prepared: list[tuple[dict, Path, str]] = []
            try:
                for review in reviews:
                    file_path = review["resolved_path"]
                    file_content = _rewrite_review_content(review, plan_targets, stem_map)
                    if not _review_content_is_valid(file_content):
                        raise ValueError("staged content failed validation")
                    prepared.append((review, file_path, file_content))
            except (OSError, ValueError) as e:
                log.error("Plan %s failed validation: %s", plan_hash, e)
                stats["failed"] += len(reviews)
                continue

            temp_paths: list[Path] = []
            committed_paths: list[Path] = []
            try:
                for _review, file_path, file_content in prepared:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = file_path.with_name(f".{file_path.name}.tmp")
                    assert_path_within(file_path.parent, tmp_path)
                    tmp_path.write_text(file_content, encoding="utf-8")
                    temp_paths.append(tmp_path)
                for (_review, file_path, _file_content), tmp_path in zip(prepared, temp_paths):
                    tmp_path.replace(file_path)
                    committed_paths.append(file_path)
                for review, file_path, _file_content in prepared:
                    store.review_approve(review["id"])
                    stats["approved"] += 1
                    stats["written"] += 1
                    stats["written_paths"].append(str(file_path))
                    log.info("Approved and wrote: %s", file_path.name)
                successful_hashes.add(plan_hash)
            except OSError as e:
                log.error("Plan %s failed during atomic write: %s", plan_hash, e)
                for tmp_path in temp_paths:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                for file_path in committed_paths:
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError:
                        log.warning("Rollback failed for %s", file_path)
                stats["failed"] += len(reviews)

    finally:
        store.close()

    if stats["written"] > 0:
        selected_pending_ids = {r["id"] for r in pending}
        fully_successful_hashes = {
            plan_hash
            for plan_hash in successful_hashes
            if all(r["id"] in selected_pending_ids for r in all_pending if r["plan_hash"] == plan_hash)
        }
        try:
            from pipeline.vault import archive_clippings, archive_inbox
            from pipeline.vault import reindex as vault_reindex
            vault_reindex(cfg)
            archive_inbox(cfg, fully_successful_hashes)
            archive_clippings(cfg, fully_successful_hashes)
        except OSError as e:
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
