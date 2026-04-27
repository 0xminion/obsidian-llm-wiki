"""Lint check functions — all check_* functions and their helpers."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline.lint.models import LintIssue, Severity
from pipeline.note_schema import (
    concept_schema,
    effective_entry_schema,
    markdown_headings,
)
from pipeline.utils import (
    extract_body as _extract_body,
)
from pipeline.utils import (
    parse_frontmatter as _parse_frontmatter,
)

log = logging.getLogger(__name__)


# ─── Shared Utilities ─────────────────────────────────────────────────────────


def _find_md_files(vault: Path, *dirs: str) -> list[Path]:
    """Find all .md files under given vault subdirectories."""
    files = []
    for d in dirs:
        dir_path = vault / d
        if dir_path.exists():
            files.extend(sorted(dir_path.glob("*.md")))
    return files


# ─── Cache-Aware Wikilink Index ─────────────────────────────────────────────

_WIKI_DIRS = ("04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources")


def _build_wikilink_index(vault: Path, cache=None) -> tuple[dict[str, Path], dict[str, set[str]], dict[str, set[str]]]:
    """Build note name index and wikilink graph.

    Uses vault cache for incremental updates — only re-reads files whose mtime
    changed since the last lint run.

    Args:
        vault: Vault root path
        cache: Optional ContentStore with vault cache methods

    Returns:
        (note_paths, incoming_links, outgoing_links) where:
        - note_paths: {note_name: file_path}
        - incoming_links: {note_name: set of notes that link TO it}
        - outgoing_links: {note_name: set of notes it links to}
    """
    note_paths: dict[str, Path] = {}
    outgoing: dict[str, set[str]] = {}  # note_name -> set of notes it links to

    # Check if we can use cached data
    use_cache = cache is not None
    if use_cache:
        for d in _WIKI_DIRS:
            if cache.cache_is_directory_stale(vault / d):
                use_cache = False
                break

    if use_cache:
        cached_links = cache.cache_get_wikilinks(vault)
        if cached_links:
            # Rebuild note_paths from current filesystem (fast — no file reads)
            for d in _WIKI_DIRS:
                dir_path = vault / d
                if not dir_path.exists():
                    continue
                for md in dir_path.glob("*.md"):
                    note_paths[md.stem] = md

            # Build incoming from cached outgoing
            incoming: dict[str, set[str]] = {name: set() for name in note_paths}
            for source, targets in cached_links.items():
                for target in targets:
                    if target in incoming:
                        incoming[target].add(source)

            log.debug("Lint: using cached wikilink index (%d notes)", len(note_paths))
            return note_paths, incoming, {}

    # Cache miss or stale — build from scratch
    for d in _WIKI_DIRS:
        dir_path = vault / d
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            note_name = md.stem
            note_paths.setdefault(note_name, md)
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                links = set(re.findall(r"\[\[([^|#\]]+)(?:[|#][^\]]*)?\]\]", content))
                outgoing.setdefault(note_name, set()).update(links)
            except OSError:
                outgoing.setdefault(note_name, set())

    # Build incoming links from outgoing
    incoming: dict[str, set[str]] = {name: set() for name in note_paths}
    for source, targets in outgoing.items():
        for target in targets:
            if target in incoming:
                incoming[target].add(source)

    # Update cache
    if cache is not None:
        cache.cache_set_wikilinks(outgoing)
        for d in _WIKI_DIRS:
            dir_path = vault / d
            if dir_path.exists():
                index = {}
                for md in dir_path.glob("*.md"):
                    try:
                        index[md.name] = md.stat().st_mtime
                    except OSError:
                        pass
                cache.cache_set_file_index(dir_path, index)
        log.debug("Lint: rebuilt wikilink index (%d notes, cached)", len(note_paths))

    return note_paths, incoming, outgoing


# ─── Check Functions ─────────────────────────────────────────────────────────

def check_orphaned_notes(vault: Path, _cache=None) -> list[LintIssue]:
    """Check 1: Files with zero incoming wikilinks.

    Uses cached wikilink index when available.
    """
    issues = []
    note_paths, incoming, _ = _build_wikilink_index(vault, _cache)

    for note_name, md_path in note_paths.items():
        if not incoming.get(note_name):
            rel = md_path.relative_to(vault)
            issues.append(LintIssue(
                check="orphaned_notes",
                severity=Severity.WARNING,
                note=note_name,
                detail=f"No incoming wikilinks — {rel}",
            ))

    return issues


def check_unreviewed_entries(vault: Path) -> list[LintIssue]:
    """Check 2: Entries with reviewed: null or empty."""
    issues = []
    entries_dir = vault / "04-Wiki" / "entries"
    if not entries_dir.exists():
        return issues

    for md in entries_dir.glob("*.md"):
        fm = _parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
        reviewed = fm.get("reviewed")
        if not reviewed or str(reviewed).strip() in ("", "null", "None"):
            date_entry = fm.get("date_entry", "unknown")
            issues.append(LintIssue(
                check="unreviewed_entries",
                severity=Severity.INFO,
                note=md.stem,
                detail=f"created: {date_entry}",
            ))

    return issues


def check_stale_reviews(vault: Path, days: int = 14) -> list[LintIssue]:
    """Check 3: status: review older than N days."""
    issues = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    for d in ("04-Wiki/entries", "04-Wiki/concepts"):
        dir_path = vault / d
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            fm = _parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
            status = str(fm.get("status", "")).strip()
            if status != "review":
                continue
            date_str = str(fm.get("updated") or fm.get("date_entry", "")).strip()
            if not date_str:
                continue
            try:
                note_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if note_date < cutoff:
                    issues.append(LintIssue(
                        check="stale_reviews",
                        severity=Severity.WARNING,
                        note=md.stem,
                        detail=f"dated: {date_str} ({days}+ days old)",
                    ))
            except ValueError:
                pass

    return issues


# ─── Knowledge Decay / Staleness Scoring ───────────────────────────────────────

_VOLATILITY_MAP: dict[str, str] = {
    "crypto": "high",
    "ai": "high",
    "blockchain": "high",
    "ethereum": "high",
    "bitcoin": "high",
    "gpt": "high",
    "llm": "high",
    "tech": "medium",
    "technology": "medium",
    "science": "medium",
    "research": "medium",
    "history": "low",
    "philosophy": "low",
    "art": "low",
    "literature": "low",
}

_VOLATILITY_DEFAULT_DAYS = 3 * 365  # 3 years default


def _parse_note_date(fm: dict, mtime: float | None = None) -> datetime | None:
    """Extract date from frontmatter or file mtime."""
    for key in ("date", "source_date"):
        val = fm.get(key)
        if val:
            s = str(val).strip()[:10]
            try:
                return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    if mtime is not None:
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    return None


def _compute_staleness(
    note_path: Path, thresholds: dict[str, int] | None = None
) -> tuple[bool, float, str]:
    """Compute staleness for a single note.

    Returns (is_stale, days_old, threshold_tag).
    """
    content = note_path.read_text(encoding="utf-8", errors="replace")
    fm = _parse_frontmatter(content)
    mtime = note_path.stat().st_mtime
    note_date = _parse_note_date(fm, mtime)

    if note_date is None:
        return False, 0.0, ""

    now = datetime.now(timezone.utc)
    days_old = (now - note_date).total_seconds() / 86400.0

    tags = fm.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    # Determine highest volatility from tags present
    max_volatility = "default"
    volatility_days = thresholds or _STALENESS_THRESHOLDS()

    for tag in tags:
        tag_str = str(tag).strip().lower()
        v = _VOLATILITY_MAP.get(tag_str)
        if v:
            # Check if this tag overrides a current max
            if _volatility_rank(v) > _volatility_rank(max_volatility):
                max_volatility = v
    # Map back to threshold tag
    tag_to_threshold = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "default": "default",
    }
    threshold_tag = tag_to_threshold.get(max_volatility, "default")
    threshold_days = volatility_days.get(threshold_tag, _VOLATILITY_DEFAULT_DAYS)

    is_stale = days_old > threshold_days
    return is_stale, days_old, threshold_tag


def _volatility_rank(volatility: str) -> int:
    return {"default": 0, "low": 1, "medium": 2, "high": 3}.get(volatility, 0)


def _STALENESS_THRESHOLDS(
    default: int = 3 * 365,
    high: int = 365,
    medium: int = 730,
    low: int = 5 * 365,
) -> dict[str, int]:
    return {
        "default": default,
        "high": high,
        "medium": medium,
        "low": low,
    }


def check_staleness(vault: Path) -> list[LintIssue]:
    """Check: Notes older than volatility-based staleness threshold."""
    issues = []
    thresholds = _STALENESS_THRESHOLDS()

    for d in ("04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/sources"):
        dir_path = vault / d
        if not dir_path.exists():
            continue
        for md in dir_path.glob("*.md"):
            try:
                is_stale, days_old, threshold_tag = _compute_staleness(md, thresholds)
            except (OSError, ValueError):
                continue
            if not is_stale:
                continue
            # Severity by volatility
            if threshold_tag == "high":
                severity = Severity.WARNING
            elif threshold_tag == "medium":
                severity = Severity.INFO
            elif threshold_tag == "low":
                severity = Severity.INFO
            else:
                severity = Severity.INFO
            threshold_days = thresholds.get(threshold_tag, thresholds["default"])
            issues.append(LintIssue(
                check="staleness",
                severity=severity,
                note=md.stem,
                detail=f"{days_old:.0f} days old (threshold {threshold_days}d, tag '{threshold_tag}')",
            ))

    return issues


def check_broken_wikilinks(vault: Path, _cache=None) -> list[LintIssue]:
    """Check 4: Wikilinks pointing to non-existent notes.

    Uses cached wikilink index when available.
    """
    issues = []
    note_paths, _, outgoing = _build_wikilink_index(vault, _cache)
    existing_names = set(note_paths.keys())

    for note_name, targets in outgoing.items():
        md = note_paths[note_name]
        rel = md.relative_to(vault)
        for target in targets:
            if target not in existing_names:
                issues.append(LintIssue(
                    check="broken_wikilinks",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail=f"In {rel} → [[{target}]]",
                ))

    return issues


def check_empty_notes(vault: Path, min_chars: int = 50) -> list[LintIssue]:
    """Check 5: Notes with < min_chars of body content."""
    issues = []
    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources"):
        content = md.read_text(encoding="utf-8", errors="replace")
        body = _extract_body(content)
        # Strip headings and whitespace
        stripped = re.sub(r"^#+\s*.*$", "", body, flags=re.MULTILINE)
        stripped = re.sub(r"\s+", "", stripped)
        if len(stripped) < min_chars:
            issues.append(LintIssue(
                check="empty_notes",
                severity=Severity.WARNING,
                note=md.stem,
                detail=f"{len(stripped)} chars body",
            ))

    return issues


def check_concept_structure(vault: Path) -> list[LintIssue]:
    """Check 6: Concepts have required sections (Core concept/核心概念, Context/背景, Links/关联)."""
    issues = []
    concepts_dir = vault / "04-Wiki" / "concepts"
    if not concepts_dir.exists():
        return issues

    for md in concepts_dir.glob("*.md"):
        content = md.read_text(encoding="utf-8", errors="replace")
        fm = _parse_frontmatter(content)
        lang = str(fm.get("language", "en")).strip()

        schema = concept_schema(lang)
        required = list(markdown_headings(schema))

        missing = [s for s in required if s not in content]
        if missing:
            issues.append(LintIssue(
                check="concept_structure",
                severity=Severity.ERROR,
                note=md.stem,
                detail=f"missing sections: {', '.join(missing)}",
            ))

    return issues


def check_entry_template_sections(vault: Path) -> list[LintIssue]:
    """Check 7: Entry sections match template type."""
    issues = []
    entries_dir = vault / "04-Wiki" / "entries"
    if not entries_dir.exists():
        return issues

    for md in entries_dir.glob("*.md"):
        content = md.read_text(encoding="utf-8", errors="replace")
        fm = _parse_frontmatter(content)
        template = str(fm.get("template", "standard")).strip() or "standard"
        language = str(fm.get("language", "en")).strip() or "en"

        required = list(markdown_headings(effective_entry_schema(language, template)))
        missing = [s for s in required if s not in content]
        if missing:
            issues.append(LintIssue(
                check="entry_template_sections",
                severity=Severity.ERROR,
                note=md.stem,
                detail=f"(template: {template}) missing: {', '.join(missing)}",
            ))

    return issues


def check_orphaned_concepts(vault: Path) -> list[LintIssue]:
    """Check 8: Concepts not referenced by any Entry."""
    issues = []
    concepts_dir = vault / "04-Wiki" / "concepts"
    entries_dir = vault / "04-Wiki" / "entries"
    if not concepts_dir.exists() or not entries_dir.exists():
        return issues

    # Build set of all wikilink references in entries (precise match)
    entry_wikilinks: set[str] = set()
    for md in entries_dir.glob("*.md"):
        content = md.read_text(encoding="utf-8", errors="replace")
        entry_wikilinks.update(
            re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", content)
        )

    for concept_md in concepts_dir.glob("*.md"):
        concept_name = concept_md.stem
        referenced = concept_name in entry_wikilinks
        if not referenced:
            # Self-references don't count
            self_ref = f"[[{concept_name}]]" in concept_md.read_text(encoding="utf-8", errors="replace")
            if not self_ref:
                issues.append(LintIssue(
                    check="orphaned_concepts",
                    severity=Severity.WARNING,
                    note=concept_name,
                    detail="No Entry links to this concept",
                ))

    return issues


def check_wiki_index_drift(vault: Path) -> list[LintIssue]:
    """Check 9: wiki-index.md counts vs actual file counts."""
    issues = []
    index_file = vault / "06-Config" / "wiki-index.md"

    if not index_file.exists():
        issues.append(LintIssue(
            check="wiki_index_drift",
            severity=Severity.ERROR,
            note="wiki-index.md",
            detail="File not found — run `pipeline reindex`",
        ))
        return issues

    content = index_file.read_text(encoding="utf-8", errors="replace")
    index_entries = len(re.findall(r"\(entry\)", content))
    index_concepts = len(re.findall(r"\(concept\)", content))

    actual_entries = len(list((vault / "04-Wiki" / "entries").glob("*.md"))) if (vault / "04-Wiki" / "entries").exists() else 0
    actual_concepts = len(list((vault / "04-Wiki" / "concepts").glob("*.md"))) if (vault / "04-Wiki" / "concepts").exists() else 0

    if index_entries != actual_entries:
        issues.append(LintIssue(
            check="wiki_index_drift",
            severity=Severity.ERROR,
            note="Entry mismatch",
            detail=f"Index: {index_entries}, actual: {actual_entries}",
        ))

    if index_concepts != actual_concepts:
        issues.append(LintIssue(
            check="wiki_index_drift",
            severity=Severity.ERROR,
            note="Concept mismatch",
            detail=f"Index: {index_concepts}, actual: {actual_concepts}",
        ))

    return issues


def check_edges_consistency(vault: Path) -> list[LintIssue]:
    """Check 10: Edges.tsv references non-existent notes."""
    issues = []
    edges_file = vault / "06-Config" / "edges.tsv"

    if not edges_file.exists():
        issues.append(LintIssue(
            check="edges_consistency",
            severity=Severity.ERROR,
            note="edges.tsv",
            detail="File not found — run `pipeline compile`",
        ))
        return issues

    existing_names = set()
    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources"):
        existing_names.add(md.stem)

    lines = edges_file.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    for line in lines[1:]:  # skip header
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        source, target = parts[0].strip(), parts[1].strip()
        if source and source not in existing_names:
            issues.append(LintIssue(
                check="edges_consistency",
                severity=Severity.ERROR,
                note=f"Edge source '{source}'",
                detail="Not found as a note",
            ))
        if target and target not in existing_names:
            issues.append(LintIssue(
                check="edges_consistency",
                severity=Severity.ERROR,
                note=f"Edge target '{target}'",
                detail="Not found as a note",
            ))

    return issues


def check_weak_links(vault: Path) -> list[LintIssue]:
    """Check 10b: Notes with incoming but zero outgoing edges (weak links).

    Severity: WARNING for notes with >3 incoming and 0 outgoing.
    """
    issues = []
    edges_file = vault / "06-Config" / "edges.tsv"

    if not edges_file.exists():
        return issues

    # Count incoming and outgoing per note
    incoming: dict[str, int] = {}
    outgoing: dict[str, int] = {}
    lines = edges_file.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        source, target = parts[0].strip(), parts[1].strip()
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1

    for note, inc_count in incoming.items():
        if inc_count > 3 and outgoing.get(note, 0) == 0:
            issues.append(LintIssue(
                check="weak_links",
                severity=Severity.WARNING,
                note=note,
                detail=f"{inc_count} incoming edges, 0 outgoing — likely weak link",
            ))

    return issues


_STUB_PATTERNS = [
    r"^>\s*待补充",
    r"^>\s*待分析",
    r"^>\s*待深入研究",
    r"^>\s*待深入",
    r"^>\s*TODO",
    r"^>\s*TBD",
    r"^>\s*FIXME",
    r"^>\s*PLACEHOLDER",
    r"^>\s*待完善",
    r"^>\s*待更新",
    r"^>\s*待定",
    r"^>\s*待处理",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bPLACEHOLDER\b",
    r"\bTBD\b",
    r"\bTo be written\b",
    r"待补充",
    r"待填",
    r"\[insert",
    r"Content goes here",
    r"Write your",
    r"Lorem ipsum",
]


def check_stubs(vault: Path) -> list[LintIssue]:
    """Check 11: Stub/placeholder detection."""
    issues = []
    pattern = re.compile("|".join(_STUB_PATTERNS), re.IGNORECASE)

    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/concepts"):
        content = md.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        current_section = ""
        for i, line in enumerate(lines, 1):
            if line.startswith("## "):
                current_section = line[3:].strip()
            if pattern.search(line):
                issues.append(LintIssue(
                    check="stubs",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail=f"Section '{current_section}': `{line.strip()[:60]}`",
                ))

    return issues


_BLOCKED_TAGS = {"x.com", "tweet", "http", "https", "rss", "feed", "url", "link"}


def check_tag_quality(vault: Path) -> list[LintIssue]:
    """Check 12: Banned tags, too-short tags, and potential synonyms."""
    issues = []

    all_tags: dict[str, list[str]] = {}  # tag -> list of file stems

    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/sources", "04-Wiki/concepts"):
        content = md.read_text(encoding="utf-8", errors="replace")
        fm = _parse_frontmatter(content)
        tags = fm.get("tags", [])
        if not isinstance(tags, list):
            continue

        for tag in tags:
            tag_str = str(tag).strip().lower()
            if tag_str in _BLOCKED_TAGS:
                issues.append(LintIssue(
                    check="tag_quality",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail=f"Blocked tag: '{tag}'",
                ))
            elif len(tag_str) <= 1:
                issues.append(LintIssue(
                    check="tag_quality",
                    severity=Severity.WARNING,
                    note=md.stem,
                    detail=f"Too-short tag: '{tag}'",
                ))
            else:
                if tag_str not in all_tags:
                    all_tags[tag_str] = []
                all_tags[tag_str].append(md.stem)

    # Synonym detection: find tags that differ only by hyphenation, pluralization, or case
    tag_list = sorted(all_tags.keys())
    for i, tag_a in enumerate(tag_list):
        for tag_b in tag_list[i + 1:]:
            # Skip if already reported
            if tag_a == tag_b:
                continue
            # Check for near-duplicates
            normalized_a = tag_a.replace("-", "").replace("_", "").rstrip("s")
            normalized_b = tag_b.replace("-", "").replace("_", "").rstrip("s")
            if normalized_a == normalized_b and tag_a != tag_b:
                files_a = ", ".join(all_tags[tag_a][:3])
                files_b = ", ".join(all_tags[tag_b][:3])
                issues.append(LintIssue(
                    check="tag_synonyms",
                    severity=Severity.INFO,
                    note=f"'{tag_a}' ↔ '{tag_b}'",
                    detail=f"Possible synonyms — used in: {files_a} / {files_b}",
                ))

    return issues


# ─── Validate-Output Checks (from validate-output.sh) ────────────────────────

def check_frontmatter_validity(vault: Path) -> list[LintIssue]:
    """Validate-check 1: YAML frontmatter parses correctly, no null values."""
    issues = []
    try:
        import yaml as _yaml
    except ImportError:
        issues.append(LintIssue(
            check="frontmatter_validity",
            severity=Severity.WARNING,
            note="(global)",
            detail="PyYAML not installed — skipping YAML validation",
        ))
        return issues

    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs", "04-Wiki/sources"):
        content = md.read_text(encoding="utf-8", errors="replace")
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not fm_match:
            issues.append(LintIssue(
                check="frontmatter_validity",
                severity=Severity.ERROR,
                note=md.stem,
                detail="No YAML frontmatter found",
            ))
            continue

        yaml_text = fm_match.group(1)
        try:
            fm = _yaml.safe_load(yaml_text)
        except _yaml.YAMLError as e:
            issues.append(LintIssue(
                check="frontmatter_validity",
                severity=Severity.ERROR,
                note=md.stem,
                detail=f"YAML parse error: {e}",
            ))
            continue

        # Check for null values
        if isinstance(fm, dict):
            for key, value in fm.items():
                if value is None and key not in ("reviewed", "review_notes"):
                    issues.append(LintIssue(
                        check="frontmatter_validity",
                        severity=Severity.ERROR,
                        note=md.stem,
                        detail=f"Null value for '{key}' — use empty string instead",
                    ))

        # Check unquoted wikilinks in YAML
        if re.search(r"^source:\s*\[\[", yaml_text, re.MULTILINE):
            if not re.search(r'^source:\s*"\[\[', yaml_text, re.MULTILINE):
                issues.append(LintIssue(
                    check="frontmatter_validity",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail='Wikilink in YAML not quoted — use source: "[[note]]"',
                ))

    return issues


def check_required_sections(vault: Path) -> list[LintIssue]:
    """Validate-check 2: Required sections per note type.

    Currently only checks MoCs. Entry and concept section checks are
    handled by check_entry_template_sections and check_concept_structure.
    """
    issues = []
    vault / "04-Wiki" / "concepts"
    mocs_dir = vault / "04-Wiki" / "mocs"

    # MoC checks
    if mocs_dir.exists():
        for md in mocs_dir.glob("*.md"):
            content = md.read_text(encoding="utf-8", errors="replace")
            if "## Overview / 概述" not in content:
                issues.append(LintIssue(
                    check="required_sections",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail="Missing required MoC section: ## Overview / 概述",
                ))
            section_count = len(re.findall(r"^## ", content, re.MULTILINE))
            if section_count < 2:
                issues.append(LintIssue(
                    check="required_sections",
                    severity=Severity.ERROR,
                    note=md.stem,
                    detail=f"MoC has only {section_count} section(s) — needs Overview + at least 1 topic",
                ))

    return issues


def check_markdown_format(vault: Path) -> list[LintIssue]:
    """Validate-check 6: H1 title, blank lines after headings."""
    issues = []

    for md in _find_md_files(vault, "04-Wiki/entries", "04-Wiki/concepts", "04-Wiki/mocs"):
        content = md.read_text(encoding="utf-8", errors="replace")
        body = _extract_body(content)
        if not body.strip():
            continue

        lines = body.split("\n")
        # Check 1: First non-empty line should be H1
        first_line = ""
        for line in lines:
            if line.strip():
                first_line = line.strip()
                break
        if first_line and not first_line.startswith("# "):
            issues.append(LintIssue(
                check="markdown_format",
                severity=Severity.ERROR,
                note=md.stem,
                detail=f"Body must start with H1 title, found: {first_line[:60]}",
            ))

        # Check 2: Blank line after ## headings
        for i, line in enumerate(lines):
            if line.startswith("## ") or line.startswith("### "):
                if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith("#"):
                    issues.append(LintIssue(
                        check="markdown_format",
                        severity=Severity.ERROR,
                        note=md.stem,
                        detail=f"Missing blank line after heading at line {i + 1}: '{line[:40]}'",
                    ))

    return issues
