"""Output validation and auto-repair for created vault files.

Validates immediately after agent creation (per-batch) and after full pipeline.
Auto-repair derives real content from the file's existing body — never stubs.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import Config

log = logging.getLogger(__name__)

# Patterns that indicate stub content
_STUB_PATTERNS = [
    re.compile(r">\s*待补充", re.IGNORECASE),
    re.compile(r">\s*TODO\b", re.IGNORECASE),
    re.compile(r"Full article text available in raw extraction", re.IGNORECASE),
    re.compile(r"\bTo be written\b\.?", re.IGNORECASE),
    re.compile(r"Key information extracted from source material", re.IGNORECASE),
    re.compile(r"See related entries and sources for broader context", re.IGNORECASE),
    re.compile(r"Full content should be embedded here from extraction", re.IGNORECASE),
    re.compile(r"Primary themes and arguments from the source", re.IGNORECASE),
]

# Tags that should never appear
_BANNED_TAGS = {"x.com", "tweet", "source", "http", "https", "url", "link", "rss", "feed"}

_REQUIRED_FM_FIELDS: dict[str, list[str]] = {
    "entry": ["title", "source", "date_entry", "status", "template", "tags"],
    "concept": ["title", "type", "status", "sources", "tags"],
}

_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "entry": ["Summary", "Core insights", "Other takeaways", "Open questions", "Linked concepts"],
    "entry_chinese": ["摘要", "核心发现", "其他要点", "开放问题", "关联概念"],
    "entry_technical": ["Summary", "Key Findings", "Data/Evidence", "Methodology", "Limitations", "Linked concepts"],
    "concept": ["Core concept", "Context", "Links"],
    "concept_chinese": ["核心概念", "背景", "关联"],
    "source": ["Original content"],
}

_MIN_BODY_LENGTHS = {
    "entry": 200,
    "concept": 150,
    "source": 200,
}

# Source body patterns that indicate just a link instead of full content
_LINK_ONLY_PATTERNS = [
    re.compile(r"^\[Original source\]\(https?://", re.MULTILINE),
    re.compile(r"^URL: https?://\s*$", re.MULTILINE),
]


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter as a dict. Returns empty dict if invalid."""
    try:
        import yaml
    except ImportError:
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1))
        return fm if isinstance(fm, dict) else {}
    except Exception:
        return {}


def _get_required_sections(fm: dict, note_type: str) -> list[str]:
    """Determine required sections based on template and language."""
    template = str(fm.get("template", "standard")).strip()
    language = str(fm.get("language", fm.get("lang", "en"))).strip()

    if note_type == "concept":
        if language == "zh":
            return _REQUIRED_SECTIONS.get("concept_chinese", _REQUIRED_SECTIONS["concept"])
        return _REQUIRED_SECTIONS["concept"]

    if note_type == "source":
        return _REQUIRED_SECTIONS["source"]

    # Entry — pick by template
    key = f"entry_{template}" if template != "standard" else "entry"
    if language == "zh":
        key = "entry_chinese"
    return _REQUIRED_SECTIONS.get(key, _REQUIRED_SECTIONS["entry"])


def validate_single_file(file_path: Path, note_type: str) -> list[str]:
    """Validate a single vault file. Returns list of violation strings.

    This is the per-file validation used immediately after agent creation.
    """
    violations = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        return [f"Cannot read file: {e}"]

    # Check frontmatter exists
    if not content.startswith("---"):
        violations.append("missing YAML frontmatter")
        return violations  # Can't check further without frontmatter

    fm_end = content.find("---", 3)
    if fm_end == -1:
        violations.append("unclosed YAML frontmatter")
        return violations

    frontmatter_str = content[3:fm_end]
    fm = _parse_frontmatter(content)

    # Check required frontmatter fields (use regex ^field: to avoid substring matches)
    for field_name in _REQUIRED_FM_FIELDS.get(note_type, []):
        if not re.search(rf"^{field_name}:\s", frontmatter_str, re.MULTILINE):
            violations.append(f"missing frontmatter field: {field_name}")

    # Check banned tags
    tags = fm.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            tag_str = str(tag).strip().lower()
            if tag_str in _BANNED_TAGS:
                violations.append(f"banned tag: {tag}")

    # Check stub content
    body = content[fm_end + 3:]
    for pattern in _STUB_PATTERNS:
        if pattern.search(body):
            violations.append(f"stub content detected: {pattern.pattern}")
            break  # One stub violation per file is enough

    # Check required sections
    required = _get_required_sections(fm, note_type)
    for section in required:
        if f"## {section}" not in content:
            violations.append(f"missing required section: ## {section}")

    # Check minimum body length (exclude headings)
    stripped_body = re.sub(r"^#+\s*.*$", "", body, flags=re.MULTILINE)
    stripped_body = re.sub(r"\s+", "", stripped_body)
    min_len = _MIN_BODY_LENGTHS.get(note_type, 100)
    if len(stripped_body) < min_len:
        violations.append(f"body too short: {len(stripped_body)} chars (min {min_len})")

    # Source-specific: check for link-only sources
    if note_type == "source":
        for pattern in _LINK_ONLY_PATTERNS:
            if pattern.search(content):
                violations.append("source has link instead of full content")
                break

    return violations


def validate_batch(files: list[tuple[Path, str]]) -> dict[str, list[str]]:
    """Validate a batch of (file_path, note_type) pairs.

    Returns {file_path_str: [violations]}.
    """
    results = {}
    for file_path, note_type in files:
        violations = validate_single_file(file_path, note_type)
        if violations:
            results[str(file_path)] = violations
    return results


def validate_output(cfg: Config, since_manifest: Path) -> list[str]:
    """Check files created after the manifest timestamp for violations.

    Validates entries, concepts, and sources directories.
    Returns list of violation strings.
    """
    violations: list[str] = []

    # Get manifest timestamp
    if since_manifest.exists():
        try:
            manifest_mtime = since_manifest.stat().st_mtime
        except OSError:
            manifest_mtime = 0
    else:
        manifest_mtime = 0

    dirs_to_check = [
        (cfg.entries_dir, "entry"),
        (cfg.concepts_dir, "concept"),
        (cfg.sources_dir, "source"),
    ]

    for dir_path, note_type in dirs_to_check:
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            if md_file.stat().st_mtime < manifest_mtime:
                continue

            file_violations = validate_single_file(md_file, note_type)
            for v in file_violations:
                violations.append(f"{note_type}:{md_file.name}: {v}")

    return violations


def _derive_section_content(section: str, content: str, note_type: str) -> str:
    """Derive real section content from the file's existing body.

    Never returns stubs. Uses actual content from the note.
    """
    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    lines = body.split("\n")

    if section in ("Summary", "摘要"):
        # Find the longest substantial paragraph that isn't a heading or list item
        best = ""
        for line in lines:
            stripped = line.strip()
            if (stripped and not stripped.startswith("#")
                    and not stripped.startswith("![")
                    and not stripped.startswith("- ")
                    and not stripped.startswith("* ")
                    and len(stripped) > 60):
                if len(stripped) > len(best):
                    best = stripped
        return best[:400] if best else ""

    if section in ("Core insights", "核心发现"):
        insights = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "1.")) and len(stripped) > 25:
                clean = re.sub(r"^\d+\.\s*", "", stripped)
                insights.append(stripped if stripped.startswith("- ") else f"- {clean}")
            if len(insights) >= 5:
                break
        return "\n".join(insights) if insights else ""

    if section in ("Linked concepts", "关联概念", "Links", "关联"):
        links = re.findall(r"\[\[([^\]]+)\]\]", body)
        if links:
            unique = list(dict.fromkeys(links))[:15]
            return "\n".join(f"- [[{link}]]" for link in unique)
        return ""

    if section in ("Core concept", "核心概念"):
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 50:
                return stripped[:500]
        return ""

    if section in ("Context", "背景"):
        # Gather all non-heading, non-list paragraphs
        paragraphs = []
        current = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                break
            if stripped and not stripped.startswith("#") and not stripped.startswith("- "):
                current.append(stripped)
            elif current:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return "\n\n".join(p for p in paragraphs if len(p) > 30)[:800]

    if section in ("Open questions", "开放问题"):
        questions = []
        for line in lines:
            stripped = line.strip()
            if "?" in stripped and len(stripped) > 15:
                if not stripped.startswith("#"):
                    questions.append(stripped if stripped.startswith("- ") else f"- {stripped}")
            if len(questions) >= 5:
                break
        return "\n".join(questions) if questions else ""

    if section in ("Other takeaways", "其他要点"):
        takeaways = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if "other takeaways" in stripped.lower() or "其他要点" in stripped:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("## "):
                    break
                if stripped.startswith(("- ", "* ")) and len(stripped) > 15:
                    takeaways.append(stripped)
                if len(takeaways) >= 5:
                    break
        return "\n".join(takeaways) if takeaways else ""

    if section == "Diagrams":
        return "n/a"

    if section == "Original content":
        # For sources, extract from the body itself
        content_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        return "\n".join(content_lines[:5]) if content_lines else ""

    # Fallback: try to find a relevant subsection in the body
    for line in lines:
        stripped = line.strip().lower()
        if section.lower() in stripped and len(stripped) > 20:
            return line.strip()

    return ""


def _repair_violations(cfg: Config, violations: list[str]) -> int:
    """Attempt to auto-repair common validation violations.

    Only repairs missing sections by deriving real content from the file.
    Does NOT create stubs — returns empty content if nothing can be derived.
    Returns count of files repaired.
    """
    repaired = 0

    for violation in violations:
        # Parse: "entry:filename.md: missing required section: ## Section"
        match = re.match(r"(\w+):(.+?): missing required section: ## (.+)", violation)
        if not match:
            continue

        note_type, filename, section = match.groups()

        dir_map = {
            "entry": cfg.entries_dir,
            "concept": cfg.concepts_dir,
            "source": cfg.sources_dir,
        }
        note_dir = dir_map.get(note_type)
        if not note_dir:
            continue

        file_path = note_dir / filename
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")

        # Skip if section already exists
        if f"## {section}" in content:
            continue

        # Derive real content
        section_content = _derive_section_content(section, content, note_type)
        if not section_content:
            log.info("Cannot derive content for ## %s in %s — skipping repair", section, filename)
            continue

        # Insert before the last section
        last_section_pos = content.rfind("\n## ")
        if last_section_pos > 0:
            new_content = (
                content[:last_section_pos]
                + f"\n\n## {section}\n\n{section_content}\n"
                + content[last_section_pos:]
            )
        else:
            new_content = content.rstrip() + f"\n\n## {section}\n\n{section_content}\n"

        file_path.write_text(new_content, encoding="utf-8")
        repaired += 1
        log.info("Auto-repaired: %s:%s — added ## %s (%d chars derived content)",
                 note_type, filename, section, len(section_content))

    return repaired
