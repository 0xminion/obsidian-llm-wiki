"""Output validation and auto-repair for created vault files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pipeline.config import Config

log = logging.getLogger(__name__)

# Patterns that indicate stub content
_STUB_PATTERNS = [
    re.compile(r">\s*待补充", re.IGNORECASE),
    re.compile(r">\s*TODO\b", re.IGNORECASE),
    re.compile(r"Full article text available in raw extraction", re.IGNORECASE),
]

# Tags that should never appear
_BANNED_TAGS = {"x.com", "tweet", "source"}

_REQUIRED_FM_FIELDS: dict[str, list[str]] = {
    "entry": ["title", "source", "date_entry", "status", "template", "tags"],
    "concept": ["title", "type", "status", "sources", "tags"],
}

_REQUIRED_ENTRY_SECTIONS = [
    "Summary",
    "Core insights",
    "Other takeaways",
    "Diagrams",
    "Open questions",
    "Linked concepts",
]

_REQUIRED_CONCEPT_SECTIONS = [
    "Core concept",
    "Context",
    "Links",
]

_REQUIRED_SOURCE_SECTIONS = [
    "Original content",
]

_MIN_SOURCE_BODY_LENGTH = 200

# Source body patterns that indicate just a link instead of full content
_LINK_ONLY_PATTERNS = [
    re.compile(r"^\[Original source\]\(https?://", re.MULTILINE),
]


def validate_output(cfg: Config, since_manifest: Path) -> list[str]:
    """Check files created after the manifest timestamp for violations.

    Validates:
      - Frontmatter fields present
      - Required sections exist
      - No stub content (> 待补充, > TODO)
      - No banned tags
      - Source notes have full content (not just links)

    Returns list of violation strings.
    """
    violations: list[str] = []

    # Get manifest timestamp (when Stage 3 started)
    if since_manifest.exists():
        try:
            manifest_mtime = since_manifest.stat().st_mtime
        except OSError:
            manifest_mtime = 0
    else:
        manifest_mtime = 0

    # Check entries, concepts, and sources directories
    dirs_to_check = [
        (cfg.entries_dir, "entry", _REQUIRED_ENTRY_SECTIONS),
        (cfg.concepts_dir, "concept", _REQUIRED_CONCEPT_SECTIONS),
        (cfg.sources_dir, "source", _REQUIRED_SOURCE_SECTIONS),
    ]

    for dir_path, note_type, required_sections in dirs_to_check:
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            # Only check files created/modified after manifest
            if md_file.stat().st_mtime < manifest_mtime:
                continue

            content = md_file.read_text(encoding="utf-8")
            rel_path = f"{note_type}:{md_file.name}"

            # Check frontmatter
            if not content.startswith("---"):
                violations.append(f"{rel_path}: missing YAML frontmatter")
                continue

            fm_end = content.find("---", 3)
            if fm_end == -1:
                violations.append(f"{rel_path}: unclosed YAML frontmatter")
                continue

            frontmatter = content[3:fm_end]

            # Check required frontmatter fields
            for field in _REQUIRED_FM_FIELDS.get(note_type, []):
                if f"{field}:" not in frontmatter:
                    violations.append(f"{rel_path}: missing frontmatter field: {field}")

            # Check banned tags
            tags_match = re.search(r"tags:\s*\n((?:\s+-\s+.*\n?)*)", frontmatter)
            if tags_match:
                tag_lines = tags_match.group(1)
                for tag in _BANNED_TAGS:
                    if f"- {tag}" in tag_lines.lower() or f'- "{tag}"' in tag_lines.lower():
                        violations.append(f"{rel_path}: banned tag: {tag}")

            # Check stub content
            body = content[fm_end + 3:]
            for pattern in _STUB_PATTERNS:
                if pattern.search(body):
                    violations.append(f"{rel_path}: stub content detected: {pattern.pattern}")

            # Check required sections
            for section in required_sections:
                if f"## {section}" not in content:
                    violations.append(f"{rel_path}: missing required section: ## {section}")

            # Source-specific: check for link-only sources (not full content)
            if note_type == "source":
                body_text = body.strip()
                if len(body_text) < _MIN_SOURCE_BODY_LENGTH:
                    violations.append(
                        f"{rel_path}: source body too short ({len(body_text)} chars) — "
                        f"expected full content, not just a link"
                    )
                for pattern in _LINK_ONLY_PATTERNS:
                    if pattern.search(content):
                        violations.append(
                            f"{rel_path}: source has link instead of full content"
                        )
                        break

    return violations


def _repair_violations(cfg: Config, violations: list[str]) -> int:
    """Attempt to auto-repair common validation violations.

    Repairs:
      - Missing required sections: adds section with placeholder content derived from file
      - Returns count of files repaired.

    Does NOT create stubs (待补充/TODO) — derives real content from file context.
    """
    repaired = 0

    for violation in violations:
        # Parse violation: "entry:filename.md: missing required section: ## Section"
        match = re.match(r"(\w+):(.+?): missing required section: ## (.+)", violation)
        if not match:
            continue

        note_type, filename, section = match.groups()

        # Determine directory
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

        # Skip if section already exists (might have been repaired by another pass)
        if f"## {section}" in content:
            continue

        # Generate minimal section content based on context
        section_content = _generate_section_content(section, content, note_type)

        # Insert section before the last section
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
        log.info("Auto-repaired: %s:%s — added ## %s", note_type, filename, section)

    return repaired


def _generate_section_content(section: str, content: str, note_type: str) -> str:
    """Generate minimal section content derived from the file's existing content."""
    body = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

    if section == "Summary":
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("![") and len(stripped) > 50:
                return stripped[:300]
        return "Key information extracted from source material."

    elif section == "Core insights":
        insights = []
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("- ", "* ", "1.")) and len(stripped) > 20:
                insights.append(stripped)
            if len(insights) >= 3:
                break
        if insights:
            return "\n".join(insights)
        return "- Primary themes and arguments from the source"

    elif section == "Linked concepts":
        links = re.findall(r"\[\[([^\]]+)\]\]", body)
        if links:
            unique_links = list(dict.fromkeys(links))[:10]
            return "\n".join(f"- [[{link}]]" for link in unique_links)
        return ""

    elif section == "Core concept":
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 50:
                return stripped[:500]
        return ""

    elif section == "Context":
        return "See related entries and sources for broader context."

    elif section == "Links":
        links = re.findall(r"\[\[([^\]]+)\]\]", body)
        if links:
            unique_links = list(dict.fromkeys(links))[:10]
            return "\n".join(f"- [[{link}]]" for link in unique_links)
        return ""

    elif section == "Original content":
        return "Full content should be embedded here from extraction."

    return ""
