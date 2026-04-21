"""Shared utility functions used across pipeline modules."""

from pathlib import Path
import re


def count_md(directory: Path) -> int:
    """Count .md files in a directory (non-recursive)."""
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.md")))


def extract_frontmatter_field(content: str, field: str) -> str:
    """Extract a single field value from YAML frontmatter."""
    pattern = rf"^{field}:\s*[\"']?(.*?)[\"']?\s*$"
    match = re.search(pattern, content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def escape_yaml(s: str) -> str:
    """Escape strings for safe YAML double-quoted values."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def strip_qmd_noise(text: str) -> str:
    """Strip cmake/build noise from qmd stdout, keeping JSON array."""
    for marker in ["[{\n", "[{", "[\n", "["]:
        idx = text.find(marker)
        if idx != -1:
            candidate = text[idx:]
            bracket = 0
            for i, ch in enumerate(candidate):
                if ch == "[":
                    bracket += 1
                elif ch == "]":
                    bracket -= 1
                    if bracket == 0:
                        return candidate[:i + 1]
    return text


def extract_body(content: str) -> str:
    """Extract body text (after YAML frontmatter) from a markdown file."""
    m = re.match(r"^---\n.*?\n---\n(.*)", content, re.DOTALL)
    return m.group(1) if m else content
