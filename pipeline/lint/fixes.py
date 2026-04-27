"""Lint fix functions — auto-fix frontmatter, markdown format, and banned tags."""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.lint.checks import _BLOCKED_TAGS


def fix_frontmatter(file_path: Path) -> bool:
    """Fix null values and unquoted wikilinks in YAML frontmatter."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    original = content

    # Fix null values: 'key: null' or 'key: ~' → 'key: ""'
    content = re.sub(r"(:\s*)(null|~)(\s*$)", r'\1""\3', content, flags=re.MULTILINE)

    # Fix unquoted wikilinks in YAML frontmatter only
    fm_match = re.match(r"^(---\n)(.*?)(---\n)", content, re.DOTALL)
    if fm_match:
        fm = fm_match.group(2)
        # Only quote wikilinks that are NOT already surrounded by quotes
        fixed_fm = re.sub(r'(?<!["\'])((?:\[\[[^\]]+\]\])|(?:\[\[[^\]]+\]\]))(?!["\'])', r'"\1"', fm)
        content = fm_match.group(1) + fixed_fm + fm_match.group(3) + content[fm_match.end():]

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        return True
    return False


def fix_markdown_format(file_path: Path) -> bool:
    """Fix H1 title and blank lines around headings."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    original = content

    fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", content, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        body = fm_match.group(2)
    else:
        frontmatter = ""
        body = content

    # Fix 1: Ensure body starts with H1
    lines = body.split("\n")
    first_nonempty = ""
    first_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_nonempty = line.strip()
            first_idx = i
            break

    if first_nonempty and not first_nonempty.startswith("# "):
        # Extract title from frontmatter
        title_match = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', frontmatter, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled"
        lines.insert(first_idx, f"# {title}")
        lines.insert(first_idx + 1, "")

    # Fix 2: Add blank line after ## headings if missing
    fixed = []
    for i, line in enumerate(lines):
        fixed.append(line)
        if line.startswith("## ") or line.startswith("### "):
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith("#"):
                fixed.append("")

    # Fix 3: Add blank line before ## headings if missing
    fixed2 = []
    for i, line in enumerate(fixed):
        if (line.startswith("## ") or line.startswith("### ")) and i > 0:
            if fixed2 and fixed2[-1].strip() != "":
                fixed2.append("")
        fixed2.append(line)

    # Normalize multiple blank lines
    body = "\n".join(fixed2)
    body = re.sub(r"\n{3,}", "\n\n", body)

    content = frontmatter + body
    if content != original:
        file_path.write_text(content, encoding="utf-8")
        return True
    return False


def fix_banned_tags(file_path: Path) -> bool:
    """Remove banned tags from YAML frontmatter."""
    content = file_path.read_text(encoding="utf-8", errors="replace")

    # Only apply within frontmatter block
    fm_match = re.match(r"^(---\n)(.*?)(---\n)", content, re.DOTALL)
    if not fm_match:
        return False

    fm = fm_match.group(2)
    fixed_fm = fm
    for tag in _BLOCKED_TAGS:
        fixed_fm = re.sub(rf"^  - {re.escape(tag)}\s*$", "", fixed_fm, flags=re.MULTILINE)

    if fixed_fm != fm:
        content = fm_match.group(1) + fixed_fm + fm_match.group(3) + content[fm_match.end():]
        file_path.write_text(content, encoding="utf-8")
        return True
    return False
