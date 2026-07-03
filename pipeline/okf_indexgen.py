"""OKF-native index and changelog generation (deterministic, no LLM).

Produces the navigation artefacts for an OKF v0.1 bundle:

* **Per-directory ``index.md``** — plain markdown, no frontmatter, listing all
  concept files in that directory with their title and description.
* **Bundle-root ``index.md``** — frontmatter carrying ``okf_version``, listing
  every sub-directory together with its concept count.
* **``log.md``** — the compilation change log grouped by ISO date, newest first.

The functions here are pure: they accept :class:`~pathlib.Path` objects (or a
list of entries) and return strings.  Persisting the results is left to the
caller (e.g. via :func:`pipeline.okf_markdown.atomic_write`).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.okf_markdown import parse_frontmatter, safe_read_file
from pipeline.okf_models import LogEntry

__all__ = [
    "generate_directory_index",
    "generate_bundle_index",
    "generate_log",
    "append_log_entry",
]

# ── Helpers ───────────────────────────────────────────────────────────


def _read_concept_meta(path: Path) -> tuple[str, str]:
    """Return ``(title, description)`` for a concept ``.md`` file.

    Falls back to the file stem for the title and an empty string for the
    description when frontmatter is missing or unparseable.
    """
    raw = safe_read_file(path)
    meta, _body = parse_frontmatter(raw)
    title = meta.get("title") or path.stem
    description = meta.get("description") or ""
    return title, description


def _format_log_line(entry: LogEntry) -> str:
    """Render a single :class:`LogEntry` as a markdown bullet."""
    return (
        f"- **{entry.action}** "
        f"[{entry.concept_id}](/{entry.concept_id}.md) - {entry.description}"
    )


# ── Per-directory index ───────────────────────────────────────────────


def generate_directory_index(directory: Path, concept_files: list[Path]) -> str:
    """Generate the ``index.md`` content for a single directory.

    The output has *no* frontmatter.  Each concept file is listed as a
    bullet linking to the file (by filename, relative to the directory)
    followed by an em-dash style description separator::

        # DirectoryName

        - [Title](filename.md) - description

    ``index.md`` and ``log.md`` are explicitly skipped when they appear in
    ``concept_files``.
    """
    skip_names = {"index.md", "log.md"}

    lines: list[str] = [f"# {directory.name}", ""]

    for path in sorted(concept_files, key=lambda p: p.name.lower()):
        if path.name in skip_names:
            continue
        title, description = _read_concept_meta(path)
        line = f"- [{title}](/{directory.name}/{path.name})"
        if description:
            line += f" - {description}"
        lines.append(line)

    return "\n".join(lines) + "\n"


# ── Bundle-root index ─────────────────────────────────────────────────


def generate_bundle_index(bundle_dir: Path, okf_version: str = "0.1") -> str:
    """Generate the root ``index.md`` for an OKF bundle.

    The output carries frontmatter with ``okf_version`` and then lists each
    sub-directory (excluding ``index.md`` / ``log.md`` themselves) with its
    concept count::

        ---
        okf_version: '0.1'
        ---
        # Knowledge Bundle

        - [dirname/](/dirname/index.md) (N concepts)
    """
    # Frontmatter block — single key, hand-rolled so quoting is predictable.
    frontmatter = f"---\nokf_version: '{okf_version}'\n---\n"

    lines: list[str] = [frontmatter, "# Knowledge Bundle", ""]

    if not bundle_dir.is_dir():
        return "\n".join(lines) + "\n"

    for sub in sorted(bundle_dir.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("."):
            continue
        # Count .md concept files, excluding index.md and log.md.
        md_files = [
            f for f in sub.glob("*.md")
            if f.name not in {"index.md", "log.md"}
        ]
        count = len(md_files)
        lines.append(f"- [{sub.name}/](/{sub.name}/index.md) ({count} concepts)")

    return "\n".join(lines) + "\n"


# ── Change log ─────────────────────────────────────────────────────────


def generate_log(entries: list[LogEntry]) -> str:
    """Generate ``log.md`` content from a list of :class:`LogEntry`.

    Entries are grouped by their ``date`` field (ISO 8601 ``YYYY-MM-DD``)
    with the newest date first.  Within a date, entries are listed in the
    order they appear in ``entries``.
    """
    lines: list[str] = ["# Change Log", ""]

    if not entries:
        return "\n".join(lines) + "\n"

    # Group preserving per-date insertion order.
    by_date: dict[str, list[LogEntry]] = {}
    for entry in entries:
        by_date.setdefault(entry.date, []).append(entry)

    # Newest date first.
    for date in sorted(by_date.keys(), reverse=True):
        lines.append(f"## {date}")
        lines.append("")
        for entry in by_date[date]:
            lines.append(_format_log_line(entry))
        lines.append("")

    # Collapse any trailing blank line into a single trailing newline.
    return "\n".join(lines).rstrip("\n") + "\n"


def append_log_entry(existing_log: str, entry: LogEntry) -> str:
    """Append ``entry`` to ``existing_log`` preserving date grouping.

    If a section for ``entry.date`` already exists, the new entry is inserted
    as the *last* bullet within that section (maintaining chronological order
    within the day).  Otherwise a new ``## YYYY-MM-DD`` section is created and
    placed immediately after the ``# Change Log`` header (i.e. it becomes the
    newest date section).
    """
    header_marker = "# Change Log"

    # Split into header, sections.
    # Each section begins with "## YYYY-MM-DD" and contains everything up to
    # the next "## " line or EOF.
    body = existing_log

    if not body.startswith(header_marker):
        # No existing header — build from scratch.
        return generate_log([entry])

    # Pull off the top-level header and any blank line(s) following it.
    after_header = body[len(header_marker):]
    after_header = after_header.lstrip("\n")

    # Collect existing date sections as (date, raw_section_text) pairs.
    sections: list[tuple[str, str]] = []
    current_date: str | None = None
    current_lines: list[str] = []
    new_line_buffer: list[str] = []

    for raw_line in after_header.splitlines(keepends=True):
        stripped = raw_line.lstrip()
        if stripped.startswith("## "):
            # Save previous section if any.
            if current_date is not None:
                sections.append((current_date, "".join(current_lines)))
            current_date = stripped[3:].strip()
            current_lines = [raw_line]
        elif current_date is None:
            # Lines before the first section — preamble after header.
            new_line_buffer.append(raw_line)
        else:
            current_lines.append(raw_line)

    # Flush the last section.
    if current_date is not None:
        sections.append((current_date, "".join(current_lines)))

    new_bullet = _format_log_line(entry) + "\n"

    # Try to insert into an existing date section.
    for idx, (date_key, section_text) in enumerate(sections):
        if date_key == entry.date:
            # Append the new bullet as the last line of this section.
            sections[idx] = (date_key, section_text.rstrip("\n") + "\n" + new_bullet)
            break
    else:
        # No matching section — create a new one and place it first (newest).
        new_section = f"## {entry.date}\n\n{new_bullet}\n"
        sections.insert(0, (entry.date, new_section))

    # Reassemble.
    out_lines = [header_marker + "\n"]
    if new_line_buffer:
        out_lines.append("".join(new_line_buffer))

    for _date_key, section_text in sections:
        out_lines.append(section_text)

    result = "".join(out_lines)
    # Normalise: single trailing newline.
    return result.rstrip("\n") + "\n"
