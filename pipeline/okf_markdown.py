"""OKF-native markdown utilities.

This module replaces the legacy ``pipeline/markdown.py`` with helpers tuned
for the OKF (Open Knowledge Format) wiki layout: standard markdown links
instead of Obsidian ``[[wikilinks]]``, YAML frontmatter via ``yaml.safe_load``,
and atomic file I/O.

Key conventions
---------------
* Frontmatter is delimited by ``---`` fences and parsed with
  ``yaml.safe_load``.
* Wikilinks (``[[slug]]`` / ``[[slug|alias]]``) are rewritten to standard
  markdown links pointing at ``/<directory>/<slug>.md``.
* Concept identifiers are slug-style path segments (e.g. ``"foo"`` or
  ``"concepts/foo"``); they are *not* required to carry a ``.md`` suffix —
  the helpers append it themselves.
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

__all__ = [
    "parse_frontmatter",
    "build_frontmatter",
    "extract_links",
    "make_absolute_link",
    "make_relative_link",
    "safe_read_file",
    "atomic_write",
    "slugify",
]


# ── Slugify ────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert arbitrary text to a filename-safe slug.

    * lowercase
    * non-alphanumeric characters removed (Unicode letters/digits kept)
    * whitespace and punctuation collapsed to single hyphens
    * leading/trailing hyphens stripped
    """
    # Strip apostrophes and smart quotes first so "don't" -> "dont".
    cleaned = text.replace("'", "").replace("\u2018", "").replace("\u2019", "")
    # Keep Unicode letters, numbers, whitespace and hyphens; drop the rest.
    cleaned = re.sub(r"[^\w\s-]", "", cleaned, flags=re.UNICODE)
    # Whitespace runs become a single hyphen.
    cleaned = re.sub(r"\s+", "-", cleaned)
    # Collapse repeated hyphens.
    cleaned = re.sub(r"-+", "-", cleaned)
    # Trim leading/trailing hyphens and lowercase.
    slug = cleaned.strip("-").lower()
    # Fallback for empty input or input that becomes empty after cleaning.
    if not slug:
        return "untitled"
    return slug


# ── Frontmatter ────────────────────────────────────────────────────────


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from ``raw``.

    Returns ``(meta, body)``. If ``raw`` has no leading ``---``-delimited
    frontmatter block, returns ``({}, raw)`` unchanged. Uses
    ``yaml.safe_load``; on any parse error the frontmatter is treated as
    empty and the original text is returned untouched.
    """
    if not raw.startswith("---\n"):
        return {}, raw

    _prefix, sep, rest = raw.partition("---\n")
    if not sep:
        return {}, raw

    yaml_block, sep2, body = rest.partition("\n---")
    if not sep2:
        # Opening fence with no closing fence — not valid frontmatter.
        return {}, raw

    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        return {}, raw

    if not isinstance(meta, dict):
        meta = {}

    # Drop the newline that immediately follows the closing fence.
    body = body.removeprefix("\n")
    return meta, body


def build_frontmatter(fm_dict: dict) -> str:
    """Serialize ``fm_dict`` to a ``---``-delimited YAML frontmatter block.

    Uses ``yaml.dump`` with ``sort_keys=False``, ``default_flow_style=False``
    and ``allow_unicode=True`` so dict ordering and non-ASCII content survive
    a round trip.
    """
    dumped = yaml.dump(
        fm_dict,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).rstrip()
    return f"---\n{dumped}\n---"


# ── Link extraction ────────────────────────────────────────────────────

# Standard markdown link: [text](url). Negative lookbehind on "!" excludes
# images ![alt](url).
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)]*)\)")

# Legacy wikilink regex moved to pipeline.migrate.


def extract_links(body: str) -> list[tuple[str, str]]:
    """Extract standard markdown ``[text](url)`` links from ``body``.

    Returns a list of ``(text, url)`` tuples in document order. Image links
    (``![alt](url)``) are intentionally excluded.
    """
    return [(m.group(1), m.group(2)) for m in _LINK_RE.finditer(body)]


# ── Wikilink → OKF conversion (moved to pipeline.migrate) ───────────────


# ── Link construction ──────────────────────────────────────────────────


def make_absolute_link(concept_id: str, display_text: str | None = None) -> str:
    """Build an absolute OKF link ``[display](/concept_id.md)``.

    ``display_text`` defaults to ``concept_id``. ``concept_id`` is used as-is
    in the path; a ``.md`` suffix is appended automatically (no doubling if
    the id already ends with ``.md``).
    """
    display = display_text if display_text is not None else concept_id
    target = concept_id if concept_id.endswith(".md") else f"{concept_id}.md"
    return f"[{display}](/{target})"


def make_relative_link(from_id: str, to_id: str,
                        display_text: str | None = None) -> str:
    """Build a relative-path OKF link from ``from_id`` to ``to_id``.

    Both ids are treated as wiki paths (with or without ``.md``). The
    relative path is computed from the *directory* of ``from_id`` to
    ``to_id`` using :func:`os.path.relpath`. ``display_text`` defaults to
    ``to_id``.
    """
    display = display_text if display_text is not None else to_id
    from_dir = os.path.dirname(from_id)
    rel = os.path.relpath(to_id, start=from_dir) if from_dir else to_id
    return f"[{display}]({rel})"


# ── File I/O ───────────────────────────────────────────────────────────


def safe_read_file(path: str | Path) -> str:
    """Read a file as UTF-8, returning ``""`` on any error."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def atomic_write(path: str | Path, content: str) -> None:
    """Atomically write ``content`` to ``path``.

    Writes to a temporary file in the *same directory* then uses
    :func:`os.replace` to swap it into place, guaranteeing readers never see
    a partially-written file. Parent directories are created as needed.
    """
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory tempfile so os.replace stays atomic on the same FS.
    fd, tmp_name = tempfile.mkstemp(
        dir=str(fp.parent), prefix=fp.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(content)
        os.replace(tmp_name, fp)
    except BaseException:
        # Clean up the orphaned tempfile on failure.
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


