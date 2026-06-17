"""OKF-native link resolver.

Rewrites markdown links to absolute bundle-relative paths.
Replaces the legacy wikilink-based ``pipeline/resolver.py`` with a resolver
that works with OKF standard markdown links: ``[text](/concepts/foo.md)``.

Link resolution rules
----------------------
* ``http://``, ``https://``, ``mailto:``, ``#``-anchor links → leave as-is
* Links starting with ``/`` → already absolute bundle-relative → leave as-is
* Bare slug (no ``/``, no ``.md``) → look up in registry, rewrite to
  ``[text](/concept_id.md)``
* Relative ``.md`` paths → resolve to absolute bundle-relative path
* Unknown slugs → leave as-is (OKF spec: tolerate broken links)
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.okf_markdown import (
    atomic_write,
    build_frontmatter,
    parse_frontmatter,
    safe_read_file,
)

__all__ = [
    "build_concept_registry",
    "resolve_links",
]

# Files to skip when scanning the bundle.
_SKIP_FILES: frozenset[str] = frozenset({"index.md", "log.md", "viz.html"})

# Standard markdown link regex (excludes images ![alt](url)).
_LINK_RE = re.compile(r"(?<!\!)\[([^\]]*)\]\(([^)]*)\)")


# ── Registry ────────────────────────────────────────────────────────────


def build_concept_registry(bundle_dir: Path) -> dict[str, str]:
    """Build a registry mapping slugs and concept ids to concept ids.

    Scans all ``.md`` files in ``bundle_dir`` (recursively), skipping
    ``index.md``, ``log.md``, and ``viz.html``.  For each file, the
    **concept id** is the path relative to ``bundle_dir`` without the
    ``.md`` suffix (e.g. ``concepts/foo``).

    Returns a dict with two kinds of entries:

    * ``slug -> concept_id`` — the filename stem maps to the concept id.
      When two files share a stem, the first one encountered wins.
    * ``concept_id -> concept_id`` — the full concept id maps to itself,
      so callers can do a single lookup regardless of whether they have
      a bare slug or a full path.
    """
    bundle = Path(bundle_dir)
    registry: dict[str, str] = {}
    for md_path in sorted(bundle.rglob("*.md")):
        if md_path.name in _SKIP_FILES:
            continue
        rel = md_path.relative_to(bundle)
        concept_id = str(rel.with_suffix(""))  # strip .md
        # Full concept_id -> concept_id (idempotent lookup).
        registry[concept_id] = concept_id
        # slug (filename stem) -> concept_id; first wins on collision.
        slug = md_path.stem
        if slug not in registry:
            registry[slug] = concept_id
    return registry


# ── Link resolution ─────────────────────────────────────────────────────


def _resolve_one(
    url: str,
    registry: dict[str, str],
    bundle: Path,
    current_file: Path,
) -> str:
    """Resolve a single link URL to an absolute bundle-relative path.

    Returns the original URL unchanged if it should be left as-is.
    """
    # External / anchor / already-absolute → leave as-is
    if (
        url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("mailto:")
        or url.startswith("#")
        or url.startswith("/")
    ):
        return url

    # Relative .md path → resolve relative to the current file's directory,
    # then express as an absolute bundle-relative path.
    if url.endswith(".md"):
        cur_dir = current_file.parent
        resolved = (cur_dir / url).resolve()
        try:
            rel_to_bundle = resolved.relative_to(bundle.resolve())
        except ValueError:
            # Path resolves outside the bundle — leave as-is.
            return url
        return f"/{rel_to_bundle.as_posix()}"

    # Bare slug or concept_id without .md → registry lookup.
    concept_id = registry.get(url)
    if concept_id is not None:
        return f"/{concept_id}.md"

    # Unknown slug → leave as-is (OKF spec: tolerate broken links).
    return url


def _resolve_all_links(
    body: str,
    registry: dict[str, str],
    bundle_dir: Path,
    current_file: Path,
) -> str:
    """Apply link resolution rules to a body string and return the result.

    See module docstring for the rules.  ``current_file`` is the absolute
    path of the file the body came from, used to resolve relative ``.md``
    paths.
    """
    bundle = Path(bundle_dir)

    def _replace(match: re.Match[str]) -> str:
        text = match.group(1)
        url = match.group(2).strip()
        new_url = _resolve_one(url, registry, bundle, current_file)
        if new_url == url:
            return match.group(0)
        return f"[{text}]({new_url})"

    return _LINK_RE.sub(_replace, body)


# ── Public entry point ──────────────────────────────────────────────────


def resolve_links(
    bundle_dir: Path | str,
    all_slugs: list[str] | None = None,
    new_slugs: list[str] | None = None,
) -> int:
    """Rewrite all markdown links in the bundle to absolute paths.

    Scans every ``.md`` file in ``bundle_dir`` (recursively), skipping
    ``index.md``, ``log.md``, and ``viz.html``.  For each file, parses
    frontmatter, resolves links in the body via
    :func:`_resolve_all_links`, and atomically writes back the file if
    it changed.

    ``all_slugs`` and ``new_slugs`` are accepted for API compatibility with
    the legacy resolver but are not currently used — the registry is
    built from the full bundle.

    Returns the count of files that were modified.
    """
    bundle = Path(bundle_dir)
    if not bundle.is_dir():
        return 0

    registry = build_concept_registry(bundle)

    modified_count = 0
    for md_path in sorted(bundle.rglob("*.md")):
        if md_path.name in _SKIP_FILES:
            continue
        raw = safe_read_file(md_path)
        if not raw:
            continue

        has_frontmatter = raw.startswith("---\n")
        meta, body = parse_frontmatter(raw)
        new_body = _resolve_all_links(body, registry, bundle, md_path)
        if new_body == body:
            continue

        if has_frontmatter:
            fm = build_frontmatter(meta)
            new_raw = f"{fm}\n{new_body}"
        else:
            new_raw = new_body

        atomic_write(md_path, new_raw)
        modified_count += 1

    return modified_count
