"""Clippings quality gate — determines whether a clipping needs Stage 1 extraction.

Scans 02-Clippings/ for markdown files and evaluates each:
- Passes gate (body > threshold AND has title) → skip Stage 1 extraction.
- Fails gate (too short or missing title) → needs Stage 1 extraction.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.config import Config
from pipeline.okf_markdown import parse_frontmatter, safe_read_file
from pipeline.okf_models import IngestedSource

# ── Public API ─────────────────────────────────────────────────────────


def check_clipping(path: Path, config: Config) -> tuple[bool, IngestedSource | None]:
    """Evaluate a single clipping file against the quality gate.

    Reads a markdown clipping, parses frontmatter, and checks:
      1. Has a title (from frontmatter: title, source_url title, or first H1).
      2. Body is longer than config.clipping_min_body_chars.

    Args:
        path: Path to a .md file in 02-Clippings/.
        config: Pipeline configuration (threshold from clipping_min_body_chars).

    Returns:
        (True, IngestedSource) if the clipping passes the gate.
        (False, None) if it needs Stage 1 extraction.
    """
    raw = safe_read_file(path)
    if not raw.strip():
        return False, None

    meta, body = parse_frontmatter(raw)

    # ── Determine title ─────────────────────────────────────────────
    title = _extract_title(meta, body)

    if not title:
        return False, None

    # ── Check body length ───────────────────────────────────────────
    body_stripped = body.strip()
    if len(body_stripped) < config.clipping_min_body_chars:
        return False, None

    return True, IngestedSource(title=title, content=body_stripped)


def collect_clippings(config: Config) -> list[tuple[Path, IngestedSource]]:
    """Scan 02-Clippings/ and run the quality gate on every .md file.

    Returns all clippings that pass the gate, ready for direct use
    (no Stage 1 extraction needed).

    Args:
        config: Pipeline configuration.

    Returns:
        List of (file_path, IngestedSource) tuples for clippings that pass.
        Files that fail the gate are silently skipped.
    """
    clippings_dir = config.clippings_dir
    if not clippings_dir.exists():
        return []

    passed: list[tuple[Path, IngestedSource]] = []

    for f in sorted(clippings_dir.iterdir()):
        if f.suffix != ".md" or not f.is_file():
            continue

        ok, source = check_clipping(f, config)
        if ok and source is not None:
            passed.append((f, source))

    return passed


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_title(meta: dict, body: str) -> str:
    """Extract a title from frontmatter or body.

    Priority:
      1. meta["title"]
      2. meta["source_url"] / meta["url"] — derive from path/filename
      3. meta["source"] — use as title directly
      4. First H1 in body
      5. First non-empty line of body (truncated)
    """
    # Direct title
    title = (meta.get("title") or "").strip()
    if title:
        return title

    # source field
    source = (meta.get("source") or "").strip()
    if source:
        return source

    # source_url / url — derive title from URL path
    source_url = (meta.get("source_url") or meta.get("url") or "").strip()
    if source_url:
        derived = _title_from_url(source_url)
        if derived:
            return derived

    # First H1 in body
    import re
    h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    # Fallback: first non-empty line (truncated)
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Use first 80 chars as title
            return stripped[:80].rstrip()

    return ""


def _title_from_url(url: str) -> str:
    """Derive a human-readable title from a URL path.

    Examples:
        https://example.com/blog/my-article → My Article
        https://example.com/docs/rust/ownership → Ownership
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        return ""

    # Take the last meaningful segment
    segments = [s for s in path.split("/") if s and not s.endswith((".html", ".htm", ".php", ".asp"))]
    if not segments:
        # Maybe it's a bare filename with extension
        segments = path.split("/")

    if segments:
        # Last segment, strip extension, replace hyphens/underscores with spaces
        last = segments[-1]
        # Strip common extensions
        for ext in (".html", ".htm", ".php", ".asp", ".aspx", ".md"):
            if last.endswith(ext):
                last = last[: -len(ext)]
                break
        # Replace hyphens and underscores with spaces, title-case
        title = last.replace("-", " ").replace("_", " ")
        return title.strip()

    return ""


# ── CLI entry point (for testing) ──────────────────────────────────────

if __name__ == "__main__":
    from pipeline.config import load_config

    config = load_config()
    passed = collect_clippings(config)

    print(f"Scanned: {config.clippings_dir}")
    print(f"Passed:  {len(passed)}")
    for path, source in passed:
        print(f"  [{len(source.content)} chars] {source.title}  ← {path.name}")
