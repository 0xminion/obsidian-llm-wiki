"""Extraction orchestrator — routes URLs to extractors and writes source files.

Stage 1 deterministic extraction: full content, never truncated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pipeline.config import Config
from pipeline.extractors.web import extract_web
from pipeline.hasher import hash_content
from pipeline.markdown import atomic_write, parse_frontmatter, safe_read_file, slugify
from pipeline.models import IngestedSource

if TYPE_CHECKING:
    pass


# ── Extractor registry ─────────────────────────────────────────────────
# Maps URL scheme/pattern → extractor function.
# Extensible: add youtube/podcast extractors here later.

_EXTRACTORS: dict[str, callable] = {}


def _init_registry() -> None:
    """Populate the extractor registry (lazy to avoid circular imports)."""
    if _EXTRACTORS:
        return
    _EXTRACTORS["http"] = extract_web
    _EXTRACTORS["https"] = extract_web


# ── Public API ─────────────────────────────────────────────────────────


def run_extraction(urls: list[str], config: Config) -> dict[str, IngestedSource]:
    """Extract full content from a list of URLs.

    For each URL:
      1. Dedup check: skip if an identical source already exists.
      2. Route to the appropriate extractor.
      3. Write source as a .md file in config.sources_dir.

    Args:
        urls: List of URLs to extract.
        config: Pipeline configuration.

    Returns:
        Dict mapping URL → IngestedSource for successfully extracted URLs.
        URLs that were skipped (dedup match) are NOT included.
    """
    _init_registry()

    config.sources_dir.mkdir(parents=True, exist_ok=True)

    # Build index of existing hashes for dedup
    existing_hashes = _build_existing_hash_index(config)

    results: dict[str, IngestedSource] = {}
    seen_hashes: set[str] = set(existing_hashes)

    for url in urls:
        try:
            source = _extract_one(url, config)
        except Exception:
            # Surface errors but continue with remaining URLs
            import sys
            print(f"[extract] FAILED: {url}", file=sys.stderr)
            continue

        content_hash = hash_content(source.content)

        # Dedup check against both persisted and in-memory (this batch)
        if content_hash in seen_hashes:
            print(f"[extract] SKIPPED (duplicate): {url}")
            continue

        seen_hashes.add(content_hash)
        _write_source_file(source, url, config)
        results[url] = source
        print(f"[extract] OK: {source.title[:60]} ({len(source.content)} chars)")

    return results


def _extract_one(url: str, config: Config) -> IngestedSource:
    """Route a URL to the correct extractor."""
    for prefix, extractor in _EXTRACTORS.items():
        if url.startswith(prefix):
            return extractor(url)
    raise ValueError(f"No extractor available for URL: {url}")


# ── File output ────────────────────────────────────────────────────────


def _write_source_file(source: IngestedSource, url: str, config: Config) -> None:
    """Write an IngestedSource to a .md file in sources_dir.

    File format:
        ---
        title: Page Title
        url: https://...
        extracted_at: 2026-05-12T07:00:00Z
        ---

        <full content>
    """
    extracted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = slugify(source.title) if source.title else slugify(url)

    # Ensure unique filename
    base = slug
    counter = 1
    while True:
        filename = f"{slug}.md"
        filepath = config.sources_dir / filename
        if not filepath.exists():
            break
        slug = f"{base}-{counter}"
        counter += 1

    frontmatter = f"---\ntitle: {source.title}\nurl: {url}\nextracted_at: {extracted_at}\n---\n\n"
    full_md = frontmatter + source.content

    atomic_write(filepath, full_md)


# ── Dedup helpers ──────────────────────────────────────────────────────


def _build_existing_hash_index(config: Config) -> set[str]:
    """Build a set of content hashes from existing source .md files.

    Strips frontmatter to hash only the body (content), matching how
    extraction produces content hashes.
    """
    hashes: set[str] = set()
    if not config.sources_dir.exists():
        return hashes

    for f in config.sources_dir.iterdir():
        if f.suffix != ".md" or not f.is_file():
            continue
        raw = safe_read_file(f)
        if not raw:
            continue
        _meta, body = parse_frontmatter(raw)
        if body.strip():
            hashes.add(hash_content(body.strip()))

    return hashes


# ── CLI entry point (for testing) ──────────────────────────────────────

if __name__ == "__main__":
    import sys

    from pipeline.config import load_config

    if len(sys.argv) < 2:
        print(f"Usage: python {__file__} <url> [url...]", file=sys.stderr)
        sys.exit(1)

    config = load_config()
    urls = sys.argv[1:]
    results = run_extraction(urls, config)

    print(f"\nExtracted {len(results)} of {len(urls)} URLs")
    for url, source in results.items():
        print(f"  {source.title} ← {url}")
