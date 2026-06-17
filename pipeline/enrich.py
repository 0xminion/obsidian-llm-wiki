"""OKF wiki enrichment agent — web-crawl pass.

The enrichment agent fetches a set of seed URLs, asks the LLM (via the
enrich prompts) to decide whether to *enrich* an existing concept, *mint*
a new reference page, or *skip* the page, writes any resulting
``references/`` docs, and follows outbound links within the allowed host
for further crawling.

The module is fully async so it can batch network I/O.  The public entry
point is :func:`run_enrichment`.

Key dataclasses:

* :class:`EnrichmentResult` — aggregate counts from a run.
* :class:`EnrichOptions` — crawl constraints (seed URLs, allowed host,
  page cap, no-web flag).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from pipeline.config import Config
from pipeline.okf_markdown import (
    atomic_write,
    build_frontmatter,
    parse_frontmatter,
    safe_read_file,
    slugify,
)
from pipeline.prompts_enrich import (
    EnrichDecision,
    build_enrich_prompt,
    parse_enrich_response,
)

__all__ = [
    "EnrichmentResult",
    "EnrichOptions",
    "run_enrichment",
]

logger = logging.getLogger("llmwiki.enrich")


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class EnrichmentResult:
    """Aggregate outcome of a single enrichment run."""

    pages_fetched: int = 0
    references_created: int = 0
    concepts_enriched: int = 0
    pages_skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class EnrichOptions:
    """Crawl constraints for the enrichment run.

    Attributes:
        seed_urls: Initial URLs to fetch.
        allowed_host: Hostname that outbound links must match to be
            followed.  An empty string means "follow all".
        max_pages: Hard cap on total pages fetched.
        no_web: When True, skip all web fetching (useful for dry-run
            tests where the LLM is mocked).
    """

    seed_urls: list[str] = field(default_factory=list)
    allowed_host: str = ""
    max_pages: int = 20
    no_web: bool = False


# ── Public entry point ─────────────────────────────────────────────────────


async def run_enrichment(
    bundle_dir: str | Path,
    config: Config,
    options: EnrichOptions,
) -> EnrichmentResult:
    """Run the enrichment agent over ``bundle_dir``.

    Args:
        bundle_dir: OKF bundle root (contains ``concepts/``, ``references/``).
        config: Pipeline configuration (used for LLM calls).
        options: Crawl constraints.

    Returns:
        :class:`EnrichmentResult` with aggregate counts.
    """
    from pipeline.extractors.web import extract_web
    from pipeline.llm_client import call_llm

    bd = Path(bundle_dir)
    bd.mkdir(parents=True, exist_ok=True)
    refs_dir = bd / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    result = EnrichmentResult()
    visited: set[str] = set()
    queue: list[str] = list(options.seed_urls)

    registry = _build_concept_registry(bd)
    existing_ids = list(registry.values())

    while queue and result.pages_fetched < options.max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        # ── Fetch page content ─────────────────────────────────────
        if options.no_web:
            logger.info("no_web=True — skipping fetch of %s", url)
            result.pages_skipped += 1
            continue

        try:
            source = extract_web(url)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            result.errors.append(f"fetch {url}: {exc}")
            continue

        result.pages_fetched += 1
        title = source.title or url
        content = source.content or ""

        # ── Ask LLM what to do ─────────────────────────────────────
        prompt = build_enrich_prompt(url, title, content, registry)
        messages = [
            {"role": "user", "content": f"Enrichment decision for {url}"},
        ]
        try:
            response = await call_llm(prompt, messages, config)
        except Exception as exc:
            logger.warning("LLM call failed for %s: %s", url, exc)
            result.errors.append(f"llm {url}: {exc}")
            continue

        decisions = parse_enrich_response(response)
        if not decisions:
            logger.info("No decisions parsed for %s — skipping", url)
            result.pages_skipped += 1
            continue

        # ── Apply decisions ────────────────────────────────────────
        for dec in decisions:
            if dec.action == "skip":
                result.pages_skipped += 1
                continue

            if dec.action == "enrich":
                concept_id = dec.concept_id
                # Look up by registry (slug -> concept_id) or directly.
                concept_path = _find_concept_path(bd, concept_id, registry)
                if not concept_path:
                    logger.warning(
                        "enrich: concept '%s' not found — minting reference instead",
                        concept_id,
                    )
                    _mint_reference(
                        refs_dir, dec, url, result
                    )
                    continue
                _enrich_existing_concept(concept_path, url, dec.addition)
                result.concepts_enriched += 1

            elif dec.action == "mint":
                _mint_reference(refs_dir, dec, url, result)

            # Follow additional links the LLM suggested.
            if dec.follow_links:
                for link in dec.follow_links:
                    abs_link = urljoin(url, link)
                    if (
                        abs_link not in visited
                        and (
                            not options.allowed_host
                            or urlparse(abs_link).hostname == options.allowed_host
                        )
                    ):
                        queue.append(abs_link)

        # Also auto-extract outbound links from the page for crawling.
        for link in _extract_outbound_links(content, url, options.allowed_host):
            if link not in visited:
                queue.append(link)

    _append_enrichment_log(bd, result)
    return result


# ── Helper: concept registry ──────────────────────────────────────────────


def _build_concept_registry(bundle_dir: str | Path) -> dict[str, str]:
    """Build a ``slug -> concept_id`` map from all concept .md files.

    The slug is derived from the concept's filename stem (or a ``slug``
    frontmatter key if present).  The concept_id is the path relative to
    the bundle root without ``.md``.
    """
    bd = Path(bundle_dir)
    registry: dict[str, str] = {}
    if not bd.is_dir():
        return registry

    for md_path in sorted(bd.rglob("*.md")):
        if md_path.name.lower() in {"index.md", "log.md", "viz.html"}:
            continue
        if md_path.name.startswith("."):
            continue

        rel = md_path.relative_to(bd)
        concept_id = str(rel.with_suffix("")).replace("\\", "/")

        raw = safe_read_file(md_path)
        meta, _body = parse_frontmatter(raw)
        slug = meta.get("slug") or slugify(md_path.stem)

        registry[slug] = concept_id
        # Also map the bare stem for convenience.
        registry[md_path.stem] = concept_id
        # And the concept_id itself.
        registry[concept_id] = concept_id

    return registry


def _find_concept_path(
    bundle_dir: Path, concept_id: str, registry: dict[str, str]
) -> Path | None:
    """Resolve a concept_id (or slug) to a filesystem path."""
    if not concept_id:
        return None
    # Direct registry lookup.
    resolved = registry.get(concept_id)
    if resolved:
        candidate = bundle_dir / f"{resolved}.md"
        if candidate.is_file():
            return candidate

    # Try as a direct path.
    candidate = bundle_dir / f"{concept_id}.md"
    if candidate.is_file():
        return candidate

    # Try slugify lookup.
    resolved = registry.get(slugify(concept_id))
    if resolved:
        candidate = bundle_dir / f"{resolved}.md"
        if candidate.is_file():
            return candidate

    return None


# ── Helper: enrich existing concept ────────────────────────────────────────


def _enrich_existing_concept(
    concept_path: Path, source_url: str, addition: str
) -> None:
    """Append a citation to an existing concept's Citations section.

    If the concept has no ``## Citations`` section, one is created at the
    end of the file.  The citation is formatted as a markdown list item
    with the source URL and timestamp.
    """
    raw = safe_read_file(concept_path)
    if not raw.strip():
        return

    meta, body = parse_frontmatter(raw)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    citation = f"- [{timestamp}] {source_url}"
    if addition:
        citation += f"\n  {addition.strip()}"
    citation += "\n"

    section_header = "## Citations"
    if section_header in body:
        # Insert after the header line.
        body = body.replace(
            section_header,
            f"{section_header}\n{citation}",
            1,
        )
    else:
        # Append a new section.
        sep = "\n\n" if body and not body.endswith("\n\n") else (
            "\n" if body and not body.endswith("\n") else ""
        )
        body = f"{body}{sep}{section_header}\n{citation}"

    # Rebuild the file with frontmatter + updated body.
    if meta:
        new_content = f"{build_frontmatter(meta)}\n\n{body}"
    else:
        new_content = body

    atomic_write(concept_path, new_content)


# ── Helper: mint a reference page ──────────────────────────────────────────


def _mint_reference(
    refs_dir: Path, dec: EnrichDecision, source_url: str, result: EnrichmentResult
) -> None:
    """Write a new reference markdown page based on an EnrichDecision."""
    slug = dec.concept_id or slugify(dec.title or source_url)
    if not slug:
        slug = "reference"
    ref_path = refs_dir / f"{slug}.md"

    meta = {
        "type": "Reference",
        "title": dec.title or slug,
        "description": dec.summary or "",
        "tags": dec.tags,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d"),
        "resource": source_url,
    }

    body = dec.body or dec.summary or ""
    if not body.strip():
        body = f"Reference page for [{dec.title or source_url}]({source_url})."

    content = f"{build_frontmatter(meta)}\n\n{body}\n"
    atomic_write(ref_path, content)
    result.references_created += 1


# ── Helper: extract outbound links ─────────────────────────────────────────


_HREF_RE = re.compile(
    r"""(?:href\s*=\s*["']([^"']+)["'])|(?:\[([^\]]*)\]\(([^)]+)\))""",
    re.IGNORECASE,
)


def _extract_outbound_links(
    content: str, base_url: str, allowed_host: str
) -> list[str]:
    """Extract crawlable outbound links from page content.

    Handles both HTML ``href="…"`` attributes and markdown
    ``[text](url)`` links.  Only same-host (when ``allowed_host`` is set)
    HTTP(S) URLs are returned.  Fragment-only links (``#…``) and
    mailto/tel links are excluded.

    Returns a de-duplicated list in document order.
    """
    if not content:
        return []

    base = urlparse(base_url)
    base_scheme = base.scheme or "http"
    links: list[str] = []
    seen: set[str] = set()

    for match in _HREF_RE.finditer(content):
        # Group 1 = HTML href, group 2 = markdown text, group 3 = markdown url.
        href = match.group(1)
        if href is None:
            href = match.group(3)
        if not href:
            continue

        href = href.strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        # Make absolute.
        abs_url = urljoin(base_url, href)

        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if allowed_host and parsed.hostname != allowed_host:
            continue
        # Drop fragment to avoid revisiting the same page.
        clean = abs_url.split("#")[0]
        if clean in seen:
            continue
        # Skip the base URL itself.
        if clean.rstrip("/") == base_url.rstrip("/"):
            continue
        seen.add(clean)
        links.append(clean)

    return links


# ── Helper: append enrichment log ──────────────────────────────────────────


def _append_enrichment_log(bundle_dir: Path, result: EnrichmentResult) -> None:
    """Append an enrichment summary entry to ``log.md``."""
    log_path = bundle_dir / "log.md"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")

    entry_lines = [
        f"## {timestamp} — Enrichment pass",
        "",
        f"- pages_fetched: {result.pages_fetched}",
        f"- references_created: {result.references_created}",
        f"- concepts_enriched: {result.concepts_enriched}",
        f"- pages_skipped: {result.pages_skipped}",
    ]
    if result.errors:
        entry_lines.append(f"- errors: {len(result.errors)}")
        for err in result.errors:
            entry_lines.append(f"  - {err}")
    entry_lines.append("")

    entry = "\n".join(entry_lines)

    existing = safe_read_file(log_path)
    if existing.strip():
        new_content = existing.rstrip() + "\n\n" + entry
    else:
        new_content = "# Enrichment Log\n\n" + entry

    atomic_write(log_path, new_content)
