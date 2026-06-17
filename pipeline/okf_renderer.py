"""OKF-native page renderers.

Functions that assemble full OKF concept documents (YAML frontmatter +
markdown body) for each of the five OKF concept types: Source, Entry,
Concept, Map of Content, and Reference.

Each render function returns the complete document text — frontmatter
fence, a blank line, then the markdown body — ready for atomic write to
disk.

Links use the standard ``[text](/path.md)`` OKF form via
:func:`pipeline.okf_markdown.make_absolute_link`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pipeline.okf_markdown import build_frontmatter, make_absolute_link
from pipeline.okf_models import OKFConceptType

__all__ = [
    "render_source_page",
    "render_entry_page",
    "render_concept_page",
    "render_moc_page",
    "render_reference_page",
]


def _default_timestamp() -> str:
    """Return a UTC ISO-style timestamp string for default use."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Source ─────────────────────────────────────────────────────────────


def render_source_page(
    title: str,
    url: str,
    content: str,
    timestamp: str | None = None,
) -> str:
    """Render a Source-type OKF page.

    Source pages store the raw, unmodified content captured from an
    external resource (web page, file, PDF, …) for provenance.

    Frontmatter fields:
      * type = ``"Source"``
      * title = *title*
      * description = ``f"Original content from {url}"``
      * resource = *url*
      * tags = ``["source"]``
      * timestamp = *timestamp* (UTC now if not supplied)

    Body:
      ``# {title}``

      {content}
    """
    ts = timestamp if timestamp is not None else _default_timestamp()
    fm = {
        "type": OKFConceptType.SOURCE.value,
        "title": title,
        "description": f"Original content from {url}",
        "resource": url,
        "tags": ["source"],
        "timestamp": ts,
    }
    body = f"# {title}\n\n{content}"
    return f"{build_frontmatter(fm)}\n\n{body}"


# ── Entry ──────────────────────────────────────────────────────────────


def render_entry_page(
    title: str,
    summary: str,
    source_concept_id: str,
    body: str,
    tags: list[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Render an Entry-type OKF page.

    An entry page is the primary synthesis document for a single source.

    Frontmatter fields:
      * type = ``"Entry"``
      * title = *title*
      * description = *summary* truncated to 200 chars
      * tags = *tags* (default ``[]``)
      * timestamp = *timestamp* (UTC now if not supplied)

    Body:
      ``# {title}``

      {summary}

      ``## Source``

      ``- {make_absolute_link(source_concept_id, 'Source')}``

      {body}
    """
    ts = timestamp if timestamp is not None else _default_timestamp()
    if tags is None:
        tags = []
    fm = {
        "type": OKFConceptType.ENTRY.value,
        "title": title,
        "description": summary[:200],
        "tags": list(tags),
        "timestamp": ts,
    }
    src_link = make_absolute_link(source_concept_id, "Source")
    body_md = (
        f"# {title}\n\n"
        f"{summary}\n\n"
        f"## Source\n\n"
        f"- {src_link}\n\n"
        f"{body}"
    )
    return f"{build_frontmatter(fm)}\n\n{body_md}"


# ── Concept ────────────────────────────────────────────────────────────


def render_concept_page(
    title: str,
    summary: str,
    body: str,
    tags: list[str] | None = None,
    source_ids: list[str] | None = None,
    citations: list[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Render a Concept-type OKF page.

    A concept page is an evergreen, atomic note on a single idea.

    Frontmatter fields:
      * type = ``"Concept"``
      * title = *title*
      * description = *summary* truncated to 200 chars
      * tags = *tags* (default ``[]``)
      * timestamp = *timestamp* (UTC now if not supplied)

    Body:
      ``# {title}``

      {summary}

      {body}

      If *source_ids* is non-empty:
        ``## Sources`` section listing each id as an absolute link.

      If *citations* is non-empty:
        ``# Citations`` section with a numbered list.
    """
    ts = timestamp if timestamp is not None else _default_timestamp()
    if tags is None:
        tags = []
    fm = {
        "type": OKFConceptType.CONCEPT.value,
        "title": title,
        "description": summary[:200],
        "tags": list(tags),
        "timestamp": ts,
    }
    parts: list[str] = [
        f"# {title}",
        "",
        summary,
        "",
        body,
    ]

    if source_ids:
        parts.append("")
        parts.append("## Sources")
        parts.append("")
        for sid in source_ids:
            parts.append(f"- {make_absolute_link(sid)}")

    if citations:
        parts.append("")
        parts.append("# Citations")
        parts.append("")
        for idx, cite in enumerate(citations, 1):
            parts.append(f"{idx}. {cite}")

    body_md = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n\n{body_md}"


# ── Map of Content ─────────────────────────────────────────────────────


def render_moc_page(
    title: str,
    summary: str,
    concept_links: list[tuple[str, str]],
    tags: list[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Render a Map of Content (MoC) OKF page.

    A MoC is a curated index that groups related concepts together.

    Frontmatter fields:
      * type = ``"Map of Content"``
      * title = *title*
      * description = *summary* truncated to 200 chars
      * tags = *tags* (default ``[]``)
      * timestamp = *timestamp* (UTC now if not supplied)

    Body:
      ``# {title}``

      {summary}

      ``## Concepts`` section listing each *(concept_id, display)* tuple
      as an absolute link.
    """
    ts = timestamp if timestamp is not None else _default_timestamp()
    if tags is None:
        tags = []
    fm = {
        "type": OKFConceptType.MOC.value,
        "title": title,
        "description": summary[:200],
        "tags": list(tags),
        "timestamp": ts,
    }
    parts: list[str] = [
        f"# {title}",
        "",
        summary,
        "",
        "## Concepts",
        "",
    ]
    for concept_id, display in concept_links:
        parts.append(f"- {make_absolute_link(concept_id, display)}")

    body_md = "\n".join(parts)
    return f"{build_frontmatter(fm)}\n\n{body_md}"


# ── Reference ──────────────────────────────────────────────────────────


def render_reference_page(
    title: str,
    url: str,
    summary: str,
    body: str,
    tags: list[str] | None = None,
    timestamp: str | None = None,
) -> str:
    """Render a Reference-type OKF page.

    A reference page is a curated, citation-backed note on an external
    resource (book, paper, tool, specification, …).

    Frontmatter fields:
      * type = ``"Reference"``
      * title = *title*
      * description = *summary* truncated to 200 chars
      * resource = *url*
      * tags = *tags* (default ``[]``)
      * timestamp = *timestamp* (UTC now if not supplied)

    Body:
      ``# {title}``

      {summary}

      {body}

      ``# Citations`` with *url* as a numbered citation.
    """
    ts = timestamp if timestamp is not None else _default_timestamp()
    if tags is None:
        tags = []
    fm = {
        "type": OKFConceptType.REFERENCE.value,
        "title": title,
        "description": summary[:200],
        "resource": url,
        "tags": list(tags),
        "timestamp": ts,
    }
    body_md = (
        f"# {title}\n\n"
        f"{summary}\n\n"
        f"{body}\n\n"
        f"# Citations\n\n"
        f"1. {url}"
    )
    return f"{build_frontmatter(fm)}\n\n{body_md}"
