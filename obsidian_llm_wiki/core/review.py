"""Small, deterministic helpers for preserving human-reviewed pages.

This module deliberately does not write pages.  Callers can use
:func:`protected_body` while constructing a replacement page so a human-marked
body is retained while separately applying explicit metadata updates.
"""

from __future__ import annotations

from obsidian_llm_wiki.render.obsidian import parse_frontmatter

__all__ = ["is_reviewed_page", "protected_body"]


def is_reviewed_page(page_content: str) -> bool:
    """Return whether a page explicitly sets the YAML value ``reviewed: true``.

    Only a YAML boolean is accepted; text such as ``reviewed: 'true'`` does not
    accidentally enable protection.
    """
    metadata, _ = parse_frontmatter(page_content)
    return metadata.get("reviewed") is True


def protected_body(existing_content: str, replacement_content: str) -> str:
    """Choose the body that an automated rewrite is allowed to use.

    A reviewed existing page retains its curated body.  An unreviewed page uses
    the replacement body.  Frontmatter is intentionally excluded so callers
    may make their own explicit, auditable metadata/provenance updates.
    """
    _, existing_body = parse_frontmatter(existing_content)
    _, replacement_body = parse_frontmatter(replacement_content)
    return existing_body if is_reviewed_page(existing_content) else replacement_body
