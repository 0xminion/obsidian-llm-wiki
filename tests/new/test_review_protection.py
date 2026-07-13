"""Tests for reviewed-page safety helpers."""

from __future__ import annotations

from obsidian_llm_wiki.core.review import is_reviewed_page, protected_body


def test_is_reviewed_page_requires_boolean_true_frontmatter_value() -> None:
    assert is_reviewed_page("---\nreviewed: true\n---\n# Curated\n") is True
    assert is_reviewed_page("---\nreviewed: false\n---\n# Draft\n") is False
    assert is_reviewed_page("---\nreviewed: 'true'\n---\n# Ambiguous\n") is False
    assert is_reviewed_page("# No frontmatter\n") is False


def test_protected_body_keeps_human_body_only_when_existing_page_is_reviewed() -> None:
    existing = "---\nreviewed: true\ntitle: Old\n---\n# Curated\n\nHuman edit.\n"
    generated = "---\ntitle: New\n---\n# Generated\n\nAutomated rewrite.\n"

    assert protected_body(existing, generated) == "# Curated\n\nHuman edit.\n"


def test_protected_body_allows_unreviewed_page_replacement() -> None:
    existing = "---\nreviewed: false\n---\n# Draft\n"
    generated = "---\ntitle: New\n---\n# Generated\n\nAutomated rewrite.\n"

    assert protected_body(existing, generated) == "# Generated\n\nAutomated rewrite.\n"
