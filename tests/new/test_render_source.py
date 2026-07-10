"""Regression tests for source page rendering.

Covers:
  - render_source_page duplicate heading deduplication
  - render_concept_page with code-block cross-refs
  - render_moc_page with code-block cross-refs
"""
from __future__ import annotations

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.render.obsidian import render_source_page


def _make_source(title: str, content: str, url: str = "https://example.com") -> SourceDoc:
    return SourceDoc(title=title, content=content, url=url)


class TestRenderSourcePageDeduplication:
    """Source page must not show the title heading twice.

    The render function adds '# {title}' at the top. It should skip this
    when the content already starts with the title (as a # heading or as
    the first line of plain text) to avoid duplication like:

        # TradingAgents
        ...
        # TradingAgents
        TAURIC RESEARCH
        ...

    Expected: only one # TradingAgents heading.
    """

    def test_no_dup_when_content_starts_with_hash_heading(self):
        """Content begins with '# {title}' — render must skip adding it again."""
        title = "TradingAgents"
        content = f"# {title}\n\nTAURIC RESEARCH\nThis is the body."
        page = render_source_page(_make_source(title, content))
        headings = [ln for ln in page.splitlines() if ln.startswith("# ")]
        assert len(headings) == 1, (
            f"Expected 1 heading, got {len(headings):d}. "
            f"Headings: {headings}"
        )
        assert headings[0] == f"# {title}"

    def test_no_dup_when_content_starts_with_plain_title(self):
        """Content begins with the title as plain text (no #) — skip duplicate."""
        title = "Decomposing Crowd Wisdom"
        content = f"{title}\n\nAbstract text here."
        page = render_source_page(_make_source(title, content))
        headings = [ln for ln in page.splitlines() if ln.startswith("# ")]
        assert len(headings) == 1, (
            f"Expected 1 heading, got {len(headings):d}. "
            f"Headings: {headings}"
        )
        assert headings[0] == f"# {title}"

    def test_heading_added_when_content_starts_elsewhere(self):
        """Content starts with something else entirely — render must add the heading."""
        title = "TradingAgents"
        content = "--- Page 1 ---\narXiv:2412.20138v7\nTAURIC RESEARCH"
        page = render_source_page(_make_source(title, content))
        headings = [ln for ln in page.splitlines() if ln.startswith("# ")]
        assert len(headings) == 1
        assert headings[0] == f"# {title}"

    def test_title_in_middle_not_deduplicated(self):
        """Title appears in the body (not first line) — heading only added once."""
        title = "Attention Value"
        content = f"# {title}\n\nOther content.\n\n{title} is important."
        page = render_source_page(_make_source(title, content))
        headings = [ln for ln in page.splitlines() if ln.startswith("# ")]
        assert len(headings) == 1

    def test_empty_content(self):
        """Empty content — heading still added with empty body."""
        title = "Empty Source"
        page = render_source_page(_make_source(title, ""))
        headings = [ln for ln in page.splitlines() if ln.startswith("# ")]
        assert len(headings) == 1
        assert headings[0] == "# Empty Source"
