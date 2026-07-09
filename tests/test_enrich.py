"""Tests for pipeline.enrich and pipeline.prompts_enrich."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.enrich import (
    EnrichmentResult,
    EnrichOptions,
    _append_enrichment_log,
    _build_concept_registry,
    _enrich_existing_concept,
    _extract_outbound_links,
)
from pipeline.prompts_enrich import (
    EnrichDecision,
    build_enrich_prompt,
    parse_enrich_response,
)

# ── EnrichOptions defaults ────────────────────────────────────────────────


def test_enrich_options_defaults():
    """EnrichOptions has correct default values."""
    opts = EnrichOptions()
    assert opts.seed_urls == []
    assert opts.allowed_host == ""
    assert opts.max_pages == 20
    assert opts.no_web is False


def test_enrich_options_custom_values():
    """EnrichOptions accepts custom values."""
    opts = EnrichOptions(
        seed_urls=["https://example.com"],
        allowed_host="example.com",
        max_pages=5,
        no_web=True,
    )
    assert opts.seed_urls == ["https://example.com"]
    assert opts.allowed_host == "example.com"
    assert opts.max_pages == 5
    assert opts.no_web is True


def test_enrichment_result_defaults():
    """EnrichmentResult has zero defaults and an empty errors list."""
    r = EnrichmentResult()
    assert r.pages_fetched == 0
    assert r.references_created == 0
    assert r.concepts_enriched == 0
    assert r.pages_skipped == 0
    assert r.errors == []


# ── _build_concept_registry ────────────────────────────────────────────────


def test_build_concept_registry_with_mock_files(tmp_path: Path):
    """_build_concept_registry maps slugs to concept IDs."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "alpha.md").write_text(
        "---\ntype: Concept\ntitle: Alpha\nslug: alpha-slug\n---\nAlpha body.\n",
        encoding="utf-8",
    )
    (bundle / "beta.md").write_text(
        "---\ntype: Concept\ntitle: Beta\n---\nBeta body.\n",
        encoding="utf-8",
    )
    # index.md should be skipped.
    (bundle / "index.md").write_text("---\ntype: MOC\n---\n# Index\n", encoding="utf-8")

    registry = _build_concept_registry(bundle)

    # The slug from frontmatter should map to the concept id.
    assert "alpha-slug" in registry
    assert registry["alpha-slug"] == "alpha"
    # The bare stem should also map.
    assert registry.get("alpha") == "alpha"
    assert registry.get("beta") == "beta"
    # index should not be in the registry.
    assert "index" not in registry


def test_build_concept_registry_empty_dir(tmp_path: Path):
    """_build_concept_registry returns empty dict for a directory with no .md files."""
    bundle = tmp_path / "empty"
    bundle.mkdir()
    registry = _build_concept_registry(bundle)
    assert registry == {}


def test_build_concept_registry_missing_dir(tmp_path: Path):
    """_build_concept_registry returns empty dict for a non-existent directory."""
    registry = _build_concept_registry(tmp_path / "nope")
    assert registry == {}


# ── _extract_outbound_links ────────────────────────────────────────────────


def test_extract_outbound_links_html_content():
    """_extract_outbound_links extracts href URLs from HTML."""
    content = (
        '<a href="/page1">Page 1</a>'
        '<a href="https://example.com/page2">Page 2</a>'
        '<a href="#fragment">skip</a>'
        '<a href="mailto:foo@bar.com">mail</a>'
    )
    links = _extract_outbound_links(
        content, "https://example.com/base", "example.com"
    )
    assert "https://example.com/page1" in links
    assert "https://example.com/page2" in links
    # Fragment and mailto links should be excluded.
    assert not any("#fragment" in l for l in links)  # noqa: E741
    assert not any("mailto" in l for l in links)  # noqa: E741


def test_extract_outbound_links_markdown_content():
    """_extract_outbound_links extracts markdown [text](url) links."""
    content = "See [Page 1](/page1) and [Page 2](https://example.com/page2)."
    links = _extract_outbound_links(
        content, "https://example.com/base", "example.com"
    )
    assert "https://example.com/page1" in links
    assert "https://example.com/page2" in links


def test_extract_outbound_links_host_filter():
    """_extract_outbound_links filters by allowed_host."""
    content = (
        '<a href="https://example.com/a">A</a>'
        '<a href="https://other.com/b">B</a>'
    )
    links = _extract_outbound_links(
        content, "https://example.com/", "example.com"
    )
    assert "https://example.com/a" in links
    assert not any("other.com" in l for l in links)  # noqa: E741


def test_extract_outbound_links_no_host_filter():
    """When allowed_host is empty, all http(s) links are returned."""
    content = (
        '<a href="https://example.com/a">A</a>'
        '<a href="https://other.com/b">B</a>'
    )
    links = _extract_outbound_links(content, "https://example.com/", "")
    assert "https://example.com/a" in links
    assert "https://other.com/b" in links


def test_extract_outbound_links_dedup():
    """Duplicate links are de-duplicated."""
    content = '<a href="/page1">A</a><a href="/page1">B</a>'
    links = _extract_outbound_links(
        content, "https://example.com/", "example.com"
    )
    assert links.count("https://example.com/page1") == 1


def test_extract_outbound_links_empty_content():
    """Empty content returns an empty list."""
    assert _extract_outbound_links("", "https://example.com/", "") == []


def test_extract_outbound_links_skips_base_url():
    """The base URL itself is excluded from the result."""
    content = '<a href="https://example.com/">Home</a>'
    links = _extract_outbound_links(
        content, "https://example.com/", "example.com"
    )
    assert links == []


# ── _enrich_existing_concept ───────────────────────────────────────────────


def test_enrich_existing_concept_adds_citation(tmp_path: Path):
    """_enrich_existing_concept appends a citation to the Citations section."""
    concept = tmp_path / "concept.md"
    concept.write_text(
        "---\ntype: Concept\ntitle: My Concept\n---\n\n# My Concept\n\nBody text.\n",
        encoding="utf-8",
    )

    _enrich_existing_concept(concept, "https://example.com/source", "New insight here.")

    content = concept.read_text(encoding="utf-8")
    assert "## Citations" in content
    assert "https://example.com/source" in content
    assert "New insight here." in content


def test_enrich_existing_concept_appends_to_existing_section(tmp_path: Path):
    """_enrich_existing_concept appends to an existing Citations section."""
    concept = tmp_path / "concept.md"
    concept.write_text(
        "---\ntype: Concept\ntitle: My Concept\n---\n\n# My Concept\n\nBody.\n\n"
        "## Citations\n\n- [2024-01-01] https://old.com\n",
        encoding="utf-8",
    )

    _enrich_existing_concept(concept, "https://example.com/new", "Fresh data.")

    content = concept.read_text(encoding="utf-8")
    assert content.count("## Citations") == 1
    assert "https://example.com/new" in content
    assert "https://old.com" in content  # old citation preserved


def test_enrich_existing_concept_preserves_frontmatter(tmp_path: Path):
    """_enrich_existing_concept preserves the frontmatter."""
    concept = tmp_path / "concept.md"
    concept.write_text(
        "---\ntype: Concept\ntitle: Preserved\n---\n\nBody.\n",
        encoding="utf-8",
    )

    _enrich_existing_concept(concept, "https://example.com/x", "added")

    content = concept.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "title: Preserved" in content


def test_enrich_existing_concept_empty_addition(tmp_path: Path):
    """_enrich_existing_concept works with an empty addition string."""
    concept = tmp_path / "concept.md"
    concept.write_text(
        "---\ntype: Concept\ntitle: T\n---\n\nBody.\n",
        encoding="utf-8",
    )

    _enrich_existing_concept(concept, "https://example.com/x", "")

    content = concept.read_text(encoding="utf-8")
    assert "## Citations" in content
    assert "https://example.com/x" in content


# ── _append_enrichment_log ─────────────────────────────────────────────────


def test_append_enrichment_log_creates_log(tmp_path: Path):
    """_append_enrichment_log creates log.md if it doesn't exist."""
    result = EnrichmentResult(
        pages_fetched=3, references_created=1, concepts_enriched=2, pages_skipped=0
    )
    _append_enrichment_log(tmp_path, result)

    log = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "Enrichment pass" in log
    assert "pages_fetched: 3" in log
    assert "references_created: 1" in log
    assert "concepts_enriched: 2" in log


def test_append_enrichment_log_appends_to_existing(tmp_path: Path):
    """_append_enrichment_log appends to an existing log.md."""
    log_path = tmp_path / "log.md"
    log_path.write_text("# Enrichment Log\n\nOld entry.\n", encoding="utf-8")

    result = EnrichmentResult(pages_fetched=1)
    _append_enrichment_log(tmp_path, result)

    content = log_path.read_text(encoding="utf-8")
    assert "Old entry." in content
    assert "Enrichment pass" in content
    assert "pages_fetched: 1" in content


def test_append_enrichment_log_includes_errors(tmp_path: Path):
    """_append_enrichment_log lists errors when present."""
    result = EnrichmentResult(
        pages_fetched=1,
        errors=["fetch https://x: timeout", "llm https://y: 500"],
    )
    _append_enrichment_log(tmp_path, result)

    content = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "errors: 2" in content
    assert "timeout" in content
    assert "500" in content


# ── prompts_enrich: build_enrich_prompt ─────────────────────────────────────


def test_build_enrich_prompt_basic():
    """build_enrich_prompt includes URL, title, content, and JSON spec."""
    prompt = build_enrich_prompt(
        url="https://example.com/page",
        title="Test Page",
        content="This is the page content.",
        existing_concepts=["alpha", "beta"],
    )
    assert "https://example.com/page" in prompt
    assert "Test Page" in prompt
    assert "This is the page content." in prompt
    assert "alpha" in prompt
    assert "beta" in prompt
    assert "enrich" in prompt
    assert "mint" in prompt
    assert "skip" in prompt
    assert "JSON" in prompt


def test_build_enrich_prompt_dict_concepts():
    """build_enrich_prompt accepts a dict of slug -> concept_id."""
    prompt = build_enrich_prompt(
        url="https://example.com/page",
        title="T",
        content="C",
        existing_concepts={"alpha-slug": "alpha"},
    )
    assert "alpha-slug" in prompt
    assert "alpha" in prompt


def test_build_enrich_prompt_no_concepts():
    """build_enrich_prompt handles no existing concepts."""
    prompt = build_enrich_prompt(
        url="https://example.com/page",
        title="T",
        content="C",
        existing_concepts=None,
    )
    assert "none yet" in prompt.lower() or "(none" in prompt.lower()


def test_build_enrich_prompt_truncates_long_content():
    """build_enrich_prompt truncates very long content."""
    long_content = "x" * 20000
    prompt = build_enrich_prompt(
        url="https://example.com",
        title="T",
        content=long_content,
    )
    assert "[truncated]" in prompt
    assert len(prompt) < 20000


# ── prompts_enrich: parse_enrich_response ──────────────────────────────────


def test_parse_enrich_response_valid_array():
    """parse_enrich_response parses a valid JSON array."""
    response = '''[
      {"action": "enrich", "concept_id": "alpha", "addition": "new info"},
      {"action": "mint", "concept_id": "new-ref", "title": "New Ref", "body": "body", "tags": ["t1"]}
    ]'''  # noqa: E501
    decisions = parse_enrich_response(response)
    assert len(decisions) == 2
    assert decisions[0].action == "enrich"
    assert decisions[0].concept_id == "alpha"
    assert decisions[0].addition == "new info"
    assert decisions[1].action == "mint"
    assert decisions[1].title == "New Ref"
    assert decisions[1].tags == ["t1"]


def test_parse_enrich_response_single_object():
    """parse_enrich_response parses a single JSON object."""
    response = '{"action": "skip"}'
    decisions = parse_enrich_response(response)
    assert len(decisions) == 1
    assert decisions[0].action == "skip"


def test_parse_enrich_response_with_code_fence():
    """parse_enrich_response strips markdown code fences."""
    response = '```json\n[{"action": "skip"}]\n```'
    decisions = parse_enrich_response(response)
    assert len(decisions) == 1
    assert decisions[0].action == "skip"


def test_parse_enrich_response_with_prose():
    """parse_enrich_response extracts JSON from surrounding prose."""
    response = 'Here are my decisions:\n[{"action": "mint", "concept_id": "x"}]\nDone.'
    decisions = parse_enrich_response(response)
    assert len(decisions) == 1
    assert decisions[0].action == "mint"
    assert decisions[0].concept_id == "x"


def test_parse_enrich_response_empty():
    """parse_enrich_response returns empty list for empty input."""
    assert parse_enrich_response("") == []
    assert parse_enrich_response("   ") == []


def test_parse_enrich_response_invalid_json():
    """parse_enrich_response returns empty list for invalid JSON."""
    assert parse_enrich_response("not json at all") == []
    assert parse_enrich_response("{{{{") == []


def test_parse_enrich_response_invalid_action_normalized():
    """parse_enrich_response normalises unknown actions to 'skip'."""
    response = '[{"action": "unknown_action"}]'
    decisions = parse_enrich_response(response)
    assert decisions[0].action == "skip"


def test_parse_enrich_response_follow_links():
    """parse_enrich_response extracts follow_links."""
    response = '''[{
      "action": "enrich",
      "concept_id": "alpha",
      "follow_links": ["https://example.com/a", "https://example.com/b"]
    }]'''
    decisions = parse_enrich_response(response)
    assert decisions[0].follow_links == ["https://example.com/a", "https://example.com/b"]


def test_enrich_decision_dataclass_defaults():
    """EnrichDecision has correct defaults."""
    d = EnrichDecision()
    assert d.action == "skip"
    assert d.concept_id == ""
    assert d.title == ""
    assert d.summary == ""
    assert d.body == ""
    assert d.addition == ""
    assert d.tags == []
    assert d.follow_links == []


# ── pytest entry ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
