"""Tests for obsidian_llm_wiki.render.obsidian — deterministic markdown rendering."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
)
from obsidian_llm_wiki.render.obsidian import (
    atomic_write,
    build_frontmatter,
    make_wikilink,
    parse_frontmatter,
    render_concept_page,
    render_entry_page,
    render_moc_page,
    render_source_page,
    render_vault,
    slugify,
)

# ── Utilities ────────────────────────────────────────────────────────────


def test_slugify():
    assert slugify("Gradient Descent") == "gradient-descent"
    assert slugify("") == "untitled"


def test_make_wikilink_plain():
    assert make_wikilink("foo") == "[[foo]]"


def test_make_wikilink_alias():
    assert make_wikilink("foo", "Foo Bar") == "[[foo|Foo Bar]]"


def test_build_and_parse_frontmatter_roundtrip():
    fm = {"type": "Concept", "title": "Test", "tags": ["a", "b"]}
    block = build_frontmatter(fm)
    assert block.startswith("---\n")
    assert block.endswith("---\n")
    meta, body = parse_frontmatter(block + "\nHello")
    assert meta["type"] == "Concept"
    assert meta["title"] == "Test"
    assert body == "Hello"


def test_atomic_write(tmp_path: Path):
    f = tmp_path / "test.md"
    atomic_write(f, "content")
    assert f.read_text() == "content"
    atomic_write(f, "overwritten")
    assert f.read_text() == "overwritten"


# ── Page renderers ───────────────────────────────────────────────────────


def test_render_source_page():
    doc = SourceDoc(title="Article", content="Full content here", url="https://example.com")
    page = render_source_page(doc, "2026-01-01T00:00:00Z")
    assert "type: Source" in page
    assert "title: Article" in page
    assert "url: https://example.com" in page
    assert "# Article" in page
    assert "Full content here" in page


def test_render_concept_page_basic():
    c = ConceptNote(
        title="Gradient Descent", slug="gradient-descent",
        summary="Optimization algorithm",
        tags=["machine-learning", "optimization"],
    )
    page = render_concept_page(c, "2026-01-01T00:00:00Z")
    assert "type: Concept" in page
    assert "title: Gradient Descent" in page
    assert "# Gradient Descent" in page
    assert "Optimization algorithm" in page
    assert "machine-learning" in page


def test_render_concept_page_with_sections():
    c = ConceptNote(
        title="Transformer", slug="transformer", summary="Attention model",
        sections=[
            BodySection(heading="Core concept", points=["Self-attention", "Multi-head"]),
            BodySection(heading="Context", prose="The transformer architecture..."),
        ],
    )
    page = render_concept_page(c)
    assert "## Core concept" in page
    assert "- Self-attention" in page
    assert "## Context" in page
    assert "The transformer architecture..." in page


def test_render_concept_page_with_related():
    c = ConceptNote(
        title="GD", slug="gd", summary="S",
        related=[ConceptLink(slug="sgd", relation="variant_of", display="SGD")],
    )
    page = render_concept_page(c)
    meta, body = parse_frontmatter(page)
    assert meta["relations"] == [{"target": "sgd", "type": "variant_of", "display": "SGD"}]
    assert "## Related Concepts" not in body
    assert "[[sgd|SGD]]" not in body


def test_render_concept_page_with_claims():
    c = ConceptNote(
        title="GD", slug="gd", summary="S",
        claims=[],
    )
    from obsidian_llm_wiki.core.models import Claim
    c.claims = [Claim(text="Learning rate controls step size")]
    page = render_concept_page(c)
    assert "## Claims" in page
    assert "Learning rate controls step size" in page


def test_render_concept_page_aliases_in_frontmatter():
    c = ConceptNote(
        title="GD", slug="gd", summary="S",
        aliases=["Gradient Method", "Batch GD"],
    )
    page = render_concept_page(c)
    meta, _ = parse_frontmatter(page)
    assert meta["aliases"] == ["Gradient Method", "Batch GD"]


def test_render_entry_page():
    synth = SourceSynthesis(
        source_title="Paper", source_summary="A great paper",
        source_tags=["ml"],
        key_points=["Finding 1", "Finding 2"],
        open_questions=["Why does X happen?"],
        concepts=[ConceptNote(title="C", slug="c", summary="s")],
    )
    page = render_entry_page(synth, "paper", ["c"], "2026-01-01T00:00:00Z")
    assert "type: Entry" in page
    assert "# Paper" in page
    assert "A great paper" in page
    assert "## Key Findings" in page
    assert "- Finding 1" in page
    assert "## Open Questions" in page
    assert "## Linked Concepts" in page
    assert "[[c]]" in page
    assert "## Source" in page


def test_render_moc_page():
    moc = MapOfContent(
        title="Optimization", slug="optimization", summary="Overview of optim methods",
        tags=["optimization"], concept_slugs=["gd", "sgd", "adam"],
    )
    page = render_moc_page(moc, "2026-01-01T00:00:00Z")
    assert "type: Map of Content" in page
    assert "# Optimization" in page
    assert "## Concepts" in page
    assert "[[gd]]" in page
    assert "[[adam]]" in page


# ── render_vault (end-to-end) ────────────────────────────────────────────


def test_render_vault_full(tmp_path: Path):
    from obsidian_llm_wiki.synth.dedupe import merge_bundle

    bundle_dir = tmp_path / "04-Wiki"
    sources = {
        "paper-a.md": SourceDoc(title="Paper A", content="Content A",
                                url="https://a.com"),
    }
    synth = SourceSynthesis(
        source_title="Paper A", source_summary="Summary A",
        source_tags=["ml"],
        key_points=["Point A"],
        concepts=[
            ConceptNote(
                title="Concept A", slug="concept-a", summary="Concept A summary",
                tags=["ml", "ai"],
                sections=[BodySection(heading="Core", points=["Detail"])],
                related=[ConceptLink(slug="concept-b")],
            ),
            ConceptNote(
                title="Concept B", slug="concept-b", summary="Concept B summary",
                tags=["ml"],
            ),
        ],
        maps=[
            MapOfContent(title="ML Topic", slug="ml-topic", summary="MOC",
                         concept_slugs=["concept-a", "concept-b"]),
        ],
    )
    bundle = merge_bundle([synth])

    written = render_vault(bundle_dir, bundle, sources)

    # Check files exist.
    assert (bundle_dir / "sources" / "paper-a.md").exists()
    assert (bundle_dir / "entries" / "paper-a.md").exists()
    assert (bundle_dir / "concepts" / "concept-a.md").exists()
    assert (bundle_dir / "concepts" / "concept-b.md").exists()
    assert (bundle_dir / "mocs" / "ml-topic.md").exists()
    assert (bundle_dir / "concepts" / "index.md").exists()
    assert (bundle_dir / "index.md").exists()

    # Check concept page has wikilinks.
    concept_a = (bundle_dir / "concepts" / "concept-a.md").read_text()
    assert "[[concept-b]]" in concept_a

    # Check entry has concept wikilinks.
    entry = (bundle_dir / "entries" / "paper-a.md").read_text()
    assert "[[concept-a]]" in entry

    # Check MOC has concept wikilinks.
    moc = (bundle_dir / "mocs" / "ml-topic.md").read_text()
    assert "[[concept-a]]" in moc
    assert "[[concept-b]]" in moc
