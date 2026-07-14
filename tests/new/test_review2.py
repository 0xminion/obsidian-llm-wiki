"""Tests for code review round 2 — embedding, MoC, Supadata, language, dedup."""

from __future__ import annotations

from unittest.mock import patch

# ── embedding.py: cosine_similarity edge cases ──────────────────────────────


def test_cosine_similarity_zero_vectors():
    from obsidian_llm_wiki.synth.embedding import cosine_similarity
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0


def test_cosine_similarity_mismatched_length():
    from obsidian_llm_wiki.synth.embedding import cosine_similarity
    assert cosine_similarity([1, 2], [1]) == 0.0


def test_cosine_similarity_known_values():
    from obsidian_llm_wiki.synth.embedding import cosine_similarity
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert abs(cosine_similarity([1, 1], [1, 0]) - 0.7071) < 0.01


# ── embedding.py: find_cross_lingual_links ────────────────────────────────


def test_find_cross_lingual_links_returns_empty_when_disabled(monkeypatch):
    """When EMBEDDINGS_ENABLED=false, returns {} without calling embed_text."""
    monkeypatch.setenv("EMBEDDINGS_ENABLED", "false")
    from obsidian_llm_wiki.synth.embedding import find_cross_lingual_links
    result = find_cross_lingual_links([])
    assert result == {}


# ── render/obsidian.py: _build_moc_cross_ref_diagram ────────────────────────


def _make_concept(slug, title, related=None, aliases=None, summary=""):
    from obsidian_llm_wiki.core.models import (
        BodySection,
        ConceptNote,
    )
    return ConceptNote(
        title=title, slug=slug, summary=summary,
        sections=[BodySection(heading="Core", points=["A point."])],
        related=related or [], aliases=aliases or [],
    )


def test_moc_cross_ref_diagram_empty_concepts():
    """_build_moc_cross_ref_diagram with <2 concepts returns []."""
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A")
    assert _build_moc_cross_ref_diagram([a], {"a": a}) == []
    assert _build_moc_cross_ref_diagram([], {}) == []


def test_moc_cross_ref_diagram_dedup():
    """Pairs already seen don't appear twice."""
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A", related=[ConceptLink(slug="b", relation="r")])
    b = _make_concept("b", "B", related=[ConceptLink(slug="a", relation="r")])
    lines = _build_moc_cross_ref_diagram([a, b], {"a": a, "b": b})
    # Should only have ONE pair — check by counting "↓" entries
    down_count = sum(1 for line in lines if "↓" in line)
    assert down_count == 1


def test_moc_cross_ref_diagram_bidirectional():
    """Bidirectional pair uses ↔ arrow in target line."""
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A", related=[ConceptLink(slug="b", relation="r")])
    b = _make_concept("b", "B", related=[ConceptLink(slug="a", relation="r")])
    lines = _build_moc_cross_ref_diagram([a, b], {"a": a, "b": b})
    joined = "\n".join(lines)
    assert "↔" in joined


def test_moc_cross_ref_diagram_unidirectional():
    """Unidirectional pair does not use ↔ arrow."""
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A", related=[ConceptLink(slug="b", relation="r")])
    b = _make_concept("b", "B", related=[])
    lines = _build_moc_cross_ref_diagram([a, b], {"a": a, "b": b})
    joined = "\n".join(lines)
    assert "↓" in joined
    assert "↔" not in joined


# ── render/obsidian.py: MoC cross-refs with no inter-concept relations ──────


def test_moc_cross_ref_diagram_no_inter_concept_relations_shows_placeholder():
    """MoC with ≥2 concepts but no inter-concept relations shows placeholder message."""
    from obsidian_llm_wiki.core.models import MapOfContent
    from obsidian_llm_wiki.render.obsidian import render_moc_page

    a = _make_concept("concept-a", "Concept A", related=[])
    b = _make_concept("concept-b", "Concept B", related=[])
    moc = MapOfContent(
        title="Empty Relations MoC", slug="empty-relations-moc",
        summary="MoC with no inter-concept relations.",
        concept_slugs=["concept-a", "concept-b"],
    )
    page = render_moc_page(
        moc, all_concepts={"concept-a": a, "concept-b": b},
    )
    # Section heading must always be present for structural consistency
    assert "## Cross-References / 关联图谱" in page
    # Placeholder message must appear
    assert "No cross-references available yet" in page
    # No code block (since there's no diagram to render)
    assert "```text" not in page


def test_moc_cross_ref_diagram_with_relations_still_works():
    """MoC with inter-concept relations renders the diagram as before."""
    from obsidian_llm_wiki.core.models import ConceptLink, MapOfContent
    from obsidian_llm_wiki.render.obsidian import render_moc_page

    a = _make_concept(
        "concept-a", "Concept A",
        related=[ConceptLink(slug="concept-b", relation="enables")],
    )
    b = _make_concept(
        "concept-b", "Concept B",
        related=[ConceptLink(slug="concept-a", relation="enabled_by")],
    )
    moc = MapOfContent(
        title="Related MoC", slug="related-moc",
        summary="MoC with inter-concept relations.",
        concept_slugs=["concept-a", "concept-b"],
    )
    page = render_moc_page(
        moc, all_concepts={"concept-a": a, "concept-b": b},
    )
    assert "## Cross-References / 关联图谱" in page
    assert "```text" in page
    assert "↔" in page
    # Placeholder should NOT appear when relations exist
    assert "No cross-references available yet" not in page


def test_moc_cross_ref_diagram_single_concept_no_section():
    """MoC with only 1 concept still has no Cross-References section."""
    from obsidian_llm_wiki.core.models import MapOfContent
    from obsidian_llm_wiki.render.obsidian import render_moc_page

    a = _make_concept("concept-a", "Concept A", related=[])
    moc = MapOfContent(
        title="Single MoC", slug="single-moc",
        summary="MoC with only one concept.",
        concept_slugs=["concept-a"],
    )
    page = render_moc_page(
        moc, all_concepts={"concept-a": a},
    )
    assert "## Cross-References / 关联图谱" not in page
    assert "No cross-references available yet" not in page


# ── render/obsidian.py: render_moc_page cross-lingual links ─────────────────


def test_render_moc_page_with_cross_lingual_links():
    """MoC with cross_lingual_links merges them into Concepts section."""
    from obsidian_llm_wiki.core.models import MapOfContent
    from obsidian_llm_wiki.render.obsidian import render_moc_page
    moc = MapOfContent(
        title="Test MOC", slug="test-moc", summary="A test MOC.",
        concept_slugs=["concept-a"],
    )
    a = _make_concept("concept-a", "Concept A")
    xling = {"concept-a": [("concept-b", 0.92, "概念B")]}
    page = render_moc_page(
        moc, all_concepts={"concept-a": a, "concept-b": _make_concept("concept-b", "概念B")},
        cross_lingual_links=xling,
    )
    # Cross-lingual links should NOT be a separate section. They must appear
    # before cross-references, otherwise Markdown nests them under the wrong
    # heading even though the page happens to contain both strings.
    assert "Cross-Lingual Links" not in page
    concept_b_index = page.index("concept-b")
    cross_ref_index = page.index("## Cross-References / 关联图谱")
    assert concept_b_index < cross_ref_index
    assert moc.concept_slugs == ["concept-a", "concept-b"]


def test_render_vault_remaps_embedding_links_after_bilingual_slug_normalization(tmp_path):
    """Pre-render links must survive the final English-first filename policy."""
    from obsidian_llm_wiki.core.models import MapOfContent, SourceSynthesis, SynthesisBundle
    from obsidian_llm_wiki.render.obsidian import render_vault

    english = _make_concept("language-model", "Language model")
    chinese = _make_concept("da-yuyan-moxing", "大语言模型")
    moc = MapOfContent(
        title="Language AI", slug="language-ai", summary="", concept_slugs=[english.slug],
    )
    source = SourceSynthesis(
        source_title="Source", source_summary="", concepts=[english, chinese], maps=[moc],
    )
    bundle = SynthesisBundle(
        sources=[source],
        concepts=[english, chinese],
        maps=[moc],
    )
    links = {
        english.slug: [(chinese.slug, 0.84, chinese.title)],
        chinese.slug: [(english.slug, 0.84, english.title)],
    }

    render_vault(tmp_path, bundle, {}, cross_lingual_links=links)

    normalized_chinese_slug = "da-yuyan-moxing-大语言模型"
    page = (tmp_path / "mocs" / "language-ai.md").read_text(encoding="utf-8")
    assert normalized_chinese_slug in moc.concept_slugs
    assert page.index(f"[[{normalized_chinese_slug}]]") < page.index(
        "## Cross-References / 关联图谱"
    )


# ── render/obsidian.py: render_source_page deduplication ───────────────────


def test_render_source_page_plain_title_dedup():
    """Content starting with title as plain text doesn't duplicate heading."""
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.render.obsidian import render_source_page
    source = SourceDoc(
        title="My Article Title",
        content="My Article Title\n\nThis is the body content that follows.",
        url="https://example.com",
    )
    page = render_source_page(source)
    # Should have exactly one # heading
    heading_count = page.count("# My Article Title")
    assert heading_count == 1


def test_render_source_page_no_duplicate_when_heading_in_content():
    """Content with # heading doesn't get another # heading added."""
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.render.obsidian import render_source_page
    source = SourceDoc(
        title="My Article",
        content="# My Article\n\nBody text here.",
        url="https://example.com",
    )
    page = render_source_page(source)
    heading_count = page.count("# My Article")
    assert heading_count == 1


# ── render/obsidian.py: concept with empty related ──────────────────────────


def test_render_concept_page_empty_related_no_cross_refs():
    """Concept with all_concepts but empty related list — no cross-refs section."""
    from obsidian_llm_wiki.render.obsidian import render_concept_page
    a = _make_concept("a", "A", related=[])
    page = render_concept_page(a, all_concepts={"a": a})
    assert "关联图谱" not in page


# ── synth/language.py: edge cases ───────────────────────────────────────────


def test_detect_language_russian():
    from obsidian_llm_wiki.synth.language import detect_language
    text = "Это текст на русском языке для тестирования. " * 10
    assert detect_language(text) == "ru"


def test_detect_language_mixed_chinese_japanese():
    """Japanese (with hiragana) should be detected before Chinese (kanji only)."""
    from obsidian_llm_wiki.synth.language import detect_language
    # Text with both kanji and hiragana
    text = (
        "これは日本語のテキストです。機械学習について説明します。"
        "ひらがなと漢字が混在しています。"
    ) * 10
    assert detect_language(text) == "ja"


def test_detect_language_very_short_text():
    """Text under 20 chars returns 'en'."""
    from obsidian_llm_wiki.synth.language import detect_language
    assert detect_language("hi") == "en"
    assert detect_language("12345") == "en"


# ── cli/ingest.py: ledger edge cases ────────────────────────────────────────


def test_update_failed_ledger_preserves_existing(tmp_path):
    """Existing entries are preserved when new failures are added."""
    from obsidian_llm_wiki.cli.ingest import _update_failed_ledger
    _update_failed_ledger(tmp_path, [("https://first.com", "Error 1")])
    _update_failed_ledger(tmp_path, [("https://second.com", "Error 2")])
    content = (tmp_path / "failed_urls.md").read_text()
    assert "https://first.com" in content
    assert "https://second.com" in content
    assert "Error 1" in content
    assert "Error 2" in content


def test_update_failed_ledger_dedup_same_url(tmp_path):
    """Same URL with new error overwrites old error (dict update semantics)."""
    from obsidian_llm_wiki.cli.ingest import _update_failed_ledger
    _update_failed_ledger(tmp_path, [("https://example.com", "old error")])
    _update_failed_ledger(tmp_path, [("https://example.com", "new error")])
    content = (tmp_path / "failed_urls.md").read_text()
    assert "new error" in content
    assert "old error" not in content


# ── ingest/extractors/youtube.py: subtitle parsing ─────────────────────


def test_parse_subtitle_file_vtt():
    """VTT subtitle file is parsed into plain text."""
    import os
    import tempfile

    from obsidian_llm_wiki.ingest.extractors.youtube import _parse_subtitle_file
    vtt_content = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:05.000\n"
        "Hello world this is a test\n\n"
        "00:00:05.000 --> 00:00:10.000\n"
        "This is another line of text\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
        f.write(vtt_content)
        f.flush()
        try:
            result = _parse_subtitle_file(f.name)
            assert "Hello world" in result
            assert "another line" in result
        finally:
            os.unlink(f.name)


def test_parse_subtitle_file_strips_html():
    """HTML tags in subtitle text are stripped."""
    import os
    import tempfile

    from obsidian_llm_wiki.ingest.extractors.youtube import _parse_subtitle_file
    vtt_content = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:05.000\n"
        "<c>Hello world this is a test that is long enough</c>\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False) as f:
        f.write(vtt_content)
        f.flush()
        try:
            result = _parse_subtitle_file(f.name)
            assert "<c>" not in result
            assert "Hello world" in result
        finally:
            os.unlink(f.name)


# ── ingest/web.py: full 6-layer failure ─────────────────────────────────────


def test_extract_web_all_layers_fail():
    """All 6 layers failing raises RuntimeError with all error messages."""
    from obsidian_llm_wiki.ingest.web import extract_web
    with patch("obsidian_llm_wiki.ingest.web._extract_trafilatura",
               side_effect=RuntimeError("trafilatura failed")), \
         patch("obsidian_llm_wiki.ingest.web._extract_defuddle",
               side_effect=RuntimeError("defuddle failed")), \
         patch("obsidian_llm_wiki.ingest.web._extract_wayback",
               side_effect=RuntimeError("wayback failed")):
        try:
            extract_web("https://example.com/page")
            raise AssertionError("Should have raised")
        except RuntimeError as e:
            assert "trafilatura" in str(e)
            assert "defuddle" in str(e)
            assert "wayback" in str(e)


# ── render/obsidian.py: deterministic bilingual title normalization ──────────


def test_render_vault_normalizes_chinese_titles_and_slugs(tmp_path):
    """Chinese-derived titles become English-first bilingual with bilingual filenames."""
    from obsidian_llm_wiki.core.models import (
        ConceptNote,
        MapOfContent,
        SourceSynthesis,
        SynthesisBundle,
    )
    from obsidian_llm_wiki.render.obsidian import render_vault

    concept = ConceptNote(
        title="侨批与家族量证明",
        slug="qiaopi-proof-of-family",
        summary="一种宗族信用机制。",
        aliases=["Proof of Family"],
    )
    moc = MapOfContent(
        title="信用与结算的演化图谱",
        slug="evolution-of-trust-and-settlement",
        summary="中文摘要。",
        concept_slugs=["qiaopi-proof-of-family"],
    )
    synthesis = SourceSynthesis(
        source_title="被套过的人，还可以买谁？",
        source_summary="中文摘要。",
        language="zh",
        concepts=[concept],
        maps=[moc],
    )
    bundle = SynthesisBundle(sources=[synthesis], concepts=[concept], maps=[moc])

    render_vault(tmp_path, bundle, {})

    concept_path = tmp_path / "concepts" / "proof-of-family-侨批与家族量证明.md"
    assert concept_path.exists()
    concept_text = concept_path.read_text()
    assert "title: Proof of Family (侨批与家族量证明)" in concept_text
    assert "# Proof of Family (侨批与家族量证明)" in concept_text

    moc_path = tmp_path / "mocs" / "evolution-of-trust-and-settlement-信用与结算的演化图谱.md"
    assert moc_path.exists()
    moc_text = moc_path.read_text()
    assert "title: Evolution of Trust and Settlement (信用与结算的演化图谱)" in moc_text
    assert "## Concepts / 概念" in moc_text
    assert "[[proof-of-family-侨批与家族量证明" in moc_text

    entry_files = list((tmp_path / "entries").glob("*.md"))
    assert any(
        "proof-of-family" in f.name and "被套过的人还可以买谁" in f.name
        for f in entry_files
    )


def test_bilingual_title_split_keeps_english_first_order():
    """Already-correct bilingual titles must not get reversed on second pass."""
    from obsidian_llm_wiki.render.obsidian import _ensure_english_first_bilingual

    assert _ensure_english_first_bilingual(
        "USDT as Settlement Layer (USDT 作为结算货币)",
        slug="usdt-as-settlement-layer",
    ) == "USDT as Settlement Layer (USDT 作为结算货币)"
    assert _ensure_english_first_bilingual(
        "USDT 作为结算货币 (USDT as Settlement Layer)",
        slug="usdt-as-settlement-layer",
    ) == "USDT as Settlement Layer (USDT 作为结算货币)"


# ── synth/dedupe.py: backlink propagation ──────────────────────────────────


def test_propagate_backlinks_adds_reverse_edges():
    """When A→B exists but B→A doesn't, propagation adds B→A."""
    from obsidian_llm_wiki.core.models import (
        ConceptLink,
        ConceptNote,
        SynthesisBundle,
    )
    from obsidian_llm_wiki.synth.dedupe import propagate_backlinks

    a = ConceptNote(
        title="A", slug="a", summary="",
        related=[ConceptLink(slug="b", relation="depends_on")],
    )
    b = ConceptNote(
        title="B", slug="b", summary="",
        related=[],
    )
    bundle = SynthesisBundle(concepts=[a, b])
    propagate_backlinks(bundle)

    # B should now have a link back to A
    assert any(r.slug == "a" for r in b.related)
    assert b.related[0].relation == "prerequisite_of"


def test_propagate_backlinks_skips_existing_reverse():
    """When B→A already exists, propagation doesn't duplicate it."""
    from obsidian_llm_wiki.core.models import (
        ConceptLink,
        ConceptNote,
        SynthesisBundle,
    )
    from obsidian_llm_wiki.synth.dedupe import propagate_backlinks

    a = ConceptNote(
        title="A", slug="a", summary="",
        related=[ConceptLink(slug="b", relation="depends_on")],
    )
    b = ConceptNote(
        title="B", slug="b", summary="",
        related=[ConceptLink(slug="a", relation="related_to")],
    )
    bundle = SynthesisBundle(concepts=[a, b])
    propagate_backlinks(bundle)

    # B should still have exactly one link to A
    a_links = [r for r in b.related if r.slug == "a"]
    assert len(a_links) == 1


def test_propagate_backlinks_makes_moc_diagrams_bidirectional():
    """After propagation, MoC cross-ref diagrams show ↔ for one-way edges."""
    from obsidian_llm_wiki.core.models import (
        ConceptLink,
        ConceptNote,
        MapOfContent,
        SourceSynthesis,
        SynthesisBundle,
    )
    from obsidian_llm_wiki.render.obsidian import render_vault
    from obsidian_llm_wiki.synth.dedupe import propagate_backlinks

    # A links to B, but B doesn't link back (simulating cross-run gap)
    a = ConceptNote(
        title="Concept A", slug="concept-a", summary="A summary.",
        related=[ConceptLink(slug="concept-b", relation="enables")],
    )
    b = ConceptNote(
        title="Concept B", slug="concept-b", summary="B summary.",
        related=[],
    )
    moc = MapOfContent(
        title="Test MoC", slug="test-moc", summary="Test.",
        concept_slugs=["concept-a", "concept-b"],
    )
    synthesis = SourceSynthesis(
        source_title="Test Source", source_summary="Test.",
        concepts=[a, b], maps=[moc],
    )
    bundle = SynthesisBundle(sources=[synthesis], concepts=[a, b], maps=[moc])

    # Before propagation: B has no link to A
    assert not any(r.slug == "concept-a" for r in b.related)

    propagate_backlinks(bundle)

    # After propagation: B has a link to A
    assert any(r.slug == "concept-a" for r in b.related)

    # Render and check the MoC diagram shows ↔ (bidirectional)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        render_vault(Path(tmp), bundle, {})
        moc_file = Path(tmp) / "mocs" / "test-moc.md"
        text = moc_file.read_text()
        # The diagram should show ↔ because both directions now exist
        assert "↔" in text
