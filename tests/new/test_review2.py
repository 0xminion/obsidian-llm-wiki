"""Tests for code review round 2 — embedding, MoC, Supadata, language, dedup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    # Should only have ONE line for the a↔b pair, not two
    assert len(lines) == 1


def test_moc_cross_ref_diagram_bidirectional():
    """Bidirectional pair uses ↔ arrow."""
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A", related=[ConceptLink(slug="b", relation="r")])
    b = _make_concept("b", "B", related=[ConceptLink(slug="a", relation="r")])
    lines = _build_moc_cross_ref_diagram([a, b], {"a": a, "b": b})
    assert "↔" in lines[0]


def test_moc_cross_ref_diagram_unidirectional():
    """Unidirectional pair uses → arrow."""
    from obsidian_llm_wiki.core.models import ConceptLink
    from obsidian_llm_wiki.render.obsidian import _build_moc_cross_ref_diagram
    a = _make_concept("a", "A", related=[ConceptLink(slug="b", relation="r")])
    b = _make_concept("b", "B", related=[])
    lines = _build_moc_cross_ref_diagram([a, b], {"a": a, "b": b})
    assert "→" in lines[0]


# ── render/obsidian.py: render_moc_page cross-lingual links ─────────────────


def test_render_moc_page_with_cross_lingual_links():
    """MoC with cross_lingual_links shows Cross-Lingual Links section."""
    from obsidian_llm_wiki.core.models import MapOfContent
    from obsidian_llm_wiki.render.obsidian import render_moc_page
    moc = MapOfContent(
        title="Test MOC", slug="test-moc", summary="A test MOC.",
        concept_slugs=["concept-a"],
    )
    a = _make_concept("concept-a", "Concept A")
    xling = {"concept-a": [("concept-b", 0.92, "概念B")]}
    page = render_moc_page(
        moc, all_concepts={"concept-a": a},
        cross_lingual_links=xling,
    )
    assert "Cross-Lingual Links / 跨语言关联" in page
    assert "概念B" in page
    assert "0.92" in page


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


# ── ingest/extractors/youtube.py: Supadata error paths ─────────────────────


def test_supadata_402_credits_error():
    """402 response raises RuntimeError with billing message."""
    from obsidian_llm_wiki.ingest.extractors import youtube as yt_mod
    mock_resp = MagicMock()
    mock_resp.status_code = 402
    mock_resp.json.return_value = {"error": "insufficient credits"}
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(yt_mod, "_get_api_key", return_value="fake-key"):
            try:
                yt_mod._supadata_transcript("https://youtube.com/watch?v=test", "fake-key")
                raise AssertionError("Should have raised")
            except RuntimeError as e:
                assert "credit" in str(e).lower() or "billing" in str(e).lower()


def test_supadata_403_restricted_video():
    """403 response raises RuntimeError with restricted message."""
    from obsidian_llm_wiki.ingest.extractors import youtube as yt_mod
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        try:
            yt_mod._supadata_transcript("https://youtube.com/watch?v=test", "fake-key")
            raise AssertionError("Should have raised")
        except RuntimeError as e:
            assert "private" in str(e).lower() or "restricted" in str(e).lower()


def test_parse_supadata_response_chunked():
    """Chunked content (list of segments) is joined correctly."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _parse_supadata_response
    data = {
        "content": [
            {"text": "Hello world this is a test transcript that is long "
             "enough to pass the minimum length check of two hundred "
             "characters. " * 3, "start": 0, "duration": 10},
        ],
        "lang": "en",
    }
    result = _parse_supadata_response(data, "https://youtube.com/watch?v=test")
    assert result is not None
    assert "Hello world" in result.content


def test_parse_supadata_response_non_english():
    """Non-English transcript gets language prefix."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _parse_supadata_response
    data = {"content": "这是一个中文的转录文本，足够长以通过最小长度检查。" * 10, "lang": "zh"}
    result = _parse_supadata_response(data, "https://youtube.com/watch?v=test")
    assert result is not None
    assert "[Transcript language: zh]" in result.content


def test_parse_supadata_response_short_transcript():
    """Transcript < 200 chars raises RuntimeError."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _parse_supadata_response
    data = {"content": "Too short.", "lang": "en"}
    try:
        _parse_supadata_response(data, "https://youtube.com/watch?v=test")
        raise AssertionError("Should have raised")
    except RuntimeError:
        pass


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
