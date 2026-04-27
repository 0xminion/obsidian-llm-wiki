"""Tests for quality scoring and clipping auto-retry features."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.cli import _validate_clipping_quality
from pipeline.extractors._shared import (
    score_defuddle,
    score_pdf,
    score_podcast,
    score_web,
    score_youtube,
)
from pipeline.models import ExtractedSource, QualityScore

# ─── Quality Scoring ─────────────────────────────────────────────────────

class TestQualityScoring:
    def test_score_defuddle_empty(self):
        assert score_defuddle("") == 0.0
        assert score_defuddle("\n\n\n") == 0.0

    def test_score_defuddle_single_paragraph(self):
        body = "Line one.\nLine two.\nLine three."
        # 1 paragraph over 3 non-empty lines
        assert score_defuddle(body) == pytest.approx(1 / 3)

    def test_score_defuddle_multiple_paragraphs(self):
        body = "Para one.\n\nPara two.\n\nPara three."
        result = score_defuddle(body)
        assert 0 < result <= 1.0

    def test_score_youtube_ideal(self):
        assert score_youtube(600, 5) == 1.0

    def test_score_youtube_zero_duration(self):
        assert score_youtube(100, 0) == 0.0

    def test_score_youtube_off_ideal(self):
        result = score_youtube(300, 5)
        assert 0.0 <= result <= 1.0

    def test_score_web_empty(self):
        assert score_web("") == 0.0

    def test_score_web_clean_content(self):
        body = "This is a clean article with paragraphs and meaningful text."
        result = score_web(body)
        assert 0.0 < result <= 1.0

    def test_score_web_with_noise(self):
        body = "Content here.\n\nnavigation\nfooter\nPrivacy Policy"
        result = score_web(body)
        import re as _re
        body_stripped = _re.sub(r"\s+", "", body)
        noise = _re.compile(
            r"(?im)^\s*(?:navigation|footer|privacy\s+policy|terms\s+of\s+use|cookie\s+notice|advertisements?)\s*$"
        )
        cleaned = noise.sub("", body)
        cleaned_stripped = _re.sub(r"\s+", "", cleaned)
        expected = len(cleaned_stripped) / len(body_stripped)
        assert pytest.approx(result, rel=1e-9) == expected
        assert result < 1.0

    def test_score_web_does_not_strip_inline_words(self):
        body = "The navigator uses cookies for authentication."
        result = score_web(body)
        assert result == 1.0

    def test_score_pdf_empty(self):
        assert score_pdf("", 1) == 0.0

    def test_score_pdf_zero_pages(self):
        assert score_pdf("abc", 0) == 0.0

    def test_score_pdf_ideal(self):
        body = "x" * 3000
        assert score_pdf(body, 1) == 1.0

    def test_score_podcast_zero_audio(self):
        assert score_podcast(100, 0) == 0.0

    def test_score_podcast_low_words(self):
        assert score_podcast(40, 60) == 0.0

    def test_score_podcast_ideal(self):
        result = score_podcast(4500, 30 * 60)
        assert pytest.approx(result) == 1.0


class TestQualityScoreDataclass:
    def test_defaults(self):
        qs = QualityScore(extractor="web", score=0.75)
        assert qs.metrics == {}

    def test_fields(self):
        qs = QualityScore(extractor="youtube", score=0.8, metrics={"wpm": 120})
        assert qs.extractor == "youtube"
        assert qs.score == 0.8
        assert qs.metrics == {"wpm": 120}


class TestExtractedSourceQuality:
    def test_defaults(self):
        src = ExtractedSource(url="https://example.com", title="T", content="")
        assert src.quality_score == 0.0
        assert src.quality_metrics == {}

    def test_to_dict_includes_quality(self):
        src = ExtractedSource(
            url="https://example.com",
            title="T",
            content="body",
            quality_score=0.9,
            quality_metrics={"source": "web"},
        )
        d = src.to_dict()
        assert d["quality_score"] == 0.9
        assert d["quality_metrics"] == {"source": "web"}

    def test_load_roundtrip(self, tmp_path: Path):
        src = ExtractedSource(
            url="https://example.com",
            title="T",
            content="body",
            quality_score=0.85,
            quality_metrics={"source": "youtube"},
        )
        path = src.save(tmp_path)
        loaded = ExtractedSource.load(path)
        assert loaded.quality_score == 0.85
        assert loaded.quality_metrics == {"source": "youtube"}


# ─── Clipping Auto-Retry ──────────────────────────────────────────────────

class TestClippingQualityValidation:
    def test_short_body_fails(self):
        content = "---\nurl: https://example.com\n---\n\nshort"
        valid, score = _validate_clipping_quality(content)
        assert not valid
        assert score == 0.0

    def test_nav_only_fails(self):
        content = (
            "---\nurl: https://example.com\n---\n\n"
            "https://a.com\nhttps://b.com\nhttps://c.com\n"
        )
        valid, score = _validate_clipping_quality(content)
        assert not valid
        assert score == 0.0

    def test_good_clipping_passes(self):
        body = "Paragraph one with enough text to pass the 200 char minimum requirement easily.\n\nParagraph two also has plenty of text and no links.\n\nParagraph three continues the trend with even more filler text to ensure we cross thresholds."
        content = f"---\nurl: https://example.com\n---\n\n{body}"
        valid, score = _validate_clipping_quality(content)
        assert valid
        assert score > 0.0


class TestClippingRetryFlow:
    def test_bad_clipping_added_to_urls(self, tmp_path: Path):
        from pipeline.cli import _collect_clipping_files, _validate_clipping_quality

        clips_dir = tmp_path / "02-Clippings"
        clips_dir.mkdir()

        # Good clipping
        good = clips_dir / "good.md"
        good.write_text(
            "---\nurl: https://example.com/good\n---\n\n"
            + "A " * 150,
            encoding="utf-8",
        )

        # Bad clipping (too short)
        bad = clips_dir / "bad.md"
        bad.write_text(
            "---\nurl: https://example.com/bad\n---\n\nshort",
            encoding="utf-8",
        )

        clipping_entries = _collect_clipping_files(clips_dir)
        retry_urls: list[str] = []
        good_clippings: list[tuple[Path, dict]] = []
        for fp, clipped in clipping_entries:
            raw_text = fp.read_text(encoding="utf-8", errors="replace")
            is_valid, score = _validate_clipping_quality(raw_text)
            if not is_valid or score < 0.5:
                retry_urls.append(clipped["url"])
            else:
                good_clippings.append((fp, clipped))

        assert "https://example.com/bad" in retry_urls
        assert len(good_clippings) == 1
        assert good_clippings[0][1]["url"] == "https://example.com/good"
