"""Tests for gibberish title detection and PDF title validation."""

import pytest

from obsidian_llm_wiki.core.text_quality import is_gibberish_title


class TestIsGibberishTitle:
    """Test the repeated-token gibberish title detector."""

    @pytest.mark.parametrize(
        "title",
        [
            "8n-8n-8n-8n-8n-8n-8-n",
            "ai-ai-ai-ai-ai-ai-ai-ai",
            "2025eth已死-2025eth已死-2025eth已死-2025eth已死",
            "2025tge-2025tge-2025tge-2025tge-2025tge-2025tge-2025tge-2025-tge",
            "test-test-test-test-test",
            "x-x-x-x-x-x-x",
        ],
    )
    def test_gibberish_titles_detected(self, title: str):
        assert is_gibberish_title(title) is True, f"Should detect '{title}' as gibberish"

    @pytest.mark.parametrize(
        "title",
        [
            "Prediction Markets Explained",
            "The Game Theory Behind Prediction Markets",
            "Decomposing Crowd Wisdom: Domain-Specific Calibration",
            "Polymarket: An Election-Driven Success Story",
            "From prediction markets to info finance",
            "Should Prediction Markets Be Charities?",
            "HBM 是什么？是金子做的吗？",  # Chinese title — not gibberish
            "2025 TGE 镰刀算法",  # Chinese with year — not gibberish
            "A Short Title",
            "两条线",
        ],
    )
    def test_real_titles_pass(self, title: str):
        assert is_gibberish_title(title) is False, f"Should accept '{title}' as real"

    def test_empty_title_is_gibberish(self):
        assert is_gibberish_title("") is True
        assert is_gibberish_title(None) is True  # type: ignore[arg-type]

    def test_short_title_not_gibberish(self):
        """Titles with < 3 tokens are not checked for repetition."""
        assert is_gibberish_title("hello") is False
        assert is_gibberish_title("two-words") is False


class TestPdfTitleValidation:
    """Test the PDF title plausibility checker."""

    @pytest.mark.parametrize(
        "title",
        [
            "35589",  # NBER paper number
            "2606.16852",  # arXiv ID
            "",  # empty
            "ab",  # too short
            "12345",  # pure number
        ],
    )
    def test_implausible_titles_rejected(self, title: str):
        from obsidian_llm_wiki.ingest.extractors.pdf import _is_plausible_pdf_title

        assert _is_plausible_pdf_title(title) is False

    @pytest.mark.parametrize(
        "title",
        [
            "Prediction Markets: Does Money Matter?",
            "DoesMoneyMatter",  # URL stem fallback
            "Self-Resolving Prediction Markets",
            "A Technical Typology",
        ],
    )
    def test_plausible_titles_accepted(self, title: str):
        from obsidian_llm_wiki.ingest.extractors.pdf import _is_plausible_pdf_title

        assert _is_plausible_pdf_title(title) is True


class TestExtractionQualityGateGibberish:
    """Test that the extraction quality gate rejects gibberish titles."""

    def test_gibberish_title_rejected_by_quality_gate(self):
        from obsidian_llm_wiki.core.models import SourceDoc
        from obsidian_llm_wiki.ingest.extractors import _check_extraction_quality

        source = SourceDoc(
            title="8n-8n-8n-8n-8n-8n-8-n",
            content="A" * 1000,  # long enough to pass content checks
            url="https://x.com/test/status/123",
        )
        passed, reason = _check_extraction_quality(source)
        assert passed is False
        assert "gibberish" in reason

    def test_normal_title_passes_quality_gate(self):
        from obsidian_llm_wiki.core.models import SourceDoc
        from obsidian_llm_wiki.ingest.extractors import _check_extraction_quality

        source = SourceDoc(
            title="Prediction Markets Explained",
            content="A" * 1000,
            url="https://example.com/article",
        )
        passed, reason = _check_extraction_quality(source)
        assert passed is True
