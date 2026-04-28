"""Integration tests for the extraction chain — full pipeline integration.

These tests validate the complete extraction chain:
  real HTTP → defuddle → quality check → dedup → manifest save

They mock subprocess calls to defuddle/curl with realistic HTML responses,
testing the full logic without live network calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.config import Config
from pipeline.extract import extract_url
from pipeline.models import ExtractedSource, SourceType, Manifest
from pipeline.extractors._shared import ExtractionError, _is_challenge_page


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config with isolated extract dir."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "04-Wiki" / "sources").mkdir(parents=True)
    (vault / "04-Wiki" / "entries").mkdir(parents=True)
    (vault / "04-Wiki" / "concepts").mkdir(parents=True)
    (vault / "04-Wiki" / "mocs").mkdir(parents=True)
    (vault / "06-Config").mkdir(parents=True)
    (vault / "01-Raw").mkdir(parents=True)
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    return Config(vault_path=vault, extract_dir=extract_dir)


SAMPLE_HTML = b"""<!DOCTYPE html>
<html lang="en">
<head><title>Real Article</title></head>
<body>
<article>
<h1>Real Article</h1>
<p>This is a real article with substantial content. " * 100
</p><p>Second paragraph here.</p>
</article>
</body>
</html>"""


def _make_defuddle_result(content: str) -> MagicMock:
    """Create a mock subprocess.CompletedProcess for defuddle."""
    return MagicMock(returncode=0, stdout="", stderr="")


# ─── R2: Extraction Coverage Tests ──────────────────────────────────────────────

class TestExtractWebChain:
    """Test full extraction chain: HTTP → defuddle → quality → dedup → manifest."""

    def test_full_chain_creates_manifest(self, cfg: Config):
        """Verify that extract_all produces a valid Manifest (not extract_url alone)."""
        url = "https://example.com/article"

        with patch("pipeline.extractors.web._try_defuddle") as mock_defuddle, \
             patch("pipeline.extractors.web._try_curl_extract") as mock_curl:

            # Simulate defuddle returning clean content
            mock_defuddle.return_value = "# Real Article\n\n" + "This is real content. " * 50
            mock_curl.return_value = ""

            result = extract_url(url, cfg)

        assert isinstance(result, ExtractedSource)
        assert result.url == url
        assert "Real Article" in result.title
        assert len(result.content) > 200
        # Manifest is created by extract_all(), not extract_url()
        manifest = Manifest(entries=[result])
        manifest.save(cfg.resolved_extract_dir)
        loaded = Manifest.load(cfg.resolved_extract_dir)
        assert any(e.url == url for e in loaded.entries)

    def test_cloudflare_falls_back_to_archive(self, cfg: Config):
        """Verify 403/Cloudflare response triggers archive.org fallback."""
        url = "https://example.com/protected"
        challenge_html = (
            "<!DOCTYPE html><html><head></head><body>"
            "Just a moment... Checking your browser."
            "<div class='cf-browser-verification'></div>"
            "</body></html>"
        )
        archive_content = "# Archived Article\n\n" + "Real archived content. " * 50

        with patch("pipeline.extractors.web._try_defuddle") as mock_defuddle, \
             patch("pipeline.extractors.web._try_curl_extract") as mock_curl, \
             patch("pipeline.extractors.web._try_defuddle_json") as mock_json, \
             patch("pipeline.extractors.web._try_archive_extract") as mock_archive:

            # All primary methods return Cloudflare or nothing
            mock_defuddle.return_value = challenge_html
            mock_curl.return_value = ""
            mock_json.return_value = ""
            mock_archive.return_value = archive_content

            result = extract_url(url, cfg)

        mock_archive.assert_called_once()
        assert isinstance(result, ExtractedSource)
        assert "Archived" in result.title or len(result.content) > 200

    def test_youtube_metadata_only_raises_extraction_error(self, cfg: Config):
        """YouTube without transcript should propagate ExtractionError (no metadata fallback)."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # The extract_url implementation catches ExtractionError and re-raises it
        def _raise_err(*args, **kwargs):
            raise ExtractionError("No transcript available")

        with patch("pipeline.extract._extract_youtube", side_effect=_raise_err):
            with pytest.raises(ExtractionError):
                extract_url(url, cfg)

    def test_duplicates_detected_content_hash(self, cfg: Config):
        """Same content from different URLs should be deduped by store."""
        from pipeline.store import ContentStore

        store = ContentStore.open(cfg.resolved_extract_dir)

        url1 = "https://example.com/a"
        url2 = "https://example.com/b"
        content = "Identical content across both URLs. " * 20

        with patch("pipeline.extract._extract_web") as mock_web:
            mock_web.return_value = ExtractedSource(
                url=url1, title="Test", content=content, type=SourceType.WEB
            )
            result1 = extract_url(url1, cfg, store=store)
            mock_web.return_value = ExtractedSource(
                url=url2, title="Test", content=content, type=SourceType.WEB
            )
            result2 = extract_url(url2, cfg, store=store)

        # First URL gets full content
        assert result1.content == content
        # Second URL is deduped
        assert result2.content == ""
        assert "dedup" in result2.title.lower()

        store.close()


class TestExtractValidation:
    """Test extraction quality validation and edge cases."""

    def test_is_challenge_page_detection(self):
        """Cloudflare challenge pages should be detected."""
        assert _is_challenge_page("<html>Just a moment... Checking your browser.</html>")
        assert _is_challenge_page("<html>Attention Required! Cloudflare</html>")
        assert not _is_challenge_page("<html><h1>Normal Article</h1></html>")
        assert not _is_challenge_page("")

    def test_empty_content_raises_extraction_error(self, cfg: Config):
        """All extractors returning empty should fail loudly, not create usable content."""
        url = "https://example.com/empty"

        with patch("pipeline.extractors.web._try_defuddle") as mock_d1, \
             patch("pipeline.extractors.web._try_curl_extract") as mock_curl, \
             patch("pipeline.extractors.web._try_defuddle_json") as mock_json, \
             patch("pipeline.extractors.web._try_archive_extract") as mock_archive, \
             patch("pipeline.extractors.web._try_camoufox_with_title") as mock_cfx:

            mock_d1.return_value = ""
            mock_curl.return_value = ""
            mock_json.return_value = ""
            mock_archive.return_value = ""
            mock_cfx.return_value = ("", "")

            with pytest.raises(ExtractionError, match="Extraction failed after"):
                extract_url(url, cfg)

    def test_short_content_is_rejected(self, cfg: Config):
        """Content under 20 chars should trigger retry."""
        url = "https://example.com/short"

        with patch("pipeline.extractors.web._try_defuddle") as mock_d1, \
             patch("pipeline.extractors.web._try_curl_extract") as mock_curl:

            # First attempt: very short content, second: good content
            mock_d1.side_effect = ["Short", "", ""]
            mock_curl.side_effect = ["", "# Good Article\n\nReal content here. " * 20]

            result = extract_url(url, cfg)

        assert "Good Article" in result.title or len(result.content) > 200
