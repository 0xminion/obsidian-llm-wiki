"""Tests for obsidian_llm_wiki.ingest.extractors — registry dispatch logic.

These tests verify the dispatch/fallback mechanics, NOT the individual
extractors (YouTube/PDF/DOCX require optional deps not installed in CI).
The registry must gracefully handle missing deps and fall back to web.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import _looks_like_file_path, extract

# ── File path detection ──────────────────────────────────────────────────


def test_looks_like_file_path_real_file(tmp_path: Path):
    """A real file path is detected as a file, not a URL."""
    f = tmp_path / "test.txt"
    f.write_text("hello")
    assert _looks_like_file_path(str(f)) is True


def test_looks_like_file_path_url():
    """URLs are not detected as file paths."""
    assert _looks_like_file_path("https://example.com/page") is False
    assert _looks_like_file_path("http://example.com/page") is False
    assert _looks_like_file_path("ftp://server/file") is False


def test_looks_like_file_path_nonexistent():
    """Non-existent paths without a scheme are NOT file paths."""
    assert _looks_like_file_path("/nonexistent/path/to/file.md") is False


# ── Plain text / markdown file extraction ──────────────────────────────────


def test_extract_txt_file(tmp_path: Path):
    """A .txt file is read as plain text."""
    f = tmp_path / "notes.txt"
    f.write_text("This is a plain text file with some content.\nLine two.")
    source = extract(str(f))
    assert source.title == "notes"
    assert "This is a plain text file" in source.content
    assert "Line two." in source.content


def test_extract_md_file(tmp_path: Path):
    """A .md file is read as markdown text."""
    f = tmp_path / "article.md"
    f.write_text("# My Article\n\nBody paragraph here.")
    source = extract(str(f))
    assert source.title == "article"
    assert "# My Article" in source.content
    assert "Body paragraph here." in source.content


# ── Unknown file extension ─────────────────────────────────────────────────


def test_extract_unknown_extension_raises(tmp_path: Path):
    """An unknown file extension raises RuntimeError."""
    f = tmp_path / "data.xyz"
    f.write_text("some data")
    with pytest.raises(RuntimeError, match="No extractor available"):
        extract(str(f))


# ── URL dispatch ───────────────────────────────────────────────────────────


def test_extract_unknown_url_falls_back_to_web():
    """Unknown URLs fall back to extract_web."""
    fake_source = SourceDoc(
        title="Web Page",
        content="Web content here that is long enough.",
        url="https://example.com/article",
    )
    with patch(
        "obsidian_llm_wiki.ingest.extractors.extract_web",
        return_value=fake_source,
    ):
        result = extract("https://example.com/article")
    assert result.title == "Web Page"
    assert "Web content" in result.content


def test_extract_youtube_url_routes_to_youtube_extractor():
    """YouTube URLs route to the YouTube extractor when TRANSCRIPT_API_KEY is set."""
    fake_source = SourceDoc(
        title="Test Video",
        content="Transcript content here that is long enough for quality gates.",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    from obsidian_llm_wiki.ingest import extractors as reg

    # Patch the function object stored in the registry (not the module namespace)
    original_extractors = list(reg._EXTRACTORS)
    for i, (matcher, fn) in enumerate(reg._EXTRACTORS):
        if fn.__name__ == "extract_youtube_video":
            reg._EXTRACTORS[i] = (matcher, lambda url: fake_source)
            break
    try:
        result = extract("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result.title == "Test Video"
    finally:
        reg._EXTRACTORS[:] = original_extractors


def test_extract_youtube_url_raises_when_no_api_key():
    """YouTube extractor raises RuntimeError when TRANSCRIPT_API_KEY is not set.

    Tests the extractor function in isolation with a mocked API key state,
    bypassing the test subprocess environment inheritance issue.
    """
    from obsidian_llm_wiki.ingest.extractors import youtube as yt_mod

    # Save and nullify the module-level key
    original_key = yt_mod._TRANSCRIPT_API_KEY
    yt_mod._TRANSCRIPT_API_KEY = None
    try:
        yt_mod.extract_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        pytest.fail("Expected RuntimeError but extract_youtube_video succeeded")
    except RuntimeError as exc:
        assert "TRANSCRIPT_API_KEY" in str(exc)
    finally:
        yt_mod._TRANSCRIPT_API_KEY = original_key


# ── Video ID extraction ─────────────────────────────────────────────────────


def test_extract_video_id_standard_url():
    """Standard YouTube watch URL → video ID."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _video_id

    assert _video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_short_url():
    """Short youtu.be URL → video ID."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _video_id

    assert _video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_embed_url():
    """Embed URL → video ID."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _video_id

    assert _video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_bare_id():
    """Bare 11-char ID is not a URL — returns None (correct new behavior).

    Bare IDs without a YouTube hostname are not routable to the YouTube
    extractor, so _video_id correctly returns None for them.
    """
    from obsidian_llm_wiki.ingest.extractors.youtube import _video_id

    assert _video_id("dQw4w9WgXcQ") is None


def test_extract_video_id_invalid():
    """Invalid URL returns None."""
    from obsidian_llm_wiki.ingest.extractors.youtube import _video_id

    assert _video_id("https://example.com/not-youtube") is None
    assert _video_id("short") is None
