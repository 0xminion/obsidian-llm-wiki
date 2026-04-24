"""Tests for pipeline/extract.py — Stage 1 extraction module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import Config
from pipeline.extract import (
    detect_source_type,
    extract_title,
    extract_url,
    extract_all,
    _extract_youtube_video_id,
    _extract_arxiv_paper_id,
    _extract_web,
    _extract_web_content,
    _extract_youtube,
    _extract_podcast,
    _try_defuddle,
    _try_defuddle_json,
    _try_curl_extract,
    _curl_get,
    _curl_post_json,
    _run,
    ExtractionError,
)
from pipeline.models import ExtractedSource, SourceType


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Create a Config pointing to a tmp vault with extract dir."""
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    return Config(vault_path=tmp_path, extract_dir=extract_dir)


# ─── detect_source_type ──────────────────────────────────────────────────────

class TestDetectSourceType:
    def test_youtube_com(self):
        assert detect_source_type("https://www.youtube.com/watch?v=abc12345678") == SourceType.YOUTUBE

    def test_youtu_be(self):
        assert detect_source_type("https://youtu.be/abc12345678") == SourceType.YOUTUBE

    def test_youtube_shorts(self):
        assert detect_source_type("https://www.youtube.com/shorts/abc12345678") == SourceType.YOUTUBE

    def test_youtube_nocookie(self):
        assert detect_source_type("https://www.youtube-nocookie.com/watch?v=abc12345678") == SourceType.YOUTUBE

    def test_apple_podcasts(self):
        assert detect_source_type("https://podcasts.apple.com/us/podcast/name/id123456") == SourceType.PODCAST

    def test_spotify_show(self):
        assert detect_source_type("https://open.spotify.com/show/abc123") == SourceType.PODCAST

    def test_spotify_episode(self):
        assert detect_source_type("https://spotify.com/episode/abc123") == SourceType.PODCAST

    def test_feed_url(self):
        assert detect_source_type("https://feeds.example.com/podcast.xml") == SourceType.PODCAST

    def test_podbean(self):
        assert detect_source_type("https://example.podbean.com/episode1") == SourceType.PODCAST

    def test_anchor_fm(self):
        assert detect_source_type("https://anchor.fm/show/episode1") == SourceType.PODCAST

    def test_generic_web(self):
        assert detect_source_type("https://example.com/article") == SourceType.WEB

    def test_arxiv_is_web(self):
        assert detect_source_type("https://arxiv.org/abs/2503.03312") == SourceType.WEB

    def test_blog_is_web(self):
        assert detect_source_type("https://blog.example.com/post") == SourceType.WEB

    def test_twitter_is_twitter(self):
        assert detect_source_type("https://x.com/user/status/123") == SourceType.TWITTER

    def test_twitter_com_is_twitter(self):
        assert detect_source_type("https://twitter.com/user/status/123") == SourceType.TWITTER


# ─── extract_title ───────────────────────────────────────────────────────────

class TestExtractTitle:
    def test_heading(self):
        content = "# My Great Article\n\nSome body text here."
        assert extract_title(content) == "My Great Article"

    def test_skips_original_content(self):
        content = "# Original content from the page\n\nActual body."
        result = extract_title(content)
        assert result != "Original content from the page"

    def test_skips_original_content_heading(self):
        content = "# Original content\n\nBody text."
        result = extract_title(content)
        assert result != "Original content"

    def test_second_heading_if_first_is_original(self):
        content = "# Original content from site\n\n## Real Title\n\nBody."
        result = extract_title(content)
        # Should find heading, but "Real Title" is ## not #
        # The function only looks for # (h1), so it should fall through to body
        assert result != "Original content from site"

    def test_fallback_to_body_line(self):
        content = "Some introductory text without a heading.\n\nThis is a meaningful paragraph about the topic at hand."
        result = extract_title(content)
        assert len(result) > 0

    def test_truncates_to_120(self):
        long_title = "# " + "A" * 200
        result = extract_title(long_title)
        assert len(result) <= 120

    def test_empty_content(self):
        assert extract_title("") == ""

    def test_none_like(self):
        assert extract_title("") == ""

    def test_skips_urls_and_images(self):
        content = "https://example.com/image.png\n![](img.png)\n# Real Title Here"
        assert extract_title(content) == "Real Title Here"

    def test_multiline_heading(self):
        content = "# The Title of the Article\n\nBody starts here."
        assert extract_title(content) == "The Title of the Article"

    def test_heading_with_markdown(self):
        content = "# **Bold** Title Here\n\nBody."
        assert extract_title(content) == "Bold Title Here"

    def test_chinese_title(self):
        content = "# 深度学习入门指南\n\n正文内容。"
        assert extract_title(content) == "深度学习入门指南"


# ─── _extract_youtube_video_id ───────────────────────────────────────────────

class TestExtractYoutubeVideoId:
    def test_watch_url(self):
        assert _extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert _extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_params(self):
        assert _extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30") == "dQw4w9WgXcQ"

    def test_empty_string(self):
        assert _extract_youtube_video_id("") == ""


# ─── _extract_arxiv_paper_id ────────────────────────────────────────────────

class TestExtractArxivPaperId:
    def test_abs_url(self):
        assert _extract_arxiv_paper_id("https://arxiv.org/abs/2503.03312") == "2503.03312"

    def test_pdf_url(self):
        assert _extract_arxiv_paper_id("https://arxiv.org/pdf/2503.03312") == "2503.03312"

    def test_html_url(self):
        assert _extract_arxiv_paper_id("https://arxiv.org/html/2503.03312v1") == "2503.03312"

    def test_long_id(self):
        # Regex matches \d{4}\.\d{4,5} — 5 digits max after dot
        assert _extract_arxiv_paper_id("https://arxiv.org/abs/2503.033125") == "2503.03312"

    def test_no_match(self):
        assert _extract_arxiv_paper_id("https://example.com/article") == ""


# ─── _run helper ──────────────────────────────────────────────────────────────

class TestRunHelper:
    def test_basic_echo(self):
        result = _run(["echo", "hello"], timeout=5)
        assert result.stdout.strip() == "hello"
        assert result.returncode == 0

    def test_timeout(self):
        with pytest.raises(subprocess.TimeoutExpired):
            _run(["sleep", "10"], timeout=1)

    def test_input_data(self):
        result = _run(["cat"], timeout=5, input_data="test input")
        assert result.stdout == "test input"


# ─── _try_defuddle ───────────────────────────────────────────────────────────

class TestTryDefuddle:
    @patch("pipeline.extractors.web._run")
    @patch("pipeline.extractors.web.Path.read_text")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("os.unlink")
    def test_success(self, mock_unlink, mock_getsize, mock_exists,
                     mock_tmpfile, mock_read_text, mock_run):
        mock_tmpfile.return_value.__enter__.return_value.name = "/tmp/test.md"
        mock_run.return_value = MagicMock(returncode=0)
        mock_exists.return_value = True
        mock_getsize.return_value = 1000
        mock_read_text.return_value = "# Title\n\nContent here."

        result = _try_defuddle("https://example.com", timeout=30)
        assert result == "# Title\n\nContent here."

    @patch("pipeline.extractors.web._run")
    def test_defuddle_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("defuddle not found")
        result = _try_defuddle("https://example.com", timeout=30)
        assert result == ""

    @patch("pipeline.extractors.web._run")
    def test_defuddle_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(["defuddle"], 30)
        result = _try_defuddle("https://example.com", timeout=30)
        assert result == ""

    @patch("pipeline.extractors.web._run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_defuddle_fails(self, mock_unlink, mock_tmpfile, mock_run):
        mock_tmpfile.return_value.__enter__.return_value.name = "/tmp/test.md"
        mock_run.return_value = MagicMock(returncode=1)
        result = _try_defuddle("https://example.com", timeout=30)
        assert result == ""


# ─── _try_defuddle_json ─────────────────────────────────────────────────────

class TestTryDefuddleJson:
    @patch("os.path.getsize", return_value=100)
    @patch("pipeline.extractors.web._run")
    def test_success(self, mock_run, _mock_getsize):
        content_text = "Extracted content from defuddle JSON mode. " * 10
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout=json.dumps({"content": content_text})),
        ]
        result = _try_defuddle_json("https://example.com", timeout=30)
        assert "Extracted content" in result

    @patch("os.path.getsize", return_value=100)
    @patch("pipeline.extractors.web._run")
    def test_empty_content(self, mock_run, _mock_getsize):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout='{"content": "short"}'),
        ]
        result = _try_defuddle_json("https://example.com", timeout=30)
        assert result == ""

    @patch("os.path.getsize", return_value=100)
    @patch("pipeline.extractors.web._run")
    def test_invalid_json(self, mock_run, _mock_getsize):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="not json"),
        ]
        result = _try_defuddle_json("https://example.com", timeout=30)
        assert result == ""

    @patch("pipeline.extractors.web._run")
    def test_defuddle_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        result = _try_defuddle_json("https://example.com", timeout=30)
        assert result == ""


# ─── _try_curl_extract ──────────────────────────────────────────────────────

class TestTryCurlExtract:
    @patch("pipeline.extractors.web._run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("os.unlink")
    def test_success(self, mock_unlink, mock_getsize, mock_exists,
                     mock_tmpfile, mock_run):
        mock_tmpfile.return_value.__enter__.return_value.name = "/tmp/test.html"
        # First call is curl, second is liteparse
        mock_run.side_effect = [
            MagicMock(returncode=0),  # curl
            MagicMock(returncode=0, stdout="Parsed content from liteparse."),  # liteparse
        ]
        mock_exists.return_value = True
        mock_getsize.return_value = 5000

        result = _try_curl_extract("https://example.com", timeout=30)
        assert "Parsed content" in result

    @patch("pipeline.extractors.web._run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_curl_fails(self, mock_unlink, mock_tmpfile, mock_run):
        mock_tmpfile.return_value.__enter__.return_value.name = "/tmp/test.html"
        mock_run.return_value = MagicMock(returncode=1)
        result = _try_curl_extract("https://example.com", timeout=30)
        assert result == ""


# ─── _extract_web ────────────────────────────────────────────────────────────

class TestExtractWeb:
    @patch("pipeline.extractors.web._extract_web_content")
    def test_returns_extracted_source(self, mock_content, cfg: Config):
        mock_content.return_value = "# Test Article\n\nThis is the body of the article with enough content."
        source = _extract_web("https://example.com/article", cfg)
        assert source.url == "https://example.com/article"
        assert source.title == "Test Article"
        assert source.type == SourceType.WEB
        assert "body of the article" in source.content

    @patch("pipeline.extractors.web._extract_web_content")
    def test_fallback_on_failure(self, mock_content, cfg: Config):
        mock_content.return_value = ""
        source = _extract_web("https://example.com/article", cfg)
        assert "extraction failed" in source.content.lower()
        assert source.type == SourceType.WEB


# ─── _extract_web_content ───────────────────────────────────────────────────

class TestExtractWebContent:
    @patch("pipeline.extractors.web._try_defuddle")
    def test_defuddle_succeeds(self, mock_defuddle):
        mock_defuddle.return_value = "# Title\n\n" + "A" * 300
        result = _extract_web_content("https://example.com", timeout=30)
        assert len(result) > 200

    @patch("pipeline.extractors.web._try_defuddle_json")
    @patch("pipeline.extractors.web._try_curl_extract")
    @patch("pipeline.extractors.web._try_defuddle")
    def test_fallback_chain(self, mock_defuddle, mock_curl, mock_json):
        mock_defuddle.return_value = "short"
        mock_curl.return_value = "also short"
        mock_json.return_value = "final content from json mode. " * 20
        result = _extract_web_content("https://example.com", timeout=30)
        assert "final content" in result

    @patch("pipeline.extractors.web._curl_get")
    @patch("pipeline.extractors.web._try_defuddle")
    def test_arxiv_html_succeeds(self, mock_defuddle, mock_curl):
        # arxiv HTML via defuddle succeeds
        mock_defuddle.return_value = "# Paper Title\n\n" + "A" * 600
        result = _extract_web_content("https://arxiv.org/abs/2503.03312", timeout=30)
        assert len(result) > 500

    @patch("pipeline.extractors.web._curl_get")
    @patch("pipeline.extractors.web._try_defuddle")
    def test_arxiv_alphaxiv_fallback(self, mock_defuddle, mock_curl):
        # defuddle fails, alphaxiv succeeds
        mock_defuddle.return_value = ""
        mock_curl.return_value = "# Paper Title\n\nFull paper text from alphaxiv. " + "A" * 600
        result = _extract_web_content("https://arxiv.org/abs/2503.03312", timeout=30)
        assert len(result) > 500


# ─── _extract_youtube ───────────────────────────────────────────────────────

class TestExtractYoutube:
    @patch("pipeline.extractors.youtube._try_youtube_transcript")
    @patch("pipeline.extractors.youtube._curl_get")
    def test_with_transcript(self, mock_curl, mock_transcript, cfg: Config):
        mock_curl.return_value = json.dumps({
            "title": "Test Video",
            "author_name": "Test Author",
        })
        mock_transcript.return_value = "This is the full transcript of the video. " * 5

        source = _extract_youtube("https://www.youtube.com/watch?v=abc12345678", cfg)
        assert source.title == "Test Video"
        assert source.author == "Test Author"
        assert source.type == SourceType.YOUTUBE
        assert "full transcript" in source.content

    @patch("pipeline.extractors.youtube._try_youtube_transcript")
    @patch("pipeline.extractors.youtube._curl_get")
    def test_without_transcript_raises_error(self, mock_curl, mock_transcript, cfg: Config):
        mock_curl.return_value = json.dumps({
            "title": "Test Video",
            "author_name": "Test Author",
        })
        mock_transcript.return_value = ""

        with pytest.raises(ExtractionError, match="transcript extraction failed"):
            _extract_youtube("https://www.youtube.com/watch?v=abc12345678", cfg)

    @patch("pipeline.extractors.youtube._try_youtube_transcript")
    @patch("pipeline.extractors.youtube._curl_get")
    def test_metadata_failure_with_no_transcript(self, mock_curl, mock_transcript, cfg: Config):
        mock_curl.return_value = "invalid json"
        mock_transcript.return_value = ""

        with pytest.raises(ExtractionError, match="transcript extraction failed"):
            _extract_youtube("https://www.youtube.com/watch?v=abc12345678", cfg)


# ─── _extract_podcast ────────────────────────────────────────────────────────

class TestExtractPodcast:
    @patch("pipeline.extractors.podcast._transcribe_podcast_audio")
    @patch("pipeline.extractors.podcast._parse_rss_episode")
    @patch("pipeline.extractors.podcast._curl_get")
    def test_with_transcript(self, mock_curl, mock_rss, mock_transcribe, cfg: Config):
        # iTunes lookup response
        mock_curl.return_value = json.dumps({
            "results": [{"feedUrl": "https://feeds.example.com/show.xml",
                         "collectionName": "Test Podcast"}]
        })
        mock_rss.return_value = ("https://cdn.example.com/ep1.mp3",
                                  "Episode description",
                                  "Episode 1: Introduction")
        mock_transcribe.return_value = "This is the podcast transcript. " * 10

        source = _extract_podcast(
            "https://podcasts.apple.com/us/podcast/test/id123456?i=789",
            cfg,
        )
        assert source.type == SourceType.PODCAST
        assert "transcript" in source.content.lower()
        assert "Test Podcast" in source.content

    @patch("pipeline.extractors.podcast._transcribe_podcast_audio")
    @patch("pipeline.extractors.podcast._parse_rss_episode")
    @patch("pipeline.extractors.podcast._curl_get")
    def test_transcription_failure_raises_error(self, mock_curl, mock_rss, mock_transcribe, cfg: Config):
        mock_curl.return_value = json.dumps({
            "results": [{"feedUrl": "https://feeds.example.com/show.xml",
                         "collectionName": "Test Podcast"}]
        })
        mock_rss.return_value = ("https://cdn.example.com/ep1.mp3",
                                  "This is the episode description. " * 10,
                                  "Episode 1")
        mock_transcribe.return_value = ""  # transcription fails

        with pytest.raises(ExtractionError, match="transcription failed"):
            _extract_podcast(
                "https://podcasts.apple.com/us/podcast/test/id123456?i=789",
                cfg,
            )

    @patch("pipeline.extractors.podcast._curl_get")
    def test_no_feed_url_raises_error(self, mock_curl, cfg: Config):
        mock_curl.return_value = json.dumps({"results": []})

        with pytest.raises(ExtractionError, match="no audio URL found"):
            _extract_podcast(
                "https://podcasts.apple.com/us/podcast/test/id123456?i=789",
                cfg,
            )


# ─── extract_url ─────────────────────────────────────────────────────────────

class TestExtractUrl:
    @patch("pipeline.extract._extract_youtube")
    def test_routes_youtube(self, mock_yt, cfg: Config):
        mock_yt.return_value = ExtractedSource(
            url="https://youtube.com/watch?v=abc", title="Video",
            content="transcript", type=SourceType.YOUTUBE,
        )
        source = extract_url("https://www.youtube.com/watch?v=abc", cfg)
        assert source.type == SourceType.YOUTUBE
        mock_yt.assert_called_once()

    @patch("pipeline.extract._extract_podcast")
    def test_routes_podcast(self, mock_pod, cfg: Config):
        mock_pod.return_value = ExtractedSource(
            url="https://podcasts.apple.com/us/p/test/id1",
            title="Episode",
            content="Podcast description with sufficient content for validation",
            type=SourceType.PODCAST,
        )
        source = extract_url("https://podcasts.apple.com/us/p/test/id1", cfg)
        assert source.type == SourceType.PODCAST
        mock_pod.assert_called_once()

    @patch("pipeline.extract._extract_web")
    def test_routes_web(self, mock_web, cfg: Config):
        mock_web.return_value = ExtractedSource(
            url="https://example.com", title="Article",
            content="Article body content with enough text for validation",
            type=SourceType.WEB,
        )
        source = extract_url("https://example.com/article", cfg)
        assert source.type == SourceType.WEB
        mock_web.assert_called_once()

    @patch("pipeline.extract._extract_youtube")
    def test_saves_json(self, mock_yt, cfg: Config):
        mock_yt.return_value = ExtractedSource(
            url="https://youtube.com/watch?v=testhash",
            title="Video", content="transcript", type=SourceType.YOUTUBE,
        )
        source = extract_url("https://youtube.com/watch?v=testhash", cfg)
        json_path = cfg.resolved_extract_dir / f"{source.hash}.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["url"] == "https://youtube.com/watch?v=testhash"
        assert data["type"] == "youtube"

    @patch("pipeline.extract._extract_web")
    def test_raises_after_retries_exhausted(self, mock_web, cfg: Config):
        mock_web.side_effect = RuntimeError("something broke")
        with pytest.raises(ExtractionError, match="Extraction failed after"):
            extract_url("https://example.com", cfg)


# ─── extract_all ─────────────────────────────────────────────────────────────

class TestExtractAll:
    @patch("pipeline.extract.extract_url")
    def test_empty_list(self, mock_extract, cfg: Config):
        manifest = extract_all([], cfg, parallel=2)
        assert len(manifest.entries) == 0

    @patch("pipeline.extract.extract_url")
    def test_creates_manifest(self, mock_extract, cfg: Config):
        mock_extract.side_effect = [
            ExtractedSource(url="https://a.com", title="A", content="a" * 100, type=SourceType.WEB),
            ExtractedSource(url="https://b.com", title="B", content="b" * 100, type=SourceType.WEB),
        ]
        manifest = extract_all(["https://a.com", "https://b.com"], cfg, parallel=2)
        assert len(manifest.entries) == 2
        assert manifest.entries[0].title == "A"
        assert manifest.entries[1].title == "B"

        # Check manifest.json was written
        manifest_path = cfg.resolved_extract_dir / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert len(data) == 2

    @patch("pipeline.extract.extract_url")
    def test_handles_failures(self, mock_extract, cfg: Config):
        mock_extract.side_effect = [
            ExtractedSource(url="https://a.com", title="A", content="a" * 100, type=SourceType.WEB),
            RuntimeError("failed"),
        ]
        manifest = extract_all(["https://a.com", "https://b.com"], cfg, parallel=2)
        assert len(manifest.entries) == 1
        assert manifest.entries[0].title == "A"

    @patch("pipeline.extract._extract_web")
    def test_failed_extraction_is_not_written_to_manifest(self, mock_web, cfg: Config):
        mock_web.side_effect = RuntimeError("broken extractor")

        with pytest.raises(ExtractionError, match="all extractions failed"):
            extract_all(["https://example.com/article"], cfg, parallel=1)

        manifest_path = cfg.resolved_extract_dir / "manifest.json"
        assert not manifest_path.exists()

    @patch("pipeline.extract.extract_url")
    def test_parallel_execution(self, mock_extract, cfg: Config):
        """Test that parallel parameter is accepted and doesn't break."""
        mock_extract.return_value = ExtractedSource(
            url="https://example.com", title="Test",
            content="content", type=SourceType.WEB,
        )
        urls = [f"https://example.com/{i}" for i in range(4)]
        manifest = extract_all(urls, cfg, parallel=4)
        assert len(manifest.entries) == 4


# ─── _curl_get / _curl_post_json ────────────────────────────────────────────

class TestCurlHelpers:
    @patch("pipeline.extractors._shared._validate_url", return_value=True)
    @patch("pipeline.extractors._shared._run")
    def test_curl_get(self, mock_run, _mock_validate):
        mock_run.return_value = MagicMock(stdout='{"key": "value"}', returncode=0)
        result = _curl_get("https://api.example.com/data", timeout=10)
        assert result == '{"key": "value"}'
        args = mock_run.call_args[0][0]
        assert "curl" in args
        assert "-s" in args
        assert "-sL" not in args
        assert "--max-redirs" in args

    @patch("pipeline.extractors._shared._validate_url", return_value=True)
    @patch("pipeline.extractors._shared._run")
    def test_curl_get_with_headers(self, mock_run, _mock_validate):
        mock_run.return_value = MagicMock(stdout="response", returncode=0)
        _curl_get("https://api.example.com", headers={"Authorization": "Bearer key"}, timeout=10)
        args = mock_run.call_args[0][0]
        assert "-H" in args
        assert "Authorization: Bearer key" in args

    @patch("pipeline.extractors._shared._validate_url", return_value=True)
    @patch("pipeline.extractors._shared._run")
    def test_curl_post_json(self, mock_run, _mock_validate):
        mock_run.return_value = MagicMock(stdout="ok", returncode=0)
        result = _curl_post_json(
            "https://api.example.com/submit",
            data={"key": "value"},
            headers={"x-api-key": "secret"},
            timeout=10,
        )
        assert result == "ok"
        args = mock_run.call_args[0][0]
        assert "-X" in args
        assert "POST" in args
        assert "-d" in args
