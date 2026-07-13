"""Regression tests for YouTube transcript extraction.

Tests the public API (extract_youtube_video) by mocking yt-dlp and the
AssemblyAI HTTP client so no real network calls are made.
"""
from __future__ import annotations

import os
from unittest import mock

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import youtube


class TestYouTubeExtraction:
    """YouTube transcript extraction with fallback chain."""

    def test_ytdlp_success_returns_transcript(self):
        """yt-dlp subtitle extraction succeeds → transcript returned."""
        long_transcript = (
            "Hello world this is the transcript. "
            "It contains multiple sentences to ensure we exceed "
            "the 200 character minimum threshold that the parser requires. "
            "This additional text ensures the content is long enough. "
            "Adding more text to make it really really long. "
            "More content here to ensure we pass the check."
        )

        with mock.patch.object(youtube.shutil, "which", return_value="/usr/bin/yt-dlp"), \
             mock.patch.object(youtube, "_parse_subtitle_file", return_value=long_transcript), \
             mock.patch.object(youtube, "_fetch_youtube_title", return_value="Test Video"), \
             mock.patch.object(youtube.subprocess, "run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            # Mock tempfile and os.listdir
            with mock.patch.object(youtube.tempfile, "mkdtemp", return_value="/tmp/fake"), \
                 mock.patch.object(youtube.os, "listdir", return_value=["sub.vtt"]):
                result = youtube._ytdlp_transcript(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )

        assert result is not None
        assert "Hello world" in result.content

    def test_oembed_fallback_on_all_failures(self):
        """All transcript methods fail → oEmbed metadata used as last resort."""
        env = {"ASSEMBLYAI_API_KEY": "test-key"}

        oembed_result = SourceDoc(
            title="Test Video",
            content="Title: Test Video\nChannel: TestChannel\n\nNote: Full transcript unavailable "
            "(yt-dlp and AssemblyAI could not extract subtitles). "
            "Only video metadata was extracted.",
            url="https://youtube.com/watch?v=dQw4w9WgXcQ",
        )

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(youtube.shutil, "which", return_value=None), \
             mock.patch.object(
                 youtube, "_assemblyai_transcript",
                 side_effect=RuntimeError("fail"),
             ), \
             mock.patch.object(youtube, "_extract_oembed", return_value=oembed_result):
            result = youtube.extract_youtube_video(
                "https://youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert result is not None
        assert "Test Video" in result.content

    def test_assemblyai_key_not_set_falls_through(self):
        """When AssemblyAI key is not set, falls through to next fallback."""
        oembed_result = SourceDoc(
            title="Fallback Video",
            content="Title: Fallback Video\nChannel: TestChannel\n\n"
            "Note: Full transcript unavailable "
            "(yt-dlp and AssemblyAI could not extract subtitles). "
            "Only video metadata was extracted.",
            url="https://youtube.com/watch?v=test",
        )

        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(youtube.shutil, "which", return_value=None), \
             mock.patch.object(youtube, "_extract_oembed", return_value=oembed_result):
            result = youtube.extract_youtube_video(
                "https://youtube.com/watch?v=test"
            )

        assert result is not None
        assert "Fallback Video" in result.content
