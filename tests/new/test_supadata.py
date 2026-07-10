"""Regression tests for Supadata YouTube transcript extraction.

Tests the public API (extract_youtube_video) by mocking the HTTP client
at the httpx level so no real network calls are made.
"""
from __future__ import annotations

import os
from unittest import mock

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import youtube


class TestSupadataExtraction:
    """Supadata transcript extraction with fallback chain."""

    def test_supadata_success_returns_transcript(self):
        """Supadata job completes → transcript returned."""
        # Transcript must be >= 200 chars to pass the minimum-length check
        long_transcript = (
            "Hello world this is the transcript. "
            "It contains multiple sentences to ensure we exceed "
            "the 200 character minimum threshold that the parser requires. "
            "This additional text ensures the content is long enough. "
            "Adding more text to make it really really long. "
            "More content here to ensure we pass the check."
        )

        def get_side_effect(url, params=None, headers=None, **kwargs):
            if "abc123" in str(url):
                return mock.Mock(
                    status_code=200,
                    json=lambda: {
                        "status": "completed",
                        "result": {"content": long_transcript, "lang": "en"},
                    },
                    raise_for_status=mock.Mock(),
                )
            return mock.Mock(
                status_code=202,
                json=lambda: {"jobId": "abc123", "status": "pending"},
                raise_for_status=mock.Mock(),
            )

        mock_client = mock.Mock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_client.get.side_effect = get_side_effect

        with mock.patch.object(
            youtube.httpx, "Client", return_value=mock_client
        ):
            result = youtube._supadata_transcript("dQw4w9WgXcQ", "fake-key")

        assert result is not None
        assert "Hello world" in result.content

    def test_oembed_fallback_on_supadata_failure(self):
        """Supadata fails → oEmbed metadata used as last resort."""
        env = {"SUPADATA_API_KEY": "sk_test_fake_key"}

        fake_response = mock.Mock(status_code=401)
        fake_response.raise_for_status.side_effect = youtube.httpx.HTTPStatusError(
            "401", request=mock.Mock(), response=fake_response,
        )

        mock_client = mock.Mock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_client.get.return_value = fake_response

        oembed_result = SourceDoc(
            title="Test Video",
            content="Title: Test Video\nChannel: TestChannel",
            url="https://youtube.com/watch?v=dQw4w9WgXcQ",
        )

        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(youtube.httpx, "Client", return_value=mock_client), \
             mock.patch.object(youtube, "_extract_oembed", return_value=oembed_result):
            result = youtube.extract_youtube_video(
                "https://youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert result is not None
        assert "Test Video" in result.content
