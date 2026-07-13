"""Tests for observability metrics, retry/resume truncation, and Supadata rate limiter.

Covers:
  - MetricsCollector: writing metrics.json, summary computation
  - Retry with truncation: failing source retries with shorter content
  - Supadata rate limiter: calls are delayed
  - Supadata usage tracking: usage file is written
  - Whisper fallback: gracefully handles missing faster_whisper
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest import mock

from obsidian_llm_wiki.core.metrics import (
    MetricsCollector,
    load_metrics,
    print_metrics_summary,
)
from obsidian_llm_wiki.ingest import supadata_utils

# ── Metrics collection tests ───────────────────────────────────────────────


class TestMetricsCollector:
    """Test MetricsCollector writes metrics.json correctly."""

    def test_metrics_json_is_written(self, tmp_path: Path):
        """MetricsCollector.save() writes a valid metrics.json file."""
        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.record_extraction(
            url="https://example.com/article",
            chars_extracted=5000,
            extractor_used="web",
            time_seconds=1.5,
            success=True,
        )
        collector.record_synthesis(
            source_file="article.md",
            pass1_time=10.0,
            concepts_extracted=3,
            success=True,
        )
        collector.record_synthesis(
            source_file="failed.md",
            pass1_time=5.0,
            success=False,
            error_type="no_output",
        )
        collector.record_rendering(
            concepts_rendered=3,
            mocs_rendered=1,
            time_seconds=2.0,
        )
        collector.record_embedding(
            model="nomic-embed-text",
            concepts_embedded=3,
            cross_lingual_matches=1,
            time_seconds=3.0,
        )
        collector.finish_run()
        collector.save()

        # Verify file exists
        metrics_file = tmp_path / "04-Wiki" / ".llmwiki" / "metrics.json"
        assert metrics_file.exists(), "metrics.json was not written"

        # Verify content is valid JSON with expected structure
        data = json.loads(metrics_file.read_text())
        # New format: {"runs": [...], "latest": {...}}
        latest = data.get("latest", data)  # backward compat
        assert "run_id" in latest
        assert "started_at" in latest
        assert "finished_at" in latest
        assert "extractions" in latest
        assert len(latest["extractions"]) == 1
        assert latest["extractions"][0]["chars_extracted"] == 5000
        assert "syntheses" in latest
        assert len(latest["syntheses"]) == 2
        assert "rendering" in latest
        assert latest["rendering"]["concepts_rendered"] == 3
        assert "embedding" in latest
        assert latest["embedding"]["model"] == "nomic-embed-text"
        # Verify history is preserved
        assert "runs" in data
        assert len(data["runs"]) == 1

    def test_metrics_summary_computation(self, tmp_path: Path):
        """Summary stats are computed correctly on finish_run."""
        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.record_synthesis(source_file="a.md", success=True, concepts_extracted=5)
        collector.record_synthesis(source_file="b.md", success=False, error_type="no_output")
        collector.record_synthesis(source_file="c.md", success=True, concepts_extracted=3)
        collector.finish_run()

        summary = collector._metrics.summary
        assert summary["total_syntheses"] == 3
        assert summary["successful_syntheses"] == 2
        assert summary["failed_syntheses"] == 1
        assert summary["total_concepts_extracted"] == 8

    def test_load_metrics_returns_none_when_missing(self, tmp_path: Path):
        """load_metrics returns None when no metrics file exists."""
        assert load_metrics(tmp_path) is None

    def test_load_metrics_returns_data(self, tmp_path: Path):
        """load_metrics returns parsed JSON when metrics file exists."""
        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.finish_run()
        collector.save()

        data = load_metrics(tmp_path)
        assert data is not None
        assert "run_id" in data

    def test_print_metrics_summary_no_file(self, tmp_path: Path, capsys):
        """print_metrics_summary prints message when no metrics found."""
        print_metrics_summary(tmp_path)
        captured = capsys.readouterr()
        assert "No metrics found" in captured.out

    def test_print_metrics_summary_with_data(self, tmp_path: Path, capsys):
        """print_metrics_summary prints stats when metrics exist."""
        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.record_synthesis(
            source_file="test.md", success=True, concepts_extracted=5, pass1_time=10.0,
        )
        collector.finish_run()
        collector.save()

        print_metrics_summary(tmp_path)
        captured = capsys.readouterr()
        assert "📊" in captured.out
        assert "test.md" in captured.out


# ── Retry with truncation tests ────────────────────────────────────────────


class TestRetryWithTruncation:
    """Test that synthesis retries with progressively shorter content."""

    def test_retry_truncates_on_failure(self):
        """When synthesis fails (returns None), retry with shorter content."""
        from obsidian_llm_wiki.config import Config
        from obsidian_llm_wiki.core.models import SourceDoc
        from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

        config = Config()
        source = SourceDoc(
            title="Test Source",
            content="x" * 60_000,  # 60K chars — larger than 50K truncation
            source_file="test.md",
        )

        # Mock _synthesize_source to track content lengths it receives
        call_lengths: list[int] = []

        async def mock_synth(config, filename, src, existing_concepts):
            call_lengths.append(len(src.content))
            # Fail on full and 50K, succeed on 20K
            if len(src.content) > 20_000:
                return None
            from obsidian_llm_wiki.core.models import SourceSynthesis
            return SourceSynthesis(
                source_title="Test",
                source_summary="Summary",
                source_file=filename,
            )

        with mock.patch(
            "obsidian_llm_wiki.core.pipeline._synthesize_source",
            side_effect=mock_synth,
        ):
            result = asyncio.run(
                _synthesize_with_retry(config, "test.md", source, [])
            )

        # Should have tried 3 times: full (60K), 50K, 20K
        assert len(call_lengths) == 3
        assert call_lengths[0] == 60_000  # full
        assert call_lengths[1] == 50_000  # truncated to 50K
        assert call_lengths[2] == 20_000  # truncated to 20K
        # Should succeed on the third attempt
        assert result is not None
        assert result.source_title == "Test"

    def test_all_levels_fail_returns_none(self):
        """When all truncation levels fail, return None."""
        from obsidian_llm_wiki.config import Config
        from obsidian_llm_wiki.core.models import SourceDoc
        from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

        config = Config()
        source = SourceDoc(
            title="Test Source",
            content="x" * 60_000,
            source_file="test.md",
        )

        async def mock_synth_always_none(config, filename, src, existing_concepts):
            return None

        with mock.patch(
            "obsidian_llm_wiki.core.pipeline._synthesize_source",
            side_effect=mock_synth_always_none,
        ):
            result = asyncio.run(
                _synthesize_with_retry(config, "test.md", source, [])
            )

        assert result is None

    def test_short_source_no_truncation(self):
        """Short sources (< 50K) don't get truncated at any level."""
        from obsidian_llm_wiki.config import Config
        from obsidian_llm_wiki.core.models import SourceDoc
        from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

        config = Config()
        source = SourceDoc(
            title="Short Source",
            content="x" * 100,  # Very short
            source_file="short.md",
        )

        call_lengths: list[int] = []

        async def mock_synth(config, filename, src, existing_concepts):
            call_lengths.append(len(src.content))
            from obsidian_llm_wiki.core.models import SourceSynthesis
            return SourceSynthesis(
                source_title="Short", source_summary="S", source_file=filename,
            )

        with mock.patch(
            "obsidian_llm_wiki.core.pipeline._synthesize_source",
            side_effect=mock_synth,
        ):
            result = asyncio.run(
                _synthesize_with_retry(config, "short.md", source, [])
            )

        # Should succeed on first attempt (full content = 100 chars)
        assert len(call_lengths) == 1
        assert call_lengths[0] == 100
        assert result is not None


# ── Supadata rate limiter tests ────────────────────────────────────────────


class TestSupadataRateLimiter:
    """Test that Supadata API calls are rate-limited."""

    def test_rate_limiter_delays_calls(self):
        """Consecutive calls to supadata_rate_limit() are delayed by ~3 seconds."""
        supadata_utils.reset_rate_limiter()

        # First call — no delay (no previous call)
        start1 = time.monotonic()
        supadata_utils.supadata_rate_limit()
        elapsed1 = time.monotonic() - start1
        assert elapsed1 < 0.1, "First call should not be delayed"

        # Second call — should be delayed by ~3 seconds
        start2 = time.monotonic()
        supadata_utils.supadata_rate_limit()
        elapsed2 = time.monotonic() - start2
        assert elapsed2 >= 2.5, (
            f"Second call should be delayed by ~3s, got {elapsed2:.1f}s"
        )

    def test_rate_limiter_no_delay_after_timeout(self):
        """After enough time passes, no delay is applied."""
        supadata_utils.reset_rate_limiter()
        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 0.01  # Very short for testing

        supadata_utils.supadata_rate_limit()
        time.sleep(0.05)  # Wait past the rate limit window

        start = time.monotonic()
        supadata_utils.supadata_rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed < 0.01, "No delay expected after rate limit window passed"

        # Restore default
        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 3.0

    def test_rate_limiter_reset(self):
        """reset_rate_limiter() clears the last call time."""
        supadata_utils.supadata_rate_limit()
        supadata_utils.reset_rate_limiter()

        start = time.monotonic()
        supadata_utils.supadata_rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, "After reset, first call should not be delayed"


class TestSupadataUsageTracking:
    """Test Supadata usage tracking."""

    def test_track_supadata_call_writes_usage(self, tmp_path: Path, monkeypatch):
        """track_supadata_call writes usage data to supadata_usage.json."""
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))

        # Create a mock response
        mock_resp = mock.Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = {}

        supadata_utils.track_supadata_call(mock_resp)

        usage_file = tmp_path / "04-Wiki" / ".llmwiki" / "supadata_usage.json"
        assert usage_file.exists(), "supadata_usage.json was not written"

        data = json.loads(usage_file.read_text())
        assert data["calls_made"] == 1
        assert "date" in data

    def test_track_supadata_429_logs_warning(self, tmp_path: Path, monkeypatch, caplog):
        """429 response triggers a warning log."""
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))

        mock_resp = mock.Mock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_resp.json.return_value = {}

        import logging
        with caplog.at_level(logging.WARNING, logger="obswiki.ingest.supadata_utils"):
            supadata_utils.track_supadata_call(mock_resp)

        assert any("rate limit" in record.message.lower() for record in caplog.records)


# ── Whisper fallback tests ─────────────────────────────────────────────────


class TestWhisperFallback:
    """Test Whisper fallback behavior."""

    def test_whisper_fallback_returns_empty_when_not_installed(self):
        """When faster_whisper is not installed, returns empty string."""
        from obsidian_llm_wiki.ingest.extractors.podcast import _whisper_fallback_transcribe

        # Mock the import to fail
        with mock.patch.dict("sys.modules", {"faster_whisper": None}):
            result = _whisper_fallback_transcribe("https://example.com/audio.mp3")

        # Should return empty string (not raise)
        assert result == ""

    def test_whisper_fallback_in_extract_podcast(self):
        """Whisper fallback is attempted when Supadata fails."""
        from obsidian_llm_wiki.ingest.extractors import podcast

        # Mock all external calls
        with mock.patch.object(
            podcast, "_fetch_defuddle_md_metadata",
            return_value={
                "title": "Test Podcast",
                "description": "A test episode.",
                "author": "Test Host",
                "published": "2024-01-01",
            },
        ), mock.patch.object(
            podcast,
            "_resolve_episode_asset",
            return_value=podcast.EpisodeAsset(
                audio_url="https://example.com/audio.mp3",
            ),
        ), mock.patch.object(
            podcast,
            "load_transcript_cache",
            return_value=None,
        ), mock.patch.object(
            podcast, "_get_supadata_key", return_value="fake-key",
        ), mock.patch.object(
            podcast, "_supadata_transcribe_audio", return_value="",
        ), mock.patch.object(
            podcast, "get_assemblyai_key", return_value="",
        ), mock.patch.object(
            podcast, "_whisper_fallback_transcribe",
            return_value="whisper transcript text",
        ):
            result = podcast._extract_podcast(
                "https://open.spotify.com/episode/test123",
                platform="spotify",
            )

        # Should contain the whisper transcript
        assert "whisper transcript text" in result.content


# ── Supadata key validation tests ──────────────────────────────────────────


class TestSupadataKeyValidation:
    """Test Supadata API key validation."""

    def test_validate_key_returns_false_when_no_key(self, monkeypatch):
        """validate_supadata_key returns False when no key is set."""
        monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
        supadata_utils.reset_rate_limiter()

        result = supadata_utils.validate_supadata_key()
        assert result is False

    def test_validate_key_returns_true_on_200(self, monkeypatch):
        """validate_supadata_key returns True when API returns 200."""
        monkeypatch.setenv("SUPADATA_API_KEY", "fake-key")
        supadata_utils.reset_rate_limiter()
        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 0.01

        mock_resp = mock.Mock(status_code=200)
        mock_resp.headers = {}
        mock_resp.json.return_value = {}

        mock_client = mock.Mock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_client.get.return_value = mock_resp

        with mock.patch.object(supadata_utils.httpx, "Client", return_value=mock_client):
            result = supadata_utils.validate_supadata_key("fake-key")

        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 3.0
        assert result is True

    def test_validate_key_returns_false_on_401(self, monkeypatch):
        """validate_supadata_key returns False when API returns 401."""
        supadata_utils.reset_rate_limiter()
        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 0.01

        mock_resp = mock.Mock(status_code=401)
        mock_resp.headers = {}

        mock_client = mock.Mock()
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)
        mock_client.get.return_value = mock_resp

        with mock.patch.object(supadata_utils.httpx, "Client", return_value=mock_client):
            result = supadata_utils.validate_supadata_key("bad-key")

        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 3.0
        assert result is False
