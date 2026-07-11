"""Tests for cache-first podcast transcript resolution and AssemblyAI fallback."""

from __future__ import annotations

import json
from unittest import mock

import httpx

from obsidian_llm_wiki.ingest.extractors import podcast
from obsidian_llm_wiki.ingest.transcript_resolver import (
    TranscriptResult,
    assemblyai_transcribe_url,
    fetch_public_transcript,
    load_transcript_cache,
    save_transcript_cache,
)

_LONG_TEXT = " ".join(["A usable transcript sentence."] * 20)


def _response(status_code: int, payload: dict | None = None, text: str = "") -> mock.Mock:
    response = mock.Mock(status_code=status_code, text=text)
    response.json.return_value = payload or {}
    response.raise_for_status.side_effect = None
    return response


def test_rss_asset_reads_podcast_transcript_tag():
    """Podcasting 2.0 transcript tags are resolved before generated ASR."""
    rss = """<?xml version='1.0'?>
    <rss xmlns:podcast='https://podcastindex.org/namespace/1.0'>
      <channel><item>
        <title>Episode Forty Two</title><guid>episode-guid-42</guid>
        <enclosure url='https://cdn.example/episode.mp3' type='audio/mpeg'/>
        <podcast:transcript url='https://publisher.example/episode.vtt'
          type='text/vtt' language='en'/>
      </item></channel>
    </rss>"""

    asset = podcast._find_episode_asset_in_rss(rss, target_title="Episode Forty Two")

    assert asset.audio_url == "https://cdn.example/episode.mp3"
    assert asset.transcript_url == "https://publisher.example/episode.vtt"
    assert asset.transcript_type == "text/vtt"
    assert asset.transcript_language == "en"
    assert asset.guid == "episode-guid-42"


def test_apple_falls_back_to_title_when_storefront_id_is_absent(monkeypatch):
    """Apple storefront episode IDs do not reliably appear in RSS GUIDs."""
    rss = """<rss><channel><item>
      <title>Prediction Markets and Beyond</title><guid>rss-guid</guid>
      <enclosure url='https://cdn.example/episode.mp3'/>
    </item></channel></rss>"""
    response = _response(200, {"results": [{"feedUrl": "https://feed.example/rss"}]})
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.return_value = response
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: rss)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        asset = podcast._find_apple_episode_asset(
            "https://podcasts.apple.com/us/podcast/example/id123?i=456",
            "Prediction Markets and Beyond",
        )

    assert asset.audio_url == "https://cdn.example/episode.mp3"
    assert asset.guid == "rss-guid"


def test_publisher_page_transcript_is_used_before_asr():
    """A real Transcript heading in publisher markdown is a free artifact."""
    result = podcast._publisher_transcript_from_metadata(
        {"body": "# Episode\n\n## Transcript\n\n" + _LONG_TEXT},
        "https://publisher.example/episode",
    )

    assert result is not None
    assert result.provider == "publisher_episode_page"
    assert result.text == _LONG_TEXT


def test_transcript_cache_round_trip(tmp_path, monkeypatch):
    """Cached transcripts are reusable without another provider call."""
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    result = TranscriptResult(
        text=_LONG_TEXT,
        provider="rss_podcast_transcript",
        artifact_url="https://publisher.example/episode.vtt",
        language="en",
        timed=True,
    )

    save_transcript_cache("episode-guid-42", result)
    loaded = load_transcript_cache("episode-guid-42")

    assert loaded == result
    cache_files = list((tmp_path / "04-Wiki/.llmwiki/transcripts").glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload["identity"] == "episode-guid-42"


def test_fetch_public_vtt_transcript_normalizes_timestamps(monkeypatch):
    """VTT publisher artifacts become clean source markdown text."""
    vtt = "WEBVTT\n\n00:00.000 --> 00:02.000\n" + _LONG_TEXT
    response = _response(200, text=vtt)
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.return_value = response

    with mock.patch(
        "obsidian_llm_wiki.ingest.transcript_resolver.httpx.Client",
        return_value=client,
    ):
        result = fetch_public_transcript(
            "https://publisher.example/episode.vtt", mime_type="text/vtt",
        )

    assert result is not None
    assert result.provider == "rss_podcast_transcript"
    assert result.timed is True
    assert "WEBVTT" not in result.text
    assert "-->" not in result.text


def test_assemblyai_remote_url_polls_to_completion_without_key_leak():
    """AssemblyAI uses provider-side fetch and returns only normalized text."""
    submitted = _response(200, {"id": "job-123"})
    running = _response(200, {"status": "processing"})
    completed = _response(
        200,
        {"status": "completed", "text": _LONG_TEXT, "language_code": "en"},
    )
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.post.return_value = submitted
    client.get.side_effect = [running, completed]

    with (
        mock.patch(
            "obsidian_llm_wiki.ingest.transcript_resolver.httpx.Client",
            return_value=client,
        ),
        mock.patch("obsidian_llm_wiki.ingest.transcript_resolver.time.sleep"),
    ):
        result = assemblyai_transcribe_url(
            "https://cdn.example/episode.mp3",
            "secret-api-key",
            poll_interval_seconds=0,
        )

    assert result is not None
    assert result.provider == "assemblyai_remote_url"
    assert result.text == _LONG_TEXT
    assert client.post.call_args.kwargs["json"]["audio_url"] == "https://cdn.example/episode.mp3"
    assert "secret-api-key" not in repr(result)


def test_podcast_prefers_cached_transcript_over_all_remote_providers(monkeypatch):
    """A local cache hit prevents transcript URL, Supadata, and AssemblyAI calls."""
    cached = TranscriptResult(text=_LONG_TEXT, provider="rss_podcast_transcript")
    asset = podcast.EpisodeAsset(
        title="Cached Episode",
        audio_url="https://cdn.example/episode.mp3",
        guid="episode-guid-42",
    )
    monkeypatch.setattr(podcast, "_fetch_defuddle_md_metadata", lambda _url: {})
    monkeypatch.setattr(podcast, "_resolve_episode_asset", lambda *_args: asset)
    monkeypatch.setattr(podcast, "load_transcript_cache", lambda _identity: cached)
    monkeypatch.setattr(
        podcast,
        "fetch_public_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("public fetch called")),
    )
    monkeypatch.setattr(
        podcast,
        "_supadata_transcribe_audio",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Supadata called")),
    )
    monkeypatch.setattr(
        podcast,
        "assemblyai_transcribe_url",
        lambda *_args: (_ for _ in ()).throw(AssertionError("AssemblyAI called")),
    )

    source = podcast._extract_podcast("https://podcasts.example/episode", platform="generic")

    assert "Transcript source: rss_podcast_transcript" in source.content
    assert _LONG_TEXT in source.content


def test_podcast_prefers_assemblyai_before_supadata(monkeypatch):
    """AssemblyAI is the first generated-transcript provider after free sources."""
    asset = podcast.EpisodeAsset(
        title="Provider Order", audio_url="https://cdn.example/episode.mp3",
    )
    assembly_result = TranscriptResult(
        text=_LONG_TEXT, provider="assemblyai_remote_url",
    )
    monkeypatch.setattr(podcast, "_fetch_defuddle_md_metadata", lambda _url: {})
    monkeypatch.setattr(podcast, "_resolve_episode_asset", lambda *_args: asset)
    monkeypatch.setattr(podcast, "load_transcript_cache", lambda _identity: None)
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "test-key")
    monkeypatch.setattr(
        podcast, "assemblyai_transcribe_url", lambda *_args: assembly_result,
    )
    monkeypatch.setattr(
        podcast,
        "_supadata_transcribe_audio",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Supadata called")),
    )

    source = podcast._extract_podcast("https://podcasts.example/episode", platform="generic")

    assert "Transcript source: assemblyai_remote_url" in source.content


def test_podcast_uses_supadata_when_assemblyai_returns_no_transcript(monkeypatch):
    """Supadata remains a second remote provider after an AssemblyAI miss."""
    asset = podcast.EpisodeAsset(
        title="Provider Fallback", audio_url="https://cdn.example/episode.mp3",
    )
    monkeypatch.setattr(podcast, "_fetch_defuddle_md_metadata", lambda _url: {})
    monkeypatch.setattr(podcast, "_resolve_episode_asset", lambda *_args: asset)
    monkeypatch.setattr(podcast, "load_transcript_cache", lambda _identity: None)
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "test-key")
    monkeypatch.setattr(podcast, "assemblyai_transcribe_url", lambda *_args: None)
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "supadata-key")
    monkeypatch.setattr(podcast, "_supadata_transcribe_audio", lambda *_args: _LONG_TEXT)

    source = podcast._extract_podcast("https://podcasts.example/episode", platform="generic")

    assert "Transcript source: supadata_remote_url" in source.content


def test_assemblyai_non_auth_http_error_returns_none(monkeypatch):
    """Provider fetch failures degrade to the next fallback without raising."""
    response = _response(503)
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "unavailable", request=mock.Mock(), response=response,
    )
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.post.return_value = response

    with mock.patch(
        "obsidian_llm_wiki.ingest.transcript_resolver.httpx.Client",
        return_value=client,
    ):
        result = assemblyai_transcribe_url("https://cdn.example/episode.mp3", "test-key")

    assert result is None
