"""Tests for optional Podcast Index canonical RSS discovery."""

from __future__ import annotations

import hashlib
from unittest import mock

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.ingest import podcast_index
from obsidian_llm_wiki.ingest.extractors import podcast
from obsidian_llm_wiki.ingest.podcast_index import PodcastIndexFeed

_RSS = """<rss><channel><item>
<title>Cross Platform Episode</title><guid>rss-guid</guid>
<enclosure url='https://cdn.example/episode.mp3'/>
</item></channel></rss>"""


def test_config_loads_podcast_index_credentials(tmp_path):
    """Podcast Index credentials stay opt-in and are available to discovery."""
    config = load_config(
        env_file=None,
        VAULT_PATH=str(tmp_path),
        PODCAST_INDEX_API_KEY="test-key",
        PODCAST_INDEX_API_SECRET="test-secret",
    )

    assert config.podcast_index_api_key == "test-key"
    assert config.podcast_index_api_secret == "test-secret"


def _client_with_response(response):
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.return_value = response
    return client


def test_discovery_is_a_noop_without_credentials(monkeypatch):
    """No configured Podcast Index credentials means no network activity."""
    monkeypatch.delenv("PODCAST_INDEX_API_KEY", raising=False)
    monkeypatch.delenv("PODCAST_INDEX_API_SECRET", raising=False)

    with mock.patch.object(podcast_index.httpx, "Client") as client:
        result = podcast_index.discover_feed_urls("Example Show")

    assert result == []
    client.assert_not_called()


def test_discovery_uses_documented_timestamped_auth(monkeypatch):
    """Discovery authenticates without leaking secrets into results or logs."""
    response = mock.Mock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "feeds": [
            {"id": 42, "title": "Example Show", "author": "Host", "url": "https://feed.example/rss"},
            {"id": 43, "title": "Duplicate", "url": "https://feed.example/rss"},
        ],
    }
    client = _client_with_response(response)
    monkeypatch.setattr(podcast_index.time, "time", lambda: 1_700_000_000)

    with mock.patch.object(podcast_index.httpx, "Client", return_value=client):
        feeds = podcast_index.discover_feed_urls(
            "Example Show", api_key="public-key", api_secret="private-secret",
        )

    assert feeds == [
        PodcastIndexFeed(
            feed_url="https://feed.example/rss",
            title="Example Show",
            author="Host",
            feed_id=42,
        ),
    ]
    headers = client.get.call_args.kwargs["headers"]
    expected = hashlib.sha1(b"public-keyprivate-secret1700000000").hexdigest()
    assert headers["X-Auth-Key"] == "public-key"
    assert headers["X-Auth-Date"] == "1700000000"
    assert headers["Authorization"] == expected
    assert "private-secret" not in repr(feeds)


def test_discovery_feed_requires_episode_title_match(monkeypatch):
    """A directory candidate only wins after its RSS has the requested episode."""
    monkeypatch.setattr(
        podcast,
        "discover_feed_urls",
        lambda _query: [PodcastIndexFeed(feed_url="https://feed.example/rss", title="Show")],
    )
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _RSS)

    match = podcast._find_asset_via_podcast_index("Cross Platform Episode", "Show")
    miss = podcast._find_asset_via_podcast_index("A Different Episode", "Show")

    assert match.audio_url == "https://cdn.example/episode.mp3"
    assert match.guid == "rss-guid"
    assert miss == podcast.EpisodeAsset()


def test_spotify_checks_podcast_index_before_itunes(monkeypatch):
    """Podcast Index saves an iTunes search when it resolves a verified episode."""
    resolved = podcast.EpisodeAsset(audio_url="https://cdn.example/episode.mp3")
    monkeypatch.setattr(podcast, "_find_asset_via_podcast_index", lambda *_args: resolved)

    with mock.patch.object(podcast.httpx, "Client") as client:
        asset = podcast._find_spotify_episode_asset(
            "https://open.spotify.com/episode/example",
            {"title": "Cross Platform Episode", "author": "Show"},
        )

    assert asset == resolved
    client.assert_not_called()


def test_generic_podcast_uses_podcast_index_discovery(monkeypatch):
    """Supported generic podcast pages gain canonical feed discovery."""
    resolved = podcast.EpisodeAsset(audio_url="https://cdn.example/episode.mp3")
    monkeypatch.setattr(podcast, "_find_asset_via_podcast_index", lambda *_args: resolved)

    asset = podcast._resolve_episode_asset(
        "https://podbean.example/episode",
        "generic",
        {"title": "Cross Platform Episode", "author": "Show"},
    )

    assert asset == resolved


def test_generic_podcast_falls_back_to_itunes_after_podcast_index_miss(monkeypatch):
    """Generic pages proceed to iTunes when Podcast Index has no match."""
    resolved = podcast.EpisodeAsset(audio_url="https://cdn.example/episode.mp3")
    monkeypatch.setattr(
        podcast,
        "_find_asset_via_podcast_index",
        lambda *_args: podcast.EpisodeAsset(),
    )
    monkeypatch.setattr(podcast, "_find_asset_via_itunes", lambda *_args: resolved)

    asset = podcast._resolve_episode_asset(
        "https://podbean.example/episode",
        "generic",
        {"title": "Cross Platform Episode", "author": "Show"},
    )

    assert asset == resolved


def test_apple_checks_podcast_index_before_itunes(monkeypatch):
    """Apple checks Podcast Index before performing the iTunes Lookup request."""
    resolved = podcast.EpisodeAsset(audio_url="https://cdn.example/episode.mp3")
    monkeypatch.setattr(podcast, "_find_asset_via_podcast_index", lambda *_args: resolved)

    with mock.patch.object(podcast.httpx, "Client") as client:
        asset = podcast._find_apple_episode_asset(
            "https://podcasts.apple.com/us/podcast/show/id123?i=456",
            "Cross Platform Episode",
            "Show",
        )

    assert asset == resolved
    client.assert_not_called()


def test_apple_falls_back_to_itunes_after_podcast_index_miss(monkeypatch):
    """A Podcast Index miss continues to Apple's exact iTunes feed lookup."""
    response = mock.Mock(status_code=200)
    response.json.return_value = {"results": [{"feedUrl": "https://feed.example/rss"}]}
    client = _client_with_response(response)
    monkeypatch.setattr(
        podcast,
        "_find_asset_via_podcast_index",
        lambda *_args: podcast.EpisodeAsset(),
    )
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _RSS)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        asset = podcast._find_apple_episode_asset(
            "https://podcasts.apple.com/us/podcast/show/id123?i=456",
            "Cross Platform Episode",
            "Show",
        )

    assert asset.audio_url == "https://cdn.example/episode.mp3"
