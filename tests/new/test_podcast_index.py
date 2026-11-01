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
    response.is_redirect = False
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


def _itunes_results(*feed_urls):
    response = mock.Mock(status_code=200)
    response.json.return_value = {"results": [{"feedUrl": url} for url in feed_urls]}
    return _client_with_response(response)


_PAGE = "https://www.buzzsprout.com/123456/episodes/7-cross-platform-episode"
_PUBLISHER_FEED = "https://feeds.buzzsprout.com/123456.rss"


def test_discovery_feed_requires_episode_title_match(monkeypatch):
    """A directory candidate only wins after its RSS has the requested episode.

    Contract change: the first-match ``_find_asset_via_podcast_index`` helper
    is gone; Podcast Index results join the page-publisher candidate pool.
    """
    monkeypatch.setattr(
        podcast,
        "discover_feed_urls",
        lambda _query: [PodcastIndexFeed(feed_url=_PUBLISHER_FEED, title="Show")],
    )
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _RSS)

    with mock.patch.object(podcast.httpx, "Client", return_value=_itunes_results()):
        match = podcast._resolve_episode_asset(
            _PAGE, "generic", {"title": "Cross Platform Episode", "author": "Show"})
        miss = podcast._resolve_episode_asset(
            _PAGE, "generic", {"title": "A Different Episode", "author": "Show"})

    assert match.audio_url == "https://cdn.example/episode.mp3"
    assert match.guid == "rss-guid"
    assert match.authority == "page_publisher_feed" and match.feed_url == _PUBLISHER_FEED
    assert miss == podcast.EpisodeAsset()


def test_spotify_podcast_index_feed_is_a_candidate_under_the_shared_policy(monkeypatch):
    """Podcast Index no longer short-circuits Spotify; its exact-show feeds are
    pooled with the iTunes show search, must be hosted by the publisher named
    by Spotify's labelled transcript link, and need the same corroboration."""
    rss = _RSS.replace("<rss><channel>", "<rss><channel><title>Show</title>").replace(
        "</guid>", "</guid><pubDate>Tue, 02 Jan 2024 10:00:00 +0000</pubDate>")
    monkeypatch.setattr(podcast, "discover_feed_urls", lambda query: [
        PodcastIndexFeed(feed_url="https://feed.example/rss", title="Show"),
        PodcastIndexFeed(feed_url="https://other.example/rss", title="Show Clips"),
        PodcastIndexFeed(feed_url="https://mirror.example/rss", title="Show"),
    ])
    fetched = []
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda url: fetched.append(url) or rss)
    response = mock.Mock(status_code=200)
    response.json.return_value = {"results": []}
    metadata = {
        "title": "Cross Platform Episode", "author": "Show", "published": "2024-01-02",
        "description": "Notes. Transcript: https://feed.example/transcript",
    }

    with mock.patch.object(podcast.httpx, "Client", return_value=_client_with_response(response)):
        asset = podcast._find_spotify_episode_asset(
            "https://open.spotify.com/episode/example", metadata,
        )
        contradicted = podcast._find_spotify_episode_asset(
            "https://open.spotify.com/episode/example", {**metadata, "published": "2019-01-02"},
        )
        unanchored = podcast._find_spotify_episode_asset(
            "https://open.spotify.com/episode/example", {**metadata, "description": "Notes."},
        )

    assert asset.audio_url == "https://cdn.example/episode.mp3" and asset.corroborated
    assert asset.authority == "spotify_publisher_feed"
    assert contradicted == podcast.EpisodeAsset() == unanchored
    # Near-name and off-publisher same-name feeds are never fetched.
    assert fetched == ["https://feed.example/rss"] * 2


def test_generic_podcast_pools_podcast_index_and_itunes_publisher_feeds(monkeypatch):
    """Contract change: Podcast Index no longer short-circuits generic pages
    by first match. Both directories list candidates; only page-publisher
    feeds are fetched, so a foreign first hit is never consulted."""
    monkeypatch.setattr(podcast, "discover_feed_urls", lambda _query: [
        PodcastIndexFeed(feed_url="https://foreign.example/rss", title="Other Show")])
    fetched = []
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda url: fetched.append(url) or _RSS)

    with mock.patch.object(podcast.httpx, "Client", return_value=_itunes_results(_PUBLISHER_FEED)):
        asset = podcast._resolve_episode_asset(
            _PAGE, "generic", {"title": "Cross Platform Episode", "author": "Show"},
        )

    assert asset.audio_url == "https://cdn.example/episode.mp3"
    assert fetched == [_PUBLISHER_FEED]


def test_generic_podcast_without_publisher_feed_fails_closed(monkeypatch):
    monkeypatch.setattr(podcast, "discover_feed_urls", lambda _query: [
        PodcastIndexFeed(feed_url="https://foreign.example/rss", title="Show")])
    fetched = []
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda url: fetched.append(url) or _RSS)

    with mock.patch.object(
        podcast.httpx, "Client",
        return_value=_itunes_results("https://feeds.buzzsprout.com/999999.rss"),
    ):
        asset = podcast._resolve_episode_asset(
            _PAGE, "generic", {"title": "Cross Platform Episode", "author": "Show"},
        )

    assert asset == podcast.EpisodeAsset() and fetched == []
    assert "www.buzzsprout.com" in asset.unresolved_reason


def test_apple_never_consults_podcast_index(monkeypatch):
    """Contract change: Apple's ``lookup?id=`` feed is the only authority.

    Previously Podcast Index ran first and its title hit won (review AG05).
    """
    response = mock.Mock(status_code=200)
    response.json.return_value = {"results": [{"feedUrl": "https://feed.example/rss"}]}
    client = _client_with_response(response)
    monkeypatch.setattr(
        podcast, "discover_feed_urls",
        lambda _query: (_ for _ in ()).throw(AssertionError("Podcast Index consulted")),
    )
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _RSS)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        asset = podcast._find_apple_episode_asset(
            "https://podcasts.apple.com/us/podcast/show/id123?i=456",
            "Cross Platform Episode",
            "Show",
        )

    assert asset.audio_url == "https://cdn.example/episode.mp3"
    assert asset.authority == "apple_lookup" and asset.feed_url == "https://feed.example/rss"
    assert client.get.call_args.args[0] == "https://itunes.apple.com/lookup?id=123"


def test_apple_lookup_for_a_different_show_id_is_not_authority(monkeypatch):
    response = mock.Mock(status_code=200)
    response.json.return_value = {"results": [
        {"collectionId": 999, "feedUrl": "https://feed.example/rss"}]}
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _RSS)

    with mock.patch.object(podcast.httpx, "Client", return_value=_client_with_response(response)):
        asset = podcast._find_apple_episode_asset(
            "https://podcasts.apple.com/us/podcast/show/id123?i=456", "Cross Platform Episode",
        )

    assert asset == podcast.EpisodeAsset() and "123" in asset.unresolved_reason


def test_generic_routes_share_one_deduplicated_feed_budget(monkeypatch):
    """Podcast Index then iTunes must not download the same feed twice, and the
    total across both routes is bounded per extraction. Only page-publisher
    feeds count; more than the budget is ambiguous without any fetch."""
    feeds = [f"https://feeds.buzzsprout.com/123456/{n}.rss" for n in range(12)]
    monkeypatch.setattr(podcast, "discover_feed_urls", lambda _query: [
        PodcastIndexFeed(feed_url=url, title="Show") for url in feeds[:6]])
    fetched = []
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda url: fetched.append(url) or _RSS)

    for upto, expected in ((8, podcast._MAX_FEEDS_PER_EXTRACTION), (9, 0)):
        fetched.clear()
        with mock.patch.object(
            podcast.httpx, "Client", return_value=_itunes_results(*feeds[4:upto]),
        ):
            asset = podcast._resolve_episode_asset(
                _PAGE, "generic", {"title": "Missing Episode", "author": "Show"},
            )
        assert asset == podcast.EpisodeAsset() and asset.ambiguous is (expected == 0)
        assert len(fetched) == len(set(fetched)) == expected
