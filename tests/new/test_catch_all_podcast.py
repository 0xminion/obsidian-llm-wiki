"""Tests for the catch-all podcast extractor and its pre-check heuristic."""

from __future__ import annotations

from unittest import mock

import pytest

from obsidian_llm_wiki.ingest.extractors import ExtractorNotApplicableError
from obsidian_llm_wiki.ingest.extractors import podcast


# ── _looks_like_podcast_page ────────────────────────────────────────────


def _html_response(body: str) -> mock.Mock:
    """Build a mock httpx response with the given HTML body."""
    resp = mock.Mock(status_code=200, text=body, is_redirect=False)
    return resp


def _client_with_response(response: mock.Mock) -> mock.Mock:
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.return_value = response
    return client


def test_podcast_page_detected_by_audio_element():
    """An <audio> tag in the HTML is a strong podcast signal."""
    html = '<html><body><audio src="episode.mp3"></audio></body></html>'
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep1") is True


def test_podcast_page_detected_by_rss_autodiscovery():
    """An RSS auto-discovery link is a podcast signal."""
    html = (
        '<html><head>'
        '<link rel="alternate" type="application/rss+xml" href="/feed.xml">'
        '</head><body>Some page</body></html>'
    )
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep1") is True


def test_podcast_page_detected_by_meta_keywords():
    """Podcast keywords in meta description trigger the heuristic."""
    html = (
        '<html><head>'
        '<meta name="description" content="Listen to this podcast episode about crypto">'
        '</head><body>Body text</body></html>'
    )
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep1") is True


def test_podcast_page_detected_by_title_keyword():
    """Podcast keyword in <title> triggers the heuristic."""
    html = '<html><head><title>My Podcast - Episode 42</title></head><body>Body</body></html>'
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep1") is True


def test_non_podcast_page_rejected():
    """A regular article page without podcast signals returns False."""
    html = (
        '<html><head><title>Understanding Prediction Markets</title>'
        '<meta name="description" content="An analysis of prediction market microstructure">'
        '</head><body>Article about prediction markets...</body></html>'
    )
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/article") is False


def test_podcast_page_fetch_error_returns_false():
    """A network error during the pre-check returns False (disclaim quickly)."""
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.side_effect = Exception("Connection refused")

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep") is False


def test_podcast_page_non_200_returns_false():
    """A non-200 status code returns False."""
    response = mock.Mock(status_code=403, text="", is_redirect=False)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/ep") is False


def test_body_text_mentioning_podcast_does_not_trigger():
    """The word 'podcast' in body text alone should NOT trigger (false positive)."""
    html = (
        '<html><head><title>Analysis of Crypto Markets</title>'
        '<meta name="description" content="Deep dive into market microstructure">'
        '</head><body>This article mentions podcast in the body text but is not one.</body></html>'
    )
    response = _html_response(html)
    client = _client_with_response(response)

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        assert podcast._looks_like_podcast_page("https://example.com/article") is False


# ── extract_catch_all_podcast ───────────────────────────────────────────


def test_catch_all_disclaims_when_not_a_podcast(monkeypatch):
    """When _looks_like_podcast_page returns False, the catch-all disclaims."""
    monkeypatch.setattr(podcast, "_looks_like_podcast_page", lambda _url: False)

    with pytest.raises(ExtractorNotApplicableError):
        podcast.extract_catch_all_podcast("https://example.com/some-article")


def test_catch_all_disclaims_when_extraction_fails(monkeypatch):
    """When _extract_podcast raises RuntimeError (no audio), the catch-all disclaims."""
    monkeypatch.setattr(podcast, "_looks_like_podcast_page", lambda _url: True)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("no audio URL resolved")

    monkeypatch.setattr(podcast, "_extract_podcast", _raise)

    with pytest.raises(ExtractorNotApplicableError):
        podcast.extract_catch_all_podcast("https://example.com/unknown-podcast/episode-1")


def test_catch_all_extracts_when_podcast_found(monkeypatch):
    """When the page looks like a podcast and extraction succeeds, return the SourceDoc."""
    from obsidian_llm_wiki.core.models import SourceDoc

    monkeypatch.setattr(podcast, "_looks_like_podcast_page", lambda _url: True)

    fake_doc = SourceDoc(
        title="Unknown Podcast Episode",
        content="Transcript content here",
        url="https://example.com/unknown-podcast/episode-1",
    )
    monkeypatch.setattr(podcast, "_extract_podcast", lambda *_a, **_kw: fake_doc)

    result = podcast.extract_catch_all_podcast("https://example.com/unknown-podcast/episode-1")
    assert result.title == "Unknown Podcast Episode"
    assert "Transcript content" in result.content


def test_catch_all_does_not_match_twitter_domains():
    """The catch-all match function excludes x.com / twitter.com so Twitter extractor runs first."""
    from urllib.parse import urlparse

    match_fn = None
    from obsidian_llm_wiki.ingest.extractors import _EXTRACTORS
    for mf, fn in _EXTRACTORS:
        if fn.__name__ == "extract_catch_all_podcast":
            match_fn = mf
            break
    assert match_fn is not None

    # Twitter URLs should NOT match the catch-all
    for url in (
        "https://x.com/agintender/status/123",
        "https://twitter.com/agintender/status/123",
        "https://www.x.com/agintender/status/123",
        "https://www.twitter.com/agintender/status/123",
    ):
        parsed = urlparse(url)
        assert match_fn(parsed, url) is False, f"catch-all should not match {url}"

    # Regular URLs SHOULD match
    for url in (
        "https://somepodcast.com/episode-1",
        "https://example.com/blog/post",
    ):
        parsed = urlparse(url)
        assert match_fn(parsed, url) is True, f"catch-all should match {url}"


def test_catch_all_registered_last_among_podcast_extractors():
    """The catch-all should be registered after all known podcast platform extractors."""
    from obsidian_llm_wiki.ingest.extractors import _EXTRACTORS

    names = [fn.__name__ for _, fn in _EXTRACTORS]
    catch_all_idx = names.index("extract_catch_all_podcast")
    spotify_idx = names.index("extract_spotify")
    apple_idx = names.index("extract_apple_podcast")
    generic_idx = names.index("extract_generic_podcast")

    assert spotify_idx < catch_all_idx
    assert apple_idx < catch_all_idx
    assert generic_idx < catch_all_idx