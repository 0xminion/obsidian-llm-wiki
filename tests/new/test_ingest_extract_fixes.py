"""Regression tests for the source → extract stage.

Each test here pins a bug found reviewing the restore-pipeline branch. They are
grouped by the defect they lock down, and every one of them fails against the
pre-fix code.
"""

from __future__ import annotations

import threading
from unittest import mock

import httpx
import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest import liteparse, supadata_utils
from obsidian_llm_wiki.ingest.extractors import ExtractorNotApplicableError, podcast

_LONG_TEXT = " ".join(["A usable transcript sentence."] * 20)

_PODCAST_FEED = """<?xml version='1.0'?>
<rss xmlns:itunes='http://www.itunes.com/dtds/podcast-1.0.dtd'>
  <channel><item>
    <title>Episode One</title><guid>guid-1</guid>
    <enclosure url='https://cdn.example/ep1.mp3' type='audio/mpeg'/>
  </item></channel>
</rss>"""

_BLOG_ATOM_FEED = """<?xml version='1.0'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <title>A Blog</title>
  <entry><title>A post</title><link href='https://blog.example/post'/></entry>
</feed>"""


# ── LiteParse must not treat every HTML page as a document ────────────────


def test_html_response_is_not_a_document():
    """text/html is a landing page to search, not a document to hand to LiteParse.

    Listing text/html in _CONTENT_TYPE_SUFFIXES made _is_document_response True
    for every web page, so LiteParse hijacked all of extract_web ahead of
    trafilatura and the citation-discovery branch below became unreachable.
    """
    response = httpx.Response(
        200, headers={"content-type": "text/html; charset=utf-8"}, text="<html></html>",
    )

    assert liteparse._is_document_response("https://example.com/blog/post", response) is False


@pytest.mark.parametrize(
    ("content_type", "url"),
    [
        ("application/pdf", "https://example.com/paper"),
        ("application/epub+zip", "https://example.com/book"),
        ("text/html", "https://example.com/paper.pdf"),  # suffix still wins
    ],
)
def test_real_documents_are_still_detected(content_type, url):
    """Narrowing the content-type map must not stop detecting actual documents."""
    response = httpx.Response(200, headers={"content-type": content_type}, content=b"%PDF-")

    assert liteparse._is_document_response(url, response) is True


def test_html_landing_page_reaches_citation_discovery(monkeypatch):
    """An HTML landing page must fall through to citation_pdf_url discovery.

    This is the path _document_candidates() exists for. Before the fix it was
    unreachable: the landing page was itself parsed as a '.html document'.
    """
    landing = "https://journal.example/articles/42"
    landing_html = '<meta name="citation_pdf_url" content="/pdf/42.pdf">'
    fetched: list[str] = []

    def fake_get(self, url, *args, **kwargs):
        fetched.append(url)
        if url == landing:
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=landing_html,
                request=httpx.Request("GET", url),
            )
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 body",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(
        liteparse,
        "parse_document",
        lambda path, *, source_url=None: SourceDoc(
            title="Paper", content="parsed body", url=source_url or "",
        ),
    )

    source = liteparse.extract_document_fallback(landing, timeout=10)

    # The discovered PDF was fetched, and provenance stays on the landing page.
    assert fetched == [landing, "https://journal.example/pdf/42.pdf"]
    assert source.url == landing
    assert source.content == "parsed body"


# ── The podcast extractor must not swallow non-podcast feeds ──────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/feed",
        "https://example.com/rss",
        "https://example.com/podcast.xml",
        "https://example.com/feeds/",
    ],
)
def test_rss_predicate_matches_feed_shaped_paths(url):
    assert podcast._is_rss_feed(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/feedback",           # substring match, not a feed
        "https://example.com/feedback/form",
        "https://news.example/rss-explained",     # article *about* rss
        "https://example.com/page?ref=/feed",     # query string, not a path
    ],
)
def test_rss_predicate_rejects_non_feed_paths(url):
    """`"/feed" in url` also matched /feedback pages and query strings."""
    assert podcast._is_rss_feed(url) is False


def test_blog_feed_is_not_a_podcast_feed():
    assert podcast._looks_like_podcast_feed(_BLOG_ATOM_FEED) is False
    assert podcast._looks_like_podcast_feed(_PODCAST_FEED) is True
    assert podcast._looks_like_podcast_feed("") is False
    assert podcast._looks_like_podcast_feed("not xml at all") is False


def test_blog_feed_is_disclaimed_so_dispatch_can_fall_through(monkeypatch):
    """A blog's Atom feed must be disclaimed, not extracted as a stub podcast.

    ExtractorNotApplicableError keeps the fail-closed policy for genuine failures
    while letting dispatch continue to extract_web for a URL that merely looked
    like a podcast feed.
    """
    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _BLOG_ATOM_FEED)

    with pytest.raises(ExtractorNotApplicableError):
        podcast.extract_podcast_rss("https://blog.example/feed")


def test_disclaimed_url_falls_through_to_extract_web(monkeypatch):
    """End-to-end: a disclaimed URL reaches extract_web instead of failing closed."""
    from obsidian_llm_wiki.ingest import extractors

    monkeypatch.setattr(podcast, "_fetch_rss_text", lambda _url: _BLOG_ATOM_FEED)
    monkeypatch.setattr(
        extractors,
        "extract_web",
        lambda url: SourceDoc(title="Blog", content="web body", url=url),
    )

    source = extractors.extract("https://blog.example/feed")

    assert source.content == "web body"


def test_podcast_without_transcript_or_description_raises(monkeypatch):
    """No transcript and no description means no source — never a stub SourceDoc.

    The old `if not content.strip()` guard was unreachable because the else-branch
    always wrote a 'Transcript unavailable' block, so a content-free stub reached
    the synthesis stage as though it were a real source.
    """
    monkeypatch.setattr(podcast, "_fetch_defuddle_md_metadata", lambda _url: {})
    monkeypatch.setattr(
        podcast, "_resolve_episode_asset", lambda *_a, **_kw: podcast.EpisodeAsset(),
    )
    monkeypatch.setattr(podcast, "load_transcript_cache", lambda _identity: None)

    with pytest.raises(RuntimeError, match="no transcript"):
        podcast._extract_podcast("https://podcasts.example/ep", platform="generic")


def test_podcast_with_description_only_is_still_a_valid_source(monkeypatch):
    """Metadata-only is fine *when there is real description text* to synthesize."""
    monkeypatch.setattr(
        podcast,
        "_fetch_defuddle_md_metadata",
        lambda _url: {"title": "Ep", "description": _LONG_TEXT},
    )
    monkeypatch.setattr(
        podcast, "_resolve_episode_asset", lambda *_a, **_kw: podcast.EpisodeAsset(),
    )
    monkeypatch.setattr(podcast, "load_transcript_cache", lambda _identity: None)

    source = podcast._extract_podcast("https://podcasts.example/ep", platform="generic")

    assert "## Episode Description" in source.content
    assert _LONG_TEXT in source.content


# ── RSS item matching must not pick the wrong episode ─────────────────────


def test_episode_id_does_not_match_unrelated_numbers_in_the_item():
    """`episode_id in item_xml` collided with durations, byte lengths, and URLs."""
    rss = """<rss><channel>
      <item>
        <title>Wrong Episode</title><guid>guid-wrong</guid>
        <enclosure url='https://cdn.example/wrong.mp3' length='1000456789'/>
      </item>
      <item>
        <title>Right Episode</title><guid>1000456789</guid>
        <enclosure url='https://cdn.example/right.mp3'/>
      </item>
    </channel></rss>"""

    asset = podcast._find_episode_asset_in_rss(rss, episode_id="1000456789")

    assert asset.audio_url == "https://cdn.example/right.mp3"


def test_item_title_is_not_shadowed_by_a_nested_image_title():
    """item.iter() descended into <image><title>, shadowing the item's own title."""
    rss = """<rss><channel><item>
      <image><title>Show Artwork</title></image>
      <title>The Real Episode Title</title>
      <guid>guid-1</guid>
      <enclosure url='https://cdn.example/ep.mp3' type='audio/mpeg'/>
    </item></channel></rss>"""

    asset = podcast._find_episode_asset_in_rss(rss, allow_first=True)

    assert asset.title == "The Real Episode Title"


# ── defuddle frontmatter parsing ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        ("published: 2024-01-15", "2024-01-15"),
        ("published:2024-01-15", "2024-01-15"),  # off-by-one dropped the '2'
    ],
)
def test_published_date_survives_a_missing_space(frontmatter, expected, monkeypatch):
    """`line[11:]` for the 10-character 'published:' ate the first character."""
    body = f"---\ntitle: Ep\n{frontmatter}\n---\nbody text here, long enough to keep."
    response = mock.Mock(status_code=200, text=body)
    client = mock.Mock()
    client.__enter__ = mock.Mock(return_value=client)
    client.__exit__ = mock.Mock(return_value=False)
    client.get.return_value = response

    with mock.patch.object(podcast.httpx, "Client", return_value=client):
        metadata = podcast._fetch_defuddle_md_metadata("https://podcasts.example/ep")

    assert metadata["published"] == expected
    assert metadata["title"] == "Ep"


# ── Supadata rate limiter ─────────────────────────────────────────────────


def test_rate_limiter_holds_the_lock_while_sleeping(monkeypatch):
    """The limiter must sleep *holding* the lock, or it does not limit anything.

    Releasing the lock around the sleep let every waiting thread read the same
    stale _last_call_time, sleep concurrently, and then issue its request in the
    same instant — earning exactly the 429s the limiter exists to avoid. Asserting
    the mutex is held across the sleep pins the invariant that makes concurrent
    callers serialize, without depending on wall-clock timing.
    """
    observed: list[bool] = []

    def spying_sleep(_duration: float) -> None:
        observed.append(supadata_utils._rate_lock.locked())

    monkeypatch.setattr(supadata_utils.time, "sleep", spying_sleep)
    supadata_utils.reset_rate_limiter()

    supadata_utils.supadata_rate_limit()  # first call: stamps, no sleep
    supadata_utils.supadata_rate_limit()  # second call: must sleep under the lock

    assert observed, "second call did not rate-limit at all"
    assert all(observed), "lock was released during the sleep — callers can burst"

    supadata_utils.reset_rate_limiter()


def test_rate_limiter_serializes_concurrent_callers(monkeypatch):
    """Concurrent callers must come out one-per-interval, not all at once.

    Driven by a fake clock so the assertion is on the limiter's arithmetic rather
    than on real elapsed time, which is flaky under a loaded suite.
    """
    monkeypatch.setattr(supadata_utils, "SUPADATA_RATE_LIMIT_SECONDS", 3.0)

    clock = 1000.0
    clock_lock = threading.Lock()
    fire_times: list[float] = []

    def fake_monotonic() -> float:
        return clock

    def fake_sleep(duration: float) -> None:
        nonlocal clock
        clock += duration  # safe: the limiter holds _rate_lock across the sleep

    monkeypatch.setattr(supadata_utils.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(supadata_utils.time, "sleep", fake_sleep)
    supadata_utils.reset_rate_limiter()

    def worker() -> None:
        supadata_utils.supadata_rate_limit()
        with clock_lock:
            fire_times.append(clock)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ordered = sorted(fire_times)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]

    assert len(gaps) == 3
    assert all(gap == 3.0 for gap in gaps), f"calls burst together: {gaps}"

    supadata_utils.reset_rate_limiter()
