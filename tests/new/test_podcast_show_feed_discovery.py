"""Spotify canonical-feed discovery from explicit show identity (live S02 #442).

Real Spotify defuddle metadata has no ``author``; it carries the show name only
in Spotify's own description template. Title-only directory searches can miss
the canonical feed (S02) or reach an unrelated show with the same episode title.
Fixtures are synthetic but reproduce the captured metadata/RSS/HTML shapes.
HTTP and DNS are isolated; RSS parsing, identity matching, URL safety,
transcript admission and cache stay real.
"""
from __future__ import annotations

import ipaddress
from xml.sax.saxutils import escape

import httpx
import pytest

from obsidian_llm_wiki.ingest import transcript_resolver as tr
from obsidian_llm_wiki.ingest import url_safety
from obsidian_llm_wiki.ingest.extractors import podcast

SOURCE = "https://open.spotify.com/episode/synthetic442"
DEFUDDLE = "https://defuddle.md/open.spotify.com/episode/synthetic442"
SHOW = "Example Show"
TITLE = "#442 – Synthetic Guest Interview"
ARTIFACT = "https://publisher.example/synthetic-guest-transcript"
FEED = "https://publisher.example/feed/podcast/"
AUDIO = "https://cdn.example/442.mp3"
GUID = "https://publisher.example/?p=442"
PUBDATE = "Tue, 03 Sep 2024 16:24:11 +0000"
SPEECH = (
    "We compare the measured results with the original assumptions, and describe "
    "which observations would change our conclusions. The uncertainty is important."
)


def itunes(term: str) -> str:
    return str(httpx.URL(
        "https://itunes.apple.com/search",
        params={"term": term, "media": "podcast", "limit": 5},
    ))


def transcript_html(number="442", title="Synthetic Guest Interview"):
    return (
        f"<html><head><title>Transcript for {title} | {SHOW} #{number} - Host</title>"
        f"</head><body><h1>Transcript for {title} | {SHOW} #{number}</h1>"
        "<h2>Introduction</h2>"
        + "".join(
            f'<div class="ts-segment"><div>{speaker}</div>'
            f"<div>({time})</div><div>{SPEECH}</div></div>"
            for speaker, time in [
                ("Synthetic Guest", "00:00:00"), ("Host Example", "00:00:30"),
                ("Synthetic Guest", "00:01:00"),
            ]
        )
        + "</body></html>"
    )


def item(*, title=TITLE, guid=GUID, audio=AUDIO, pubdate=PUBDATE, link=ARTIFACT):
    notes = "<p>Guest is a synthetic example.</p>"
    if link:
        notes += f'<p><b>Transcript:</b><br /><a href="{link}">{link}</a></p>'
    return (
        f"<item><title>{escape(title)}</title><guid isPermaLink=\"false\">{guid}</guid>"
        + (f"<pubDate>{pubdate}</pubDate>" if pubdate else "")
        + f"<description><![CDATA[{notes}]]></description>"
        f'<enclosure url="{audio}" length="1" type="audio/mpeg"/></item>'
    )


def feed(*items, show=SHOW):
    return (
        '<?xml version="1.0"?><rss version="2.0" '
        'xmlns:podcast="https://podcastindex.org/namespace/1.0" '
        'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'
        f"<title>{escape(show)}</title><link>https://publisher.example/</link>"
        "<language>en-US</language>"
        + "".join(items or (item(),))
        + "</channel></rss>"
    )


def spotify_metadata(*, show=SHOW, title=TITLE, published="2024-09-03", link=ARTIFACT):
    prefix = f"Listen to this episode from {show} on Spotify. " if show else ""
    description = (
        f"{prefix}Guest is a synthetic example.Thank you for listening ❤ "
        f"See below for timestamps, transcript, and contact. "
        + (f"Transcript:{link} " if link else "")
        +         "CONTACT HOST:Feedback - https://publisher.example/survey"
    )
    return (
        f'---\ntitle: "{title}"\nsite: "Spotify"\npublished: {published}\n'
        f'source: "{SOURCE}"\ndomain: "open.spotify.com"\nlanguage: "en"\n'
        f'description: "{description}"\n---\n'
        f"![]({SOURCE}.jpg)\n\n[![](https://i.scdn.co/x)\n\n{show}\n\n"
        f"](https://open.spotify.com/show/synthetic)\n\nSep 3, 2024\n\n{description}\n"
    )


@pytest.fixture
def acquisition(monkeypatch, tmp_path):
    routes: dict[str, object] = {}
    requests: list[str] = []
    asr: list[str] = []
    client_type = httpx.Client

    def respond(request):
        url = str(request.url)
        requests.append(url)
        if url not in routes:
            pytest.fail(f"Unexpected acquisition: {url}")
        value = routes[url]
        if isinstance(value, httpx.Response):
            return value
        if isinstance(value, dict):
            return httpx.Response(200, json=value)
        return httpx.Response(200, text=value, headers={"content-type": "text/html"})

    def client(**kwargs):
        kwargs.pop("proxy", None)
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(url_safety, "_resolved_addresses", lambda host: {
        ipaddress.ip_address("10.1.2.3" if host == "private.example" else "93.184.216.34"),
    })
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    for key in ("PODCAST_INDEX_API_KEY", "PODCAST_INDEX_API_SECRET", "DEEP_SEARCH_FALLBACK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "")
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "")
    monkeypatch.setattr(podcast, "_whisper_fallback_transcribe", lambda url: asr.append(url) or "")

    # Captured S02 shape: a title-only search returns unrelated shows only.
    routes[DEFUDDLE] = spotify_metadata()
    routes[itunes(TITLE)] = {"results": [
        {"collectionName": "Unrelated Interviews", "feedUrl": "https://other.example/rss"},
    ]}
    routes["https://other.example/rss"] = feed(item(title="Unrelated"), show="Unrelated Interviews")
    routes[itunes(SHOW)] = {"results": [
        {"collectionName": SHOW, "feedUrl": FEED},
        {"collectionName": f"{SHOW} Clips", "feedUrl": "https://clips.example/rss"},
        {"collectionName": f"{SHOW} | 5 minute summaries", "feedUrl": "https://sum.example/rss"},
    ]}
    routes[FEED] = feed()
    routes[ARTIFACT] = transcript_html()
    return routes, requests, asr


def bound_cache(artifact):
    """The cache entry bound to (episode identity, ``artifact``); epoch-3 keys."""
    import json

    hits = []
    for path in tr._cache_dir().glob("*.json"):
        identity = json.loads(path.read_text()).get("identity", "")
        head, _, body = identity.partition("\n")
        bound = head == "olw-podcast-transcript-binding-v3"
        if bound and json.loads(body)["artifact"][1] == artifact:
            hits.append(identity)
    assert len(hits) <= 1, hits
    return tr.load_transcript_cache(hits[0]) if hits else None


def _fails_without_artifact(routes, requests, tmp_path):
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert ARTIFACT not in requests
    assert not list(tmp_path.rglob("*.json"))


def test_s02_show_identity_resolves_canonical_feed_and_explicit_link(acquisition, tmp_path):
    routes, requests, asr = acquisition

    source = podcast.extract_spotify(SOURCE)

    assert source.title == TITLE
    assert source.content.count(SPEECH) == 3
    assert "(00:00:00)" in source.content and "(00:01:00)" in source.content
    assert f"Audio: {AUDIO}" in source.content
    assert source.provenance.extracted_url == ARTIFACT
    assert source.provenance.extractor_chain == ("publisher_transcript_link",)
    # Near-name directory results are never fetched.
    assert "https://clips.example/rss" not in requests
    assert "https://sum.example/rss" not in requests
    assert requests.count(FEED) == 1 and requests.count(ARTIFACT) == 1
    assert asr == []
    cached = bound_cache(ARTIFACT)
    assert cached is not None
    assert cached.provider == "publisher_transcript_link"
    assert cached.artifact_url == ARTIFACT and cached.language == "en-US"
    del routes[ARTIFACT]
    assert podcast.extract_spotify(SOURCE).content == source.content
    assert requests.count(ARTIFACT) == 1


def test_identical_mirror_listing_of_the_same_item_is_not_ambiguous(acquisition):
    routes, requests, _asr = acquisition
    routes[itunes(SHOW)]["results"].insert(
        1, {"collectionName": SHOW, "feedUrl": "https://mirror.example/rss"},
    )
    routes["https://mirror.example/rss"] = feed()
    source = podcast.extract_spotify(SOURCE)
    assert source.provenance.extracted_url == ARTIFACT


@pytest.mark.parametrize("variant", [
    "wrong-episode-number", "wrong-title", "channel-title-differs", "near-name-only",
    "ambiguous-in-feed", "ambiguous-across-feeds", "date-contradiction",
    "link-contradiction", "no-corroboration", "feed-blocked-private", "feed-403",
    "no-show-template",
])
def test_show_route_negative_controls_fail_closed(acquisition, tmp_path, variant):
    routes, requests, _asr = acquisition
    if variant == "wrong-episode-number":
        routes[FEED] = feed(item(title="#441 – Synthetic Guest Interview"))
    elif variant == "wrong-title":
        routes[FEED] = feed(item(title="#442 – Synthetic Guest Interview (Part 2)"))
    elif variant == "channel-title-differs":
        routes[FEED] = feed(show=f"{SHOW} Clips")
    elif variant == "near-name-only":
        routes[itunes(SHOW)] = {"results": [
            {"collectionName": f"{SHOW} Clips", "feedUrl": "https://clips.example/rss"},
        ]}
    elif variant == "ambiguous-in-feed":
        routes[FEED] = feed(item(), item(guid="dup", audio="https://cdn.example/dup.mp3"))
    elif variant == "ambiguous-across-feeds":
        # Only publisher-hosted feeds are candidates; two of them disagree.
        routes[itunes(SHOW)]["results"].insert(
            1, {"collectionName": SHOW, "feedUrl": "https://publisher.example/feed/alt/"},
        )
        routes["https://publisher.example/feed/alt/"] = feed(
            item(guid="alt-442", audio="https://cdn.example/alt-442.mp3"),
        )
    elif variant == "date-contradiction":
        routes[FEED] = feed(item(pubdate="Tue, 03 Sep 2019 16:24:11 +0000"))
    elif variant == "link-contradiction":
        routes[FEED] = feed(item(link="https://publisher.example/other-transcript"))
    elif variant == "no-corroboration":
        routes[FEED] = feed(item(pubdate="", link=""))
    elif variant == "feed-blocked-private":
        routes[itunes(SHOW)] = {"results": [
            {"collectionName": SHOW, "feedUrl": "https://private.example/rss"},
        ]}
    elif variant == "feed-403":
        routes[FEED] = httpx.Response(403, text="denied")
    elif variant == "no-show-template":
        routes[DEFUDDLE] = spotify_metadata(show="")
    _fails_without_artifact(routes, requests, tmp_path)
    assert "https://private.example/rss" not in requests
    assert "https://clips.example/rss" not in requests
    # Each control reaches the identity check it targets, not an earlier miss.
    show_searched = variant != "no-show-template"
    assert (itunes(SHOW) in requests) is show_searched
    feed_fetched = variant not in {"near-name-only", "feed-blocked-private", "no-show-template"}
    assert (FEED in requests) is feed_fetched


def test_wrong_show_exact_episode_title_is_not_accepted_from_title_search(acquisition, tmp_path):
    """A same-titled item in another show is not this Spotify episode."""
    routes, requests, _asr = acquisition
    routes[DEFUDDLE] = spotify_metadata(link="")
    routes[itunes(TITLE)] = {"results": [
        {"collectionName": "Another Show", "feedUrl": "https://other.example/rss"},
    ]}
    routes["https://other.example/rss"] = feed(
        item(guid="other-442", audio="https://other.example/442.mp3",
             link="https://other.example/transcript"),
        show="Another Show",
    )
    routes["https://other.example/transcript"] = transcript_html()
    routes[FEED] = feed(item(title="#441 – Different"))
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert "https://other.example/transcript" not in requests
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("html", [
    transcript_html(number="441"),
    transcript_html(title="Different Guest Interview"),
    f"<title>Transcript for Synthetic Guest Interview | {SHOW} #442</title><p>{SPEECH * 20}</p>",
], ids=["wrong-number", "wrong-title", "article-not-turns"])
def test_resolved_show_route_keeps_artifact_admission(acquisition, tmp_path, html):
    routes, requests, _asr = acquisition
    routes[ARTIFACT] = html
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert requests.count(ARTIFACT) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_unresolved_identity_with_advertised_link_fails_with_clear_diagnostic(acquisition):
    routes, requests, _asr = acquisition
    routes[FEED] = feed(item(title="#441 – Different"))
    with pytest.raises(RuntimeError) as error:
        podcast.extract_spotify(SOURCE)
    message = str(error.value)
    assert "transcript link" in message and "episode identity" in message
    assert ARTIFACT not in requests


def test_show_route_request_budget_is_bounded(acquisition):
    """More publisher-hosted feeds than the budget cannot be resolved: none is fetched.

    Contract change: previously the first three were fetched and searched.
    Only feeds on the labelled link's publisher host count toward the budget.
    """
    routes, requests, _asr = acquisition
    routes[itunes(SHOW)] = {"results": [
        {"collectionName": SHOW, "feedUrl": f"https://publisher.example/m{n}/rss"}
        for n in range(5)
    ]}
    for n in range(5):
        routes[f"https://publisher.example/m{n}/rss"] = feed(item(title="Unrelated"))
    with pytest.raises(RuntimeError, match="unambiguous episode identity"):
        podcast.extract_spotify(SOURCE)
    assert sum(url.startswith("https://publisher.example/m") for url in requests) == 0
    # Exactly the budget is still searched.
    routes[itunes(SHOW)]["results"] = routes[itunes(SHOW)]["results"][:3]
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert sum(url.startswith("https://publisher.example/m") for url in requests) == 3


@pytest.mark.parametrize(
    "variant", ["link-contradiction", "date-contradiction", "no-corroboration"],
)
def test_contradicted_identity_does_not_authorize_asr_audio(acquisition, variant):
    """Identity is enforced at resolution: a rejected item's audio is never ASR input."""
    routes, _requests, asr = acquisition
    routes[FEED] = feed({
        "link-contradiction": item(link="https://publisher.example/other-transcript"),
        "date-contradiction": item(pubdate="Tue, 03 Sep 2019 16:24:11 +0000"),
        "no-corroboration": item(pubdate="", link=""),
    }[variant])
    metadata = podcast._fetch_defuddle_md_metadata(SOURCE)
    assert podcast._find_spotify_episode_asset(SOURCE, metadata) == podcast.EpisodeAsset()
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert asr == []



# ── Shared identity policy across every Spotify discovery route ───────────────
# Independent review B1-B3: with the show known, the legacy iTunes-title and
# Podcast Index routes admitted a same-name mirror with no date/link
# corroboration and no cross-feed ambiguity check (first feed won). These run
# through public ``extract()``/CLI with ASR made reachable by recording doubles.

MIRROR = "https://mirror.example/feed.xml"
MIRROR_AUDIO = "https://mirror.example/audio/442.mp3"
MIRROR_VTT = "https://mirror.example/442.vtt"
MIRROR_DATE = "Mon, 01 Jan 2018 10:00:00 +0000"
FABRICATED = "Fabricated mirror speech that is not this episode"
FAKE_VTT = "WEBVTT\n\n" + "".join(
    f"00:00:{n:02d}.000 --> 00:00:{n + 5:02d}.000\n<v Impostor>{FABRICATED}, line {n}.\n\n"
    for n in range(0, 30, 5)
)
ASR_TEXT = "ASR-DOUBLE speech recognised from a mirror enclosure. " * 20
PODCAST_INDEX = "https://api.podcastindex.org/api/1.0/search/byterm"


def podcast_index(query: str) -> str:
    return str(httpx.URL(PODCAST_INDEX, params={"q": query, "max": 10}))


def mirror_item(*, pubdate=MIRROR_DATE, link="", vtt=True, guid="mirror-442"):
    xml = item(guid=guid, audio=MIRROR_AUDIO, pubdate=pubdate, link=link)
    if vtt:
        xml = xml.replace(
            "</item>", f'<podcast:transcript url="{MIRROR_VTT}" type="text/vtt"/></item>',
        )
    return xml


@pytest.fixture
def recorded_asr(acquisition, monkeypatch):
    """Make every ASR stage reachable; a call 'succeeds' with marked text."""
    from obsidian_llm_wiki.ingest.transcript_resolver import TranscriptResult

    routes, requests, asr = acquisition
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "DUMMY-TEST-KEY")
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "DUMMY-TEST-KEY")

    def assemblyai(url, key):
        asr.append(url)
        return TranscriptResult(text=ASR_TEXT, provider="assemblyai", artifact_url=url)

    monkeypatch.setattr(podcast, "assemblyai_transcribe_url", assemblyai)
    monkeypatch.setattr(
        podcast, "_supadata_transcribe_audio", lambda url, key: asr.append(url) or "",
    )
    routes[MIRROR_VTT] = httpx.Response(200, text=FAKE_VTT, headers={"content-type": "text/vtt"})
    return routes, requests, asr


def _public_extract():
    from obsidian_llm_wiki.ingest import extractors

    return extractors.extract(SOURCE)


def _assert_canonical(source, requests, asr):
    assert source.provenance.extracted_url == ARTIFACT
    assert source.content.count(SPEECH) == 3
    assert f"Audio: {AUDIO}" in source.content
    assert FABRICATED not in source.content and "ASR-DOUBLE" not in source.content
    assert asr == []
    assert requests.count(FEED) == 1


@pytest.mark.parametrize("variant", [
    "rss-transcript-contradictory-date", "conflicting-link-reaches-asr",
    "borrows-spotify-link", "relisting-same-guid-audio-injected-vtt",
])
def test_title_search_same_name_mirror_is_never_fetched_or_admitted(
    recorded_asr, tmp_path, variant,
):
    """B1 T1/T2/T3: a same-name feed off the publisher host is not a candidate.

    Contract change: previously pooled as ambiguous (fail closed); now the
    mirror is never fetched and the publisher-hosted canonical item resolves.
    """
    routes, requests, asr = recorded_asr
    routes[MIRROR] = feed({
        "rss-transcript-contradictory-date": mirror_item(),
        "conflicting-link-reaches-asr": mirror_item(
            vtt=False, link="https://mirror.example/other-transcript"),
        "borrows-spotify-link": mirror_item(vtt=False),
        "relisting-same-guid-audio-injected-vtt": item().replace(
            "</item>", f'<podcast:transcript url="{MIRROR_VTT}" type="text/vtt"/></item>'),
    }[variant])
    routes[itunes(TITLE)] = {"results": [{"collectionName": SHOW, "feedUrl": MIRROR}]}

    _assert_canonical(_public_extract(), requests, asr)

    assert requests.count(itunes(TITLE)) == 1 and MIRROR not in requests
    assert MIRROR_VTT not in requests and MIRROR_AUDIO not in asr
    assert bound_cache(ARTIFACT) is not None and bound_cache(MIRROR_VTT) is None


def test_unnamed_title_search_entry_is_not_a_candidate(recorded_asr):
    """B1 T9: an empty ``collectionName`` is not identity, even if the channel matches."""
    routes, requests, asr = recorded_asr
    routes[MIRROR] = feed(mirror_item())
    routes[itunes(TITLE)] = {"results": [{"collectionName": "", "feedUrl": MIRROR}]}

    source = _public_extract()

    _assert_canonical(source, requests, asr)
    assert MIRROR not in requests and MIRROR_VTT not in requests
    cached = bound_cache(ARTIFACT)
    assert cached is not None and cached.artifact_url == ARTIFACT


def test_title_search_alone_may_list_the_canonical_feed(recorded_asr):
    """Prior live controls (S01/F01) reached the canonical feed only by title."""
    routes, requests, asr = recorded_asr
    routes[itunes(SHOW)] = {"results": []}
    routes[itunes(TITLE)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}

    _assert_canonical(_public_extract(), requests, asr)


@pytest.mark.parametrize("route", ["itunes-author-equals-show", "podcast-index"])
@pytest.mark.parametrize("mirror_date", [MIRROR_DATE, PUBDATE], ids=["contradictory", "same-day"])
def test_same_name_mirror_in_any_candidate_route_is_never_a_candidate(
    recorded_asr, monkeypatch, tmp_path, route, mirror_date,
):
    """B1 T4/T6/P1 and B2: no first-feed-wins and no same-name mirror is fetched.

    Contract change: previously ambiguous (fail closed); a same-name feed off
    the publisher host is now not a candidate, so the canonical item resolves.
    """
    routes, requests, asr = recorded_asr
    routes[MIRROR] = feed(mirror_item(pubdate=mirror_date))
    if route == "itunes-author-equals-show":
        routes[DEFUDDLE] = routes[DEFUDDLE].replace(
            'site: "Spotify"', f'site: "Spotify"\nauthor: "{SHOW}"')
        routes[itunes(SHOW)]["results"].insert(0, {"collectionName": SHOW, "feedUrl": MIRROR})
    else:
        monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
        monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
        for query in (SHOW, TITLE):
            routes[podcast_index(query)] = {"feeds": [{"url": MIRROR, "title": SHOW}]}

    _assert_canonical(_public_extract(), requests, asr)

    assert MIRROR not in requests and MIRROR_VTT not in requests
    assert max(requests.count(url) for url in requests) == 1


@pytest.mark.parametrize("variant", ["contradictory-date", "disjoint-link", "no-evidence"])
def test_podcast_index_route_requires_the_same_corroboration(
    recorded_asr, monkeypatch, tmp_path, variant,
):
    routes, requests, asr = recorded_asr
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    for query in (SHOW, TITLE):
        routes[podcast_index(query)] = {"feeds": [{"url": FEED, "title": SHOW}]}
    routes[itunes(SHOW)] = {"results": []}
    routes[FEED] = feed({
        "contradictory-date": item(pubdate=MIRROR_DATE),
        "disjoint-link": item(link="https://publisher.example/other-transcript"),
        "no-evidence": item(pubdate="", link=""),
    }[variant])

    with pytest.raises(RuntimeError, match="no transcript"):
        _public_extract()

    assert ARTIFACT not in requests and asr == []
    assert not list(tmp_path.rglob("*.json"))


def test_podcast_index_only_canonical_feed_is_a_valid_candidate(recorded_asr, monkeypatch):
    routes, requests, asr = recorded_asr
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    routes[podcast_index(SHOW)] = {"feeds": [{"url": FEED, "title": SHOW}]}
    routes[podcast_index(TITLE)] = {"feeds": []}
    routes[itunes(SHOW)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}

    _assert_canonical(_public_extract(), requests, asr)


def test_template_without_space_after_period_keeps_show_identity(acquisition):
    """Review V19: ``on Spotify.Guest`` must not drop to the title-only route."""
    routes, requests, asr = acquisition
    routes[DEFUDDLE] = routes[DEFUDDLE].replace(
        f"from {SHOW} on Spotify. Guest", f"from {SHOW} on Spotify.Guest")
    assert "on Spotify.Guest" in routes[DEFUDDLE]

    _assert_canonical(_public_extract(), requests, asr)


@pytest.mark.parametrize("author", ["", "Unrelated Host"])
def test_unavailable_show_identity_fails_closed_before_any_directory_request(
    recorded_asr, tmp_path, author,
):
    routes, requests, asr = recorded_asr
    routes[DEFUDDLE] = spotify_metadata(show="").replace(
        'site: "Spotify"', f'site: "Spotify"\nauthor: "{author}"' if author else 'site: "Spotify"')
    routes[itunes(TITLE)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}
    if author:
        routes[itunes(author)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}

    with pytest.raises(RuntimeError, match="no transcript"):
        _public_extract()
    # The connector boundary truncates diagnostics; read the extractor's own.
    with pytest.raises(RuntimeError, match="show identity") as error:
        podcast.extract_spotify(SOURCE)

    if not author:
        assert requests == [DEFUDDLE, DEFUDDLE]  # metadata only, once per call
    assert FEED not in requests and ARTIFACT not in requests and asr == []
    assert "link not followed" in str(error.value)
    assert not list(tmp_path.rglob("*.json"))


def test_feed_retrieval_is_deduplicated_and_bounded_across_routes(recorded_asr, monkeypatch):
    """One budget for all routes: each feed at most once, at most three in total."""
    routes, requests, asr = recorded_asr
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    mirrors = [f"https://publisher.example/m{n}/rss" for n in range(2)]
    for query in (SHOW, TITLE):
        routes[podcast_index(query)] = {"feeds": [
            {"url": url, "title": SHOW} for url in (FEED, mirrors[0])]}
    routes[itunes(TITLE)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}
    routes[itunes(SHOW)] = {"results": [
        {"collectionName": SHOW, "feedUrl": url} for url in (mirrors[0], FEED, mirrors[1])]}
    routes[FEED] = feed(item(title="#441 – Different"))
    for url in mirrors:
        routes[url] = feed(item(title="Unrelated", guid=url))

    with pytest.raises(RuntimeError, match="no transcript"):
        _public_extract()

    feeds = [url for url in requests if url in (FEED, *mirrors)]
    assert sorted(feeds) == sorted({FEED, *mirrors}) and len(feeds) == 3
    assert requests.count(itunes(TITLE)) == requests.count(itunes(SHOW)) == 1
    assert requests.count(podcast_index(SHOW)) == 1 and podcast_index(TITLE) not in requests
    assert asr == []


def test_more_exact_name_feeds_than_the_budget_is_ambiguous_without_fetching(
    recorded_asr, tmp_path,
):
    routes, requests, asr = recorded_asr
    routes[itunes(SHOW)] = {"results": [
        {"collectionName": SHOW, "feedUrl": url}
        for url in (FEED, *(f"https://publisher.example/m{n}/rss" for n in range(3)))]}

    with pytest.raises(RuntimeError, match="unambiguous episode identity"):
        _public_extract()

    assert not [url for url in requests if url.endswith("rss") or url == FEED]
    assert ARTIFACT not in requests and asr == []
    assert not list(tmp_path.rglob("*.json"))


def test_spotify_metadata_link_is_not_borrowed_by_an_uncorroborated_enclosure(
    recorded_asr, monkeypatch, tmp_path,
):
    """Defence in depth at extraction: only a corroborated Spotify item may borrow."""
    routes, requests, asr = recorded_asr
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "")
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "")
    uncorroborated = podcast.EpisodeAsset(
        title=TITLE, audio_url=MIRROR_AUDIO, guid="mirror-442", notes="Reupload.")
    monkeypatch.setattr(podcast, "_resolve_episode_asset", lambda *a, **k: uncorroborated)

    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)

    assert ARTIFACT not in requests
    assert not list(tmp_path.rglob("*.json"))


def _cli(monkeypatch, tmp_path, name):
    """Public ``olw ingest VAULT --url SOURCE --preview --json`` in a fresh vault."""
    import json

    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app

    vault = tmp_path / name
    vault.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(vault))
    result = CliRunner().invoke(app, ["ingest", str(vault), "--url", SOURCE, "--preview", "--json"])
    events = []
    for line in result.output.splitlines():
        with __import__("contextlib").suppress(ValueError):
            events.append(json.loads(line))
    return result, [e for e in events if isinstance(e, dict) and e.get("type") == "source"]


def test_public_cli_rejects_same_name_mirrors_and_keeps_canonical_control(
    recorded_asr, monkeypatch, tmp_path,
):
    routes, requests, asr = recorded_asr
    good, good_events = _cli(monkeypatch, tmp_path, "control")
    assert good.exit_code == 0, good.output
    assert [e.get("status") for e in good_events] == ["succeeded"]

    routes[MIRROR] = feed(mirror_item())
    routes[itunes(TITLE)] = {"results": [{"collectionName": SHOW, "feedUrl": MIRROR}]}
    title_route, title_events = _cli(monkeypatch, tmp_path, "title-route")
    routes[itunes(TITLE)] = {"results": []}
    routes[DEFUDDLE] = routes[DEFUDDLE].replace(
        'site: "Spotify"', f'site: "Spotify"\nauthor: "{SHOW}"')
    routes[itunes(SHOW)]["results"].insert(0, {"collectionName": SHOW, "feedUrl": MIRROR})
    show_route, show_events = _cli(monkeypatch, tmp_path, "author-equals-show")

    # Contract change: the mirror is no longer a candidate (it was ambiguous).
    for result, events in ((title_route, title_events), (show_route, show_events)):
        assert result.exit_code == 0, result.output
        assert [e.get("status") for e in events] == ["succeeded"]
        assert FABRICATED not in result.output and "ASR-DOUBLE" not in result.output
    assert MIRROR not in requests and MIRROR_VTT not in requests and asr == []
