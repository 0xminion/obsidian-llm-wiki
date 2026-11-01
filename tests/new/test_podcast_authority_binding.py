"""Podcast identity authority and cache binding (final-review FB1/FB2/AG/CB/SC).

One model for every route: a discovered feed is only *authoritative* when it is
bound to evidence from the requested page or platform, never by a matching
show name, date or a copied link alone.

* Direct RSS URL: the requested feed.
* Apple: the ``lookup?id=`` feed for the URL's show ID (directories unused).
* Spotify: a feed hosted by the publisher of the single explicitly labelled
  transcript link in Spotify's own episode description.
* Generic pages: a feed hosted by the requested page's own publisher (shared
  podcast platforms additionally need the page's tenant in the feed URL).

Caches are keyed by that binding plus the selected artifact. HTTP runs through
real ``httpx`` clients on a ``MockTransport``; DNS is a table so private targets
are rejected before transport. ASR stages are recording doubles.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
from typing import Any
from xml.sax.saxutils import escape

import httpx
import pytest

from obsidian_llm_wiki.ingest import extractors, url_safety
from obsidian_llm_wiki.ingest import transcript_resolver as tr
from obsidian_llm_wiki.ingest.extractors import podcast

SPEECH = (
    "We compare the measured results with the original assumptions, and describe "
    "which observations would change our conclusions. The uncertainty is important."
)
PUBDATE = "Tue, 03 Sep 2024 16:24:11 +0000"


def itunes(term: str) -> str:
    return str(
        httpx.URL(
            "https://itunes.apple.com/search", params={"term": term, "media": "podcast", "limit": 5}
        )
    )


def pi_url(query: str) -> str:
    return str(
        httpx.URL(
            "https://api.podcastindex.org/api/1.0/search/byterm", params={"q": query, "max": 10}
        )
    )


def vtt(tag: str) -> httpx.Response:
    cues = "".join(
        f"00:00:{n * 5:02d}.000 --> 00:00:{n * 5 + 5:02d}.000\n<v Speaker{n % 2}>{tag} cue {n}: "
        "this sentence is synthetic speech used only to identify the admitted artifact.\n\n"
        for n in range(8)
    )
    return httpx.Response(200, text="WEBVTT\n\n" + cues, headers={"content-type": "text/vtt"})


def rss(show: str, *items: dict) -> str:
    def one(i):
        s = f"<item><title>{escape(i['title'])}</title>"
        if i.get("date"):
            s += f"<pubDate>{i['date']}</pubDate>"
        if i.get("guid"):
            s += f'<guid isPermaLink="false">{i["guid"]}</guid>'
        if i.get("link"):
            s += f"<link>{i['link']}</link>"
        s += f"<description><![CDATA[{i.get('notes', '')}]]></description>"
        if i.get("audio"):
            s += f'<enclosure url="{i["audio"]}" length="1" type="audio/mpeg"/>'
        if i.get("vtt"):
            s += f'<podcast:transcript url="{i["vtt"]}" type="text/vtt"/>'
        return s + "</item>"

    return (
        '<?xml version="1.0"?><rss version="2.0" '
        'xmlns:podcast="https://podcastindex.org/namespace/1.0" '
        'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"><channel>'
        f"<title>{escape(show)}</title><language>en</language>"
        + "".join(one(i) for i in items)
        + "</channel></rss>"
    )


def defuddle(title, *, author="", published="", description="d", body="placeholder body " * 5):
    fm = [f'title: "{title}"']
    if author:
        fm.append(f'author: "{author}"')
    if published:
        fm.append(f"published: {published}")
    fm.append(f'description: "{description}"')
    return "---\n" + "\n".join(fm) + "\n---\n" + body


def transcript_html(show, number, title):
    return (
        f"<html><head><title>Transcript for {title} | {show} #{number}</title></head><body>"
        f"<h1>Transcript for {title} | {show} #{number}</h1>"
        + "".join(
            f'<div class="ts-segment"><div>{who}</div><div>({t})</div><div>{SPEECH}</div></div>'
            for who, t in [("Guest", "00:00:00"), ("Host", "00:00:30"), ("Guest", "00:01:00")]
        )
        + "</body></html>"
    )


@pytest.fixture
def world(monkeypatch, tmp_path):
    routes: dict[Any, Any] = {}
    requests: list[str] = []
    asr: list[str] = []
    real_client = httpx.Client

    def respond(request):
        url = str(request.url)
        requests.append(url)
        value = routes.get(url)
        if value is None:
            return httpx.Response(404, text="")
        if isinstance(value, httpx.Response):
            return value
        if isinstance(value, dict):
            return httpx.Response(200, json=value)
        ctype = "application/rss+xml" if value.lstrip().startswith("<?xml") else "text/html"
        if value.startswith("---"):
            ctype = "text/markdown"
        return httpx.Response(200, text=value, headers={"content-type": ctype})

    def client(*args, **kwargs):
        kwargs.pop("proxy", None)
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    private = {
        "internal.example": "10.0.0.5",
        "meta.example": "169.254.169.254",
        "loop.example": "127.0.0.1",
    }
    monkeypatch.setattr(
        url_safety,
        "_resolved_addresses",
        lambda host: {ipaddress.ip_address(private.get(host, "93.184.216.34"))},
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(vault))
    for key in ("PODCAST_INDEX_API_KEY", "PODCAST_INDEX_API_SECRET", "DEEP_SEARCH_FALLBACK"):
        monkeypatch.delenv(key, raising=False)

    def assemblyai(url, key):
        asr.append(url)
        text = routes.get(("asr", url), f"ASR-DOUBLE text for {url}. " * 12)
        return tr.TranscriptResult(text=text, provider="assemblyai_remote_url", artifact_url=url)

    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "DUMMY-TEST-KEY")
    monkeypatch.setattr(podcast, "assemblyai_transcribe_url", assemblyai)
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "")
    monkeypatch.setattr(podcast, "_whisper_fallback_transcribe", lambda url: asr.append(url) or "")
    return routes, requests, asr, vault


def cache_files(vault):
    return sorted((vault / "04-Wiki" / ".llmwiki" / "transcripts").glob("*.json"))


def cli(monkeypatch, tmp_path, url, name="cli"):
    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app

    vault = tmp_path / name
    vault.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(vault))
    result = CliRunner().invoke(app, ["ingest", str(vault), "--url", url, "--preview", "--json"])
    events = []
    for line in result.output.splitlines():
        with contextlib.suppress(ValueError):
            events.append(json.loads(line))
    return result, [e for e in events if isinstance(e, dict) and e.get("type") == "source"]


# ── Spotify (FB1, FB2) ────────────────────────────────────────────────────────
S_URL = "https://open.spotify.com/episode/synthetic442"
S_DEF = "https://defuddle.md/open.spotify.com/episode/synthetic442"
SHOW = "Example Show"
TITLE = "#442 – Synthetic Guest Interview"
LINK = "https://publisher.example/synthetic-guest-transcript"
FEED = "https://publisher.example/feed/podcast/"
AUDIO = "https://cdn.example/442.mp3"
GUID = "https://publisher.example/?p=442"
MIRROR = "https://feeds.mirror.example/shows/example"
MIRROR_VTT = "https://mirror.example/442.vtt"
MIRROR_AUDIO = "https://mirror.example/442.mp3"
LINK_NOTES = f'<p><b>Transcript:</b><br/><a href="{LINK}">{LINK}</a></p>'


def spotify_page(*, link=LINK, published="2024-09-03"):
    desc = (
        f"Listen to this episode from {SHOW} on Spotify. Guest is a synthetic example. "
        + (f"Transcript:{link} " if link else "")
        + "CONTACT:https://publisher.example/c"
    )
    return defuddle(TITLE, published=published, description=desc)


def canonical_item(**kw):
    i = {"title": TITLE, "date": PUBDATE, "guid": GUID, "audio": AUDIO, "notes": LINK_NOTES}
    i.update(kw)
    return i


def mirror_item(**kw):
    i = {
        "title": TITLE,
        "date": PUBDATE,
        "guid": "mirror-442",
        "audio": MIRROR_AUDIO,
        "vtt": MIRROR_VTT,
    }
    i.update(kw)
    return {k: v for k, v in i.items() if v is not None}


@pytest.fixture
def spotify(world):
    routes, requests, asr, vault = world
    routes[S_DEF] = spotify_page()
    routes[itunes(SHOW)] = {
        "results": [
            {"collectionName": SHOW, "feedUrl": FEED},
            {"collectionName": SHOW, "feedUrl": MIRROR},
        ]
    }
    routes[itunes(TITLE)] = {"results": []}
    routes[FEED] = rss(SHOW, canonical_item())
    routes[MIRROR] = rss(SHOW, mirror_item())
    routes[LINK] = transcript_html(SHOW, "442", "Synthetic Guest Interview")
    routes[MIRROR_VTT] = vtt("FABRICATED mirror")
    return world


def _no_mirror(requests, asr, content=""):
    assert MIRROR not in requests and MIRROR_VTT not in requests
    assert MIRROR_AUDIO not in asr and "FABRICATED" not in content


def test_spotify_anchored_feed_wins_over_same_name_mirror(spotify):
    """SP03 (was ambiguous DoS): only the publisher-host feed is a candidate."""
    routes, requests, asr, vault = spotify
    source = extractors.extract(S_URL)
    assert source.provenance.extracted_url == LINK
    assert f"Audio: {AUDIO}" in source.content and source.content.count(SPEECH) == 3
    _no_mirror(requests, asr, source.content)
    assert asr == [] and requests.count(FEED) == 1


@pytest.mark.parametrize("canonical", ["502", "truncated", "private-redirect", "redirect-loop"])
@pytest.mark.parametrize(
    "mirror",
    ["same-day-own-vtt", "no-date-copied-link-own-vtt", "copied-link-no-vtt", "same-day-asr-only"],
)
def test_spotify_unavailable_or_truncated_canonical_never_promotes_mirror(
    spotify,
    monkeypatch,
    tmp_path,
    canonical,
    mirror,
):
    """FB2 SP01/SP02/SP14/SP15/SP16: fail closed; no mirror tag, enclosure, Audio or cache."""
    routes, requests, asr, vault = spotify
    routes[FEED] = {
        "502": httpx.Response(502, text=""),
        "truncated": rss(
            SHOW,
            canonical_item(
                title="#441 – Other", guid="p441", audio="https://cdn.example/441.mp3", notes=""
            ),
        ),
        "private-redirect": httpx.Response(302, headers={"location": "http://meta.example/x"}),
        "redirect-loop": httpx.Response(302, headers={"location": FEED}),
    }[canonical]
    routes[MIRROR] = rss(
        SHOW,
        {
            "same-day-own-vtt": mirror_item(),
            "no-date-copied-link-own-vtt": mirror_item(date="", notes=LINK_NOTES),
            "copied-link-no-vtt": mirror_item(date="", vtt=None, notes=LINK_NOTES),
            "same-day-asr-only": mirror_item(vtt=None),
        }[mirror],
    )
    with pytest.raises(RuntimeError, match="no transcript") as error:
        extractors.extract(S_URL)
    _no_mirror(requests, asr)
    assert asr == [] and LINK not in requests
    assert not any("meta.example" in url for url in requests)
    assert requests.count(FEED) <= 6  # one feed, shared five-redirect cap
    assert cache_files(vault) == []
    # The identity reason survives the connector's 240-character truncation.
    assert "publisher.example" in str(error.value)
    result, events = cli(monkeypatch, tmp_path, S_URL)
    assert result.exit_code == 1 and [e.get("status") for e in events] == ["failed"]
    assert "publisher.example" in (events[0].get("error") or "")
    assert "FABRICATED" not in result.output and asr == []


def test_spotify_without_labelled_link_has_no_feed_authority(spotify, monkeypatch, tmp_path):
    """Name + date alone never authenticate a directory feed (F02-type control)."""
    routes, requests, asr, vault = spotify
    routes[S_DEF] = spotify_page(link="")
    with pytest.raises(RuntimeError, match="no transcript") as error:
        extractors.extract(S_URL)
    assert requests == [S_DEF]  # no directory or feed request at all
    assert asr == [] and cache_files(vault) == []
    assert "transcript link" in str(error.value)
    result, events = cli(monkeypatch, tmp_path, S_URL)
    assert result.exit_code == 1 and "transcript link" in (events[0].get("error") or "")


def test_spotify_unlisted_publisher_feed_fails_with_visible_reason(spotify):
    routes, requests, asr, vault = spotify
    routes[itunes(SHOW)] = {"results": [{"collectionName": SHOW, "feedUrl": MIRROR}]}
    with pytest.raises(RuntimeError, match="no transcript") as error:
        extractors.extract(S_URL)
    _no_mirror(requests, asr)
    assert "publisher.example" in str(error.value)


def test_spotify_private_directory_feed_is_never_fetched(spotify):
    routes, requests, asr, vault = spotify
    routes[itunes(SHOW)]["results"].insert(
        0, {"collectionName": SHOW, "feedUrl": "http://loop.example/feed.xml"}
    )
    source = extractors.extract(S_URL)
    assert source.provenance.extracted_url == LINK
    assert not any("loop.example" in url for url in requests)


def test_spotify_raw_url_legacy_cache_is_never_consulted(spotify, monkeypatch, tmp_path):
    """FB1: an unresolved Spotify identity must not read the raw-URL legacy key."""
    routes, requests, asr, vault = spotify
    legacy = tr.TranscriptResult(
        text="FABRICATED legacy mirror transcript. " * 20,
        provider="rss_podcast_transcript",
        artifact_url=MIRROR_VTT,
    )
    tr.save_transcript_cache(S_URL, legacy)  # current admission version, raw-URL key
    [path] = cache_files(vault)
    before = path.read_bytes()
    routes[itunes(SHOW)] = httpx.Response(500, text="")
    routes[itunes(TITLE)] = httpx.Response(500, text="")
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(S_URL)
    routes[FEED] = httpx.Response(502, text="")
    routes[itunes(SHOW)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(S_URL)
    assert path.read_bytes() == before and cache_files(vault) == [path]  # preserved
    assert asr == []


def test_spotify_corroborated_legacy_media_cache_is_not_reused_automatically(spotify):
    """Epoch 3: old guid|audio entries name media, not an episode (final-check H1).

    Even an authoritative, uniquely keyed item misses its version-1 entry once
    (documented refresh cost) and the legacy file stays byte-identical. Reuse
    without re-transcription requires the operator-attested import
    (``tests/new/test_podcast_binding_epoch.py``).
    """
    routes, requests, asr, vault = spotify
    legacy = tr.TranscriptResult(
        text="Legacy ASR of the canonical enclosure. " * 12,
        provider="assemblyai_remote_url",
        artifact_url=AUDIO,
    )
    tr.save_transcript_cache(f"{GUID}|{AUDIO}", legacy)
    [legacy_path] = cache_files(vault)
    before = legacy_path.read_bytes()
    routes[FEED] = rss(SHOW, canonical_item())
    del routes[LINK]
    routes[itunes(SHOW)] = {"results": [{"collectionName": SHOW, "feedUrl": FEED}]}
    source = extractors.extract(S_URL)
    assert "Legacy ASR of the canonical enclosure." not in source.content
    assert asr == [AUDIO]  # one fresh acquisition (recording double)
    assert legacy_path.read_bytes() == before
    assert len(cache_files(vault)) == 2  # fresh epoch-3 entry beside the legacy file
    extractors.extract(S_URL)
    assert asr == [AUDIO]  # the fresh epoch-3 entry is reused warm


def test_spotify_cache_binding_two_episodes_shared_audio(world):
    """SC01->SC02: corroborated anchored episodes with no GUID and one enclosure."""
    routes, requests, asr, vault = world
    show, feed, audio = (
        "Synth Show",
        "https://synth.example/feed.xml",
        "https://synth.example/a.mp3",
    )
    pages = {}
    for n, (title, date, iso) in enumerate(
        [
            ("Ep 1 Alpha", "Mon, 01 Jan 2024 00:00:00 +0000", "2024-01-01"),
            ("Ep 2 Beta", "Thu, 01 Feb 2024 00:00:00 +0000", "2024-02-01"),
        ],
        1,
    ):
        url = f"https://open.spotify.com/episode/SYNTH{n}"
        link = f"https://synth.example/{n}/transcript"
        desc = f"Listen to this episode from {show} on Spotify. Notes. Transcript:{link} End."
        routes["https://defuddle.md/" + url[8:]] = defuddle(title, published=iso, description=desc)
        routes[itunes(title)] = {"results": []}
        pages[n] = (url, title, date, link)
    routes[itunes(show)] = {"results": [{"collectionName": show, "feedUrl": feed}]}
    routes[feed] = rss(
        show,
        *[
            {
                "title": t,
                "date": d,
                "audio": audio,
                "vtt": f"https://synth.example/{n}.vtt",
                "notes": f'<p>Transcript: <a href="{lk}">{lk}</a></p>',
            }
            for n, (_u, t, d, lk) in pages.items()
        ],
    )
    routes["https://synth.example/1.vtt"] = vtt("EPISODE-ONE")
    routes["https://synth.example/2.vtt"] = vtt("EPISODE-TWO")
    one = extractors.extract(pages[1][0])
    two = extractors.extract(pages[2][0])
    assert "EPISODE-ONE" in one.content and "EPISODE-TWO" not in one.content
    assert "EPISODE-TWO" in two.content and "EPISODE-ONE" not in two.content
    assert two.provenance.extracted_url == "https://synth.example/2.vtt"
    del routes["https://synth.example/1.vtt"], routes["https://synth.example/2.vtt"]
    assert extractors.extract(pages[1][0]).content == one.content  # warm, no refetch
    assert extractors.extract(pages[2][0]).content == two.content
    assert asr == []


# ── Apple (AG05, AG07, CB) ────────────────────────────────────────────────────
A_SHOW_ID = "999"
A_FEED = "https://cb.example/feed.xml"
A_AUDIO = "https://cb.example/constant.mp3"


def apple_url(ep):
    return f"https://podcasts.apple.com/us/podcast/cb/id{A_SHOW_ID}?i={ep}"


@pytest.fixture
def apple(world):
    routes, requests, asr, vault = world
    routes[f"https://itunes.apple.com/lookup?id={A_SHOW_ID}"] = {
        "results": [{"collectionId": int(A_SHOW_ID), "feedUrl": A_FEED}]
    }
    for ep, title, iso in (
        ("1001", "Episode A", "2024-01-01"),
        ("1002", "Episode B", "2024-02-01"),
    ):
        routes["https://defuddle.md/" + apple_url(ep)[8:]] = defuddle(title, published=iso)
    routes[A_FEED] = rss(
        "CB Show",
        {
            "title": "Episode A",
            "date": "Mon, 01 Jan 2024 00:00:00 +0000",
            "audio": A_AUDIO,
            "vtt": "https://cb.example/a.vtt",
        },
        {
            "title": "Episode B",
            "date": "Thu, 01 Feb 2024 00:00:00 +0000",
            "audio": A_AUDIO,
            "vtt": "https://cb.example/b.vtt",
        },
    )
    routes["https://cb.example/a.vtt"] = vtt("EPISODE-A")
    routes["https://cb.example/b.vtt"] = vtt("EPISODE-B")
    return world


@pytest.mark.parametrize("guid", ["", "G1"], ids=["no-guid", "reused-guid"])
def test_apple_cache_never_serves_episode_a_for_b(apple, guid):
    """CB01->CB02 and CB05: same enclosure (and GUID), distinct episodes."""
    routes, requests, asr, vault = apple
    if guid:
        routes[A_FEED] = routes[A_FEED].replace(
            "<pubDate>", f'<guid isPermaLink="false">{guid}</guid><pubDate>'
        )
    a = extractors.extract(apple_url("1001"))
    b = extractors.extract(apple_url("1002"))
    assert "EPISODE-A" in a.content and "EPISODE-B" in b.content and "EPISODE-A" not in b.content
    assert b.provenance.extracted_url == "https://cb.example/b.vtt"
    vtt_requests = [u for u in requests if u.endswith(".vtt")]
    assert extractors.extract(apple_url("1001")).content == a.content  # warm A after B
    assert extractors.extract(apple_url("1002")).content == b.content
    assert [u for u in requests if u.endswith(".vtt")] == vtt_requests  # no refetch
    assert asr == []


def test_public_cli_ingest_a_then_b_never_persists_a_for_b(apple, monkeypatch, tmp_path):
    """CB through the public CLI (persisting ingest, shared vault cache)."""
    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app
    from obsidian_llm_wiki.config import Config

    routes, requests, asr, vault = apple
    notes = []
    for ep in ("1001", "1002", "1001"):
        result = CliRunner().invoke(app, [
            "ingest", str(vault), "--parallel", "0", "--json", "--skip-synthesis",
            "--url", apple_url(ep),
        ])
        assert result.exit_code == 0, result.output
        files = sorted(Config(vault_path=str(vault)).sources_dir.glob("*.md"))
        notes.append({f.name: f.read_text() for f in files})
    texts = list(notes[1].values())
    assert any("EPISODE-B" in t and "EPISODE-A" not in t for t in texts)
    assert any("EPISODE-A" in t and "EPISODE-B" not in t for t in texts)
    assert notes[2] == notes[1]  # duplicate intake preserves both episode notes
    # Refresh no longer replaces human input files. Exercise the warm resolver
    # explicitly so duplicate indexing cannot conceal a poisoned episode cache.
    warm = extractors.extract(apple_url("1001"))
    assert "EPISODE-A" in warm.content and "EPISODE-B" not in warm.content
    assert [u for u in requests if u.endswith(".vtt")] == [
        "https://cb.example/a.vtt", "https://cb.example/b.vtt",
    ]
    assert asr == []


def test_apple_asr_cache_is_bound_to_the_episode_not_the_audio(apple):
    """CB06->CB07: ASR-only items sharing one enclosure are separate episodes."""
    routes, requests, asr, vault = apple
    routes[A_FEED] = (
        routes[A_FEED]
        .replace('<podcast:transcript url="https://cb.example/a.vtt" type="text/vtt"/>', "")
        .replace('<podcast:transcript url="https://cb.example/b.vtt" type="text/vtt"/>', "")
    )
    a = extractors.extract(apple_url("1001"))
    b = extractors.extract(apple_url("1002"))
    assert asr == [A_AUDIO, A_AUDIO]  # B was transcribed, not served A's cache
    assert a.content == extractors.extract(apple_url("1001")).content
    assert b.content == extractors.extract(apple_url("1002")).content
    assert len(asr) == 2


def test_cache_is_bound_to_the_selected_artifact(apple):
    routes, requests, asr, vault = apple
    first = extractors.extract(apple_url("1001"))
    routes[A_FEED] = routes[A_FEED].replace("https://cb.example/a.vtt", "https://cb.example/a2.vtt")
    routes["https://cb.example/a2.vtt"] = vtt("EPISODE-A-REVISED")
    second = extractors.extract(apple_url("1001"))
    assert "EPISODE-A" in first.content and "EPISODE-A-REVISED" in second.content
    assert second.provenance.extracted_url == "https://cb.example/a2.vtt"


def test_apple_lookup_precedes_and_excludes_directory_candidates(apple, monkeypatch, tmp_path):
    """AG05/AG07: a Podcast Index title hit from another show is never consulted."""
    routes, requests, asr, vault = apple
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    foreign = "https://wrong-apple.example/feed.xml"
    routes[pi_url("Episode A")] = {"feeds": [{"url": foreign, "title": "Some Other Show"}]}
    routes[foreign] = rss(
        "Some Other Show",
        {
            "title": "Episode A",
            "audio": "https://wrong-apple.example/a.mp3",
            "vtt": "https://wrong-apple.example/a.vtt",
        },
    )
    routes["https://wrong-apple.example/a.vtt"] = vtt("WRONGSHOW")
    source = extractors.extract(apple_url("1001"))
    assert "EPISODE-A" in source.content and "WRONGSHOW" not in source.content
    assert not any("wrong-apple" in u or "podcastindex" in u for u in requests)
    result, events = cli(monkeypatch, tmp_path, apple_url("1001"))
    assert result.exit_code == 0 and [e.get("status") for e in events] == ["succeeded"]


@pytest.mark.parametrize("lookup", ["down", "empty"])
def test_apple_without_authoritative_lookup_fails_closed(apple, monkeypatch, lookup):
    routes, requests, asr, vault = apple
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    routes[pi_url("Episode A")] = {"feeds": [{"url": A_FEED, "title": "CB Show"}]}
    routes[f"https://itunes.apple.com/lookup?id={A_SHOW_ID}"] = (
        httpx.Response(503, text="") if lookup == "down" else {"results": []}
    )
    with pytest.raises(RuntimeError, match="no transcript") as error:
        extractors.extract(apple_url("1001"))
    assert A_FEED not in requests and asr == [] and cache_files(vault) == []
    assert "Apple" in str(error.value) and A_SHOW_ID in str(error.value)


# ── Generic pages (AG01/AG02/AG04, LM01) ─────────────────────────────────────
G_URL = "https://www.buzzsprout.com/111111/episodes/222222-episode-12-pricing"
G_TITLE, G_AUTHOR = "Episode 12: Pricing", "Jane Doe"
WRONGF, RIGHTF = "https://wrong.example/feed.xml", "https://feeds.buzzsprout.com/111111.rss"
OTHER_TENANT = "https://feeds.buzzsprout.com/999999.rss"


@pytest.fixture
def generic(world):
    routes, requests, asr, vault = world
    routes["https://defuddle.md/" + G_URL[8:]] = defuddle(
        G_TITLE, author=G_AUTHOR, published="2024-05-01", description="Jane Doe talks pricing."
    )
    routes[itunes(G_AUTHOR)] = {
        "results": [
            {"collectionName": "Jane Doe Comedy Hour", "feedUrl": WRONGF},
            {"collectionName": "Jane Doe Talks", "feedUrl": RIGHTF},
        ]
    }
    routes[WRONGF] = rss(
        "Jane Doe Comedy Hour",
        {
            "title": G_TITLE,
            "date": "Wed, 01 Jan 2014 00:00:00 +0000",
            "guid": "w12",
            "audio": "https://wrong.example/12.mp3",
            "vtt": "https://wrong.example/12.vtt",
        },
    )
    routes["https://wrong.example/12.vtt"] = vtt("WRONGSHOW")
    routes[RIGHTF] = rss(
        "Jane Doe Talks",
        {
            "title": G_TITLE,
            "date": "Wed, 01 May 2024 00:00:00 +0000",
            "guid": "r12",
            "audio": "https://right.example/12.mp3",
            "vtt": "https://right.example/12.vtt",
        },
    )
    routes["https://right.example/12.vtt"] = vtt("RIGHTSHOW")
    return world


def _never_wrong(requests, asr, content=""):
    assert WRONGF not in requests and not any("wrong.example" in u for u in requests)
    assert not any("wrong.example" in u for u in asr) and "WRONGSHOW" not in content


@pytest.mark.parametrize("order", ["wrong-first", "right-first"])
def test_generic_title_first_match_cannot_select_an_unrelated_show(
    generic, monkeypatch, tmp_path, order
):
    """AG01/AG03: only the page publisher's (tenant-bound) feed is a candidate."""
    routes, requests, asr, vault = generic
    if order == "right-first":
        routes[itunes(G_AUTHOR)]["results"].reverse()
    source = extractors.extract(G_URL)
    assert "RIGHTSHOW" in source.content
    assert source.provenance.extracted_url == "https://right.example/12.vtt"
    _never_wrong(requests, asr, source.content)
    result, events = cli(monkeypatch, tmp_path, G_URL)
    assert result.exit_code == 0 and "WRONGSHOW" not in result.output


def test_generic_unrelated_show_audio_never_reaches_asr(generic, monkeypatch, tmp_path):
    """AG02: with only foreign feeds listed, fail closed before any ASR."""
    routes, requests, asr, vault = generic
    routes[itunes(G_AUTHOR)]["results"] = [
        {"collectionName": "Jane Doe Comedy Hour", "feedUrl": WRONGF},
        {"collectionName": "Jane Doe Pricing", "feedUrl": OTHER_TENANT},
    ]
    routes[OTHER_TENANT] = rss(
        "Jane Doe Pricing", {"title": G_TITLE, "audio": "https://other.example/12.mp3"}
    )
    with pytest.raises(RuntimeError, match="no transcript") as error:
        extractors.extract(G_URL)
    _never_wrong(requests, asr)
    assert OTHER_TENANT not in requests  # same platform, different tenant
    assert asr == [] and cache_files(vault) == []
    assert "buzzsprout.com" in str(error.value)
    result, events = cli(monkeypatch, tmp_path, G_URL)
    assert result.exit_code == 1 and [e.get("status") for e in events] == ["failed"] and asr == []


def test_generic_podcast_index_first_hit_is_not_authority(generic, monkeypatch):
    """AG04: Podcast Index ordering cannot promote a foreign feed."""
    routes, requests, asr, vault = generic
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    routes[pi_url(G_AUTHOR)] = {"feeds": [{"url": WRONGF, "title": "Jane Doe Comedy Hour"}]}
    routes[pi_url(G_TITLE)] = {"feeds": []}
    routes[itunes(G_AUTHOR)]["results"] = [{"collectionName": "Jane Doe Talks", "feedUrl": RIGHTF}]
    source = extractors.extract(G_URL)
    assert "RIGHTSHOW" in source.content
    _never_wrong(requests, asr, source.content)


def test_generic_anchored_feed_date_contradiction_and_ambiguity_fail_closed(generic):
    routes, requests, asr, vault = generic
    routes[RIGHTF] = routes[RIGHTF].replace("Wed, 01 May 2024", "Wed, 01 May 2019")
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(G_URL)
    assert "https://right.example/12.vtt" not in requests and asr == []
    second = "https://feeds.buzzsprout.com/111111/alt.rss"
    routes[RIGHTF] = routes[RIGHTF].replace("Wed, 01 May 2019", "Wed, 01 May 2024")
    routes[itunes(G_AUTHOR)]["results"].append(
        {"collectionName": "Jane Doe Talks", "feedUrl": second}
    )
    routes[second] = rss(
        "Jane Doe Talks",
        {"title": G_TITLE, "guid": "alt", "audio": "https://right.example/alt.mp3"},
    )
    with pytest.raises(RuntimeError, match="unambiguous episode identity"):
        extractors.extract(G_URL)
    assert asr == []


def test_generic_self_hosted_publisher_feed_is_anchored(world):
    routes, requests, asr, vault = world
    url = "https://www.publisher.example/episodes/12-pricing"
    feed = "https://feeds.publisher.example/podcast.xml"
    routes[url] = "<title>Podcast episode</title><audio></audio>"
    routes["https://defuddle.md/" + url[8:]] = defuddle(
        G_TITLE, author=G_AUTHOR, description="Pricing."
    )
    routes[itunes(G_AUTHOR)] = {
        "results": [
            {"collectionName": "X", "feedUrl": "https://publisher.example.evil.test/f.xml"},
            {"collectionName": "Y", "feedUrl": feed},
        ]
    }
    routes["https://publisher.example.evil.test/f.xml"] = rss(
        "X", {"title": G_TITLE, "vtt": "https://evil.test/x.vtt"}
    )
    routes[feed] = rss(
        "Y",
        {
            "title": G_TITLE,
            "guid": "g",
            "audio": "https://cdn.example/12.mp3",
            "vtt": "https://feeds.publisher.example/12.vtt",
        },
    )
    routes["https://feeds.publisher.example/12.vtt"] = vtt("RIGHTSHOW")
    source = extractors.extract(url)
    assert "RIGHTSHOW" in source.content
    assert not any("evil.test" in u for u in requests)


def test_generic_discovery_requests_are_bounded_with_unanchored_loops(generic, monkeypatch):
    """LM01: 20 Podcast Index + 5 iTunes unanchored feeds (redirect loops) fetch nothing."""
    routes, requests, asr, vault = generic
    monkeypatch.setenv("PODCAST_INDEX_API_KEY", "dummy-key")
    monkeypatch.setenv("PODCAST_INDEX_API_SECRET", "dummy-secret")
    for query, pre in ((G_AUTHOR, "pil"), (G_TITLE, "pit")):
        routes[pi_url(query)] = {
            "feeds": [
                {"url": f"https://{pre}{i}.example/feed.xml", "title": f"{pre}{i}"}
                for i in range(10)
            ]
        }
        for i in range(10):
            u = f"https://{pre}{i}.example/feed.xml"
            routes[u] = httpx.Response(302, headers={"location": u})
    routes[itunes(G_AUTHOR)]["results"] = [
        {"collectionName": f"S{i}", "feedUrl": f"https://lim{i}.example/f.xml"} for i in range(5)
    ]
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(G_URL)
    assert not any(u.endswith(("feed.xml", "f.xml")) for u in requests)
    assert len(requests) <= 4 and asr == []
