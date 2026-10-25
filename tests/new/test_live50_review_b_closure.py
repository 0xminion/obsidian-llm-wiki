"""Review B admission regressions through public dispatch and CLI persistence.

Synthetic original-language fixtures; only HTTP, DNS and optional acquisition
are isolated. No parser, admission check, cache or writer is replaced.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from xml.sax.saxutils import escape

import httpx
import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.ingest import extractors
from obsidian_llm_wiki.ingest import transcript_resolver as tr
from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

URL = "https://podcasts.apple.com/us/podcast/review-show/id24680?i=71"
FEED = "https://publisher.example/rss"
ART = "https://publisher.example/review-transcript"
TITLE = "Episode 71: Careful measurement"
SPEECH = (
    "We repeat the calibration because independent measurements should retain their "
    "observations, rather than silently merging separate events."
)
SENTENCE = "Budget <5MB and literal <tag> survive; 中文预算<十元。"
IDENTITY = "71|https://cdn.example/71.mp3"


def item(title=TITLE, guid="71", notes="", artifact=""):
    return (
        f"<item><title>{escape(title)}</title><guid>{guid}</guid>"
        f"<description>{escape(notes)}</description>"
        f'<enclosure url="https://cdn.example/{guid}.mp3" type="audio/mpeg"/>'
        + (f'<podcast:transcript url="{artifact}" type="text/html" language="zh-Hans"/>'
           if artifact else "") + "</item>"
    )


def feed(*items):
    return (
        '<rss xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>'
        '<title>Review Show</title><language>zh-Hans</language>'
        + "".join(items) + "</channel></rss>"
    )


def html(title=TITLE):
    return (
        "<html><head><title>Transcript for " + escape(title) + "</title></head><body>"
        + "".join(
            f"<div>{speaker}</div><div>(00:0{i}:00)</div>"
            f"<p>{escape(SENTENCE)} {SPEECH}</p>"
            for i, speaker in enumerate(["Alice", "Bob", "Alice"])
        ) + "</body></html>"
    )


def metadata(title=TITLE, body="Only ordinary show notes."):
    return (
        f"---\ntitle: {title}\nauthor: Review Show\nlanguage: zh-Hans\n"
        f"description: Ordinary episode metadata, not full speech.\n---\n{body}"
    )


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PROXY")):
            monkeypatch.delenv(key, raising=False)
    for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "TMPDIR"):
        directory = tmp_path / key.lower()
        directory.mkdir()
        monkeypatch.setenv(key, str(directory))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    def deny(*args, **kwargs):
        pytest.fail("Real socket forbidden")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
    ])
    routes = {
        URL: "<title>Publisher page</title><p>Public shell.</p>",
        "https://defuddle.md/" + URL.removeprefix("https://"): metadata(),
        "https://itunes.apple.com/lookup?id=24680": {"results": [{"feedUrl": FEED}]},
        FEED: feed(item()),
        ART: html(),
    }
    requests = []
    real_client = httpx.Client

    def respond(request):
        url = str(request.url)
        requests.append(url)
        assert url in routes, f"Unexpected acquisition: {url}"
        value = routes[url]
        if isinstance(value, httpx.Response):
            return value
        if isinstance(value, dict):
            return httpx.Response(200, json=value)
        return httpx.Response(200, text=value, headers={"content-type": "text/html"})

    def client(*args, **kwargs):
        kwargs.pop("transport", None)
        kwargs["trust_env"] = False
        return real_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "extract-vault"))
    return routes, requests


def run_cli(monkeypatch, tmp_path, url=URL):
    vault = tmp_path / "cli-vault"
    vault.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(vault))
    result = CliRunner().invoke(app, [
        "ingest", str(vault), "--parallel", "0", "--json", "--skip-synthesis", "--url", url,
    ])
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    files = list(Config(vault_path=str(vault)).sources_dir.glob("*.md"))
    return result, events, files


def rejected(monkeypatch, tmp_path, url=URL):
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(url)
    assert not list(tr._cache_dir().glob("*.json"))
    result, events, files = run_cli(monkeypatch, tmp_path, url)
    assert result.exit_code == 1, result.output
    assert any(event.get("event") == "failed" for event in events), result.output
    assert files == []
    assert not list(tr._cache_dir().glob("*.json"))


def test_canonical_rss_title_wins_over_contradictory_storefront(offline, monkeypatch, tmp_path):
    routes, _ = offline
    routes["https://defuddle.md/" + URL.removeprefix("https://")] = metadata(
        "Episode 72: Wrong storefront metadata",
    )
    routes[FEED] = feed(item(artifact=ART))
    source = extractors.extract(URL)
    result, events, files = run_cli(monkeypatch, tmp_path)
    assert result.exit_code == 0, result.output
    assert any(event.get("event") == "completed" for event in events)
    assert len(files) == 1
    frontmatter, _ = parse_frontmatter(files[0].read_text())
    assert source.title == frontmatter["title"] == TITLE
    assert source.content in files[0].read_text()
    assert source.content.count(SENTENCE) == source.content.count(SPEECH) == 3
    assert source.provenance.extracted_url == ART
    cached = tr.load_transcript_cache(IDENTITY)
    assert cached is not None and cached.language == "zh-Hans"


@pytest.mark.parametrize("route", ["apple", "spotify", "generic"])
def test_ambiguous_identity_never_authorizes_inline(offline, monkeypatch, tmp_path, route):
    routes, _ = offline
    url = URL
    if route != "apple":
        url = ("https://open.spotify.com/episode/71" if route == "spotify"
               else "https://publisher.example/episodes/71")
        routes[url] = '<title>Podcast episode</title><audio></audio>'
        routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
            "results": [{"feedUrl": FEED}],
        }
    routes["https://defuddle.md/" + url.removeprefix("https://")] = metadata(
        body="## Transcript\n\n" + (SPEECH + "\n") * 8,
    )
    routes[FEED] = feed(item(), item("Duplicate identity" if route == "apple" else TITLE))
    rejected(monkeypatch, tmp_path, url)


@pytest.mark.parametrize("variant", ["apple-show", "spotify-show", "unresolved-apple"])
def test_show_or_unresolved_canonical_episode_cannot_publish_inline(
    offline, monkeypatch, tmp_path, variant,
):
    routes, _ = offline
    url = URL
    if variant == "apple-show":
        url = URL.split("?")[0]
    elif variant == "spotify-show":
        url = "https://open.spotify.com/show/review"
        routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
            "results": [{"feedUrl": FEED}],
        }
    else:
        routes[FEED] = feed(item("An unrelated episode", "unrelated"))
    # Even a title identical to an RSS item must not turn a show URL into an episode.
    routes[url] = '<title>Podcast page</title><audio></audio>'
    routes["https://defuddle.md/" + url.removeprefix("https://")] = metadata(
        body="## Transcript\n\n" + (SPEECH + "\n") * 8,
    )
    rejected(monkeypatch, tmp_path, url)


@pytest.mark.parametrize("label", [
    "[Transcript](http://127.0.0.1/private)",
    '<a href="http://127.0.0.1/private">Full transcript</a>',
    "Transcript: file:///private/transcript",
    f"Transcript: {ART}\nTranscript: https://publisher.example/other",
])
def test_link_and_padding_are_not_inline_speech(offline, monkeypatch, tmp_path, label):
    routes, requests = offline
    routes["https://defuddle.md/" + URL.removeprefix("https://")] = metadata(
        body="## Transcript\n\n" + label + "\n\n"
        + "Sponsor notes and chapter summaries are not verbatim episode speech. " * 12,
    )
    rejected(monkeypatch, tmp_path)
    assert ART not in requests
    assert not any("127.0.0.1" in request for request in requests)


@pytest.mark.parametrize("header", ["title", "h1"])
def test_rss_tag_does_not_authorize_explicitly_conflicting_document(
    offline, monkeypatch, tmp_path, header,
):
    routes, _ = offline
    routes[FEED] = feed(item(artifact=ART))
    routes[ART] = html("Episode 72: Another experiment")
    if header == "h1":
        routes[ART] = routes[ART].replace("<title>", "<h1>").replace("</title>", "</h1>")
    rejected(monkeypatch, tmp_path)


@pytest.mark.parametrize("provider", [
    "rss_podcast_transcript", "publisher_episode_page", "publisher_transcript_link",
])
def test_pre_admission_public_cache_misses_without_mutating_evidence(
    offline, monkeypatch, tmp_path, provider,
):
    routes, requests = offline
    cache = tr._cache_dir() / f"{tr._cache_key(IDENTITY)}.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({
        "text": SPEECH * 8, "provider": provider, "artifact_url": ART,
        "normalization_version": 2,
    }))
    before = cache.read_bytes()
    assert tr.load_transcript_cache(IDENTITY) is None
    assert cache.read_bytes() == before
    routes[FEED] = feed(item(artifact=ART))
    routes[ART] = html("Episode 72: Another experiment")
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(URL)
    assert cache.read_bytes() == before  # no destructive migration on failed refresh
    routes[ART] = html()
    source = extractors.extract(URL)
    assert SENTENCE in source.content
    saved = json.loads(cache.read_text())
    assert saved["normalization_version"] == 2
    assert saved["podcast_admission_version"] == 1
    cached = tr.load_transcript_cache(IDENTITY)
    assert cached is not None and cached.document_title == "Transcript for " + TITLE
    requests.clear()
    assert extractors.extract(URL).content == source.content
    assert ART not in requests


@pytest.mark.parametrize("path,title", [
    ("/show/review", TITLE), ("/episodes/71", ""), ("/", TITLE),
])
def test_unknown_generic_inline_identity_fails_closed(
    offline, monkeypatch, tmp_path, path, title,
):
    routes, _ = offline
    url = "https://publisher.example" + path
    routes[url] = '<title>Podcast page</title><audio></audio>'
    routes["https://defuddle.md/" + url.removeprefix("https://")] = metadata(
        title, "## Transcript\n\n" + (SPEECH + "\n") * 8,
    )
    routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
        "results": [],
    }
    rejected(monkeypatch, tmp_path, url)


@pytest.mark.parametrize("url", [
    "https://publisher.example/episodes/71", "https://open.spotify.com/episode/71",
    "https://publisher.example/shows/review/episodes/71",
])
def test_explicit_inline_episode_without_rss_retains_exact_body_and_language(
    offline, monkeypatch, tmp_path, url,
):
    routes, requests = offline
    speech = (SENTENCE + " " + SPEECH + "\n") * 8
    speech += "\n### Discussion of Episode 72\n\n" + SPEECH
    routes[url] = '<title>Podcast episode</title><audio></audio>'
    routes["https://defuddle.md/" + url.removeprefix("https://")] = metadata(
        body="## Transcript\n\n" + speech,
    )
    routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
        "results": [],
    }
    source = extractors.extract(url)
    assert source.content.endswith(speech)
    cached = tr.load_transcript_cache(url)
    assert cached is not None and cached.text == speech
    assert cached.language == "zh-Hans"
    assert extractors.extract(url).content == source.content
    result, events, files = run_cli(monkeypatch, tmp_path, url)
    assert result.exit_code == 0, result.output
    assert len(files) == 1 and source.content in files[0].read_text()
    assert any(event.get("event") == "completed" for event in events)
    assert ART not in requests


def test_rss_bound_artifact_may_use_show_number_instead_of_unnumbered_episode_title(offline):
    routes, _ = offline
    routes[FEED] = feed(item("All the careful measurements", artifact=ART))
    routes[ART] = html("Review Show #71")
    source = extractors.extract(URL)
    assert source.title == "All the careful measurements"
    assert source.content.count(SENTENCE) == source.content.count(SPEECH) == 3


def test_spoken_episode_references_are_not_document_identity(offline):
    routes, _ = offline
    routes[FEED] = feed(item(artifact=ART))
    speech = "In Episode 72 we discussed an unrelated experiment. " + SPEECH
    routes[ART] = html().replace(SPEECH, speech)
    source = extractors.extract(URL)
    assert source.content.count(speech) == 3
    assert source.content.count(SENTENCE) == 3


@pytest.mark.parametrize("matched_rss", [False, True])
def test_generic_show_inline_cannot_escape_through_web_fallback(
    offline, monkeypatch, tmp_path, matched_rss,
):
    routes, _ = offline
    url = "https://publisher.example/show/review"
    speech = (SPEECH + "\n") * 8
    routes[url] = "<title>Podcast page</title><audio></audio><article>" + speech + "</article>"
    routes["https://defuddle.md/" + url.removeprefix("https://")] = (
        f"---\ntitle: {TITLE}\nauthor: Review Show\n---\n## Transcript\n\n{speech}"
    )  # No description field: rejection must remain a claimed-podcast failure.
    routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
        "results": [{"feedUrl": FEED}] if matched_rss else [],
    }
    rejected(monkeypatch, tmp_path, url)


def test_no_rss_inline_cache_cannot_relabel_a_different_document(offline):
    routes, _ = offline
    url = "https://open.spotify.com/episode/71"
    routes[url] = "<title>Publisher page</title>"
    routes["https://defuddle.md/" + url.removeprefix("https://")] = metadata()
    routes["https://itunes.apple.com/search?term=Review+Show&media=podcast&limit=5"] = {
        "results": [],
    }
    tr.save_transcript_cache(url, tr.TranscriptResult(
        text=SPEECH * 8, provider="publisher_episode_page", artifact_url=url,
        document_title="An entirely different experiment",
    ))
    cache = tr._cache_dir() / f"{tr._cache_key(url)}.json"
    before = cache.read_bytes()
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(url)
    assert cache.read_bytes() == before


def test_current_inline_cache_must_match_its_own_document_title(offline):
    tr.save_transcript_cache(IDENTITY, tr.TranscriptResult(
        text=SPEECH * 8, provider="publisher_episode_page", artifact_url=URL,
        document_title="An entirely different experiment",
    ))
    cache = tr._cache_dir() / f"{tr._cache_key(IDENTITY)}.json"
    before = cache.read_bytes()
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(URL)
    assert cache.read_bytes() == before


def test_wrong_metadata_inline_never_reaches_cache_or_publication(offline, monkeypatch, tmp_path):
    routes, _ = offline
    routes["https://defuddle.md/" + URL.removeprefix("https://")] = metadata(
        "Episode 72: Another experiment", "## Transcript\n\n" + (SPEECH + "\n") * 8,
    )
    rejected(monkeypatch, tmp_path)
