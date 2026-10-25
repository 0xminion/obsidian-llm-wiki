"""Capture-shaped S01 publisher-link regressions, with HTTP/DNS isolated.

Fixtures are synthetic (not copies of the interview); their metadata and HTML
structure reproduce the explicit label, matched RSS item and speaker/timestamps.
"""
from __future__ import annotations

import ipaddress
from xml.sax.saxutils import escape

import httpx
import pytest

from obsidian_llm_wiki.ingest import url_safety
from obsidian_llm_wiki.ingest.extractors import podcast

SOURCE = "https://open.spotify.com/episode/synthetic452"
ARTIFACT = "https://publisher.example/research-transcript"
TITLE = "#452 – Researcher Example: Models, evidence & uncertainty"
AUDIO = "https://cdn.example/452.mp3"
GUID = "publisher-452"
SPEECH = (
    "We compare the measured results with the original assumptions, and describe "
    "which observations would change our conclusions. The uncertainty is important."
)


def transcript_html(title=TITLE):
    return (
        f"<html><head><title>Transcript for {escape(title)}</title></head>\n<body>"
        "<h2>Transcript</h2>\n"
        + "".join(
            f'<div class="ts-segment"><div>{speaker}</div>'
            f'<div>({time})</div><div>{SPEECH}</div></div>'
            for speaker, time in [
                ("Researcher Example", "00:00:00"), ("Host Example", "00:00:30"),
                ("Researcher Example", "00:01:00"),
            ]
        )
        + "</body></html>"
    )


def rss(notes="", *, title=TITLE):
    return (
        '<rss><channel><title>Example Show</title><language>en</language><item>'
        f'<title>{escape(title)}</title><guid>{GUID}</guid>'
        f'<description>{escape(notes)}</description>'
        f'<enclosure url="{AUDIO}" type="audio/mpeg"/>'
        '</item></channel></rss>'
    )


@pytest.fixture
def acquisition(monkeypatch, tmp_path):
    routes = {}
    requests = []
    asr = []
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
    routes["https://itunes.apple.com/search?term=Example+Show&media=podcast&limit=5"] = {
        "results": [{"feedUrl": "https://publisher.example/rss"}],
    }
    routes["https://publisher.example/rss"] = rss()
    return routes, requests, asr


def metadata(routes, notes, *, title=TITLE):
    routes["https://defuddle.md/open.spotify.com/episode/synthetic452"] = (
        f"---\ntitle: {title}\nauthor: Example Show\nlanguage: en\n"
        f"description: {notes}\n---\n{notes}"
    )


@pytest.mark.parametrize("variant", [
    "article", "wrong-episode", "wrong-number", "login", "chapters",
])
def test_200_content_must_be_actual_same_episode_speech(acquisition, tmp_path, variant):
    routes, requests, _asr = acquisition
    metadata(routes, f"Transcript:{ARTIFACT}")
    html = transcript_html()
    if variant == "article":
        html = f"<title>Transcript for {escape(TITLE)}</title>\n<p>{SPEECH * 20}</p>"
    elif variant == "wrong-episode":
        html = transcript_html("#452 – Different guest and topic")
    elif variant == "wrong-number":
        html = transcript_html(TITLE.replace("#452", "#453"))
    elif variant == "login":
        html = html.replace("<h2>Transcript</h2>", "<p>Sign in to read the full transcript.</p>")
    elif variant == "chapters":
        html = html.replace(SPEECH, "Chapter summary.").replace(
            "<h2>Transcript</h2>", f"<p>{SPEECH * 8}</p>",
        )
    routes[ARTIFACT] = html

    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)

    assert requests.count(ARTIFACT) == 1
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/transcript", "http://localhost/transcript", "http://[::1]/transcript",
    "http://169.254.169.254/transcript", "https://private.example/transcript",
    "file:///private/transcript", "https://name:secret@publisher.example/transcript",
    "https://publisher.example/transcript(partial)",
])
def test_unsafe_or_ambiguous_url_syntax_is_never_requested(acquisition, url):
    routes, requests, _asr = acquisition
    metadata(routes, f"Transcript:{url}")
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert len(requests) == 3  # metadata, directory and RSS only


@pytest.mark.parametrize("notes", [
    f"Article: {ARTIFACT}", f"Visit {ARTIFACT}", f"[Article]({ARTIFACT})",
    f"Transcript archive: {ARTIFACT}",
    f"Transcript: {ARTIFACT}\nTranscript: https://other.example/transcript",
    "\n".join(f"Transcript: https://publisher.example/transcript-{n}" for n in range(12)),
])
def test_arbitrary_or_multiple_distinct_labels_are_not_discovery(acquisition, notes):
    routes, requests, _asr = acquisition
    metadata(routes, notes)
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert len(requests) == 3


@pytest.mark.parametrize("mode", ["unmatched", "ambiguous"])
def test_unresolved_episode_identity_does_not_authorize_metadata_links(acquisition, mode):
    routes, requests, _asr = acquisition
    metadata(routes, f"Transcript:{ARTIFACT}")
    feed = rss(title="Another episode") if mode == "unmatched" else rss()
    if mode == "ambiguous":
        item = feed[feed.index("<item>"):feed.index("</item>") + len("</item>")]
        feed = feed.replace("</channel>", item + "</channel>")
    routes["https://publisher.example/rss"] = feed
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert ARTIFACT not in requests


@pytest.mark.parametrize("destination", ["https://cdn.example/public-transcript", "http://127.0.0.1/private"])
def test_redirects_use_existing_public_url_validation(acquisition, destination):
    routes, requests, asr = acquisition
    metadata(routes, f"Transcript:{ARTIFACT}")
    routes[ARTIFACT] = httpx.Response(302, headers={"location": destination})
    if destination.startswith("https://"):
        routes[destination] = transcript_html()
        source = podcast.extract_spotify(SOURCE)
        assert SPEECH in source.content
        assert source.provenance.extracted_url == ARTIFACT
        assert requests.count(destination) == 1
        assert asr == []
    else:
        with pytest.raises(RuntimeError, match="no transcript"):
            podcast.extract_spotify(SOURCE)
        assert destination not in requests
    assert requests.count(ARTIFACT) == 1


def test_redirect_loop_has_finite_request_budget(acquisition):
    routes, requests, _asr = acquisition
    metadata(routes, f"Transcript:{ARTIFACT}")
    routes[ARTIFACT] = httpx.Response(302, headers={"location": ARTIFACT})
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert requests.count(ARTIFACT) == 6  # initial URL plus the shared five-hop redirect cap


def test_link_heading_and_long_sponsors_are_not_an_inline_transcript(acquisition, tmp_path):
    routes, requests, _asr = acquisition
    metadata(routes, f"## Transcript\n\n[Transcript]({ARTIFACT})\n\n## Sponsors\n{SPEECH * 12}")
    routes[ARTIFACT] = f"<title>Not a transcript</title><p>{SPEECH * 10}</p>"

    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)

    assert requests.count(ARTIFACT) == 1
    assert not list(tmp_path.rglob("*.json"))


def test_inline_transcript_subsections_are_not_truncated(acquisition):
    routes, _requests, asr = acquisition
    speech = f"{SPEECH * 3}\n\n### Further discussion\n\nSecond topic: {SPEECH * 3}"
    metadata(routes, f"## Transcript\n\n{speech}")

    source = podcast.extract_spotify(SOURCE)

    assert speech in source.content
    assert asr == []


@pytest.mark.parametrize("notes", [
    f"<script>Transcript: {ARTIFACT}</script>",
    f"<style>Transcript: {ARTIFACT}</style>",
    f"[Transcript: {ARTIFACT}](https://other.example/article)",
])
def test_hidden_or_nested_link_text_does_not_authorize_a_request(acquisition, notes):
    routes, requests, _asr = acquisition
    metadata(routes, notes)
    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify(SOURCE)
    assert len(requests) == 3


def test_plain_label_acquires_same_episode_before_asr_with_provenance(acquisition):
    routes, requests, asr = acquisition
    metadata(routes, f"Sponsor links. Transcript:{ARTIFACT} CONTACT: https://other.example")
    routes[ARTIFACT] = transcript_html()

    source = podcast.extract_spotify(SOURCE)

    assert source.url == SOURCE
    assert source.title == TITLE
    assert SPEECH in source.content
    assert source.provenance.extracted_url == ARTIFACT
    assert source.provenance.resolved_url == ARTIFACT
    assert source.provenance.extractor_chain == ("publisher_transcript_link",)
    cached = podcast.load_transcript_cache(f"{GUID}|{AUDIO}")
    assert cached is not None
    assert cached.artifact_url == ARTIFACT
    assert cached.provider == "publisher_transcript_link"
    assert cached.language == "en"
    assert cached.timed and cached.speaker_labeled
    assert requests.count(ARTIFACT) == 1
    assert asr == []
    del routes[ARTIFACT]
    assert podcast.extract_spotify(SOURCE).content == source.content
    assert requests.count(ARTIFACT) == 1


@pytest.mark.parametrize("field", ["description", "encoded", "summary"])
def test_labels_in_only_the_matched_rss_item_are_available(acquisition, field):
    routes, requests, asr = acquisition
    metadata(routes, "Description without a transcript link.")
    feed = rss(f"Transcript: {ARTIFACT}")
    feed = feed.replace("<description>", f"<{field}>").replace(
        "</description>", f"</{field}>",
    ).replace("<language>en</language>", "<language>pt-BR</language>")
    routes["https://publisher.example/rss"] = feed
    routes[ARTIFACT] = transcript_html()

    source = podcast.extract_spotify(SOURCE)

    assert SPEECH in source.content
    assert requests.count(ARTIFACT) == 1
    assert asr == []
    cached = podcast.load_transcript_cache(f"{GUID}|{AUDIO}")
    assert cached is not None and cached.language == "pt-BR"


@pytest.mark.parametrize("notes", [
    f'<p><b>Transcript:</b><br /><a href="{ARTIFACT}">Read here</a></p>',
    f'<p><a href="{ARTIFACT}">Full transcript</a></p>',
])
def test_rss_html_labels_use_anchor_destination(acquisition, notes):
    routes, requests, asr = acquisition
    metadata(routes, "Show notes only.")
    routes["https://publisher.example/rss"] = rss(notes)
    routes[ARTIFACT] = transcript_html()

    source = podcast.extract_spotify(SOURCE)

    assert SPEECH in source.content
    assert requests.count(ARTIFACT) == 1
    assert asr == []


def test_capture_shaped_title_and_transcript_region_preserve_all_speech(acquisition):
    routes, requests, asr = acquisition
    metadata(routes, f"Transcript:{ARTIFACT}")
    html = transcript_html().replace(
        f"Transcript for {escape(TITLE)}",
        "Transcript for Researcher Example: Models, evidence &amp; uncertainty"
        " | Example Show #452 - Host Example",
    ).replace(
        "<h2>Transcript</h2>",
        '<script>tracking_code_not_speech = true;</script><h2>Transcript</h2>',
    ).replace(
        "</body>",
        '<div>This entry was posted in transcripts on a date by Host Example.</div>'
        '<script>footer_tracking_not_speech = true;</script></body>',
    )
    routes[ARTIFACT] = html

    source = podcast.extract_spotify(SOURCE)

    assert source.content.count(SPEECH) == 3
    assert "(00:00:00)" in source.content and "(00:01:00)" in source.content
    assert "tracking_code_not_speech" not in source.content
    assert "footer_tracking_not_speech" not in source.content
    assert "This entry was posted" not in source.content
    assert asr == []
    assert requests.count(ARTIFACT) == 1


@pytest.mark.parametrize("notes", [
    f"[Transcript]({ARTIFACT})",
    f"[Full transcript]({ARTIFACT})",
    f"Transcript: [Read here]({ARTIFACT})",
    f"Transcript: <{ARTIFACT}>",
])
def test_explicit_markdown_link_forms(acquisition, notes):
    routes, requests, asr = acquisition
    metadata(routes, notes)
    routes[ARTIFACT] = transcript_html()

    source = podcast.extract_spotify(SOURCE)

    assert SPEECH in source.content
    assert requests.count(ARTIFACT) == 1
    assert asr == []
