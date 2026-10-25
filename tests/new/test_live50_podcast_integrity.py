"""Offline, acquisition-boundary regressions for podcast source integrity."""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

import httpx
import pytest

from obsidian_llm_wiki.ingest.extractors import podcast


def _item(title, guid, *, link="", transcript="", language="en"):
    artifact = (
        f'<podcast:transcript url={quoteattr(transcript)} type="text/vtt" '
        f'language={quoteattr(language)}/>' if transcript else ""
    )
    return (
        f"<item><title>{escape(title)}</title><guid>{escape(guid)}</guid>"
        f"<link>{escape(link)}</link>"
        f'<enclosure url={quoteattr(f"https://cdn.example/{guid}.mp3")} type="audio/mpeg"/>'
        f"{artifact}</item>"
    )


def _feed(*items):
    return (
        '<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">'
        '<channel><title>Example Show</title>' + "".join(items) + "</channel></rss>"
    )


@pytest.fixture
def public_http(monkeypatch, tmp_path):
    """Real parsers/resolvers/cache; only HTTP/DNS and ASR acquisition are isolated."""
    routes = {}
    requests = []
    client_type = httpx.Client

    def respond(request):
        url = str(request.url)
        requests.append(url)
        if url not in routes:
            pytest.fail(f"Unexpected HTTP acquisition: {url}")
        payload = routes[url]
        if isinstance(payload, dict):
            return httpx.Response(200, json=payload, request=request)
        return httpx.Response(200, text=payload, request=request)

    def client(**kwargs):
        kwargs.pop("proxy", None)
        return client_type(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.connectors.validate_remote_url", lambda _url: None,
    )
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    for key in ("PODCAST_INDEX_API_KEY", "PODCAST_INDEX_API_SECRET", "DEEP_SEARCH_FALLBACK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(podcast, "get_assemblyai_key", lambda: "")
    monkeypatch.setattr(podcast, "_get_supadata_key", lambda: "")
    monkeypatch.setattr(
        podcast, "_whisper_fallback_transcribe",
        lambda _url: pytest.fail("Unexpected ASR acquisition"),
    )
    return routes, requests


def test_exact_identity_beats_an_earlier_exact_title():
    rss = _feed(
        _item("Episode One", "wrong-guid"),
        _item("Canonical Episode One", "episode1"),
    )

    asset = podcast._find_episode_asset_in_rss(
        rss, episode_id="episode1", target_title="Episode One",
    )

    assert asset.guid == "episode1"
    assert asset.audio_url == "https://cdn.example/episode1.mp3"


@pytest.mark.parametrize("field", ["guid", "link"])
@pytest.mark.parametrize("value", [
    "episode10", "prefix-episode1-suffix", "longerepisode1",
    "https://publisher.example/episodes/episode10",
    "https://publisher.example/episode?i=episode10",
    "https://publisher.example/episode?length=episode1",
    "https://episode1.example/unrelated",
])
def test_episode_id_is_not_a_substring_or_unrelated_url_number(field, value):
    item = _item("Other Episode", value if field == "guid" else "other-guid",
                 link=value if field == "link" else "")

    asset = podcast._find_episode_asset_in_rss(_feed(item), episode_id="episode1")

    assert asset == podcast.EpisodeAsset()


@pytest.mark.parametrize("field", ["guid", "link"])
@pytest.mark.parametrize("value", [
    "episode1", "https://publisher.example/episodes/episode1/",
    "https://podcasts.apple.com/us/podcast/show/id123?i=episode1",
    "https://publisher.example/watch?episode_id=episode1",
])
def test_complete_identity_fields_and_url_episode_identifiers_still_match(field, value):
    item = _item("Correct Episode", value if field == "guid" else "correct-guid",
                 link=value if field == "link" else "")

    asset = podcast._find_episode_asset_in_rss(_feed(item), episode_id="episode1")

    assert asset.title == "Correct Episode"


@pytest.mark.parametrize(("requested", "other"), [
    ("Episode 1", "Episode 10"),
    ("Episode 10", "Episode 1"),
    ("AI", "Daily Science"),
    ("Markets", "Prediction Markets and Beyond"),
    ("Episode 1", "Episode 1 - Bonus"),
    ("Episode 1 - Bonus", "Episode 1"),
    ("Episode 1-0", "Episode 10"),
])
def test_titles_require_normalized_equality_not_substrings(requested, other):
    asset = podcast._find_episode_asset_in_rss(
        _feed(_item(other, "other-guid")), target_title=requested,
    )

    assert asset == podcast.EpisodeAsset()


@pytest.mark.parametrize("requested", [
    "episode 1", "  EPISODE   1! ",
    "Episode 1 - Podcast Episode - Spotify",
    "Episode 1 | Podcast on Spotify",
    "Episode 1 on Apple Podcasts",
    "Episode 1 - Apple Podcasts",
    "Episode 1 - Example Show | Spotify",
    "Episode 1 - Example Show",
])
def test_exact_normalization_and_explicit_platform_show_suffixes(requested):
    rss = _feed(_item("Episode 10", "wrong"), _item("Episode 1", "correct"))

    asset = podcast._find_episode_asset_in_rss(rss, target_title=requested)

    assert asset.guid == "correct"


@pytest.mark.parametrize("allow_first", [False, True])
@pytest.mark.parametrize("identity", [False, True])
def test_ambiguous_best_rank_never_uses_feed_order_or_weaker_title(identity, allow_first):
    rss = _feed(
        _item("Episode One", "episode1"),
        _item("Different Title" if identity else "EPISODE ONE!",
              "episode1" if identity else "episode2"),
    )

    asset = podcast._find_episode_asset_in_rss(
        rss, episode_id="episode1" if identity else "",
        target_title="Episode One", allow_first=allow_first,
    )

    assert asset == podcast.EpisodeAsset()


@pytest.mark.parametrize("target", [{"episode_id": "missing"}, {"target_title": "Missing Title"}])
def test_allow_first_cannot_override_an_explicit_unmatched_target(target):
    asset = podcast._find_episode_asset_in_rss(
        _feed(_item("Other Episode", "other")), allow_first=True, **target,
    )

    assert asset == podcast.EpisodeAsset()


def test_apple_lookup_does_not_retry_ambiguous_identity_as_a_title(public_http):
    routes, _requests = public_http
    routes["https://itunes.apple.com/lookup?id=123"] = {
        "results": [{"feedUrl": "https://publisher.example/rss"}],
    }
    routes["https://publisher.example/rss"] = _feed(
        _item("Unique Title", "456"), _item("Different Title", "456"),
    )

    asset = podcast._find_apple_episode_asset(
        "https://podcasts.apple.com/us/podcast/show/id123?i=456", "Unique Title",
    )

    assert asset == podcast.EpisodeAsset()


@pytest.mark.parametrize(
    "description", ["Brief show notes.", "Detailed show notes. " * 150], ids=["brief", "long"],
)
def test_episode_description_without_transcript_fails_actionably(
    public_http, monkeypatch, tmp_path, description,
):
    routes, _requests = public_http
    url = "https://publisher.example/rss"
    routes[url] = _feed(_item("Episode One", "episode1"))
    routes["https://defuddle.md/publisher.example/rss"] = (
        f"---\ntitle: Episode One\ndescription: {description}\n---\nShow notes only."
    )
    monkeypatch.setattr(podcast, "_whisper_fallback_transcribe", lambda _url: "")

    with pytest.raises(RuntimeError, match="no transcript") as error:
        podcast.extract_podcast_rss(url)

    assert "public transcript" in str(error.value)
    assert "clipping" in str(error.value)
    assert url in str(error.value)
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("resolved_media", [True, False])
def test_catch_all_keeps_missing_transcript_failure_instead_of_disclaiming(
    public_http, monkeypatch, resolved_media,
):
    routes, _requests = public_http
    url = "https://publisher.example/episode-1"
    routes[url] = '<html><head><title>Podcast Episode One</title></head><audio></audio></html>'
    routes["https://defuddle.md/publisher.example/episode-1"] = (
        "---\ntitle: Episode One\nauthor: Show\n"
        "description: Episode summary only.\n---\nShow notes."
    )
    routes["https://itunes.apple.com/search?term=Show&media=podcast&limit=5"] = {
        "results": [{"feedUrl": "https://publisher.example/rss"}] if resolved_media else [],
    }
    routes["https://publisher.example/rss"] = _feed(_item("Episode One", "episode1"))
    monkeypatch.setattr(podcast, "_whisper_fallback_transcribe", lambda _url: "")

    with pytest.raises(RuntimeError, match="no transcript") as error:
        podcast.extract_catch_all_podcast(url)

    assert not isinstance(error.value, podcast.ExtractorNotApplicableError)
    assert "clipping" in str(error.value)


def test_direct_feed_prefers_first_transcript_bearing_item_in_publisher_order():
    rss = _feed(
        _item("Latest audio only", "latest"),
        _item("Latest public transcript", "public", transcript="https://publisher.example/public.vtt"),
        _item("Older public transcript", "older", transcript="https://publisher.example/older.vtt"),
    )

    asset = podcast._resolve_episode_asset("https://publisher.example/rss", "rss", {}, rss_text=rss)

    assert asset.guid == "public"
    assert asset.transcript_url == "https://publisher.example/public.vtt"


def test_direct_feed_without_artifacts_preserves_first_item_for_existing_asr():
    rss = _feed(_item("Latest audio only", "latest"), _item("Older audio only", "older"))

    asset = podcast._resolve_episode_asset("https://publisher.example/rss", "rss", {}, rss_text=rss)

    assert asset.guid == "latest"
    assert asset.audio_url == "https://cdn.example/latest.mp3"


@pytest.mark.parametrize(("language", "title", "sentence"), [
    ("de", "Über Forschung", "Sprecher: Wir besprechen verlässliche Ergebnisse ohne Übersetzung."),
    ("zh-Hans", "研究方法", "主持人：我们讨论研究方法、实验结果与科学证据，不改变原文语言。"),
    (
        "pt-BR", "Pesquisa pública",
        "Apresentador: Discutimos os resultados e a evidência científica disponível.",
    ),
], ids=["de", "zh-Hans", "pt-BR"])
def test_public_transcript_provenance_and_language_survive_dispatch_and_cache(
    public_http, language, title, sentence,
):
    from obsidian_llm_wiki.ingest import extractors

    routes, requests = public_http
    url = "https://publisher.example/rss"
    artifact = "https://publisher.example/transcript.vtt"
    transcript = "\n".join([sentence] * 30)
    routes[url] = _feed(_item(title, "canonical-id", transcript=artifact, language=language))
    routes["https://defuddle.md/publisher.example/rss"] = ""
    routes[artifact] = "WEBVTT\n\n00:00.000 --> 00:30.000\n" + transcript

    source = extractors.extract(url)

    assert source.url == url
    assert source.title == title
    assert transcript in source.content
    assert "WEBVTT" not in source.content
    assert source.provenance.requested_url == url
    assert source.provenance.extracted_url == artifact
    assert source.provenance.resolved_url == artifact
    assert source.provenance.extractor_chain == ("rss_podcast_transcript", "extract_podcast_rss")
    assert source.provenance.content_type == "text/vtt"
    cached = podcast.load_transcript_cache("canonical-id|https://cdn.example/canonical-id.mp3")
    assert cached is not None
    assert cached.language == language
    assert cached.artifact_url == artifact
    assert cached.text == transcript
    del routes[artifact]
    repeated = extractors.extract(url)
    assert repeated.content == source.content
    assert repeated.provenance.extracted_url == artifact
    assert requests.count(artifact) == 1


def test_empty_spotify_metadata_is_not_fetched_twice(public_http):
    routes, requests = public_http
    metadata_url = "https://defuddle.md/open.spotify.com/episode/episode1"
    routes[metadata_url] = ""

    with pytest.raises(RuntimeError, match="no transcript"):
        podcast.extract_spotify("https://open.spotify.com/episode/episode1")

    assert requests == [metadata_url]


@pytest.mark.parametrize("title", ["Life on Spotify", "Life - Example Show"])
def test_literal_episode_title_wins_before_platform_suffix_interpretation(title):
    rss = _feed(_item("Life", "shorter"), _item(title, "literal"))

    asset = podcast._find_episode_asset_in_rss(rss, target_title=title)

    assert asset.guid == "literal"


def _ingest(monkeypatch, config, *args):
    import importlib

    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app

    module = importlib.import_module("obsidian_llm_wiki.cli.ingest")
    monkeypatch.setattr(module, "resolve_vault", lambda _vault: (config.vault, config))
    return CliRunner().invoke(
        app, ["ingest", str(config.vault), "--parallel", "0", "--json", "--skip-synthesis", *args],
    )


def test_missing_transcript_never_publishes_a_durable_source(public_http, monkeypatch, tmp_path):
    import json

    from obsidian_llm_wiki.config import Config

    routes, _requests = public_http
    url = "https://publisher.example/rss"
    routes[url] = _feed(_item("Episode One", "episode1"))
    routes["https://defuddle.md/publisher.example/rss"] = (
        "---\ntitle: Episode One\ndescription: "
        + "Long show notes. " * 100 + "\n---\nNo transcript."
    )
    monkeypatch.setattr(podcast, "_whisper_fallback_transcribe", lambda _url: "")
    config = Config(vault_path=str(tmp_path))

    result = _ingest(monkeypatch, config, "--url", url)

    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert any(event.get("status") == "failed" for event in events), result.output
    assert "no transcript" in result.output
    assert "clipping" in result.output
    assert list(config.sources_dir.glob("*.md")) == []


def test_explicit_manual_description_clipping_still_publishes(public_http, monkeypatch, tmp_path):
    from obsidian_llm_wiki.config import Config

    _routes, requests = public_http
    config = Config(vault_path=str(tmp_path))
    config.clippings_dir.mkdir(parents=True)
    # Keep the existing clipping length gate: explicit does not mean content-free.
    description = "## Episode Description\n\n" + (
        "Show notes deliberately imported as a description, not as an episode transcript. " * 25
    ).strip()
    (config.clippings_dir / "episode-description.md").write_text(
        "---\ntitle: Manual Episode Description\n---\n" + description, encoding="utf-8",
    )

    result = _ingest(monkeypatch, config)

    assert result.exit_code == 0, result.output
    sources = list(config.sources_dir.glob("*.md"))
    assert len(sources) == 1, result.output
    assert description in sources[0].read_text(encoding="utf-8")
    assert requests == []


def test_catch_all_without_episode_evidence_still_disclaims(public_http):
    routes, _requests = public_http
    url = "https://publisher.example/blog"
    routes[url] = '<html><title>News about a podcast</title><body>Article.</body></html>'
    routes["https://defuddle.md/publisher.example/blog"] = ""

    with pytest.raises(podcast.ExtractorNotApplicableError):
        podcast.extract_catch_all_podcast(url)


def test_inline_public_transcript_succeeds_without_rss_or_asr(public_http):
    routes, _requests = public_http
    url = "https://open.spotify.com/episode/episode1"
    transcript = "A speaker discusses evidence and the limits of this research. " * 25
    routes["https://defuddle.md/open.spotify.com/episode/episode1"] = (
        "---\ntitle: Episode One\nauthor: Show\n---\n## Transcript\n\n" + transcript
    )
    routes["https://itunes.apple.com/search?term=Show&media=podcast&limit=5"] = {"results": []}

    source = podcast.extract_spotify(url)

    assert transcript.strip() in source.content
    assert source.url == url
    assert source.provenance.extracted_url == url
    assert source.provenance.extractor_chain == ("publisher_episode_page",)
