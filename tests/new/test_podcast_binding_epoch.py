"""Podcast cache binding epoch v3, operator-attested import and identity contradictions.

Closes the independent final-review findings:

* H1 - a version-1 media-keyed entry (``guid|enclosure`` / ``guid`` / ``enclosure``)
  written by the real HEAD code for episode A was served for episode B in eight
  variants, including feed rotation after B's ``pubDate``; the previous migration
  also copied that text under B's version-2 key with a fresh ``acquired_at`` and
  no migration marker, so a poisoned v2 entry is indistinguishable from a fresh
  one. Neither v1 nor v2 entries are read automatically any more; only epoch-3
  entries carrying explicit episode + artifact provenance are served.
* H2 - an exact unique title admitted a feed item whose own ``<link>``/``<guid>``
  named another same-host episode ID or another Apple ``?i=`` episode, and an Apple
  title-only match whose ``pubDate`` contradicted the page by two years.

All HTTP runs through real ``httpx`` clients on a ``MockTransport``; DNS is a
table; ASR is a recording double. Nothing here contacts a network or provider.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
import pytest

from obsidian_llm_wiki.ingest import extractors, url_safety
from obsidian_llm_wiki.ingest import transcript_resolver as tr
from obsidian_llm_wiki.ingest.extractors import podcast

FEED = "https://pub.example/feed.xml"
AUDIO = "https://cdn.pub.example/episode.mp3"
VTT_URL = "https://pub.example/transcript.vtt"
DATE_A = "Mon, 01 Jan 2024 00:00:00 +0000"
DATE_B = "Sat, 01 Jun 2024 00:00:00 +0000"
DATE_C = "Mon, 01 Jul 2024 00:00:00 +0000"
A_TEXT = "EPISODE-A-SPOKEN alpha discussion of measured results and assumptions. " * 6
B_TEXT = "EPISODE-B-SPOKEN beta discussion of entirely different subject matter. " * 6


def vtt_body(tag: str) -> str:
    return "WEBVTT\n\n" + "".join(
        f"00:00:{n * 5:02d}.000 --> 00:00:{n * 5 + 5:02d}.000\n<v S{n % 2}>{tag} cue {n}: "
        "this sentence is synthetic speech used only to identify the artifact.\n\n"
        for n in range(8)
    )


def rss(show: str, *items: dict) -> str:
    def one(i):
        s = f"<item><title>{escape(i['title'])}</title><pubDate>{i['date']}</pubDate>"
        if i.get("guid"):
            s += f'<guid isPermaLink="false">{escape(i["guid"])}</guid>'
        if i.get("link"):
            s += f"<link>{escape(i['link'])}</link>"
        s += "<description>d</description>"
        if i.get("audio"):
            s += f'<enclosure url="{i["audio"]}" length="1" type="audio/mpeg"/>'
        if i.get("vtt"):
            s += f'<podcast:transcript url="{i["vtt"]}" type="text/vtt"/>'
        return s + "</item>"

    return (
        '<?xml version="1.0"?><rss version="2.0" '
        'xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel>'
        f"<title>{escape(show)}</title><language>en</language>"
        + "".join(one(i) for i in items)
        + "</channel></rss>"
    )


def defuddle(title: str, published: str = "", author: str = "") -> str:
    fm = [f'title: "{title}"']
    if author:
        fm.append(f'author: "{author}"')
    if published:
        fm.append(f"published: {published}")
    fm.append('description: "d"')
    return "---\n" + "\n".join(fm) + "\n---\n" + "placeholder body " * 5


def apple_url(ep: str) -> str:
    return f"https://podcasts.apple.com/us/podcast/x/id999?i={ep}"


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
        if value.startswith("WEBVTT"):
            return httpx.Response(200, text=value, headers={"content-type": "text/vtt"})
        ctype = "application/rss+xml" if value.lstrip().startswith("<?xml") else "text/html"
        if value.startswith("---"):
            ctype = "text/markdown"
        return httpx.Response(200, text=value, headers={"content-type": ctype})

    def client(*args, **kwargs):
        kwargs.pop("proxy", None)
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(
        url_safety, "_resolved_addresses", lambda host: {ipaddress.ip_address("93.184.216.34")}
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


def cache_dir(vault: Path) -> Path:
    return vault / "04-Wiki" / ".llmwiki" / "transcripts"


def snapshot(vault: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(cache_dir(vault).glob("*.json"))}


def write_head_entry(identity: str, text: str, *, provider: str, artifact_url: str,
                     acquired_at: str) -> Path:
    """Byte-format of the real HEAD 7158bf2 ``save_transcript_cache`` payload."""
    path = tr._cache_dir() / f"{tr._cache_key(identity)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text, "provider": provider, "artifact_url": artifact_url, "language": "",
        "timed": False, "speaker_labeled": False, "document_title": "", "identity": identity,
        "normalization_version": 2, "podcast_admission_version": 1, "acquired_at": acquired_at,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apple_routes(routes, feed_xml):
    routes["https://itunes.apple.com/lookup?id=999"] = {
        "results": [{"collectionId": 999, "feedUrl": FEED}]
    }
    routes[FEED] = feed_xml
    for ep, title, day in (("1001", "Episode A Alpha", "2024-01-01"),
                           ("1002", "Episode B Beta", "2024-06-01")):
        routes["https://defuddle.md/" + apple_url(ep)[8:]] = defuddle(title, day)


A = {"title": "Episode A Alpha", "date": DATE_A}
B = {"title": "Episode B Beta", "date": DATE_B}

# (name, route, guid, legacy identity, artifact kind, acquired_at)
H1_CASES = [
    ("rss_guid_audio", "rss", "G1", f"G1|{AUDIO}", "audio", "2024-01-05T00:00:00Z"),
    ("apple_guid_audio", "apple", "G1", f"G1|{AUDIO}", "audio", "2024-01-05T00:00:00Z"),
    ("rss_audio_only", "rss", "", AUDIO, "audio", "2024-01-05T00:00:00Z"),
    ("apple_audio_only", "apple", "", AUDIO, "audio", "2024-01-05T00:00:00Z"),
    ("rss_tag_guid_only", "rss", "G1", "G1", "tag", "2024-01-05T00:00:00Z"),
    # Rotation after B's pubDate: the entry post-dates B, a timestamp guard cannot help.
    ("apple_cb_rotation_late", "apple", "", AUDIO, "audio", "2024-06-05T00:00:00Z"),
    ("apple_guidaudio_rotation_late", "apple", "G1", f"G1|{AUDIO}", "audio",
     "2024-06-05T00:00:00Z"),
    ("apple_cb_rotation", "apple", "", AUDIO, "audio", "2024-01-05T00:00:00Z"),
]


def _h1_world(world, route, guid, kind):
    routes, requests, asr, vault = world
    if kind == "tag":
        item_b = {**B, "guid": guid, "vtt": VTT_URL}
        routes[VTT_URL] = vtt_body("EPISODE-B-SPOKEN")
    else:
        item_b = {**B, "guid": guid, "audio": AUDIO}
        routes[("asr", AUDIO)] = B_TEXT
    if route == "rss":
        routes[FEED] = rss("Pub", item_b)
        return FEED
    apple_routes(routes, rss("Pub", item_b))
    return apple_url("1002")


@pytest.mark.parametrize("name,route,guid,legacy_id,kind,acquired", H1_CASES,
                         ids=[c[0] for c in H1_CASES])
def test_h1_head_v1_entry_of_episode_a_never_serves_episode_b(
    world, name, route, guid, legacy_id, kind, acquired,
):
    routes, requests, asr, vault = world
    url = _h1_world(world, route, guid, kind)
    if kind == "tag":
        write_head_entry(legacy_id, A_TEXT, provider="rss_podcast_transcript",
                         artifact_url=VTT_URL, acquired_at=acquired)
    else:
        write_head_entry(legacy_id, A_TEXT, provider="assemblyai_remote_url",
                         artifact_url=AUDIO, acquired_at=acquired)
    legacy = snapshot(vault)
    cold = extractors.extract(url)
    warm = extractors.extract(url)
    for source in (cold, warm):
        assert "EPISODE-A" not in source.content
        assert "EPISODE-B-SPOKEN" in source.content
    # Fresh acquisition exactly once (cold), then the fresh epoch-3 entry is reused.
    assert len(asr) == (1 if kind == "audio" else 0)
    if kind == "tag":
        assert requests.count(VTT_URL) == 1
    after = snapshot(vault)
    assert all(after[k] == v for k, v in legacy.items())  # legacy byte-identical


def _v2_identity(binding: dict, kind: str, url: str) -> str:
    return "olw-podcast-transcript-binding-v2\n" + json.dumps(
        {**binding, "artifact": [kind, url]}, sort_keys=True, ensure_ascii=False)


def test_h1_already_poisoned_v2_entry_is_never_served(world):
    """Old migration copied A's text under B's v2 key with a fresh acquired_at."""
    routes, requests, asr, vault = world
    url = _h1_world(world, "apple", "G1", "audio")
    write_head_entry(f"G1|{AUDIO}", A_TEXT, provider="assemblyai_remote_url",
                     artifact_url=AUDIO, acquired_at="2024-06-05T00:00:00Z")
    binding_b = {"episode": "feed-item", "feed": FEED, "guid": "G1", "audio": AUDIO,
                 "title": "episode b beta", "published": DATE_B}
    write_head_entry(_v2_identity(binding_b, "audio", AUDIO), A_TEXT,
                     provider="assemblyai_remote_url", artifact_url=AUDIO,
                     acquired_at="2026-10-27T22:42:45Z")
    before = snapshot(vault)
    for _ in range(2):
        source = extractors.extract(url)
        assert "EPISODE-A" not in source.content and "EPISODE-B-SPOKEN" in source.content
    assert len(asr) == 1
    after = snapshot(vault)
    assert all(after[k] == v for k, v in before.items())


def test_genuine_v2_entry_of_the_same_episode_also_misses_once(world):
    """A v2 entry cannot be told apart from a poisoned one: no automatic promotion."""
    routes, requests, asr, vault = world
    url = _h1_world(world, "rss", "G1", "audio")
    binding_b = {"episode": "feed-item", "feed": FEED, "guid": "G1", "audio": AUDIO,
                 "title": "episode b beta", "published": DATE_B}
    write_head_entry(_v2_identity(binding_b, "audio", AUDIO), B_TEXT,
                     provider="assemblyai_remote_url", artifact_url=AUDIO,
                     acquired_at="2026-10-27T22:42:45Z")
    extractors.extract(url)
    extractors.extract(url)
    assert len(asr) == 1


def test_same_episode_v1_control_is_not_reused_automatically(world):
    """Positive control of the review (feed gained C): reuse now requires an import."""
    routes, requests, asr, vault = world
    item_a = {**A, "guid": "G1", "audio": AUDIO}
    item_c = {"title": "Episode C Gamma", "date": DATE_C, "guid": "G3",
              "audio": "https://cdn.pub.example/c.mp3"}
    apple_routes(routes, rss("Pub", item_c, item_a))
    routes[("asr", AUDIO)] = A_TEXT
    write_head_entry(f"G1|{AUDIO}", A_TEXT, provider="assemblyai_remote_url",
                     artifact_url=AUDIO, acquired_at="2024-01-05T00:00:00Z")
    source = extractors.extract(apple_url("1001"))
    assert "EPISODE-A-SPOKEN" in source.content
    assert asr == [AUDIO]  # documented one-time refresh cost


def bound_payloads(vault: Path) -> list[dict]:
    out = []
    for path in cache_dir(vault).glob("*.json"):
        payload = json.loads(path.read_text())
        if payload.get("identity", "").startswith("olw-podcast-transcript-binding-v3\n"):
            out.append(payload)
    return out


def test_fresh_v3_entry_records_episode_and_artifact_provenance_and_is_reused(world):
    routes, requests, asr, vault = world
    url = _h1_world(world, "apple", "G1", "audio")
    first = extractors.extract(url)
    second = extractors.extract(url)
    assert first.content == second.content and len(asr) == 1
    [payload] = bound_payloads(vault)
    prov = payload["podcast_binding"]
    assert prov["epoch"] == 3 and prov["origin"] == "acquired"
    assert prov["artifact"] == ["audio", AUDIO]
    assert prov["episode"]["guid"] == "G1" and prov["episode"]["feed"] == FEED
    assert prov["episode"]["title"] == "episode b beta"
    assert "operator-attested" not in first.content


def test_v3_key_without_matching_provenance_is_a_miss(world):
    """A hand-copied/old payload under a v3 file name does not masquerade as fresh."""
    routes, requests, asr, vault = world
    url = _h1_world(world, "rss", "G1", "audio")
    extractors.extract(url)
    [path] = list(cache_dir(vault).glob("*.json"))
    payload = json.loads(path.read_text())
    payload["text"] = A_TEXT
    del payload["podcast_binding"]
    path.write_text(json.dumps(payload))
    source = extractors.extract(url)
    assert "EPISODE-A" not in source.content and len(asr) == 2


# ── Operator-attested import ─────────────────────────────────────────────

def _control_world(world):
    routes, requests, asr, vault = world
    item_a = {**A, "guid": "G1", "audio": AUDIO}
    apple_routes(routes, rss("Pub", item_a))
    legacy = write_head_entry(f"G1|{AUDIO}", A_TEXT, provider="assemblyai_remote_url",
                              artifact_url=AUDIO, acquired_at="2024-01-05T00:00:00Z")
    return apple_url("1001"), legacy


def test_describe_reports_exact_bindings_and_legacy_candidates_without_acquiring(world):
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    url, legacy = _control_world(world)
    before = snapshot(vault)
    report = rec.describe(url)
    assert asr == [] and VTT_URL not in requests
    [audio] = [b for b in report["bindings"] if b["artifact"] == ["audio", AUDIO]]
    assert audio["identity"].startswith("olw-podcast-transcript-binding-v3\n")
    assert audio["digest"] == hashlib.sha256(audio["identity"].encode()).hexdigest()
    assert audio["v3_entry"] is None
    [cand] = audio["legacy_candidates"]
    assert cand["file"] == str(legacy) and cand["epoch"] == 1
    assert cand["sha256"] == hashlib.sha256(legacy.read_bytes()).hexdigest()
    assert cand["acquired_at"] == "2024-01-05T00:00:00Z"
    assert cand["acknowledgement"] == rec.acknowledgement(cand["sha256"], audio["digest"])
    assert snapshot(vault) == before  # read-only


def test_operator_import_requires_exact_attestation_and_preserves_provenance(world):
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    url, legacy = _control_world(world)
    original = legacy.read_bytes()
    report = rec.describe(url)
    [audio] = [b for b in report["bindings"] if b["artifact"] == ["audio", AUDIO]]
    sha = hashlib.sha256(original).hexdigest()
    ack = rec.acknowledgement(sha, audio["digest"])
    with pytest.raises(rec.ImportRefusedError, match="sha256"):
        rec.import_legacy(legacy, legacy_sha256="0" * 64, binding_identity=audio["identity"],
                          acknowledgement=ack)
    with pytest.raises(rec.ImportRefusedError, match="acknowledgement"):
        rec.import_legacy(legacy, legacy_sha256=sha, binding_identity=audio["identity"],
                          acknowledgement=ack.replace("attest", "agree"))
    with pytest.raises(rec.ImportRefusedError, match="binding"):
        rec.import_legacy(legacy, legacy_sha256=sha,
                          binding_identity=audio["identity"].replace("v3", "v2", 1),
                          acknowledgement=ack)
    record = rec.import_legacy(legacy, legacy_sha256=sha, binding_identity=audio["identity"],
                               acknowledgement=ack, operator="test-operator")
    assert legacy.read_bytes() == original
    payload = json.loads(Path(record["file"]).read_text())
    assert payload["acquired_at"] == "2024-01-05T00:00:00Z"  # never re-stamped as fresh
    prov = payload["podcast_binding"]
    assert prov["origin"] == "operator_import" and prov["artifact"] == ["audio", AUDIO]
    imp = prov["import"]
    assert imp["legacy_epoch"] == 1 and imp["legacy_identity"] == f"G1|{AUDIO}"
    assert imp["legacy_sha256"] == sha and imp["operator"] == "test-operator"
    assert imp["original_acquired_at"] == "2024-01-05T00:00:00Z" and imp["imported_at"]
    with pytest.raises(rec.ImportRefusedError, match="already"):
        rec.import_legacy(legacy, legacy_sha256=sha, binding_identity=audio["identity"],
                          acknowledgement=ack)
    source = extractors.extract(url)
    assert asr == []  # reused without paid re-transcription
    assert "EPISODE-A-SPOKEN" in source.content
    assert "operator-attested import" in source.content
    assert "2024-01-05T00:00:00Z" in source.content
    assert any("operator-attested" in d for d in source.provenance.diagnostics)


def test_import_refuses_legacy_entry_for_a_different_artifact_or_identity(world):
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    url, legacy = _control_world(world)
    other = write_head_entry("G9|https://cdn.pub.example/other.mp3", A_TEXT,
                             provider="assemblyai_remote_url",
                             artifact_url="https://cdn.pub.example/other.mp3",
                             acquired_at="2024-01-05T00:00:00Z")
    report = rec.describe(url)
    [audio] = [b for b in report["bindings"] if b["artifact"] == ["audio", AUDIO]]
    sha = hashlib.sha256(other.read_bytes()).hexdigest()
    with pytest.raises(rec.ImportRefusedError, match="artifact|identity"):
        rec.import_legacy(other, legacy_sha256=sha, binding_identity=audio["identity"],
                          acknowledgement=rec.acknowledgement(sha, audio["digest"]))
    assert bound_payloads(vault) == []


def test_import_of_a_v2_entry_records_its_epoch_and_contamination_risk(world, tmp_path):
    """A v2 entry can be imported only by attestation; it stays labelled as legacy."""
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    item_a = {**A, "guid": "G1", "audio": AUDIO}
    apple_routes(routes, rss("Pub", item_a))
    url = apple_url("1001")
    [audio] = [b for b in rec.describe(url)["bindings"] if b["artifact"][0] == "audio"]
    episode = {k: v for k, v in json.loads(audio["identity"].split("\n", 1)[1]).items()
               if k not in ("artifact", "authority")}
    v2 = write_head_entry(_v2_identity(episode, "audio", AUDIO), A_TEXT,
                          provider="assemblyai_remote_url", artifact_url=AUDIO,
                          acquired_at="2026-10-27T22:42:45Z")
    [cand] = [b for b in rec.describe(url)["bindings"]
              if b["artifact"][0] == "audio"][0]["legacy_candidates"]
    assert cand["epoch"] == 2 and "former automatic migration" in cand["warning"]
    binding_file = tmp_path / "binding.txt"
    binding_file.write_text(audio["identity"] + "\n", encoding="utf-8")
    code = rec.main([
        "import", "--legacy-file", str(v2), "--legacy-sha256", cand["sha256"],
        "--binding-file", str(binding_file), "--acknowledgement", cand["acknowledgement"],
        "--operator", "cli-test",
    ])
    assert code == 0
    [payload] = bound_payloads(vault)
    assert payload["podcast_binding"]["import"]["legacy_epoch"] == 2
    assert payload["acquired_at"] == "2026-10-27T22:42:45Z"
    source = extractors.extract(url)
    assert asr == [] and "legacy epoch-2" in source.content
    assert "not freshly acquired" in source.content


def test_cli_import_refusal_writes_nothing(world, tmp_path, capsys):
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    url, legacy = _control_world(world)
    [audio] = [b for b in rec.describe(url)["bindings"] if b["artifact"][0] == "audio"]
    binding_file = tmp_path / "binding.txt"
    binding_file.write_text(audio["identity"], encoding="utf-8")
    before = snapshot(vault)
    code = rec.main([
        "import", "--legacy-file", str(legacy), "--legacy-sha256",
        audio["legacy_candidates"][0]["sha256"], "--binding-file", str(binding_file),
        "--acknowledgement", "yes",
    ])
    assert code == 2 and "acknowledgement" in capsys.readouterr().err
    assert snapshot(vault) == before


def test_import_refuses_pre_admission_public_entry(world):
    """Public artifacts that current admission rejects are refetched free, not imported."""
    from obsidian_llm_wiki.ingest import podcast_cache_recovery as rec

    routes, requests, asr, vault = world
    item = {**A, "guid": "G1", "vtt": VTT_URL}
    routes[FEED] = rss("Pub", item)
    routes[VTT_URL] = vtt_body("EPISODE-A-SPOKEN")
    path = write_head_entry("G1", A_TEXT, provider="rss_podcast_transcript",
                            artifact_url=VTT_URL, acquired_at="2024-01-05T00:00:00Z")
    payload = json.loads(path.read_text())
    del payload["podcast_admission_version"]
    path.write_text(json.dumps(payload))
    [tag] = [b for b in rec.describe(FEED)["bindings"] if b["artifact"][0] == "tag"]
    [cand] = tag["legacy_candidates"]
    assert cand["admissible"] is False
    with pytest.raises(rec.ImportRefusedError, match="admission"):
        rec.import_legacy(path, legacy_sha256=cand["sha256"], binding_identity=tag["identity"],
                          acknowledgement=cand["acknowledgement"])
    assert bound_payloads(vault) == []


# ── H2: explicit item identity contradicting the requested episode ─────

TITLE = "Ask Me Anything"
AUTHOR = "Jane Doe"
PAGE = "https://www.buzzsprout.com/111111/episodes/333333-ask-me-anything"
OTHER = "https://www.buzzsprout.com/111111/episodes/444444-ask-me-anything"
GFEED = "https://feeds.buzzsprout.com/111111.rss"
ITEM_VTT = "https://pub.example/item.vtt"
ITEM_DATE = "Sat, 01 Mar 2025 00:00:00 +0000"
APPLE_1001 = apple_url("1001")


def _item(**kw):
    return {"title": TITLE, "date": ITEM_DATE, "audio": "https://cdn.pub.example/x.mp3",
            "vtt": ITEM_VTT, **kw}


def _generic(world, *, published="", **item):
    routes, requests, asr, vault = world
    routes["https://defuddle.md/" + PAGE[8:]] = defuddle(TITLE, published, AUTHOR)
    routes[str(httpx.URL("https://itunes.apple.com/search",
                         params={"term": AUTHOR, "media": "podcast", "limit": 5}))] = {
        "results": [{"collectionName": "Jane Doe Talks", "feedUrl": GFEED}]}
    routes[GFEED] = rss("Jane Doe Talks", _item(**item))
    routes[ITEM_VTT] = vtt_body("FEED-ITEM")
    return PAGE


def _apple(world, ep, *, published="", items):
    routes, requests, asr, vault = world
    url = apple_url(ep)
    routes["https://defuddle.md/" + url[8:]] = defuddle(TITLE, published)
    routes["https://itunes.apple.com/lookup?id=999"] = {
        "results": [{"collectionId": 999, "feedUrl": FEED}]}
    routes[FEED] = rss("Jane Doe Talks", *items)
    routes[ITEM_VTT] = vtt_body("FEED-ITEM")
    routes["https://pub.example/other.vtt"] = vtt_body("ID-ITEM")
    return url


H2_REJECT = {
    "G-wrong link+guid undated": lambda w: _generic(w, link=OTHER, guid="Buzzsprout-444444"),
    "G-wrong link undated": lambda w: _generic(w, link=OTHER),
    "G-wrong link same day": lambda w: _generic(w, published="2025-03-01", link=OTHER),
    "G-wrong guid URL same host": lambda w: _generic(w, guid=OTHER),
    "G-wrong short-path link": lambda w: _generic(
        w, link="https://buzzsprout.com/111111/444444"),
    "A-wrong guid ?i=1001": lambda w: _apple(w, "1002", published="2023-01-10",
                                             items=[_item(guid=APPLE_1001)]),
    "A-wrong link ?i=1001": lambda w: _apple(w, "1002", published="2023-01-10",
                                             items=[_item(link=APPLE_1001, guid="opaque-1")]),
    "A-wrong ?i undated page": lambda w: _apple(w, "1002", items=[_item(guid=APPLE_1001)]),
    "A-wrong date title-only": lambda w: _apple(w, "1002", published="2023-01-10",
                                                items=[_item(guid="opaque-1")]),
}


@pytest.mark.parametrize("case", list(H2_REJECT), ids=list(H2_REJECT))
def test_h2_contradictory_item_identity_fails_closed(world, case):
    routes, requests, asr, vault = world
    url = H2_REJECT[case](world)
    with pytest.raises(RuntimeError, match="no transcript"):
        extractors.extract(url)
    assert asr == [] and ITEM_VTT not in requests
    assert not list(cache_dir(vault).glob("*.json"))


H2_ADMIT = {
    "G-alias exact": lambda w: _generic(w, link=PAGE, guid="Buzzsprout-333333"),
    "G-alias http": lambda w: _generic(w, link=PAGE.replace("https", "http")),
    "G-alias no-www": lambda w: _generic(w, link=PAGE.replace("www.", "")),
    "G-alias slash+utm": lambda w: _generic(w, link=PAGE + "/?utm_source=rss"),
    "G-alias publisher site": lambda w: _generic(
        w, link="https://janedoe.example/episodes/444444-ask-me-anything"),
    "G-alias short id path": lambda w: _generic(
        w, link="https://www.buzzsprout.com/111111/333333"),
    "G-alias no link": lambda w: _generic(w),
    "G-alias opaque guid only": lambda w: _generic(w, guid="Buzzsprout-444444"),
    "G-unrelated Apple guid": lambda w: _generic(w, guid=APPLE_1001),
    "A-alias guid ?i=1002": lambda w: _apple(w, "1002", published="2025-03-01",
                                             items=[_item(guid=apple_url("1002"))]),
    "A-alias no IDs same day": lambda w: _apple(w, "1002", published="2025-03-01",
                                                items=[_item(guid="opaque-1")]),
    "A-alias undated page": lambda w: _apple(w, "1002", items=[_item(guid="opaque-1")]),
    "A-unrelated publisher link": lambda w: _apple(w, "1002", published="2025-03-01", items=[
        _item(guid="opaque-1", link="https://www.buzzsprout.com/111111/episodes/1001-x")]),
}


@pytest.mark.parametrize("case", list(H2_ADMIT), ids=list(H2_ADMIT))
def test_h2_legitimate_aliases_are_preserved(world, case):
    url = H2_ADMIT[case](world)
    source = extractors.extract(url)
    assert "FEED-ITEM" in source.content


def test_h2_apple_id_match_beats_title_and_skips_title_date_rule(world):
    url = _apple(world, "1002", published="2023-01-10", items=[
        _item(title="Different", guid="1002", vtt="https://pub.example/other.vtt",
              audio="https://cdn.pub.example/y.mp3"),
        _item(guid="opaque-1")])
    source = extractors.extract(url)
    assert "ID-ITEM" in source.content


def test_h2_contradiction_reason_leads_the_error(world):
    url = _apple(world, "1002", items=[_item(guid=APPLE_1001)])
    with pytest.raises(RuntimeError) as error:
        extractors.extract(url)
    assert "names another Apple episode" in str(error.value)
