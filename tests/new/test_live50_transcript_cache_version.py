"""Parser fixes must not keep serving known-old normalized public artifacts."""

import json

from obsidian_llm_wiki.ingest import transcript_resolver as resolver


def test_old_public_transcript_cache_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "_cache_dir", lambda: tmp_path)
    identity = "episode-guid|https://example.org/episode.mp3"
    path = tmp_path / f"{resolver._cache_key(identity)}.json"
    path.write_text(
        json.dumps(
            {
                "text": "Previously normalized public transcript with potentially lost prose. "
                * 10,
                "provider": "rss_podcast_transcript",
                "artifact_url": "https://example.org/transcript",
            }
        )
    )
    before = path.read_bytes()
    assert resolver.load_transcript_cache(identity) is None
    assert path.read_bytes() == before  # A miss is not destructive migration.


def test_current_public_cache_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "_cache_dir", lambda: tmp_path)
    result = resolver.TranscriptResult(
        text=("Correctly normalized public transcript. " * 10).strip(),
        provider="rss_podcast_transcript",
        artifact_url="https://example.org/transcript",
    )
    resolver.save_transcript_cache("episode", result)
    assert resolver.load_transcript_cache("episode") == result


def test_legacy_asr_cache_still_loads_without_retranscription(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "_cache_dir", lambda: tmp_path)
    text = "Existing ASR text was not affected by HTML parsing. " * 10
    path = tmp_path / f"{resolver._cache_key('episode')}.json"
    path.write_text(json.dumps({"text": text, "provider": "assemblyai_remote_url"}))
    result = resolver.load_transcript_cache("episode")
    assert result is not None and result.text == text.strip()
