"""Lossless transcript regressions distilled from live A03/Y08 captures.

All committed prose is original test text, not captured copyrighted captions.
Only transport/file acquisition is substituted; public extraction stays real.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import httpx
import pytest

from obsidian_llm_wiki.ingest import transcript_resolver
from obsidian_llm_wiki.ingest.extractors import youtube


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("Transcript fidelity regressions must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda url: None,
    )


def _publisher_transcript(monkeypatch, raw, mime_type):
    url = "https://publisher.example.test/episode/transcript"
    real_client = httpx.Client

    def respond(request):
        assert str(request.url) == url
        return httpx.Response(200, content=raw.encode(), headers={"content-type": mime_type})

    monkeypatch.setattr(
        transcript_resolver.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = transcript_resolver.fetch_public_transcript(url, mime_type=mime_type)
    assert result is not None
    assert result.artifact_url == url
    return result.text


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        ("&lt;5MB RAM &amp; &gt;0 cores", "<5MB RAM & >0 cores"),
        ("&#60;5MB RAM &#x26; &#62;0 cores", "<5MB RAM & >0 cores"),
        ("内存&lt;五兆，预算&#x3c;十元 &amp; 成功", "内存<五兆，预算<十元 & 成功"),
        ("literal < 5 and 8 > 3", "literal < 5 and 8 > 3"),
        ("&lt;em&gt;literal&lt;/em&gt; &amp;lt;", "<em>literal</em> &lt;"),
    ],
)
def test_public_html_preserves_literal_angles_and_following_prose(monkeypatch, encoded, decoded):
    before = "The engineer recorded a resource budget during the morning experiment. "
    after = (
        ". The next paragraph explains why the measurement matters to the operator. "
        "A later observation records the same result without discarding any of the original prose."
    )
    raw = f"<article><p>{before}<strong>{encoded}</strong>{after}</p></article>"
    assert _publisher_transcript(monkeypatch, raw, "text/html") == before + decoded + after


_CONTEXT = (
    "The field engineer records each observation independently, even when a speaker repeats "
    "the same instruction. The transcript should retain these separate events, preserve their "
    "order, and avoid treating presentation metadata as part of the spoken explanation."
)


def _youtube_text(monkeypatch, tmp_path, raw, suffix=".vtt"):
    """Replay subtitle acquisition through the real public YouTube extractor."""
    url = "https://www.youtube.com/watch?v=testvideo01"
    real_client = httpx.Client
    downloads = []

    def respond(request):
        assert request.url.path == "/oembed"
        return httpx.Response(200, json={"title": "Original transcript fidelity example"})

    def download(cmd, **kwargs):
        assert "--skip-download" in cmd
        assert cmd[-1] == url
        path = Path(cmd[cmd.index("-o") + 1]).parent / f"original.en{suffix}"
        path.write_text(raw, encoding="utf-8")
        downloads.append(path)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        youtube.httpx, "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(
        youtube.shutil, "which", lambda name: "/offline/yt-dlp" if name == "yt-dlp" else None,
    )
    monkeypatch.setattr(youtube.subprocess, "run", download)
    source = youtube.extract_youtube_video(url)
    assert len(downloads) == 1
    assert not downloads[0].exists(), "The real extractor must clean up its download"
    assert source.url == url
    return source.content


@pytest.mark.parametrize("speech", ["Please repeat the measurement.", "请重复这次测量。"])
@pytest.mark.parametrize("gap", [0, 1])
def test_nonoverlapping_vtt_preserves_all_six_repetitions(monkeypatch, tmp_path, speech, gap):
    cues = [
        f"00:00:{i * (2 + gap):02d}.000 --> 00:00:{i * (2 + gap) + 2:02d}.000\n{speech}"
        for i in range(6)
    ]
    cues.append(f"00:01:00.000 --> 00:02:00.000\n{_CONTEXT}")
    raw = "WEBVTT\nKind: captions\nLanguage: en\n\n" + "\n\n".join(cues)
    expected = " ".join([speech] * 6 + [_CONTEXT])
    assert _youtube_text(monkeypatch, tmp_path, raw) == expected


def test_vtt_metadata_blocks_are_not_speech_but_numeric_payload_is(monkeypatch, tmp_path):
    raw = (
        "\ufeffWEBVTT - example\r\nKind: captions\r\nLanguage: zh\r\n\r\n"
        "NOTE editorial note\r\nDo not speak this note.\r\n\r\n"
        "STYLE\r\n::cue { color: lime; }\r\n\r\n"
        "REGION\r\nid:upper\r\nwidth:40%\r\n\r\n"
        "measurement-cue\r\n00:00:01.000 --> 00:00:02.000 align:start position:0%\r\n"
        "42\r\nNOTE is spoken here, not a comment block.\r\n\r\n"
        f"00:00:02.000 --> 00:00:09.000\r\n{_CONTEXT}"
    )
    expected = f"42 NOTE is spoken here, not a comment block. {_CONTEXT}"
    assert _youtube_text(monkeypatch, tmp_path, raw) == expected


def test_srt_retains_numeric_speech_and_repeated_cues(monkeypatch, tmp_path):
    raw = (
        "1\n00:00:01,000 --> 00:00:03,000\n2026\nRead that again.\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\n2026\nRead that again.\n\n"
        f"3\n00:00:05,000 --> 00:00:09,000\n{_CONTEXT}\n"
    )
    expected = f"2026 Read that again. 2026 Read that again. {_CONTEXT}"
    assert _youtube_text(monkeypatch, tmp_path, raw, ".srt") == expected


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        ("<v Engineer><c.green>Budget &lt;5MB &amp; safe</c></v>", "Budget <5MB & safe"),
        ("<lang zh>内存&#60;五兆，成本&#x3c;十元。</lang>", "内存<五兆，成本<十元。"),
        ("<b>Literal < 5</b> and 8 > 3", "Literal < 5 and 8 > 3"),
        ("&lt;c&gt;literal&lt;/c&gt; &amp;lt;", "<c>literal</c> &lt;"),
        ("<c>First</c><00:00:01.500><c> &amp; second</c>", "First & second"),
    ],
)
def test_vtt_decodes_entities_once_after_cue_markup(monkeypatch, tmp_path, encoded, decoded):
    raw = (
        f"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n{encoded}\n\n"
        f"00:00:03.000 --> 00:00:09.000\n{_CONTEXT}\n"
    )
    assert _youtube_text(monkeypatch, tmp_path, raw) == f"{decoded} {_CONTEXT}"


@pytest.mark.parametrize(
    ("first", "second", "third", "expected"),
    [
        (
            "We measure the current", "the current every morning", "every morning before sunrise",
            "We measure the current every morning before sunrise",
        ),
        (
            "第一句。\n第二句。", "第二句。\n第三句。", "第三句。\n第四句。",
            "第一句。 第二句。 第三句。 第四句。",
        ),
        (
            "Repeated display.", "Repeated display.", "Repeated display. A new observation.",
            "Repeated display. A new observation.",
        ),
    ],
)
def test_rolling_cues_merge_only_overlapping_suffix_prefix(
    monkeypatch, tmp_path, first, second, third, expected,
):
    raw = (
        f"WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n{first}\n\n"
        f"00:00:02.000 --> 00:00:05.000\n{second}\n\n"
        f"00:00:03.000 --> 00:00:06.000\n{third}\n\n"
        f"00:00:06.000 --> 00:00:09.000\n{third}\n\n"
        f"00:00:10.000 --> 00:00:20.000\n{_CONTEXT}\n"
    )
    # The last display is spoken again at a nonoverlapping time, not deduplicated.
    assert _youtube_text(monkeypatch, tmp_path, raw) == (
        f"{expected} {third.replace(chr(10), ' ')} {_CONTEXT}"
    )


def test_identical_overlapping_speech_from_different_voices_is_not_collapsed(monkeypatch, tmp_path):
    raw = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<v Alice>Ready now.</v>\n\n"
        "00:00:02.000 --> 00:00:05.000\n<v Bob>Ready now.</v>\n\n"
        f"00:00:05.000 --> 00:00:09.000\n{_CONTEXT}\n"
    )
    assert _youtube_text(monkeypatch, tmp_path, raw) == f"Ready now. Ready now. {_CONTEXT}"


def test_repetition_inside_one_cue_is_not_collapsed(monkeypatch, tmp_path):
    raw = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nAgain.\nAgain.\n再说一遍。\n再说一遍。\n\n"
        f"00:00:05.000 --> 00:00:09.000\n{_CONTEXT}\n"
    )
    expected = f"Again. Again. 再说一遍。 再说一遍。 {_CONTEXT}"
    assert _youtube_text(monkeypatch, tmp_path, raw) == expected


def test_public_vtt_does_not_leak_inline_word_timestamps(monkeypatch):
    raw = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:09.000\n"
        "<c>The field</c><00:00:01.500><c> engineer</c>"
        f"{_CONTEXT.removeprefix('The field engineer')}"
    )
    assert _publisher_transcript(monkeypatch, raw, "text/vtt") == _CONTEXT
