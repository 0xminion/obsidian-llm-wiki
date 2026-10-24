"""Offline extraction boundary regressions from the baseline audit."""

from __future__ import annotations

import sys

import pytest

from obsidian_llm_wiki.config import Config


def test_liteparse_overflow_cannot_return_success():
    from obsidian_llm_wiki.ingest.liteparse import _run_liteparse

    with pytest.raises(RuntimeError, match="stdout.*exceed"):
        _run_liteparse(
            [sys.executable, "-c", "print('Evidence sentence. ' * 100)"],
            Config(max_parser_stdout_bytes=10),
        )


def test_scientific_discovery_respects_candidate_budget(monkeypatch):
    from obsidian_llm_wiki.ingest.extractors import scientific

    monkeypatch.setenv("MAX_DOCUMENT_CANDIDATES", "2")
    html = "".join(f'<a href="/{index}.pdf">PDF</a>' for index in range(12))
    monkeypatch.setattr(scientific, "_fetch_public_html", lambda *_args: html)
    attempts = []

    def unavailable(url):
        attempts.append(url)
        raise RuntimeError("fixture unavailable")

    monkeypatch.setattr(scientific, "_extract_pdf", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        scientific.extract_discovered_scientific_document("https://example.test/paper")
    assert len(attempts) == 2


def test_xml_podcast_reaches_rss_extractor_instead_of_jats(monkeypatch):
    import httpx

    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.ingest import connectors, extractors
    from obsidian_llm_wiki.ingest.extractors import podcast

    feed = (
        "<rss><channel><title>Show</title><item><title>Episode</title><description>"
        + "Description evidence. " * 40
        + '</description><enclosure url="https://cdn.example.test/audio.mp3" '
        'type="audio/mpeg"/></item></channel></rss>'
    )
    client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=feed, request=request))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None
    )
    monkeypatch.setattr(connectors, "validate_remote_url", lambda _url: None)
    monkeypatch.setattr(
        podcast,
        "_extract_podcast",
        lambda url, **_kw: SourceDoc(title="Episode", content="Actual transcript. " * 40, url=url),
    )
    source = extractors.extract("https://example.test/podcast.xml")
    assert "extract_podcast_rss" in source.provenance.extractor_chain
    assert source.title == "Episode"


@pytest.mark.parametrize(
    ("title", "content", "extractor"),
    [
        (
            "Video",
            "Note: Full transcript unavailable. Only video metadata was extracted.",
            "extract_youtube_video",
        ),
        ("Access denied", "Please sign in to continue. " * 40, "extract_jats"),
    ],
)
def test_remote_specialists_cannot_admit_explicit_unavailable_shells(title, content, extractor):
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.ingest.extractors import _stamp_extracted_source

    with pytest.raises(RuntimeError, match="unavailable|restricted"):
        _stamp_extracted_source(
            SourceDoc(title=title, content=content), "https://example.test/source", extractor
        )


def test_html_redirect_provenance_records_the_actual_document(monkeypatch):
    import httpx
    import trafilatura

    from obsidian_llm_wiki.ingest import web

    requested = "https://example.test/old"
    resolved = "https://example.test/current"
    client = httpx.Client

    def handle(request):
        if str(request.url) == requested:
            return httpx.Response(302, headers={"location": resolved}, request=request)
        return httpx.Response(
            200,
            text="<title>Actual article</title>" + "Body " * 200,
            headers={"content-type": "text/html"},
            request=request,
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw)
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None
    )
    monkeypatch.setattr(trafilatura, "extract", lambda *_a, **_k: "Actual evidence. " * 40)
    monkeypatch.setattr(trafilatura, "extract_metadata", lambda *_a: None)
    source = web._extract_trafilatura(requested, 1)
    assert source.provenance.requested_url == requested
    assert source.provenance.resolved_url == resolved
    assert source.provenance.extracted_url == resolved
    assert source.provenance.content_type == "text/html"


def test_concurrent_transcript_cache_writes_use_independent_temporary_files(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from threading import Barrier

    from obsidian_llm_wiki.ingest.transcript_resolver import (
        TranscriptResult,
        load_transcript_cache,
        save_transcript_cache,
    )

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    barrier = Barrier(2)
    replace = Path.replace

    def synchronized_replace(path, destination):
        barrier.wait(timeout=5)
        return replace(path, destination)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    results = [
        TranscriptResult(text=(f"Transcript {i}. " * 40).strip(), provider="fixture")
        for i in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save_transcript_cache, "same-episode", result) for result in results]
        for future in futures:
            future.result()
    assert load_transcript_cache("same-episode") in results
    assert not list(tmp_path.rglob("*.tmp"))


def test_docx_extra_extracts_real_document_without_liteparse(tmp_path, monkeypatch):
    docx = pytest.importorskip("docx")
    from obsidian_llm_wiki.ingest import extractors, liteparse

    path = tmp_path / "note.docx"
    document = docx.Document()
    document.core_properties.title = "Native document"
    document.add_paragraph("Evidence from a Word file. " * 40)
    document.save(path)
    monkeypatch.setattr(liteparse.shutil, "which", lambda _name: None)
    source = extractors.extract(str(path))
    assert source.title == "Native document"
    assert "Evidence from a Word file." in source.content


def test_scientific_html_uses_configured_stream_limit(monkeypatch):
    import httpx

    from obsidian_llm_wiki.ingest.extractors import scientific

    monkeypatch.setenv("MAX_HTML_BYTES", "10")
    client = httpx.Client
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text="<html>" + "untrusted data " * 100, request=request
        )
    )
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None
    )
    with pytest.raises(RuntimeError, match="exceed"):
        scientific._fetch_public_html("https://example.test/paper", 1)


def test_document_discovery_rejects_oversized_landing_response(monkeypatch):
    import httpx

    from obsidian_llm_wiki.ingest import documents

    client = httpx.Client
    html = '<a href="/paper.pdf">PDF</a>' + "padding " * 100
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=html, request=request))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None
    )
    attempts = []

    def no_download(url, **_kw):
        attempts.append(url)
        raise documents.DocumentError("fixture failed")

    monkeypatch.setattr(documents, "download_document", no_download)
    with pytest.raises(documents.DocumentError):
        documents.extract_discovered_document(
            "https://example.test/landing", 1, config=Config(max_html_bytes=10)
        )
    assert attempts == []
