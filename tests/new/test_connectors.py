"""Contract tests for source connector dispatch."""

from __future__ import annotations

import httpx

from obsidian_llm_wiki.core.models import SourceDoc


def test_generic_web_connector_returns_typed_failure_for_rejected_url():
    """The generic connector rejects an unsafe URL before invoking extraction."""
    from obsidian_llm_wiki.ingest.connectors import (
        ConnectorFailureKind,
        GenericWebConnector,
    )

    called = False

    def extractor(_url: str) -> SourceDoc:
        nonlocal called
        called = True
        return SourceDoc(title="unexpected", content="unexpected")

    result = GenericWebConnector(extractor=extractor).extract("http://127.0.0.1/private")

    assert result.source is None
    assert result.failure is not None
    assert result.failure.kind is ConnectorFailureKind.INVALID_URL
    assert called is False


def test_dispatcher_prefers_matching_specialist_over_generic_web_extraction():
    """A matching specialist keeps its established precedence over generic web."""
    from obsidian_llm_wiki.ingest.connectors import (
        CallableSourceConnector,
        GenericWebConnector,
        SourceConnectorDispatcher,
    )

    generic_called = False

    def specialized(_url: str) -> SourceDoc:
        return SourceDoc(title="Video", content="specialized transcript", url="https://example.com/final")

    def generic(_url: str) -> SourceDoc:
        nonlocal generic_called
        generic_called = True
        return SourceDoc(title="Web", content="generic page")

    dispatcher = SourceConnectorDispatcher(
        [
            CallableSourceConnector(
                "video", lambda _parsed, _url: True, specialized, validated_redirects=True
            )
        ],
        GenericWebConnector(extractor=generic),
    )

    result = dispatcher.dispatch("https://example.com/video")

    assert result.source is not None
    assert result.source.content == "specialized transcript"
    assert generic_called is False



def test_dispatcher_does_not_hide_specialist_failure_with_generic_content():
    """A failed matching specialist remains fail-closed instead of falling through."""
    from obsidian_llm_wiki.ingest.connectors import (
        CallableSourceConnector,
        ConnectorFailureKind,
        GenericWebConnector,
        SourceConnectorDispatcher,
    )

    generic_called = False

    def generic(_url: str) -> SourceDoc:
        nonlocal generic_called
        generic_called = True
        return SourceDoc(title="Web", content="generic page")

    dispatcher = SourceConnectorDispatcher(
        [
            CallableSourceConnector(
                "video",
                lambda _parsed, _url: True,
                lambda _url: (_ for _ in ()).throw(RuntimeError("transcript unavailable")),
                validated_redirects=True,
            )
        ],
        GenericWebConnector(extractor=generic),
    )

    result = dispatcher.dispatch("https://example.com/video")

    assert result.source is None
    assert result.failure is not None
    assert result.failure.kind is ConnectorFailureKind.EXTRACTION_FAILED
    assert result.connector_name == "specialist_dispatch"
    assert generic_called is False


def test_public_extract_stamps_generic_connector_provenance(monkeypatch):
    """Generic fallback is recorded by its connector, not as an ad-hoc branch."""
    from obsidian_llm_wiki.ingest import extractors

    monkeypatch.setattr(
        extractors,
        "extract_web",
        lambda url: SourceDoc(title="Article", content="body " * 150, url=f"{url}/final"),
    )

    source = extractors.extract("https://example.com/article")

    assert source.provenance.requested_url == "https://example.com/article"
    assert source.provenance.resolved_url == "https://example.com/article/final"
    assert source.provenance.extractor_chain == ("generic_web",)


def test_generic_web_connector_rejects_private_redirect_before_second_request(monkeypatch):
    """A generic HTML fetch checks each redirect hop before opening it."""
    from obsidian_llm_wiki.ingest import web
    from obsidian_llm_wiki.ingest.connectors import ConnectorFailureKind, GenericWebConnector

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

    real_client = httpx.Client
    monkeypatch.setattr(
        web.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 0))],
    )

    result = GenericWebConnector(
        extractor=lambda url: web._extract_trafilatura(url, timeout=10)
    ).extract("https://example.com/start")

    assert result.source is None
    assert result.failure is not None
    assert result.failure.kind is ConnectorFailureKind.EXTRACTION_FAILED
    assert requested == ["https://example.com/start"]


def test_generic_web_connector_reports_bounded_html_overflow(monkeypatch):
    """Chunked HTML over the configured cap returns a typed extraction failure."""
    from obsidian_llm_wiki.ingest import web
    from obsidian_llm_wiki.ingest.connectors import ConnectorFailureKind, GenericWebConnector

    class ChunkedBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"<html>"
            yield b"x" * 5
            raise AssertionError("stream read past configured byte cap")

    real_client = httpx.Client
    monkeypatch.setenv("MAX_HTML_BYTES", "10")
    monkeypatch.setattr(
        web.httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=ChunkedBody(), request=request)
            ),
            **kwargs,
        ),
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.url_safety.validate_remote_url", lambda _url: None
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.connectors.validate_remote_url", lambda _url: None
    )

    result = GenericWebConnector(
        extractor=lambda url: web._extract_trafilatura(url, timeout=10)
    ).extract("https://example.com/oversized")

    assert result.source is None
    assert result.failure is not None
    assert result.failure.kind is ConnectorFailureKind.EXTRACTION_FAILED
    assert "exceeded 10 bytes" in result.failure.message
