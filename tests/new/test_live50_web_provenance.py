"""Regressions derived from live archive and empty hosted-response captures."""

from contextlib import contextmanager

import httpx
import pytest

from obsidian_llm_wiki.ingest import web


def test_wayback_keeps_snapshot_url_and_requested_source_separate(monkeypatch):
    original = "https://medium.com/example/article"
    snapshot = "https://web.archive.org/web/20250505060754/" + original
    response = httpx.Response(
        200,
        text="<html><title>An archived article</title><p>Original archived prose.</p></html>",
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", snapshot),
    )
    monkeypatch.setattr(web, "get_with_validated_redirects", lambda *_: response)
    result = web._extract_wayback(original, 5)
    assert result.url == original
    assert result.provenance.requested_url == original
    assert result.provenance.resolved_url == snapshot
    assert result.provenance.extracted_url == snapshot
    assert "wayback" in result.provenance.extractor_chain
    assert "archived_snapshot: 20250505060754" in result.provenance.diagnostics


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_hosted_frontmatter_without_body_is_rejected_inside_strategy(monkeypatch, ending):
    text = '---\ntitle: "A valid document title"\ndescription: "' + "Metadata only. " * 20
    text += '"\n---' + ending
    response = httpx.Response(
        200, text=text, request=httpx.Request("GET", "https://defuddle.md/doc")
    )

    @contextmanager
    def stream(*_args):
        yield response

    monkeypatch.setattr(web, "stream_with_validated_redirects", stream)
    with pytest.raises(RuntimeError, match="content too short"):
        web._extract_defuddle_md("https://example.org/document", 5)


def test_hosted_yaml_title_and_fetch_provenance(monkeypatch):
    requested = "https://example.org/article"
    hosted = "https://defuddle.md/example.org/article"
    body = "A complete paragraph with original observations and evidence. " * 20
    response = httpx.Response(
        200,
        text='---\r\ntitle: "A: quoted --- title"\r\n---\r\n' + body,
        request=httpx.Request("GET", hosted),
    )

    @contextmanager
    def stream(*_args):
        yield response

    monkeypatch.setattr(web, "stream_with_validated_redirects", stream)
    result = web._extract_defuddle_md(requested, 5)
    assert result.title == "A: quoted --- title"
    assert result.content == body.strip()
    assert result.provenance.requested_url == requested
    assert result.provenance.extracted_url == hosted
    assert "defuddle.md" in result.provenance.extractor_chain
