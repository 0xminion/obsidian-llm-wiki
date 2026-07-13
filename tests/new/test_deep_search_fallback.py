"""Tests for the deep search fallback in ingest/extractors/__init__.py.

Covers:
  - _deep_search_enabled: env-gated, off by default
  - _title_from_url: deriving a search query from a URL
  - _deep_search_fallback: picks longest-content candidate, provenance tagging,
    all-providers-fail → RuntimeError
  - Per-provider parsers (Semantic Scholar, OpenAlex, arXiv, Crossref) with
    mocked HTTP responses
  - extract() integration: disabled by default (no network), stub replacement
    when enabled, recovery after specialized-extractor failure when enabled
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import (
    _deep_search_arxiv,
    _deep_search_crossref,
    _deep_search_enabled,
    _deep_search_fallback,
    _deep_search_openalex,
    _deep_search_semantic_scholar,
    _title_from_url,
    extract,
)

# ── _deep_search_enabled ────────────────────────────────────────────────


def test_deep_search_disabled_by_default(monkeypatch):
    """No env var → disabled (keeps CI offline and deterministic)."""
    monkeypatch.delenv("DEEP_SEARCH_FALLBACK", raising=False)
    assert _deep_search_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "True", "YES"])
def test_deep_search_enabled_truthy_values(monkeypatch, val):
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", val)
    assert _deep_search_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "  "])
def test_deep_search_disabled_falsy_values(monkeypatch, val):
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", val)
    assert _deep_search_enabled() is False


# ── _title_from_url ──────────────────────────────────────────────────────


def test_title_from_url_arxiv_abs():
    """arXiv abstract URL → last path segment as the query."""
    assert _title_from_url("https://arxiv.org/abs/1706.03762") == "1706.03762"


def test_title_from_url_pdf_extension_stripped():
    """PDF extensions are stripped from the query."""
    url = "https://example.com/papers/attention-is-all-you-need.pdf"
    assert _title_from_url(url) == "attention is all you need"


def test_title_from_url_html_extension_stripped():
    """HTML extensions are stripped from the query."""
    assert _title_from_url("https://example.com/articles/my-article.html") == "my article"


def test_title_from_url_query_string_ignored():
    """Query strings are ignored."""
    assert _title_from_url("https://example.com/paper?foo=bar") == "paper"


def test_title_from_url_bare_path():
    """A bare path with no host still produces a query."""
    assert _title_from_url("/some/deep-path-to-paper") == "deep path to paper"


def test_title_from_url_fallback_to_host():
    """When the path is empty, fall back to the hostname."""
    assert _title_from_url("https://example.com/") == "example.com"


# ── HTTP client mock helper ──────────────────────────────────────────────


class _FakeResponse:
    """Minimal httpx.Response stub for mocking deep search HTTP calls."""

    def __init__(self, *, status_code: int = 200, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://fake"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._json


class _FakeClient:
    """Context-manager httpx.Client stub that returns a canned response."""

    def __init__(self, response: _FakeResponse, **_kwargs):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, _url, **_kwargs):
        return self._response


# ── Semantic Scholar provider ────────────────────────────────────────────


def test_semantic_scholar_success():
    """A well-formed Semantic Scholar response produces a SourceDoc."""
    payload = {
        "data": [{
            "title": "Attention Is All You Need",
            "abstract": "We propose a new architecture... " * 10,
            "year": 2017,
            "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
            "externalIds": {"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762"},
            "url": "https://api.semanticscholar.org/paper/123",
        }],
    }
    resp = _FakeResponse(json_data=payload)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        doc = _deep_search_semantic_scholar("Attention Is All You Need", timeout=10)
    assert doc is not None
    assert "Attention Is All You Need" in doc.content
    assert "Vaswani" in doc.content
    assert "2017" in doc.content
    assert "1706.03762" in doc.content
    assert len(doc.content) >= 100


def test_semantic_scholar_no_results():
    """Empty data list → None (not raise)."""
    resp = _FakeResponse(json_data={"data": []})
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        assert _deep_search_semantic_scholar("Nonexistent Paper", timeout=10) is None


def test_semantic_scholar_404():
    """404 → None."""
    resp = _FakeResponse(status_code=404)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        assert _deep_search_semantic_scholar("Nonexistent", timeout=10) is None


def test_semantic_scholar_network_error_returns_none():
    """Network errors are swallowed and return None."""
    class _ErrorClient:
        def __init__(self, **_kw): pass
        def __enter__(self): return self
        def __exit__(self, *_a): return None
        def get(self, _url, **_kw): raise httpx.ConnectError("connection refused")
    with patch("httpx.Client", side_effect=lambda **kw: _ErrorClient(**kw)):
        assert _deep_search_semantic_scholar("Anything", timeout=10) is None


# ── OpenAlex provider ────────────────────────────────────────────────────


def test_openalex_success():
    """A well-formed OpenAlex response produces a SourceDoc."""
    payload = {
        "results": [{
            "title": "Deep Residual Learning for Image Recognition",
            "abstract_inverted_index": {
                "Deep": [0], "residual": [1], "learning": [2],
                "framework": [3],
            },
            "publication_year": 2016,
            "authorships": [
                {"author": {"display_name": "Kaiming He"}},
                {"author": {"display_name": "Xiangyu Zhang"}},
            ],
            "doi": "10.1109/cvpr.2016.90",
            "ids": {"openalex": "https://openalex.org/W123"},
        }],
    }
    resp = _FakeResponse(json_data=payload)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        doc = _deep_search_openalex("Deep Residual Learning", timeout=10)
    assert doc is not None
    assert "Deep Residual Learning" in doc.content
    assert "Kaiming He" in doc.content
    assert "2016" in doc.content
    assert "Deep residual learning framework" in doc.content
    assert len(doc.content) >= 100


def test_openalex_no_results():
    """Empty results → None."""
    resp = _FakeResponse(json_data={"results": []})
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        assert _deep_search_openalex("Nonexistent", timeout=10) is None


# ── arXiv provider ────────────────────────────────────────────────────────


def test_arxiv_success():
    """A well-formed arXiv Atom feed entry produces a SourceDoc."""
    xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v1</id>
    <title>Attention Is All You Need</title>
    <summary>Sequence transduction models based on neural networks.</summary>
    <published>2017-06-15T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""
    resp = _FakeResponse(text=xml)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        doc = _deep_search_arxiv("Attention Is All You Need", timeout=10)
    assert doc is not None
    assert "Attention Is All You Need" in doc.content
    assert "Vaswani" in doc.content
    assert "2017" in doc.content
    assert "transduction models" in doc.content.lower()
    assert len(doc.content) >= 100


def test_arxiv_no_entries():
    """arXiv feed with no <entry> → None."""
    xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    resp = _FakeResponse(text=xml)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        assert _deep_search_arxiv("Nonexistent", timeout=10) is None


# ── Crossref provider ─────────────────────────────────────────────────────


def test_crossref_success():
    """A well-formed Crossref response produces a SourceDoc."""
    payload = {
        "message": {
            "items": [{
                "title": ["Batch Normalization"],
                "abstract": "<jats:p>Internal covariate shift is a problem.</jats:p>",
                "published": {"date-parts": [[2015]]},
                "author": [{"given": "Sergey", "family": "Ioffe"}],
                "DOI": "10.5555/3045392.3045594",
                "URL": "https://doi.org/10.5555/3045392.3045594",
            }],
        },
    }
    resp = _FakeResponse(json_data=payload)
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        doc = _deep_search_crossref("Batch Normalization", timeout=10)
    assert doc is not None
    assert "Batch Normalization" in doc.content
    assert "Ioffe" in doc.content
    assert "2015" in doc.content
    assert "Internal covariate shift" in doc.content
    assert "jats:p" not in doc.content  # JATS tags stripped
    assert len(doc.content) >= 100


def test_crossref_no_items():
    """Empty items → None."""
    resp = _FakeResponse(json_data={"message": {"items": []}})
    with patch("httpx.Client", side_effect=lambda **kw: _FakeClient(resp, **kw)):
        assert _deep_search_crossref("Nonexistent", timeout=10) is None


# ── _deep_search_fallback orchestration ──────────────────────────────────


def test_deep_search_fallback_picks_longest_content():
    """When multiple providers return results, the longest-content wins."""
    short_doc = SourceDoc(title="Short", content="A" * 200, url="https://short")
    long_doc = SourceDoc(title="Long", content="B" * 5000, url="https://long")

    def fake_semantic_scholar(_title, _timeout):
        return short_doc

    def fake_openalex(_title, _timeout):
        return long_doc

    def fake_arxiv(_title, _timeout):
        return None

    def fake_crossref(_title, _timeout):
        return None

    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (fake_semantic_scholar, fake_openalex, fake_arxiv, fake_crossref),
    ):
        result = _deep_search_fallback("Some Paper", "https://example.com/paper")

    # The longer OpenAlex result should win.
    assert len(result.content) == 5000
    assert result.url == "https://example.com/paper"  # original URL preserved
    assert "deep_search_fallback" in result.provenance.extractor_chain
    assert "deep_search_fallback" in result.provenance.diagnostics[0]
    assert "fake_openalex" in result.provenance.extractor_chain


def test_deep_search_fallback_all_providers_fail():
    """When all providers return None, RuntimeError is raised."""
    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (lambda _t, _to: None, lambda _t, _to: None),
    ), pytest.raises(RuntimeError, match="no accessible alternative"):
        _deep_search_fallback("Nonexistent", "https://example.com/x")


def test_deep_search_fallback_provider_exception_swallowed():
    """A provider that raises (not just returns None) is swallowed."""
    good_doc = SourceDoc(title="Good", content="C" * 300, url="https://good")

    def raising_provider(_title, _timeout):
        raise RuntimeError("network down")

    def good_provider(_title, _timeout):
        return good_doc

    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (raising_provider, good_provider),
    ):
        result = _deep_search_fallback("Paper", "https://example.com")
    assert result.content == "C" * 300


def test_deep_search_fallback_stops_after_two_candidates():
    """The fallback stops querying once it has 2 candidates (latency bound)."""
    doc1 = SourceDoc(title="A", content="A" * 200, url="")
    doc2 = SourceDoc(title="B", content="B" * 300, url="")
    call_count = {"n": 0}

    def provider1(_title, _timeout):
        call_count["n"] += 1
        return doc1

    def provider2(_title, _timeout):
        call_count["n"] += 1
        return doc2

    def provider3(_title, _timeout):
        call_count["n"] += 1
        return SourceDoc(title="C", content="C" * 99999, url="")

    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (provider1, provider2, provider3),
    ):
        result = _deep_search_fallback("Paper", "https://example.com")
    # Only the first two providers should have been called.
    assert call_count["n"] == 2
    # The longer of the first two wins (doc2, 300 chars), NOT doc3.
    assert len(result.content) == 300


def test_deep_search_fallback_derives_title_from_url_when_empty():
    """When title is empty, the query is derived from the URL."""
    good_doc = SourceDoc(title="Found", content="D" * 200, url="")

    def fake_provider(title, _timeout):
        # Verify the title was derived from the URL, not empty.
        assert title == "1706.03762"
        return good_doc

    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (fake_provider,),
    ):
        result = _deep_search_fallback("", "https://arxiv.org/abs/1706.03762")
    assert result.content == "D" * 200


def test_deep_search_fallback_provenance_tagged():
    """The returned SourceDoc carries deep_search_fallback provenance."""
    doc = SourceDoc(title="X", content="Y" * 200, url="")
    with patch(
        "obsidian_llm_wiki.ingest.extractors._DEEP_SEARCH_PROVIDERS",
        (lambda _t, _to: doc,),
    ):
        result = _deep_search_fallback("X", "https://example.com/x")
    assert "deep_search_fallback" in result.provenance.extractor_chain
    assert result.provenance.diagnostics
    assert "deep_search_fallback" in result.provenance.diagnostics[0]
    assert result.provenance.requested_url == "https://example.com/x"


# ── extract() integration ────────────────────────────────────────────────


def test_extract_no_deep_search_when_disabled(monkeypatch):
    """With DEEP_SEARCH_FALLBACK unset, stubs are returned as-is (no network)."""
    monkeypatch.delenv("DEEP_SEARCH_FALLBACK", raising=False)
    stub = SourceDoc(title="Stub", content="Too short.", url="https://example.com/x")
    with patch("obsidian_llm_wiki.ingest.extractors.extract_web", return_value=stub):
        result = extract("https://example.com/x")
    # The stub is returned unchanged — no deep search attempted.
    assert result.content == "Too short."


def test_extract_deep_search_replaces_stub_when_enabled(monkeypatch):
    """With DEEP_SEARCH_FALLBACK=1, a stub is replaced by a better deep search result."""
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "1")
    stub = SourceDoc(title="Stub Title", content="Too short.", url="https://example.com/x")
    good = SourceDoc(
        title="Real Paper",
        content="This is a full abstract recovered via deep search. " * 10,
        url="https://example.com/x",
    )
    with patch("obsidian_llm_wiki.ingest.extractors.extract_web", return_value=stub), \
         patch(
             "obsidian_llm_wiki.ingest.extractors._deep_search_fallback",
             return_value=good,
         ) as mock_ds:
        result = extract("https://example.com/x")
    mock_ds.assert_called_once()
    assert "Real Paper" in result.title
    assert "full abstract recovered" in result.content
    assert "deep_search_fallback" in result.provenance.extractor_chain


def test_extract_deep_search_not_called_when_content_is_good(monkeypatch):
    """With good content, deep search is not invoked even when enabled."""
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "1")
    good_content = "## Introduction\n\nThis is a well-extracted article. " * 20
    good = SourceDoc(title="Good Article", content=good_content, url="https://example.com/x")
    with patch("obsidian_llm_wiki.ingest.extractors.extract_web", return_value=good), \
         patch(
             "obsidian_llm_wiki.ingest.extractors._deep_search_fallback",
         ) as mock_ds:
        result = extract("https://example.com/x")
    mock_ds.assert_not_called()
    assert result.content == good_content


def test_extract_deep_search_handles_fallback_failure(monkeypatch):
    """When deep search itself fails, the original stub is returned."""
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "1")
    stub = SourceDoc(title="Stub", content="Too short.", url="https://example.com/x")
    with patch("obsidian_llm_wiki.ingest.extractors.extract_web", return_value=stub), \
         patch(
             "obsidian_llm_wiki.ingest.extractors._deep_search_fallback",
             side_effect=RuntimeError("all providers failed"),
         ):
        result = extract("https://example.com/x")
    # The stub is returned unchanged because deep search failed.
    assert result.content == "Too short."


def test_extract_deep_search_recovers_after_specialized_failure(monkeypatch):
    """When specialized extractors fail and deep search is enabled, it recovers."""
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "1")
    good = SourceDoc(
        title="Recovered Paper",
        content="This is a recovered abstract from Semantic Scholar. " * 10,
        url="https://example.com/paper",
    )

    # Register a fake specialized extractor that always fails.
    from obsidian_llm_wiki.ingest import extractors as reg

    def always_match(_parsed, _raw):
        return True

    def always_fail(_raw):
        raise RuntimeError("extractor boom")

    original = list(reg._EXTRACTORS)
    reg._EXTRACTORS.insert(0, (always_match, always_fail))
    try:
        with patch(
            "obsidian_llm_wiki.ingest.extractors._deep_search_fallback",
            return_value=good,
        ) as mock_ds:
            result = extract("https://example.com/paper")
        mock_ds.assert_called_once()
        assert "Recovered Paper" in result.title
        assert "deep_search_fallback" in result.provenance.extractor_chain
    finally:
        reg._EXTRACTORS[:] = original


def test_extract_deep_search_does_not_replace_when_shorter(monkeypatch):
    """Deep search result shorter than the stub is not used."""
    monkeypatch.setenv("DEEP_SEARCH_FALLBACK", "1")
    # A stub that's just barely under 500 chars.
    stub_content = "A" * 450
    stub = SourceDoc(title="Stub", content=stub_content, url="https://example.com/x")
    shorter = SourceDoc(title="Shorter", content="B" * 100, url="https://example.com/x")
    with patch("obsidian_llm_wiki.ingest.extractors.extract_web", return_value=stub), \
         patch(
             "obsidian_llm_wiki.ingest.extractors._deep_search_fallback",
             return_value=shorter,
         ):
        result = extract("https://example.com/x")
    # The original stub is kept because the deep search result was shorter.
    assert result.content == stub_content
