"""Tests for official, accessible scientific document extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from obsidian_llm_wiki.core.models import SourceDoc


def test_arxiv_uses_official_accessible_html_before_pdf() -> None:
    """An available arXiv HTML conversion is preferred to the PDF rendition."""
    from obsidian_llm_wiki.ingest.extractors.scientific import extract_arxiv

    html = """
    <html>
      <head><title>Accessible Transformer Paper</title></head>
      <body>
        <main>
          <h1>Accessible Transformer Paper</h1>
          <p>This official HTML conversion contains the complete scientific report.</p>
          <p>It preserves headings, equations, and prose for accessible extraction.</p>
        </main>
      </body>
    </html>
    """

    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            return_value=html,
        ) as fetch_html,
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._extract_pdf",
        ) as extract_pdf,
    ):
        result = extract_arxiv("https://arxiv.org/abs/1706.03762")

    fetch_html.assert_called_once_with("https://arxiv.org/html/1706.03762", 45)
    extract_pdf.assert_not_called()
    assert result.title == "Accessible Transformer Paper"
    assert "complete scientific report" in result.content
    assert result.url == "https://arxiv.org/html/1706.03762"


def test_discover_scientific_documents_reads_publication_metadata() -> None:
    """A scientific landing page exposes same-site HTML and PDF candidates."""
    from obsidian_llm_wiki.ingest.extractors.scientific import discover_scientific_documents

    html = """
    <html><head>
      <meta name="citation_fulltext_html_url" content="/article/full-text">
      <meta name="citation_pdf_url" content="https://journals.example.edu/files/paper.pdf">
    </head><body>
      <a href="/downloads/supplement.pdf" type="application/pdf">Download PDF</a>
      <a href="https://unlicensed-mirror.example/paper.pdf">Mirror</a>
      <a href="https://unlicensed-mirror.journals.example.edu/paper.pdf">Subdomain mirror</a>
    </body></html>
    """

    candidates = discover_scientific_documents(
        html,
        "https://journals.example.edu/article/10.1000/example",
    )

    assert candidates == [
        ("html", "https://journals.example.edu/article/full-text"),
        ("pdf", "https://journals.example.edu/files/paper.pdf"),
        ("pdf", "https://journals.example.edu/downloads/supplement.pdf"),
    ]


def test_arxiv_html_unavailable_falls_back_to_official_pdf() -> None:
    """Missing arXiv HTML uses the direct official PDF document extractor."""
    from obsidian_llm_wiki.ingest.extractors.scientific import extract_arxiv

    pdf_source = SourceDoc(
        title="PDF Paper",
        content="Text from the official PDF document extractor.",
        url="https://arxiv.org/pdf/1706.03762",
    )
    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            side_effect=RuntimeError("arXiv HTML conversion is unavailable"),
        ),
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._extract_pdf",
            return_value=pdf_source,
        ) as extract_pdf,
    ):
        result = extract_arxiv("https://arxiv.org/abs/1706.03762v2")

    extract_pdf.assert_called_once_with("https://arxiv.org/pdf/1706.03762v2")
    assert result is pdf_source


def test_public_scientific_fetch_sends_no_cookie_or_auth_headers() -> None:
    """Accessible-document fetches do not attempt authenticated access bypasses."""
    from obsidian_llm_wiki.ingest.extractors.scientific import _fetch_public_html

    response = MagicMock()
    response.text = "<html><body>" + ("public article text " * 10) + "</body></html>"
    client = MagicMock()
    client.get.return_value = response
    client.__enter__.return_value = client
    client.__exit__.return_value = False

    with patch(
        "obsidian_llm_wiki.ingest.extractors.scientific.httpx.Client",
        return_value=client,
    ) as http_client:
        assert "public article" in _fetch_public_html("https://papers.ssrn.com/article", 12)

    headers = http_client.call_args.kwargs["headers"]
    assert not {"authorization", "cookie", "referer"} & {name.lower() for name in headers}
    client.get.assert_called_once_with("https://papers.ssrn.com/article")
    response.raise_for_status.assert_called_once()


def test_registry_imports_scientific_extractor_before_pdf_extractor() -> None:
    """arXiv dispatch must get a chance before legacy /abs/ to /pdf/ handling."""
    from obsidian_llm_wiki.ingest import extractors as registry

    names = [extractor.__name__ for _, extractor in registry._EXTRACTORS]
    assert names.index("extract_arxiv") < names.index("extract_pdf")


def test_registry_forwards_original_arxiv_abstract_url_to_specialized_route() -> None:
    """The scientific route sees /abs/ rather than a prematurely rewritten PDF URL."""
    from obsidian_llm_wiki.ingest import extractors as registry

    received: list[str] = []
    source = SourceDoc(title="HTML paper", content="Accessible report text.", url="unused")

    def match_arxiv(parsed, raw: str) -> bool:
        return parsed.hostname == "arxiv.org"

    def extract_specialized(raw_url: str) -> SourceDoc:
        received.append(raw_url)
        return source

    original_extractors = list(registry._EXTRACTORS)
    registry._EXTRACTORS.insert(0, (match_arxiv, extract_specialized))
    try:
        assert registry.extract("https://arxiv.org/abs/1706.03762?ref=reader") is source
    finally:
        registry._EXTRACTORS[:] = original_extractors

    assert received == ["https://arxiv.org/abs/1706.03762?ref=reader"]


def test_discovered_public_pdf_uses_existing_document_extractor() -> None:
    """Publisher metadata routes a public direct PDF to the PDF extractor."""
    from obsidian_llm_wiki.ingest.extractors.scientific import (
        extract_discovered_scientific_document,
    )

    pdf_source = SourceDoc(
        title="Publisher PDF",
        content="Text from a publicly linked publisher PDF.",
        url="https://journals.example.edu/download/paper.pdf",
    )
    landing_html = """
    <meta name="citation_pdf_url" content="/download/paper.pdf">
    """
    with (
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._fetch_public_html",
            return_value=landing_html,
        ) as fetch_html,
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific._extract_pdf",
            return_value=pdf_source,
        ) as extract_pdf,
    ):
        result = extract_discovered_scientific_document(
            "https://journals.example.edu/article/10.1000/example",
            timeout=17,
        )

    fetch_html.assert_called_once_with("https://journals.example.edu/article/10.1000/example", 17)
    extract_pdf.assert_called_once_with("https://journals.example.edu/download/paper.pdf")
    assert result is pdf_source


def test_ssrn_falls_to_semantic_scholar_when_no_public_document_is_available() -> None:
    """SSRN does not bypass access controls after public document discovery fails."""
    from obsidian_llm_wiki.ingest.web import extract_web

    ssrn_url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522"
    abstract_source = SourceDoc(
        title="Semantic Scholar Abstract",
        content="Public metadata and abstract only.",
        url=ssrn_url,
    )
    unavailable = RuntimeError("landing page is inaccessible")
    with (
        patch("obsidian_llm_wiki.ingest.web._extract_defuddle_md", side_effect=unavailable),
        patch("obsidian_llm_wiki.ingest.web._extract_trafilatura", side_effect=unavailable),
        patch("obsidian_llm_wiki.ingest.web._extract_defuddle", side_effect=unavailable),
        patch(
            "obsidian_llm_wiki.ingest.extractors.scientific.extract_discovered_scientific_document",
            side_effect=unavailable,
        ) as discover_document,
        patch(
            "obsidian_llm_wiki.ingest.alt_source.extract_via_semantic_scholar",
            return_value=abstract_source,
        ) as semantic_scholar,
    ):
        result = extract_web(ssrn_url, timeout=9)

    discover_document.assert_called_once_with(ssrn_url, 9)
    semantic_scholar.assert_called_once_with(ssrn_url, 9)
    assert result is abstract_source
