"""Accessible-first extraction for public scientific reports.

arXiv publishes an official HTML rendition for a growing subset of papers.
This module prefers that structured, accessible rendition and only then asks the
existing PDF extractor for the official PDF.  It never uses cookie, login, or
mirror-based access workarounds.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.scientific")

_ARXIV_HOSTS = frozenset(("arxiv.org", "www.arxiv.org", "export.arxiv.org"))
_MIN_DOCUMENT_CHARS = 100


def _is_arxiv_url(parsed, raw: str) -> bool:
    """Return whether *raw* is an official arXiv abstract, HTML, or PDF URL."""
    del raw
    return (parsed.hostname or "").lower() in _ARXIV_HOSTS


def arxiv_paper_id(url: str) -> str | None:
    """Parse a canonical arXiv paper identifier from an official arXiv URL."""
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() not in _ARXIV_HOSTS:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts or parts[0] not in {"abs", "html", "pdf"}:
        return None
    identifier = "/".join(parts[1:])
    if identifier.endswith(".pdf"):
        identifier = identifier[:-4]
    if not identifier:
        return None

    # Modern identifiers are YYMM.NNNNN[vN].  Keep legacy category/NNNNNNN
    # identifiers too; official arXiv routes support both forms.
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", identifier):
        return identifier
    if re.fullmatch(r"[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7}(?:v\d+)?", identifier):
        return identifier
    return None


def _fetch_public_html(url: str, timeout: int) -> str:
    """Fetch a public document without cookies, credentials, or browser bypasses."""
    with httpx.Client(
        **make_client_kwargs(timeout=timeout, follow_redirects=True),
        headers=BROWSER_HEADERS,
    ) as client:
        response = client.get(url)
    response.raise_for_status()
    html = response.text
    if len(html.strip()) < _MIN_DOCUMENT_CHARS:
        raise RuntimeError(f"public HTML response was too short ({len(html)} chars)")
    return html


def _title_from_html(html: str) -> str:
    """Return a document title without relying on an optional extractor."""
    for pattern in (
        r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\']([^"\']+)',
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
    ):
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            from obsidian_llm_wiki.ingest.web import _strip_tags

            return _strip_tags(match.group(1)).strip()
    return ""


class _ScientificLinkParser(HTMLParser):
    """Collect scholarly citation metadata and direct document links."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag == "meta":
            name = attributes.get("name", "").lower()
            content = attributes.get("content", "").strip()
            if name == "citation_fulltext_html_url" and content:
                self.links.append(("html", content))
            elif name == "citation_pdf_url" and content:
                self.links.append(("pdf", content))
            return

        if tag == "a":
            href = attributes.get("href", "").strip()
            link_type = attributes.get("type", "").lower()
            is_pdf_url = urlparse(href).path.lower().endswith(".pdf")
            if href and (is_pdf_url or link_type == "application/pdf"):
                self.links.append(("pdf", href))


def _same_official_site(candidate_url: str, landing_url: str) -> bool:
    """Keep discovery on the known publisher host, never broad domain families."""
    candidate_host = (urlparse(candidate_url).hostname or "").lower()
    landing_host = (urlparse(landing_url).hostname or "").lower()
    if not candidate_host or not landing_host:
        return False
    if candidate_host == landing_host:
        return True

    # SSRN's official publicly linked documents use several ssrn.com
    # subdomains (for example papers.ssrn.com and deliverypdf.ssrn.com).
    # Keep this narrow exception rather than treating every sibling subdomain
    # as official, which could silently admit an unlicensed mirror.
    return candidate_host.endswith(".ssrn.com") and landing_host.endswith(".ssrn.com")


def discover_scientific_documents(html: str, landing_url: str) -> list[tuple[str, str]]:
    """Find public same-publisher full-text HTML and PDF links in a landing page.

    Discovery considers citation metadata and explicit PDF links, but deliberately
    rejects off-site candidates so a landing page cannot route extraction through
    an unlicensed mirror or a third-party access workaround.
    """
    parser = _ScientificLinkParser()
    parser.feed(html)

    candidates: list[tuple[str, str]] = []
    for kind, raw_url in parser.links:
        candidate_url = urljoin(landing_url, raw_url)
        scheme = urlparse(candidate_url).scheme.lower()
        if scheme not in {"http", "https"} or not _same_official_site(candidate_url, landing_url):
            continue
        candidate = (kind, candidate_url)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_scientific_html(html: str, source_url: str) -> SourceDoc:
    """Structurally extract a public scientific HTML document."""
    import trafilatura

    content = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not content or len(content.strip()) < _MIN_DOCUMENT_CHARS:
        from obsidian_llm_wiki.ingest.web import _strip_tags

        content = _strip_tags(html)
    if not content or len(content.strip()) < _MIN_DOCUMENT_CHARS:
        raise RuntimeError("scientific HTML extraction returned empty/short content")

    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title or "").strip() if metadata else ""
    title = title or _title_from_html(html) or source_url
    return SourceDoc(title=title, content=content.strip(), url=source_url)


def extract_discovered_scientific_document(
    landing_url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> SourceDoc:
    """Follow a publicly advertised, same-publisher scientific document link.

    This is intentionally a narrow fallback for landing pages that do not expose
    useful body text.  It only follows publisher-provided direct HTML/PDF links;
    failed or authenticated links raise so the caller can use its normal fallback
    chain (for example, SSRN's Semantic Scholar abstract fallback).
    """
    landing_html = _fetch_public_html(landing_url, timeout)
    candidates = discover_scientific_documents(landing_html, landing_url)
    if not candidates:
        raise RuntimeError(f"No public scientific document links found at {landing_url}")

    errors: list[str] = []
    for kind, document_url in candidates:
        try:
            if kind == "html":
                html = _fetch_public_html(document_url, timeout)
                return extract_scientific_html(html, document_url)
            return _extract_pdf(document_url)
        except Exception as exc:
            errors.append(f"{document_url}: {exc}")

    raise RuntimeError(
        f"Public scientific document links were unavailable for {landing_url}: "
        + "; ".join(errors)
    )


def _extract_pdf(url: str) -> SourceDoc:
    """Delegate direct official PDFs to the existing PDF document extractor."""
    from obsidian_llm_wiki.ingest.extractors.pdf import extract_pdf

    return extract_pdf(url)


@register_extractor(_is_arxiv_url)
def extract_arxiv(raw_url: str) -> SourceDoc:
    """Extract arXiv through official accessible HTML, then official PDF."""
    paper_id = arxiv_paper_id(raw_url)
    if not paper_id:
        raise RuntimeError(f"Could not parse arXiv paper ID from {raw_url}")

    html_url = f"https://arxiv.org/html/{paper_id}"
    try:
        html = _fetch_public_html(html_url, DEFAULT_TIMEOUT)
        return extract_scientific_html(html, html_url)
    except Exception as exc:
        logger.info("Official arXiv HTML unavailable for %s: %s; trying PDF", paper_id, exc)

    return _extract_pdf(f"https://arxiv.org/pdf/{paper_id}")
