"""Accessible-first extraction for public scientific reports.

arXiv publishes an official HTML rendition for a growing subset of papers.
This module prefers that structured, accessible rendition and only then asks the
existing PDF extractor for the official PDF.  It never uses cookie, login, or
mirror-based access workarounds.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS, DEFAULT_TIMEOUT

logger = logging.getLogger("obswiki.ingest.extractors.scientific")

_ARXIV_HOSTS = frozenset(("arxiv.org", "www.arxiv.org", "export.arxiv.org"))
_MIN_DOCUMENT_CHARS = 100
_MIN_SUBSTANTIVE_FULLTEXT_CHARS = 500
_SCIENTIFIC_HOST_MARKERS = frozenset(
    (
        "academic",
        "arxiv",
        "journals",
        "journal",
        "papers",
        "pubmed",
        "research",
        "science",
        "sciences",
    )
)
_SCIENTIFIC_OFFICIAL_DOMAINS = frozenset(
    (
        "arxiv.org",
        "bmj.com",
        "cell.com",
        "jamanetwork.com",
        "nature.com",
        "nih.gov",
        "plos.org",
        "sciencedirect.com",
        "science.org",
        "springer.com",
        "ssrn.com",
        "tandfonline.com",
        "thelancet.com",
        "wiley.com",
    )
)
_MAX_CANDIDATE_ERRORS = 3
_MAX_CANDIDATE_ERROR_CHARS = 240


def _is_arxiv_url(parsed, raw: str) -> bool:
    """Return whether *raw* is an arXiv paper URL (abs/html/pdf with a valid ID).

    Non-paper pages (``/help``, ``/list``, ``/year``) do not match — they fall
    through to ``extract_web`` as expected.
    """
    if (parsed.hostname or "").lower() not in _ARXIV_HOSTS:
        return False
    return arxiv_paper_id(raw) is not None


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
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        headers=BROWSER_HEADERS,
    ) as client:
        from obsidian_llm_wiki.ingest.url_safety import get_with_validated_redirects
        response = get_with_validated_redirects(client, url)
    response.raise_for_status()
    resolved_url = str(getattr(response, "url", ""))
    if resolved_url.startswith(("http://", "https://")) and not _same_official_site(
        resolved_url, url
    ):
        raise RuntimeError("public document redirected off the official site")
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


def is_likely_scientific_landing_page(url: str) -> bool:
    """Return whether a URL merits the narrow scientific preflight.

    The check uses explicit scholarly URL markers rather than fetching every
    page only to inspect metadata.  Ordinary blogs therefore retain their
    generic extraction path without an additional request.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    labels = frozenset(host.split("."))
    is_known_scientific_domain = any(
        host == domain or host.endswith(f".{domain}")
        for domain in _SCIENTIFIC_OFFICIAL_DOMAINS
    )
    return bool(labels & _SCIENTIFIC_HOST_MARKERS or is_known_scientific_domain)


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
    # Strip arXiv acknowledgment artifacts (††thanks: ...) that trafilatura
    # sometimes folds into the title metadata.
    title = re.split(r"\u2020\u2020thanks:|††thanks:", title, flags=re.IGNORECASE)[0].strip()
    return SourceDoc(title=title, content=content.strip(), url=source_url)


def _is_substantive_fulltext(document: SourceDoc) -> bool:
    """Reject thin or untitled HTML candidates that are likely abstracts."""
    title = document.title.strip()
    if len(title) < 4 or title.startswith(("http://", "https://", "<")):
        return False
    content = document.content.strip()
    return len(content) >= _MIN_SUBSTANTIVE_FULLTEXT_CHARS and len(content.split()) >= 80


def _record_selection(document: SourceDoc, landing_url: str, kind: str) -> SourceDoc:
    """Preserve a bounded scientific-selection decision in existing provenance."""
    provenance = document.provenance
    diagnostics = (*provenance.diagnostics, f"scientific selection: official {kind} candidate")
    document.provenance = replace(
        provenance,
        requested_url=provenance.requested_url or landing_url,
        extracted_url=provenance.extracted_url or document.url or "",
        extractor_chain=(*provenance.extractor_chain, f"scientific_public_{kind}"),
        diagnostics=diagnostics,
    )
    return document


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
    ranked_candidates = sorted(
        enumerate(candidates), key=lambda item: (0 if item[1][0] == "html" else 1, item[0])
    )
    for _, (kind, document_url) in ranked_candidates:
        try:
            if kind == "html":
                html = _fetch_public_html(document_url, timeout)
                document = extract_scientific_html(html, document_url)
                if not _is_substantive_fulltext(document):
                    raise RuntimeError("public HTML candidate was not substantive full text")
                return _record_selection(document, landing_url, kind)
            return _record_selection(_extract_pdf(document_url), landing_url, kind)
        except Exception as exc:
            if len(errors) < _MAX_CANDIDATE_ERRORS:
                errors.append(f"{document_url}: {str(exc)[:_MAX_CANDIDATE_ERROR_CHARS]}")

    raise RuntimeError(
        f"Public scientific document links were unavailable for {landing_url}: "
        + "; ".join(errors or ["no usable candidates"])
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
