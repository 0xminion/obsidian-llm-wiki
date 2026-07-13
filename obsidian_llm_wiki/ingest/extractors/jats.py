"""JATS/XML extractor — extracts text from academic XML articles.

Handles JATS (Journal Article Tag Suite) XML commonly served by academic
publishers (akjournals.com, PubMed Central, etc.).

Dependency: ``defusedxml`` (installed by default).
"""

from __future__ import annotations

import logging

import httpx
from defusedxml import ElementTree as DET  # noqa: N814

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

logger = logging.getLogger("obswiki.ingest.extractors.jats")

# ── JATS namespace handling ─────────────────────────────────────────────
# JATS XML uses namespaces like {http://www.ncbi.nlm.nih.gov/JATS}article
# We strip namespaces for simpler matching.

# Hosts known to serve JATS XML
_JATS_HOSTS = frozenset((
    "akjournals.com",
    "www.akjournals.com",
))


def _try_publisher_pdf(raw_url: str) -> SourceDoc | None:
    """Try to download a PDF from the publisher's direct PDF URL.

    Many academic publishers (akjournals, etc.) serve PDFs at a
    ``/downloadpdf/view/`` path even when the XML/HTML endpoints are
    captcha-walled. This constructs the PDF URL from the article URL
    and attempts to download and extract it via the PDF extractor.
    """
    from urllib.parse import urlparse

    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").lower()
    path = parsed.path

    # Only attempt for known JATS publishers with PDF endpoints.
    if "akjournals" not in host:
        return None

    # akjournals pattern:
    #   /view/journals/2054/9/3/article-p294.xml
    # → /downloadpdf/view/journals/2054/9/3/article-p294.pdf
    if "/view/journals/" not in path:
        return None

    pdf_path = path.replace("/view/", "/downloadpdf/view/")
    if pdf_path.endswith(".xml"):
        pdf_path = pdf_path[:-4] + ".pdf"
    elif not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"

    pdf_url = f"{parsed.scheme}://{host}{pdf_path}"

    try:
        from obsidian_llm_wiki.ingest.documents import dispatch_document
        logger.info("Trying publisher PDF: %s", pdf_url)
        source = dispatch_document(pdf_url)
        logger.info("Publisher PDF extracted %d chars", len(source.content))
        return source
    except Exception as exc:
        logger.debug("Publisher PDF failed: %s", exc)
        return None


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag: {ns}local → local."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _is_xml_jats(parsed, raw: str) -> bool:
    """Match URLs pointing to XML/JATS articles.

    Matches:
    - URLs with .xml suffix
    - Known JATS hosts (akjournals.com, etc.)
    """
    path = (parsed.path or "").lower()
    if path.endswith(".xml"):
        return True
    host = (parsed.hostname or "").lower()
    return host in _JATS_HOSTS


# ── Registration ────────────────────────────────────────────────────────

@register_extractor(_is_xml_jats)
def extract_jats(raw_url: str) -> SourceDoc:
    """Download and extract text from a JATS/XML article.

    Parses JATS article structure: title, abstract, body sections.
    Falls back to generic XML text extraction if JATS elements aren't found.

    If the server rejects the XML URL (405/403/404), strips the .xml
    suffix and falls back to extract_web on the HTML article page.
    """
    try:
        with httpx.Client(
            **make_client_kwargs(follow_redirects=True, timeout=45),
            headers=BROWSER_HEADERS,
        ) as client:
            resp = client.get(raw_url)
            resp.raise_for_status()
            xml_text = resp.text
    except httpx.HTTPStatusError as exc:
        # 405/403/404 — server doesn't serve XML to bots.
        # Strip .xml suffix and try the HTML article page via extract_web.
        if raw_url.lower().endswith(".xml"):
            html_url = raw_url[: -len(".xml")]
            logger.warning(
                "XML endpoint returned %s for %s; falling back to HTML: %s",
                exc.response.status_code, raw_url, html_url,
            )
            try:
                from obsidian_llm_wiki.ingest.web import extract_web
                return extract_web(html_url)
            except Exception:
                pass
            # Try the publisher's direct PDF download URL.
            # Many publishers (akjournals, etc.) serve a PDF at a
            # /downloadpdf/view/... path even when the XML/HTML is
            # captcha-walled. The residential proxy (RESIDENTIAL_PROXY_URL)
            # bypasses the bot detection that blocks datacenter IPs.
            pdf_source = _try_publisher_pdf(raw_url)
            if pdf_source is not None:
                return pdf_source
            # If PDF also failed, try Semantic Scholar
            # title search as a last-resort academic fallback.
            try:
                from obsidian_llm_wiki.ingest.alt_source import _semantic_scholar_search
                # Extract a search query from the URL path
                path_part = html_url.rsplit("/article-", 1)[-1]
                search_title = path_part.replace("-", " ").strip()
                if search_title and len(search_title) > 10:
                    return _semantic_scholar_search(search_title, raw_url)
            except Exception:
                pass
        raise

    if not xml_text.strip():
        raise RuntimeError(f"Empty XML response from {raw_url}")

    # Detect HTML masquerading as XML before parsing.
    content_type = (resp.headers.get("content-type") or "").lower()
    looks_like_html = (
        "html" in content_type
        or xml_text.lstrip()[:20].lower().startswith(("<!doctype", "<html"))
    )
    if looks_like_html:
        # Salvage the already-downloaded HTML body — do NOT re-fetch
        # (the stripped .xml URL may 404 while the .xml response itself
        # contains the full HTML article page).
        return _salvage_html_body(raw_url, xml_text)

    # Parse with defusedxml (XXE-safe).
    # If the XML is malformed (e.g. HTML served as .xml), salvage the
    # body as HTML instead of crashing.
    try:
        root = DET.fromstring(xml_text)
    except Exception as parse_exc:
        logger.warning(
            "XML parse failed for %s: %s; salvaging as HTML",
            raw_url, parse_exc,
        )
        return _salvage_html_body(raw_url, xml_text)

    # ── Extract title ─────────────────────────────────────────────
    title = _find_jats_title(root)
    if not title:
        title = raw_url

    # ── Extract abstract ──────────────────────────────────────────
    abstract = _find_jats_abstract(root)

    # ── Extract body ──────────────────────────────────────────────
    body = _find_jats_body(root)

    # ── Assemble content ──────────────────────────────────────────
    parts: list[str] = []
    if abstract:
        parts.append(f"## Abstract\n\n{abstract}")
    if body:
        parts.append(f"## Full Text\n\n{body}")

    if not parts:
        # Fallback: generic XML text extraction
        generic_text = _generic_xml_text(root)
        if not generic_text.strip():
            raise RuntimeError(f"No extractable text from XML: {raw_url}")
        content = generic_text
    else:
        content = "\n\n".join(parts)

    if len(content.strip()) < 50:
        raise RuntimeError(f"XML extraction too short ({len(content)} chars): {raw_url}")

    return SourceDoc(
        title=title,
        content=content.strip(),
        url=raw_url,
    )


# ── HTML salvage (when .xml returns HTML body) ──────────────────────────


def _salvage_html_body(raw_url: str, html: str) -> SourceDoc:
    """Extract text from an HTML body already downloaded (e.g. .xml that was HTML).

    Never re-fetches: stripped .xml URLs may 404 while the original response
    body already contains the full article page.
    """
    import re as _re

    # Prefer trafilatura on the in-memory HTML
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        extracted = None

    if extracted and extracted.strip():
        # Title from <title> or og:title
        title = ""
        m = _re.search(r"<title[^>]*>([^<]+)</title>", html, _re.IGNORECASE)
        if m:
            title = m.group(1).strip()
        if not title:
            m = _re.search(
                r'property=["\']og:title["\'][^>]*content=["\']([^"\']+)',
                html,
                _re.IGNORECASE,
            )
            if m:
                title = m.group(1).strip()
        if not title:
            title = raw_url

        if len(extracted.strip()) < 50:
            raise RuntimeError(
                f"HTML salvage too short ({len(extracted)} chars): {raw_url}"
            )
        return SourceDoc(title=title, content=extracted.strip(), url=raw_url)

    # Regex fallback: strip tags manually
    from obsidian_llm_wiki.ingest.web import _extract_title_from_html, _strip_tags

    title = _extract_title_from_html(html) or raw_url
    text = _strip_tags(html)
    if not text or len(text.strip()) < 50:
        raise RuntimeError(f"HTML salvage produced no usable text: {raw_url}")
    return SourceDoc(title=title, content=text.strip(), url=raw_url)


# ── JATS element extraction helpers ─────────────────────────────────────


def _find_jats_title(root) -> str:
    """Find article title in JATS XML."""
    # JATS: //article-meta/article-title or //front/article-meta/article-title
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "article-title" and elem.text:
            return " ".join(elem.itertext()).strip()
    # Fallback: <title> element
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "title" and elem.text:
            return " ".join(elem.itertext()).strip()
    return ""


def _find_jats_abstract(root) -> str:
    """Find abstract in JATS XML."""
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "abstract":
            text = _collect_text(elem)
            if text.strip():
                return text.strip()
    return ""


def _find_jats_body(root) -> str:
    """Find body text in JATS XML."""
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag == "body":
            text = _collect_text(elem)
            if text.strip():
                return text.strip()
    return ""


def _collect_text(elem) -> str:
    """Collect all text from an element and its descendants, preserving
    paragraph/section structure."""
    parts: list[str] = []
    for child in elem.iter():
        tag = _strip_ns(child.tag)
        if tag in ("p", "sec", "title", "caption", "list-item", "li"):
            text = " ".join(child.itertext()).strip()
            if text:
                parts.append(text)
    if not parts:
        # No structured elements found — get all text
        return " ".join(elem.itertext()).strip()
    return "\n\n".join(parts)


def _generic_xml_text(root) -> str:
    """Fallback: extract all text from any XML structure."""
    parts: list[str] = []
    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag in ("p", "para", "paragraph", "section", "sec", "abstract", "body", "description"):
            text = " ".join(elem.itertext()).strip()
            if text:
                parts.append(text)
    if not parts:
        return " ".join(root.itertext()).strip()
    return "\n\n".join(parts)
