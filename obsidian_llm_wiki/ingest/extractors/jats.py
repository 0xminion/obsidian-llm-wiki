"""JATS/XML extractor — extracts text from academic XML articles.

Handles JATS (Journal Article Tag Suite) XML commonly served by academic
publishers (akjournals.com, PubMed Central, etc.).

Dependency: ``defusedxml`` (installed by default).
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx
from defusedxml import ElementTree as DET

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS

logger = logging.getLogger("obswiki.ingest.extractors.jats")

# ── JATS namespace handling ─────────────────────────────────────────────
# JATS XML uses namespaces like {http://www.ncbi.nlm.nih.gov/JATS}article
# We strip namespaces for simpler matching.

_NS_STRIP = True

# Hosts known to serve JATS XML
_JATS_HOSTS = frozenset((
    "akjournals.com",
    "www.akjournals.com",
))


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
    if host in _JATS_HOSTS:
        return True
    return False


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
            follow_redirects=True, timeout=45, headers=BROWSER_HEADERS
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
            from obsidian_llm_wiki.ingest.web import extract_web
            return extract_web(html_url)
        raise

    if not xml_text.strip():
        raise RuntimeError(f"Empty XML response from {raw_url}")

    # Parse with defusedxml (XXE-safe)
    root = DET.fromstring(xml_text)

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