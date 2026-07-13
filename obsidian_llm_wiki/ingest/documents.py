"""Safe, format-aware dispatch for local and remote documents.

All binary document downloads pass through this module.  It deliberately keeps
remote documents out of HTML extraction: a URL that claims to be a document
must validate as that document before a parser is invoked.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
from obsidian_llm_wiki.ingest.liteparse import parse_document
from obsidian_llm_wiki.ingest.provenance import stamp_source
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs

DOCUMENT_SUFFIXES = frozenset((".pdf", ".doc", ".docx", ".epub", ".ppt", ".pptx", ".xls", ".xlsx"))
TEXT_SUFFIXES = frozenset((".txt", ".md", ".markdown", ".rst"))

_CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/epub+zip": ".epub",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}
_GENERIC_BINARY_TYPES = frozenset(("", "application/octet-stream", "application/binary"))


class DocumentError(RuntimeError):
    """Base error for safe document download and dispatch failures."""


class DocumentTooLargeError(DocumentError):
    """A document exceeded the configured byte boundary."""


class InvalidDocumentError(DocumentError):
    """A response was not the declared or expected document type."""


@dataclass(frozen=True)
class DownloadedDocument:
    """A bounded temporary document download owned by the caller."""

    path: Path
    source_url: str
    resolved_url: str
    content_type: str
    suffix: str


def is_document_path(value: str | Path) -> bool:
    """Whether a path or URL has one of the supported binary document suffixes."""
    return Path(urlparse(str(value)).path).suffix.lower() in DOCUMENT_SUFFIXES


def is_direct_document_url(value: str) -> bool:
    """Whether *value* is an HTTP(S) URL that explicitly names a document."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and is_document_path(value)


def dispatch_document(
    value: str | Path,
    *,
    config: Config | None = None,
    source_url: str | None = None,
) -> SourceDoc:
    """Extract a supported local file or direct remote document safely.

    PDF text extraction remains PyMuPDF-first.  Other supported formats route to
    LiteParse.  Callers must use this function for known binary URLs rather than
    falling through to the generic HTML extractor.
    """
    cfg = config or load_config()
    raw = str(value)
    if urlparse(raw).scheme in {"http", "https"}:
        downloaded = download_document(raw, config=cfg)
        try:
            source = _parse_local_document(
                downloaded.path, source_url=source_url or downloaded.resolved_url, config=cfg
            )
            return _stamp_download_provenance(
                source,
                downloaded,
                requested_url=raw,
                extracted_url=raw,
            )
        finally:
            _remove_temp(downloaded.path)

    path = Path(os.path.expanduser(raw))
    if not path.is_file():
        raise DocumentError(f"Document file not found: {raw}")
    if not is_document_path(path):
        raise DocumentError(f"Unsupported document type: {path.suffix.lower()}")
    return _parse_local_document(path, source_url=source_url or str(path), config=cfg)


def extract_discovered_document(
    page_url: str, timeout: int, *, config: Config | None = None
) -> SourceDoc:
    """Discover and parse bounded same-site document candidates from a landing page."""
    cfg = config or load_config()
    with httpx.Client(
        **make_client_kwargs(timeout=timeout, follow_redirects=True), headers=BROWSER_HEADERS
    ) as client:
        try:
            response = client.get(page_url)
            response.raise_for_status()
        except Exception as exc:
            raise DocumentError(f"Could not retrieve document landing page {page_url}") from exc

        candidates = document_candidates(
            response.text, page_url, max_candidates=cfg.max_document_candidates
        )

    if not candidates:
        raise DocumentError("No same-site document candidate found")

    errors: list[str] = []
    official_host = (urlparse(page_url).hostname or "").lower()
    for candidate in candidates:
        try:
            downloaded = download_document(
                candidate, config=cfg, timeout=timeout, required_host=official_host
            )
            try:
                source = _parse_local_document(downloaded.path, source_url=page_url, config=cfg)
                return _stamp_download_provenance(
                    source,
                    downloaded,
                    requested_url=page_url,
                    extracted_url=candidate,
                    discovered=True,
                )
            finally:
                _remove_temp(downloaded.path)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    error = DocumentError("Document candidates failed: " + "; ".join(errors))
    if errors:
        raise error
    raise DocumentError("No same-site document candidate found")


def download_document(
    url: str,
    *,
    config: Config | None = None,
    timeout: int = 60,
    required_host: str | None = None,
) -> DownloadedDocument:
    """Stream one remote binary document to a byte-bounded temporary file.

    The returned path is temporary storage owned by the caller.  On every
    validation/download failure it is removed before the original cause is
    re-raised through a document-specific exception.
    """
    cfg = config or load_config()
    expected_suffix = _suffix_from_url(url)
    temp_path: Path | None = None
    try:
        with (
            httpx.Client(
                **make_client_kwargs(timeout=timeout, follow_redirects=True),
                headers=BROWSER_HEADERS,
            ) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            resolved_url = str(response.url)
            if (
                required_host
                and (urlparse(resolved_url).hostname or "").lower() != required_host
            ):
                raise InvalidDocumentError("Document redirect left the official site")
            content_type = _content_type(response.headers)
            mime_suffix = _CONTENT_TYPE_SUFFIXES.get(content_type)
            suffix = expected_suffix or mime_suffix
            if suffix not in DOCUMENT_SUFFIXES:
                raise InvalidDocumentError(
                    f"Unsupported document MIME {content_type or 'missing'} for {resolved_url}"
                )
            _validate_mime(content_type, suffix)
            declared_size = _content_length(response.headers)
            if declared_size is not None and declared_size > cfg.max_document_bytes:
                raise DocumentTooLargeError(
                    f"Document Content-Length {declared_size} "
                    f"exceeds {cfg.max_document_bytes} bytes"
                )

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temp_path = Path(handle.name)
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > cfg.max_document_bytes:
                        raise DocumentTooLargeError(
                            f"Document stream exceeded {cfg.max_document_bytes} bytes"
                        )
                    handle.write(chunk)

        if temp_path is None or temp_path.stat().st_size == 0:
            raise InvalidDocumentError(f"Downloaded empty document from {url}")
        _validate_signature(temp_path, suffix)
        return DownloadedDocument(
            path=temp_path,
            source_url=url,
            resolved_url=resolved_url,
            content_type=content_type,
            suffix=suffix,
        )
    except DocumentError:
        _remove_temp(temp_path)
        raise
    except Exception as exc:
        _remove_temp(temp_path)
        raise DocumentError(f"Could not download document {url}") from exc


def document_candidates(html: str, page_url: str, *, max_candidates: int) -> list[str]:
    """Return at most ``max_candidates`` same-site document links from HTML."""
    if max_candidates <= 0:
        return []

    parser = _DocumentLinkParser()
    parser.feed(html)
    parser.close()

    official_host = (urlparse(page_url).hostname or "").lower()
    seen: set[str] = set()
    candidates: list[str] = []
    for raw_candidate in parser.candidates:
        resolved = urljoin(page_url, raw_candidate)
        if not raw_candidate or resolved in seen:
            continue
        if (urlparse(resolved).hostname or "").lower() != official_host:
            continue
        seen.add(resolved)
        candidates.append(resolved)
        if len(candidates) >= max(0, max_candidates):
            break
    return candidates


def _parse_local_document(path: Path, *, source_url: str, config: Config) -> SourceDoc:
    if path.suffix.lower() == ".pdf":
        from obsidian_llm_wiki.ingest.extractors.pdf import _extract_local_pdf

        return _extract_local_pdf(str(path), source_url=source_url, config=config)
    return parse_document(path, source_url=source_url, config=config)


def _stamp_download_provenance(
    source: SourceDoc,
    downloaded: DownloadedDocument,
    *,
    requested_url: str,
    extracted_url: str,
    discovered: bool = False,
) -> SourceDoc:
    """Attach the complete retrieval chain after the final extracted text exists."""
    common = {
        "requested_url": requested_url,
        "resolved_url": downloaded.resolved_url,
        "extracted_url": extracted_url,
        "content_type": downloaded.content_type,
        "document_format": downloaded.suffix.removeprefix("."),
    }
    parsed = source
    if discovered:
        source = stamp_source(source, extractor="document-discovery", **common)
    stamped = stamp_source(source, extractor="document-download", **common)
    # Parsers historically return the object that registry callers retain. Keep
    # that identity while replacing its frozen provenance value with the final
    # immutable stamp.
    parsed.provenance = stamped.provenance
    return parsed


def _validate_mime(content_type: str, suffix: str) -> None:
    if content_type in _GENERIC_BINARY_TYPES:
        return
    declared_suffix = _CONTENT_TYPE_SUFFIXES.get(content_type)
    if declared_suffix != suffix:
        raise InvalidDocumentError(f"MIME {content_type} does not match expected {suffix} document")


def _validate_signature(path: Path, suffix: str) -> None:
    with path.open("rb") as handle:
        header = handle.read(8)
    if suffix == ".pdf" and not header.startswith(b"%PDF-"):
        raise InvalidDocumentError("PDF signature validation failed")
    if suffix in {".docx", ".epub", ".pptx", ".xlsx"} and not header.startswith(b"PK\x03\x04"):
        raise InvalidDocumentError(f"{suffix} ZIP signature validation failed")
    if suffix in {".doc", ".ppt", ".xls"} and not header.startswith(b"\xd0\xcf\x11\xe0"):
        raise InvalidDocumentError(f"{suffix} OLE signature validation failed")


def _suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in DOCUMENT_SUFFIXES else ""


def _content_type(headers: httpx.Headers) -> str:
    return headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _content_length(headers: httpx.Headers) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _remove_temp(path: Path | None) -> None:
    if path is not None:
        with suppress(OSError):
            os.unlink(path)


class _DocumentLinkParser(HTMLParser):
    """Collect candidate citation/document links without another dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        href = attributes.get("href", "").strip()
        if tag == "meta":
            name = attributes.get("name", attributes.get("property", "")).lower()
            content = attributes.get("content", "").strip()
            if name in {"citation_pdf_url", "citation_fulltext_html_url"} and content:
                self.candidates.append(content)
        elif tag in {"a", "link"} and href:
            relation = " ".join(
                (attributes.get("rel", ""), attributes.get("type", ""), attributes.get("title", ""))
            ).lower()
            if is_document_path(href) or any(
                token in relation
                for token in (
                    "pdf",
                    "epub",
                    "word",
                    "excel",
                    "spreadsheet",
                    "powerpoint",
                    "presentation",
                )
            ):
                self.candidates.append(href)
