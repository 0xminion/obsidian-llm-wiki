"""Safe, format-aware dispatch for local and remote documents.

All binary document downloads pass through this module.  It deliberately keeps
remote documents out of HTML extraction: a URL that claims to be a document
must validate as that document before a parser is invoked.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

from obsidian_llm_wiki.config import Config, load_config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.http_headers import BROWSER_HEADERS
from obsidian_llm_wiki.ingest.liteparse import parse_document
from obsidian_llm_wiki.ingest.provenance import stamp_source
from obsidian_llm_wiki.ingest.proxy import make_client_kwargs
from obsidian_llm_wiki.ingest.url_safety import (
    stream_with_validated_redirects,
    validate_remote_url,
)

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
    """Whether a URL routes to documents, including supported public file viewers.

    The historical name is retained for registry callers. Only Drive's exact
    file-viewer route is recognized here, never Google Docs editors or folders.
    """
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and (
        is_document_path(value) or bool(_drive_viewer_id(value))
    )


def _drive_https_url(value: str, host: str) -> bool:
    """Reject alternate authorities and URL normalization tricks before fetching."""
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == host
        and not parsed.params
        and not parsed.fragment
        and not any(ord(char) <= 32 or char == "\\" for char in value)
    )


def _drive_viewer_id(value: str) -> str:
    if not _drive_https_url(value, "drive.google.com"):
        return ""
    parsed = urlparse(value)
    match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]{1,200})/view", parsed.path)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if not match or set(query) - {"usp"} or any(len(v) != 1 for v in query.values()):
        return ""
    return match.group(1)


def _is_drive_download(value: str, file_id: str) -> bool:
    if not _drive_https_url(value, "drive.usercontent.google.com"):
        return False
    parsed = urlparse(value)
    return parsed.path in {"/uc", "/download"} and parse_qs(
        parsed.query, keep_blank_values=True
    ) == {"id": [file_id], "export": ["download"]}


def _remaining_drive_time(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise DocumentError("Public Drive download exceeded its time budget")
    return remaining


def _drive_request_guard(
    file_id: str, *, deadline: float, viewer: bool = False
) -> Callable[[httpx.Request], None]:
    def guard(request: httpx.Request) -> None:
        remaining = _remaining_drive_time(deadline)
        request.extensions["timeout"] = dict.fromkeys(
            ("connect", "read", "write", "pool"), remaining
        )
        url = str(request.url)
        allowed = _drive_viewer_id(url) == file_id if viewer else _is_drive_download(url, file_id)
        if not allowed:
            raise InvalidDocumentError("Drive redirect left the public file download route")
        # This connector is intentionally anonymous, including between redirects.
        for name in ("cookie", "authorization", "proxy-authorization"):
            request.headers.pop(name, None)

    return guard


def extract_public_drive_document(
    page_url: str, *, config: Config | None = None, timeout: int = 60
) -> SourceDoc:
    """Follow an explicitly advertised public download, without auth or confirmation.

    Never synthesize a download URL from an ID. Missing links, private viewers,
    confirmation pages and non-document downloads all fail closed.
    """
    file_id = _drive_viewer_id(page_url)
    if not file_id:
        raise InvalidDocumentError("Not a supported public Drive file viewer URL")
    cfg = config or load_config()
    deadline = monotonic() + timeout

    with (
        httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=BROWSER_HEADERS,
            event_hooks={
                "request": [_drive_request_guard(file_id, deadline=deadline, viewer=True)]
            },
        ) as client,
        stream_with_validated_redirects(client, page_url) as response,
    ):
        response.raise_for_status()
        if _content_type(response.headers) != "text/html":
            raise InvalidDocumentError("Drive viewer did not return HTML")
        html = _read_drive_viewer(response, cfg.max_html_bytes, deadline)

    parser = _DriveLinkParser(file_id)
    parser.feed(html)
    parser.close()
    if parser.has_form:
        raise InvalidDocumentError("Drive viewer requires a login or confirmation form")
    if not parser.download_url:
        raise InvalidDocumentError("No official public Drive download link found")
    downloaded = download_document(
        parser.download_url,
        config=cfg,
        timeout=timeout,
        _drive_file_id=file_id,
        _drive_deadline=deadline,
    )
    try:
        source = _parse_local_document(downloaded.path, source_url=page_url, config=cfg)
        source = _stamp_download_provenance(
            source, downloaded, requested_url=page_url, extracted_url=parser.download_url
        )
        return stamp_source(source, requested_url=page_url, extractor="google-drive-public-viewer")
    finally:
        _remove_temp(downloaded.path)


def _read_drive_viewer(response: httpx.Response, max_bytes: int, deadline: float) -> str:
    declared_size = _content_length(response.headers)
    if declared_size is not None and declared_size > max_bytes:
        raise DocumentTooLargeError("Drive viewer Content-Length exceeds HTML byte limit")
    body = bytearray()
    for chunk in response.iter_bytes():
        _remaining_drive_time(deadline)
        if len(body) + len(chunk) > max_bytes:
            raise DocumentTooLargeError("Drive viewer stream exceeded HTML byte limit")
        body.extend(chunk)
    _remaining_drive_time(deadline)
    return body.decode("utf-8", errors="replace")


class _DriveLinkParser(HTMLParser):
    """Read links from markup or JSON strings without executing viewer scripts."""

    def __init__(self, file_id: str) -> None:
        super().__init__()
        self.file_id = file_id
        self.download_url = ""
        self.in_script = False
        self.has_form = False

    def _candidate(self, value: str) -> None:
        if not self.download_url and _is_drive_download(value, self.file_id):
            self.download_url = value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._candidate(dict(attrs).get("href") or "")
        elif tag == "script":
            self.in_script = True
        elif tag == "form":
            self.has_form = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and not self.download_url:
            for match in re.finditer(r'"https:(?:[^"\\]|\\.)*"', data):
                try:
                    self._candidate(json.loads(match.group()))
                except ValueError:
                    continue


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
    if _drive_viewer_id(raw):
        return extract_public_drive_document(raw, config=cfg)
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
    from obsidian_llm_wiki.ingest.web import _read_bounded_html

    with httpx.Client(
        **make_client_kwargs(timeout=timeout, follow_redirects=False), headers=BROWSER_HEADERS
    ) as client:
        try:
            with stream_with_validated_redirects(client, page_url) as response:
                response.raise_for_status()
                html = _read_bounded_html(response, cfg.max_html_bytes)
        except Exception as exc:
            raise DocumentError(f"Could not retrieve document landing page {page_url}") from exc

        candidates = document_candidates(html, page_url, max_candidates=cfg.max_document_candidates)

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
    _drive_file_id: str | None = None,
    _drive_deadline: float | None = None,
) -> DownloadedDocument:
    """Stream one remote binary document to a byte-bounded temporary file.

    The returned path is temporary storage owned by the caller.  On every
    validation/download failure it is removed before the original cause is
    re-raised through a document-specific exception.
    """
    cfg = config or load_config()
    validate_remote_url(url)
    expected_suffix = _suffix_from_url(url)
    temp_path: Path | None = None
    client_kwargs = make_client_kwargs(timeout=timeout, follow_redirects=False)
    if _drive_file_id:
        if _drive_deadline is None:
            _drive_deadline = monotonic() + timeout
        client_kwargs = {
            "timeout": timeout,
            "follow_redirects": False,
            "trust_env": False,
            "event_hooks": {
                "request": [_drive_request_guard(_drive_file_id, deadline=_drive_deadline)]
            },
        }
    try:
        with (
            httpx.Client(**client_kwargs, headers=BROWSER_HEADERS) as client,
            stream_with_validated_redirects(client, url) as response,
        ):
            response.raise_for_status()
            resolved_url = str(response.url)
            if required_host and (urlparse(resolved_url).hostname or "").lower() != required_host:
                raise InvalidDocumentError("Document redirect left the official site")
            content_type = _content_type(response.headers)
            mime_suffix = _CONTENT_TYPE_SUFFIXES.get(content_type)
            suffix = expected_suffix or mime_suffix
            if _drive_file_id and not suffix:
                # Use only a supported suffix; never use the supplied filename
                # as a filesystem path. Signature and MIME checks still apply.
                disposition = Message()
                disposition["content-disposition"] = response.headers.get("content-disposition", "")
                suffix = Path(disposition.get_filename() or "").suffix.lower()
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
                    if _drive_deadline is not None:
                        _remaining_drive_time(_drive_deadline)
                    total += len(chunk)
                    if total > cfg.max_document_bytes:
                        raise DocumentTooLargeError(
                            f"Document stream exceeded {cfg.max_document_bytes} bytes"
                        )
                    handle.write(chunk)

        if _drive_deadline is not None:
            _remaining_drive_time(_drive_deadline)
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
    if path.suffix.lower() == ".docx":
        from obsidian_llm_wiki.ingest.extractors import docx

        if docx._DEPS_AVAILABLE:
            source = docx.extract_docx(str(path))
            source.url = source_url
            return source
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
