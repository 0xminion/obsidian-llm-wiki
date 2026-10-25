"""Regression tests for live50 document acquisition (real HTTPX decoding)."""

from __future__ import annotations

import gzip
import ipaddress
import json
import zlib

import httpx
import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.ingest import documents, url_safety


@pytest.fixture
def transport(monkeypatch, tmp_path):
    """Replace only the network; retain HTTPX headers, streams and decoders."""
    real_client = httpx.Client
    monkeypatch.setattr(
        url_safety, "_resolved_addresses", lambda _host: {ipaddress.ip_address("8.8.8.8")}
    )
    monkeypatch.setattr(documents.tempfile, "tempdir", str(tmp_path))
    requests = []

    def install(handler):
        def capture(request):
            requests.append(request)
            assert "authorization" not in request.headers
            assert "cookie" not in request.headers
            return handler(request)

        def client(**kwargs):
            kwargs.pop("proxy", None)
            kwargs["trust_env"] = False
            return real_client(transport=httpx.MockTransport(capture), **kwargs)

        monkeypatch.setattr(documents.httpx, "Client", client)
        return requests

    return install


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
def test_pdf_wire_negotiation_only_advertises_standard_decoders(transport, encoding):
    payload = b"%PDF-1.7\nvalid signature control\n%%EOF"
    compress = gzip.compress if encoding == "gzip" else zlib.compress

    def serve(request):
        # A server may select any advertised encoding: no optional decoder
        # should be advertised even when the test response picks gzip/deflate.
        assert request.headers["accept-encoding"] == "gzip, deflate"
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf", "content-encoding": encoding},
            stream=httpx.ByteStream(compress(payload)),
        )

    transport(serve)
    downloaded = documents.download_document("https://example.com/report.pdf")
    try:
        assert downloaded.path.read_bytes() == payload
    finally:
        downloaded.path.unlink()


FILE_ID = "19YPsXg_bChXEq0o_D-hUpGqtz8Pz9nk5"
VIEWER = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"
DOWNLOAD = f"https://drive.usercontent.google.com/uc?id={FILE_ID}&export=download"
ARTIFACT = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download"


def _viewer(link=DOWNLOAD):
    # Matches the live viewer's JSON-escaped script string, not a guessed URL.
    quoted = json.dumps(link).replace("&", r"\u0026").replace("=", r"\u003d")
    return f'<html><script>window.viewerData = [null,"application/pdf",{quoted}];</script></html>'


@pytest.fixture
def pdf_bytes():
    pymupdf = pytest.importorskip("pymupdf")

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "A public document with native PDF text.")
        pdf.set_metadata({"title": "Public document"})
        return pdf.tobytes()


def test_public_drive_viewer_dispatches_official_link_with_full_provenance(
    transport, pdf_bytes, tmp_path, monkeypatch
):
    from obsidian_llm_wiki.ingest import extractors

    def no_html(_url):
        raise AssertionError("Drive viewer incorrectly reached generic HTML extraction")

    monkeypatch.setattr(extractors, "extract_web", no_html)
    extract = extractors.extract

    def serve(request):
        if str(request.url) == VIEWER:
            return httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
        if str(request.url) == DOWNLOAD:
            return httpx.Response(303, headers={"location": ARTIFACT})
        assert str(request.url) == ARTIFACT
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "content-disposition": 'attachment; filename="public.pdf"',
            },
            stream=httpx.ByteStream(pdf_bytes),
        )

    requests = transport(serve)
    source = extract(VIEWER)
    assert "A public document with native PDF text." in source.content
    assert source.title == "Public document"
    assert source.url == VIEWER
    assert source.provenance.requested_url == VIEWER
    assert source.provenance.extracted_url == DOWNLOAD
    assert source.provenance.resolved_url == ARTIFACT
    assert source.provenance.content_type == "application/octet-stream"
    assert source.provenance.document_format == "pdf"
    assert "google-drive-public-viewer" in source.provenance.extractor_chain
    assert "document-download" in source.provenance.extractor_chain
    assert [str(request.url) for request in requests] == [VIEWER, DOWNLOAD, ARTIFACT]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "page_url",
    [
        VIEWER.replace("https:", "http:"),
        VIEWER.replace("drive.google.com", "drive.google.com.evil.example"),
        VIEWER.replace("drive.google.com", "drive.google.com@evil.example"),
        VIEWER.replace("drive.google.com", "user:secret@drive.google.com"),
        VIEWER.replace("drive.google.com", "drive.google.com:444"),
        VIEWER.replace("drive.google.com", "127.0.0.1"),
        VIEWER.replace("/view?", "/edit?"),
        VIEWER + "&authuser=0",
        VIEWER + "&confirm=t",
        VIEWER.replace(FILE_ID, "%2e%2e"),
        VIEWER + "\n",
        f"https://docs.google.com/document/d/{FILE_ID}/edit",
    ],
)
def test_hostile_or_nonpublic_viewer_urls_never_request(transport, page_url):
    requests = transport(lambda request: pytest.fail(f"Unexpected network: {request.url}"))
    assert not documents.is_direct_document_url(page_url)
    with pytest.raises(documents.InvalidDocumentError, match="viewer URL"):
        documents.extract_public_drive_document(page_url)
    assert not requests


@pytest.mark.parametrize(
    "link",
    [
        "http://127.0.0.1/private.pdf",
        "https://evil.example/file.pdf",
        DOWNLOAD.replace(
            "drive.usercontent.google.com", "drive.usercontent.google.com.evil.example"
        ),
        DOWNLOAD.replace(
            "drive.usercontent.google.com", "user:secret@drive.usercontent.google.com"
        ),
        DOWNLOAD.replace("drive.usercontent.google.com", "docs.google.com"),
        DOWNLOAD.replace("/uc?", "/arbitrary-api?"),
        DOWNLOAD.replace(FILE_ID, "another_file"),
        DOWNLOAD.replace("https:", "http:"),
        DOWNLOAD + "&confirm=t",
        DOWNLOAD + "&uuid=secret",
        DOWNLOAD + "&id=another_file",
        DOWNLOAD + "&authuser=0",
        DOWNLOAD + "&resourcekey=secret",
        DOWNLOAD + "#download",
        "//drive.usercontent.google.com/uc?id=" + FILE_ID + "&export=download",
        "javascript:alert(1)",
    ],
)
def test_hostile_download_links_are_not_followed(transport, link):
    requests = transport(
        lambda request: httpx.Response(
            200, text=_viewer(link), headers={"content-type": "text/html"}
        )
    )
    with pytest.raises(documents.InvalidDocumentError, match="No official public"):
        documents.extract_public_drive_document(VIEWER)
    assert [str(r.url) for r in requests] == [VIEWER]


@pytest.mark.parametrize(
    "html",
    [
        "<html>Loading...</html>",
        "<html><title>Sign in - Google Accounts</title><p>You need access</p></html>",
        '<a href="https://accounts.google.com/ServiceLogin">Sign in</a>',
    ],
)
def test_missing_or_private_download_is_not_guessed(transport, html):
    requests = transport(
        lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"})
    )
    with pytest.raises(documents.InvalidDocumentError, match="No official public"):
        documents.extract_public_drive_document(VIEWER)
    assert [str(r.url) for r in requests] == [VIEWER]


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1/internal.pdf",
        "http://169.254.169.254/latest/meta-data",
        "https://accounts.google.com/ServiceLogin",
        "https://evil.example/file.pdf",
        ARTIFACT.replace(FILE_ID, "different_file"),
        ARTIFACT + "&confirm=t",
        "https://drive.google.com/arbitrary-api",
    ],
)
@pytest.mark.parametrize("hop", ["viewer", "download"])
def test_redirect_policy_rejects_target_before_request(transport, target, hop):
    def serve(request):
        if str(request.url) == VIEWER and hop == "download":
            return httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
        return httpx.Response(303, headers={"location": target})

    requests = transport(serve)
    with pytest.raises((documents.DocumentError, ValueError)):
        documents.extract_public_drive_document(VIEWER)
    assert [str(r.url) for r in requests] == ([VIEWER] if hop == "viewer" else [VIEWER, DOWNLOAD])


@pytest.mark.parametrize(
    "content_type, payload, message",
    [
        ("text/html", b"<form>Confirm download</form>", "MIME"),
        ("application/pdf", b"<html>Sign in</html>", "signature"),
        ("application/octet-stream", b"<html>You need access</html>", "signature"),
    ],
)
def test_drive_artifact_mime_and_signature_fail_closed(
    transport, tmp_path, content_type, payload, message
):
    def serve(request):
        if str(request.url) == VIEWER:
            return httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
        return httpx.Response(
            200,
            headers={
                "content-type": content_type,
                "content-disposition": 'attachment; filename="file.pdf"',
            },
            stream=httpx.ByteStream(payload),
        )

    transport(serve)
    with pytest.raises(documents.InvalidDocumentError, match=message):
        documents.extract_public_drive_document(VIEWER)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["viewer", "artifact", "shared"])
def test_drive_download_has_one_total_time_budget(
    transport, pdf_bytes, monkeypatch, tmp_path, phase
):
    clock = [0.0]
    monkeypatch.setattr(documents, "monotonic", lambda: clock[0], raising=False)

    class Delayed(httpx.SyncByteStream):
        def __init__(self, payload, delay):
            self.payload, self.delay = payload, delay

        def __iter__(self):
            clock[0] += self.delay
            yield self.payload

    def serve(request):
        if str(request.url) == VIEWER:
            delay = 6 if phase == "viewer" else 2
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                stream=Delayed(_viewer().encode(), delay),
            )
        delay = 6 if phase == "artifact" else 4
        return httpx.Response(
            200, headers={"content-type": "application/pdf"}, stream=Delayed(pdf_bytes, delay)
        )

    transport(serve)
    with pytest.raises(documents.DocumentError, match="time.*budget|deadline"):
        documents.extract_public_drive_document(VIEWER, timeout=5)
    assert not list(tmp_path.iterdir())


def test_drive_viewer_deadline_checked_between_small_wire_chunks(transport, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(documents, "monotonic", lambda: clock[0])

    class DripStream(httpx.SyncByteStream):
        def __iter__(self):
            clock[0] = 6.0
            yield b"<html>"
            pytest.fail("Expired viewer stream consumed another network chunk")

    transport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, stream=DripStream()
        )
    )
    with pytest.raises(documents.DocumentError, match="time budget"):
        documents.extract_public_drive_document(VIEWER, timeout=5)


def test_drive_cookie_redirects_remain_anonymous(transport, pdf_bytes):
    def serve(request):
        if str(request.url) == VIEWER:
            return httpx.Response(
                200,
                text=_viewer(),
                headers={
                    "content-type": "text/html",
                    "set-cookie": "session=private; Path=/",
                },
            )
        if str(request.url) == DOWNLOAD:
            return httpx.Response(
                303,
                headers={
                    "location": ARTIFACT,
                    "set-cookie": "download_warning=token; Path=/",
                },
            )
        return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})

    requests = transport(serve)
    assert documents.extract_public_drive_document(VIEWER).title == "Public document"
    assert len(requests) == 3
    assert all("cookie" not in request.headers for request in requests)
    assert all(0 < request.extensions["timeout"]["read"] <= 60 for request in requests)


@pytest.mark.parametrize("phase", ["viewer", "artifact", "direct"])
@pytest.mark.parametrize("declared", [True, False])
def test_document_and_viewer_byte_caps_count_decoded_bytes(transport, tmp_path, phase, declared):
    payload = (b"<html>" if phase == "viewer" else b"%PDF-1.7\n") + b"x" * 2048

    def serve(request):
        if phase == "artifact" and str(request.url) == VIEWER:
            return httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
        headers = {
            "content-type": "text/html" if phase == "viewer" else "application/pdf",
            "content-encoding": "gzip",
        }
        if declared:
            headers["content-length"] = "2048"
        return httpx.Response(200, headers=headers, stream=httpx.ByteStream(gzip.compress(payload)))

    requests = transport(serve)
    with pytest.raises(documents.DocumentTooLargeError, match="Content-Length|stream"):
        if phase == "direct":
            documents.download_document(
                "https://example.com/test.pdf", config=Config(max_document_bytes=1024)
            )
        else:
            documents.extract_public_drive_document(
                VIEWER,
                config=Config(
                    max_document_bytes=1024,
                    max_html_bytes=1024,
                ),
            )
    assert len(requests) == (2 if phase == "artifact" else 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["viewer", "artifact"])
def test_drive_dns_resolution_still_rejects_private_addresses(transport, monkeypatch, phase):
    def resolve(host):
        is_target = host == (
            "drive.google.com" if phase == "viewer" else "drive.usercontent.google.com"
        )
        return {ipaddress.ip_address("127.0.0.1" if is_target else "8.8.8.8")}

    requests = transport(
        lambda request: httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
    )
    monkeypatch.setattr(url_safety, "_resolved_addresses", resolve)
    with pytest.raises(ValueError, match="non-public"):
        documents.extract_public_drive_document(VIEWER)
    assert len(requests) == (0 if phase == "viewer" else 1)


@pytest.mark.parametrize("phase", ["viewer", "artifact"])
def test_drive_redirect_loops_are_capped(transport, phase):
    def serve(request):
        if str(request.url) == VIEWER and phase == "artifact":
            return httpx.Response(200, text=_viewer(), headers={"content-type": "text/html"})
        return httpx.Response(303, headers={"location": str(request.url)})

    requests = transport(serve)
    with pytest.raises((ValueError, documents.DocumentError)):
        documents.extract_public_drive_document(VIEWER)
    assert len(requests) <= 7


@pytest.mark.parametrize("encoding", [None, "gzip", "deflate"])
def test_direct_pdf_normal_control_is_parsed(transport, pdf_bytes, tmp_path, encoding):
    from obsidian_llm_wiki.ingest.extractors import extract

    data = pdf_bytes
    headers = {"content-type": "application/pdf"}
    if encoding:
        headers["content-encoding"] = encoding
        data = (gzip.compress if encoding == "gzip" else zlib.compress)(data)
    transport(lambda request: httpx.Response(200, headers=headers, stream=httpx.ByteStream(data)))
    url = "https://example.com/report.pdf"
    source = extract(url)
    assert "native PDF text" in source.content
    assert source.provenance.requested_url == url
    assert source.provenance.resolved_url == url
    assert source.provenance.document_format == "pdf"
    assert not list(tmp_path.iterdir())


def test_drive_literal_html_anchor_download(transport, pdf_bytes):
    def serve(request):
        if str(request.url) == VIEWER:
            return httpx.Response(
                200,
                text='<a href="' + DOWNLOAD.replace("&", "&amp;") + '">Download</a>',
                headers={"content-type": "text/html"},
            )
        return httpx.Response(200, content=pdf_bytes, headers={"content-type": "application/pdf"})

    transport(serve)
    assert documents.extract_public_drive_document(VIEWER).provenance.extracted_url == DOWNLOAD


def test_drive_confirmation_form_is_not_bypassed_even_with_download_link(transport):
    html = '<form action="/download"><input name="confirm" value="t"></form>' + _viewer()

    def serve(request):
        if str(request.url) == VIEWER:
            return httpx.Response(200, text=html, headers={"content-type": "text/html"})
        return httpx.Response(403)

    requests = transport(serve)
    with pytest.raises(documents.InvalidDocumentError, match="confirmation|form"):
        documents.extract_public_drive_document(VIEWER)
    assert [str(r.url) for r in requests] == [VIEWER]
