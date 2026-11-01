"""Deterministic contracts for the legacy Word (.doc) LibreOffice route.

The LibreOffice subprocess is replaced at the ``subprocess.Popen`` boundary.
Fake conversions write a python-docx-authored DOCX so expected text is known
independently of the code under test.  No test here needs LibreOffice.
"""

from __future__ import annotations

import hashlib
import io
import struct
import subprocess
import zlib
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

docx = pytest.importorskip("docx")

from obsidian_llm_wiki.config import Config  # noqa: E402
from obsidian_llm_wiki.core.models import SourceDoc  # noqa: E402

FREE, END, FAT_SECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD


def _dir_entry(name: str, kind: int, start: int, size: int, child: int = FREE) -> bytes:
    encoded = (name + "\0").encode("utf-16-le")
    entry = bytearray(128)
    entry[: len(encoded)] = encoded
    struct.pack_into("<HBB", entry, 0x40, len(encoded), kind, 1)
    struct.pack_into("<III", entry, 0x44, FREE, FREE, child)
    struct.pack_into("<II", entry, 0x74, start, size)
    return bytes(entry)


def make_cfb(stream_name: str = "WordDocument", *, flags: int = 0, n_fib: int = 0x00C1) -> bytes:
    """Build a minimal Compound File with one 4096-byte Word stream."""
    stream = bytearray(4096)
    struct.pack_into("<HH", stream, 0, 0xA5EC, n_fib)
    struct.pack_into("<H", stream, 0x0A, flags)
    header = bytearray(512)
    header[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<HHHHH", header, 0x18, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<I", header, 0x2C, 1)  # one FAT sector
    struct.pack_into("<I", header, 0x30, 1)  # directory starts in sector 1
    struct.pack_into("<I", header, 0x38, 4096)
    struct.pack_into("<IIII", header, 0x3C, END, 0, END, 0)
    difat = [0] + [FREE] * 108
    struct.pack_into("<109I", header, 0x4C, *difat)
    fat = [FAT_SECT, END, *range(3, 10), END] + [FREE] * (128 - 10)
    directory = (
        _dir_entry("Root Entry", 5, END, 0, child=1)
        + _dir_entry(stream_name, 2, 2, len(stream))
        + bytes(256)
    )
    return bytes(header) + struct.pack("<128I", *fat) + directory + bytes(stream)


def _authored_docx(path: Path, *, image: bool = False) -> None:
    document = docx.Document()
    document.core_properties.title = ""  # python-docx otherwise defaults to "Word Document"
    document.add_heading("Quarterly “Smart” Report", level=1)
    document.add_paragraph("First item of third item’s nested list")
    document.add_paragraph("中文段落：保留弯引号“示例”。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Red Sox"
    table.cell(0, 1).text = "Boston"
    if image:
        document.add_picture(io.BytesIO(_tiny_png()))
    document.save(path)


def _tiny_png() -> bytes:
    """A valid 1x1 PNG, generated so no binary fixture is needed."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


class _FakeProcess:
    """A Popen stand-in that performs (or withholds) a LibreOffice conversion."""

    def __init__(
        self,
        command: list[str],
        *,
        returncode: int = 0,
        write_output: bool = True,
        out_bytes: bytes = b"",
        err_bytes: bytes = b"",
        hang: bool = False,
        image: bool = False,
        **kwargs,
    ) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = returncode
        self.stdout = io.BytesIO(out_bytes)
        self.stderr = io.BytesIO(err_bytes)
        self.hang = hang
        if write_output:
            outdir = Path(command[command.index("--outdir") + 1])
            source = Path(command[-1])
            _authored_docx(outdir / f"{source.stem}.docx", image=image)

    def wait(self, timeout: float | None = None) -> int:
        if self.hang:
            self.hang = False
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def kill(self) -> None:
        return None


class _Recorder:
    def __init__(self, **behaviour) -> None:
        self.behaviour = behaviour
        self.calls: list[_FakeProcess] = []

    def __call__(self, command, **kwargs):
        process = _FakeProcess(list(command), **self.behaviour, **kwargs)
        self.calls.append(process)
        return process


@pytest.fixture(autouse=True)
def _no_real_process_group_signals(monkeypatch: pytest.MonkeyPatch) -> list:
    """Fake PIDs must never reach the real ``os.killpg``."""
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.office.os.killpg", lambda pid, sig: sent.append((pid, sig))
    )
    return sent


@pytest.fixture
def word97(tmp_path: Path) -> Path:
    path = tmp_path / "Quarterly Report.doc"
    path.write_bytes(make_cfb())
    return path


@pytest.fixture
def soffice(monkeypatch: pytest.MonkeyPatch) -> str:
    from obsidian_llm_wiki.ingest import office

    monkeypatch.setattr(office, "find_libreoffice", lambda: "/opt/lo/program/soffice")
    return "/opt/lo/program/soffice"


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route conversion temp dirs to an observable location."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr("obsidian_llm_wiki.ingest.office.tempfile.tempdir", str(root))
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Word binary inspection ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "verdict"),
    [
        (make_cfb(), "word97"),
        (make_cfb(flags=0x0100), "encrypted"),
        (make_cfb("EncryptedPackage"), "encrypted"),
        (make_cfb(n_fib=0x0065), "pre-word97"),
        (make_cfb("Workbook"), "not-word"),
    ],
    ids=["word97", "fib-encrypted", "encrypted-package", "word95", "excel"],
)
def test_inspect_word_binary_classifies_compound_files(
    tmp_path: Path, payload: bytes, verdict: str
) -> None:
    from obsidian_llm_wiki.ingest.office import inspect_word_binary

    path = tmp_path / "probe.doc"
    path.write_bytes(payload)

    assert inspect_word_binary(path) == verdict


def test_inspect_word_binary_rejects_non_ole_bytes(tmp_path: Path) -> None:
    from obsidian_llm_wiki.ingest.office import inspect_word_binary

    path = tmp_path / "fake.doc"
    path.write_bytes(b"<html>not a word file</html>")

    assert inspect_word_binary(path) == "not-ole"


# ── Conversion boundary ───────────────────────────────────────────────


def test_conversion_uses_isolated_profile_strict_filter_and_a_copy(
    word97: Path, soffice: str, scratch: Path
) -> None:
    from obsidian_llm_wiki.ingest.office import extract_legacy_doc

    before = _sha(word97)
    recorder = _Recorder()
    with patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", recorder):
        source = extract_legacy_doc(word97, source_url=str(word97), config=Config())

    [call] = recorder.calls
    command = call.command
    assert command[0] == soffice
    profile = next(arg for arg in command if arg.startswith("-env:UserInstallation=file://"))
    assert str(scratch) in profile
    for flag in ("--headless", "--norestore", "--nolockcheck"):
        assert flag in command
    assert "--infilter=MS Word 97" in command
    assert command[command.index("--convert-to") + 1] == "docx:MS Word 2007 XML"
    converted_input = Path(command[-1])
    assert converted_input != word97 and str(scratch) in str(converted_input)
    assert call.kwargs.get("start_new_session") is True
    assert call.kwargs.get("stdin") == subprocess.DEVNULL
    assert "PYTHONPATH" not in call.kwargs["env"] and "PYTHONHOME" not in call.kwargs["env"]

    assert "First item of third item’s nested list" in source.content
    assert "中文段落：保留弯引号“示例”。" in source.content
    assert "Red Sox | Boston" in source.content
    assert source.title == "Quarterly “Smart” Report"
    assert "libreoffice-docx" in source.provenance.extractor_chain
    assert _sha(word97) == before
    assert sorted(p.name for p in word97.parent.iterdir()) == ["Quarterly Report.doc", "tmp"]
    assert list(scratch.iterdir()) == []


def test_conversion_reports_embedded_assets_as_not_extracted(
    word97: Path, soffice: str, scratch: Path
) -> None:
    from obsidian_llm_wiki.ingest.office import extract_legacy_doc

    with patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", _Recorder(image=True)):
        source = extract_legacy_doc(word97, source_url=str(word97), config=Config())

    assert any(
        "1 image" in item and "not extracted" in item for item in source.provenance.diagnostics
    )


@pytest.mark.parametrize(
    ("behaviour", "message"),
    [
        ({"write_output": False}, "no DOCX output"),
        (
            {
                "returncode": 1,
                "write_output": False,
                "err_bytes": b"Error: source file could not be loaded",
            },
            "could not be loaded",
        ),
        ({"returncode": 81, "write_output": True}, "exited 81"),
    ],
    ids=["exit0-no-output", "load-failure", "nonzero-with-output"],
)
def test_conversion_failures_never_claim_success(
    word97: Path, soffice: str, scratch: Path, behaviour: dict, message: str
) -> None:
    from obsidian_llm_wiki.ingest.office import OfficeConversionError, extract_legacy_doc

    before = _sha(word97)
    with (
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", _Recorder(**behaviour)),
        pytest.raises(OfficeConversionError, match=message),
    ):
        extract_legacy_doc(word97, source_url=str(word97), config=Config())

    assert _sha(word97) == before
    assert list(scratch.iterdir()) == []


def test_conversion_timeout_kills_process_group_and_cleans_up(
    word97: Path, soffice: str, scratch: Path
) -> None:
    from obsidian_llm_wiki.ingest.office import OfficeConversionError, extract_legacy_doc

    killed: list[tuple[int, int]] = []
    with (
        patch(
            "obsidian_llm_wiki.ingest.office.subprocess.Popen",
            _Recorder(hang=True, write_output=False),
        ),
        patch(
            "obsidian_llm_wiki.ingest.office.os.killpg", lambda pid, sig: killed.append((pid, sig))
        ),
        pytest.raises(OfficeConversionError, match="timed out after 3"),
    ):
        extract_legacy_doc(word97, source_url=str(word97), config=Config(parser_timeout_seconds=3))

    assert killed and killed[0][0] == 4242
    assert list(scratch.iterdir()) == []


def test_conversion_stdout_is_byte_capped(word97: Path, soffice: str, scratch: Path) -> None:
    from obsidian_llm_wiki.ingest.office import OfficeConversionError, extract_legacy_doc

    noisy = _Recorder(out_bytes=b"x" * 64)
    with (
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", noisy),
        pytest.raises(OfficeConversionError, match="stdout exceeded 16 bytes"),
    ):
        extract_legacy_doc(
            word97, source_url=str(word97), config=Config(max_parser_stdout_bytes=16)
        )

    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (make_cfb(flags=0x0100), "EncryptedDocumentError", "password-protected"),
        (make_cfb(n_fib=0x0065), "UnsupportedDocumentError", "Word 6/95"),
        (
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 40,
            "UnsupportedDocumentError",
            "not a Word",
        ),
    ],
    ids=["encrypted", "word95", "truncated-ole"],
)
def test_rejected_inputs_never_start_libreoffice(
    tmp_path: Path, soffice: str, payload: bytes, error: str, message: str
) -> None:
    from obsidian_llm_wiki.ingest import office

    path = tmp_path / "locked.doc"
    path.write_bytes(payload)
    recorder = _Recorder()
    with (
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", recorder),
        pytest.raises(getattr(office, error), match=message),
    ):
        office.extract_legacy_doc(path, source_url=str(path), config=Config())

    assert recorder.calls == []
    assert path.read_bytes() == payload


def test_missing_libreoffice_is_reported_as_unavailable(
    word97: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from obsidian_llm_wiki.ingest import office

    monkeypatch.setattr(office.shutil, "which", lambda _name: None)
    with pytest.raises(office.LibreOfficeUnavailableError, match="LibreOffice"):
        office.extract_legacy_doc(word97, source_url=str(word97), config=Config())


# ── Document dispatch integration ─────────────────────────────────────


class _StreamingClient:
    def __init__(self, response: httpx.Response, **_kwargs) -> None:
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def stream(self, _method: str, _url: str, **_kwargs):
        response = self.response

        class _Stream:
            def __enter__(self):
                return response

            def __exit__(self, *_args) -> None:
                response.close()

        return _Stream()


def _download(url: str, content: bytes, content_type: str):
    response = httpx.Response(
        200,
        content=content,
        headers={"content-type": content_type},
        request=httpx.Request("GET", url),
    )
    return patch(
        "obsidian_llm_wiki.ingest.documents.httpx.Client",
        lambda **kwargs: _StreamingClient(response, **kwargs),
    )


def test_remote_doc_uses_native_route_not_liteparse_and_keeps_provenance(
    soffice: str, scratch: Path
) -> None:
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    url = "https://filesamples.com/samples/document/doc/sample2.doc"
    with (
        _download(url, make_cfb(), "application/msword"),
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", _Recorder(image=True)),
        patch(
            "obsidian_llm_wiki.ingest.documents.parse_document",
            side_effect=AssertionError("LiteParse must not handle .doc when LibreOffice exists"),
        ),
    ):
        source = dispatch_document(url, config=Config())

    assert "First item of third item’s nested list" in source.content
    assert source.url == url
    provenance = source.provenance
    assert provenance.requested_url == url
    assert provenance.resolved_url == url
    assert provenance.extracted_url == url
    assert provenance.document_format == "doc"
    assert provenance.content_type == "application/msword"
    assert provenance.extractor_chain == ("libreoffice-docx", "document-download")
    assert provenance.content_sha256 == hashlib.sha256(source.content.encode()).hexdigest()
    assert any("not extracted" in item for item in provenance.diagnostics)
    assert list(scratch.iterdir()) == []


def test_remote_doc_falls_back_to_url_filename_not_temp_name(
    soffice: str, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document without metadata or text title never inherits the temp name."""
    from obsidian_llm_wiki.ingest import office
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    def untitled(path: str) -> SourceDoc:
        return SourceDoc(title=Path(path).stem, content="Body only", url=path)

    monkeypatch.setattr(office, "_extract_converted_docx", untitled)
    url = "https://example.com/files/Annual%20Plan%202024.doc"
    with (
        _download(url, make_cfb(), "application/msword"),
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", _Recorder()),
    ):
        source = dispatch_document(url, config=Config())

    assert source.title == "Annual Plan 2024"


def test_liteparse_temp_stem_title_is_replaced_by_url_filename() -> None:
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    url = "https://example.com/books/field-guide.epub"

    def parse(path, *, source_url=None, **_kwargs) -> SourceDoc:
        return SourceDoc(title=Path(path).stem, content="Body", url=source_url)

    with (
        _download(url, b"PK\x03\x04epub", "application/epub+zip"),
        patch("obsidian_llm_wiki.ingest.documents.parse_document", side_effect=parse),
    ):
        source = dispatch_document(url, config=Config())

    assert source.title == "field-guide"


def test_remote_encrypted_doc_is_invalid_and_temp_download_removed(
    soffice: str, scratch: Path
) -> None:
    from obsidian_llm_wiki.ingest.documents import InvalidDocumentError, dispatch_document

    seen: list[Path] = []
    real_named = __import__("tempfile").NamedTemporaryFile

    def tracking(*args, **kwargs):
        handle = real_named(*args, **kwargs)
        seen.append(Path(handle.name))
        return handle

    url = "https://example.com/locked.doc"
    with (
        _download(url, make_cfb(flags=0x0100), "application/msword"),
        patch("obsidian_llm_wiki.ingest.documents.tempfile.NamedTemporaryFile", tracking),
        patch("obsidian_llm_wiki.ingest.office.subprocess.Popen", _Recorder()),
        pytest.raises(InvalidDocumentError, match="password-protected"),
    ):
        dispatch_document(url, config=Config())

    assert seen and not seen[0].exists()


def test_doc_without_libreoffice_keeps_liteparse_route_with_truthful_diagnostic(
    word97: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from obsidian_llm_wiki.ingest import office
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    monkeypatch.setattr(office.shutil, "which", lambda _name: None)

    def parse(path, *, source_url=None, **_kwargs) -> SourceDoc:
        return SourceDoc(title="Parsed", content="Body", url=source_url)

    with patch("obsidian_llm_wiki.ingest.documents.parse_document", side_effect=parse):
        source = dispatch_document(word97, config=Config())

    assert any("LibreOffice unavailable" in item for item in source.provenance.diagnostics)
