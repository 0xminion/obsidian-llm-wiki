"""Legacy Word (.doc) extraction through an isolated, bounded LibreOffice conversion.

Word 97-2003 binaries are converted natively to DOCX by LibreOffice and then
read by the existing python-docx extractor.  This keeps the document's own
Unicode text (curly quotes, CJK, dashes) and table cells instead of
re-deriving text from a rendered PDF layout.

Safety contract:

* the input is inspected first; encrypted, pre-Word-97 and non-Word compound
  files are rejected before LibreOffice starts;
* LibreOffice runs on a private copy, with a throw-away user profile, a forced
  ``MS Word 97`` import filter (otherwise corrupt bytes are silently imported
  as plain text), no stdin, bounded stdout/stderr and a wall-clock timeout
  that kills the whole process group;
* exit status 0 is not trusted on its own: a missing, empty or non-DOCX output
  is a failure;
* every temporary file and profile is removed, and the original file is never
  written to.
"""

from __future__ import annotations

import os
import shutil
import signal
import struct
import subprocess
import tempfile
import zipfile
from contextlib import suppress
from pathlib import Path

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
from obsidian_llm_wiki.ingest.liteparse import _BoundedPipe

EXTRACTOR_STAGE = "libreoffice-docx"

_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_WORD_IDENT = 0xA5EC
_WORD97_MIN_NFIB = 0x00C0
_FIB_ENCRYPTED = 0x0100
_FREE_SECT = 0xFFFFFFFF
_END_OF_CHAIN = 0xFFFFFFFE
_MAX_DIRECTORY_SECTORS = 4096
_MAX_DIFAT_SECTORS = 4096
_INPUT_FILTER = "MS Word 97"
_OUTPUT_FILTER = "docx:MS Word 2007 XML"
# document.xml may legitimately be larger than the zipped DOCX, but never
# arbitrarily so; this bounds decompression of converter output.
_MAX_XML_EXPANSION = 10
_STRIPPED_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")


class OfficeConversionError(RuntimeError):
    """A legacy Office document could not be converted faithfully."""


class OfficeRouteUnavailableError(OfficeConversionError):
    """The native conversion route is not installed in this environment."""


class LibreOfficeUnavailableError(OfficeRouteUnavailableError):
    """No LibreOffice ``soffice`` executable could be found."""


class EncryptedDocumentError(OfficeConversionError):
    """The document is password-protected or encrypted."""


class UnsupportedDocumentError(OfficeConversionError):
    """The file is not a Word 97-2003 binary document."""


# ── Inspection ─────────────────────────────────────────────────────────


def find_libreoffice() -> str | None:
    """Return the LibreOffice CLI on PATH, preferring ``soffice``."""
    return shutil.which("soffice") or shutil.which("libreoffice")


def inspect_word_binary(path: str | Path) -> str:
    """Classify a ``.doc`` candidate without executing any parser.

    Returns one of ``word97``, ``encrypted``, ``pre-word97``, ``not-word``,
    ``not-ole`` or ``unreadable``.  ``unreadable`` means the compound-file
    structure could not be walked with these bounded reads; the strict
    LibreOffice import filter then remains the final judge.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(512)
        if len(header) < 512 or not header.startswith(_OLE_SIGNATURE):
            return "not-ole"
        try:
            return _inspect_compound_file(handle, header, size)
        except (struct.error, ValueError):
            return "unreadable"


def _inspect_compound_file(handle, header: bytes, size: int) -> str:
    sector_shift = struct.unpack_from("<H", header, 0x1E)[0]
    if sector_shift not in (9, 12):
        raise ValueError("unsupported sector size")
    sector_size = 1 << sector_shift
    num_fat, first_dir = struct.unpack_from("<II", header, 0x2C)
    mini_cutoff, _first_minifat, _num_minifat, first_difat, num_difat = struct.unpack_from(
        "<IIIII", header, 0x38
    )
    max_sectors = max(0, size // sector_size - 1)
    if num_fat > max_sectors or num_difat > _MAX_DIFAT_SECTORS:
        raise ValueError("FAT larger than file")

    def read_sector(sector: int) -> bytes:
        if sector >= max_sectors:
            raise ValueError("sector outside file")
        handle.seek((sector + 1) * sector_size)
        data = handle.read(sector_size)
        if len(data) != sector_size:
            raise ValueError("short sector")
        return data

    fat_sectors = [s for s in struct.unpack_from("<109I", header, 0x4C) if s != _FREE_SECT]
    difat, seen = first_difat, set()
    while difat not in (_END_OF_CHAIN, _FREE_SECT) and len(fat_sectors) < num_fat:
        if difat in seen or len(seen) >= _MAX_DIFAT_SECTORS:
            raise ValueError("DIFAT loop")
        seen.add(difat)
        entries = struct.unpack(f"<{sector_size // 4}I", read_sector(difat))
        fat_sectors.extend(s for s in entries[:-1] if s != _FREE_SECT)
        difat = entries[-1]
    fat: list[int] = []
    for sector in fat_sectors[:num_fat]:
        fat.extend(struct.unpack(f"<{sector_size // 4}I", read_sector(sector)))

    streams: dict[str, tuple[int, int, int]] = {}
    sector, visited = first_dir, set()
    while sector != _END_OF_CHAIN:
        if sector in visited or len(visited) >= _MAX_DIRECTORY_SECTORS or sector >= len(fat):
            raise ValueError("directory chain invalid")
        visited.add(sector)
        data = read_sector(sector)
        for offset in range(0, sector_size, 128):
            entry = data[offset : offset + 128]
            name_length, kind = struct.unpack_from("<HB", entry, 0x40)
            if kind == 0 or not 2 <= name_length <= 64:
                continue
            name = entry[: name_length - 2].decode("utf-16-le", errors="replace")
            start, stream_size = struct.unpack_from("<II", entry, 0x74)
            streams.setdefault(name, (kind, start, stream_size))
        sector = fat[sector]

    if "EncryptedPackage" in streams:
        return "encrypted"  # an encrypted OOXML package inside an OLE container
    word = streams.get("WordDocument")
    if word is None or word[0] != 2:
        return "not-word"
    _kind, start, stream_size = word
    if stream_size < max(mini_cutoff, 0x0C):
        raise ValueError("WordDocument stream in the mini stream is not inspected")
    fib = read_sector(start)
    ident, n_fib = struct.unpack_from("<HH", fib, 0)
    flags = struct.unpack_from("<H", fib, 0x0A)[0]
    if ident != _WORD_IDENT:
        return "not-word"
    if flags & _FIB_ENCRYPTED:
        return "encrypted"
    if n_fib < _WORD97_MIN_NFIB:
        return "pre-word97"
    return "word97"


def _reject(verdict: str, path: Path) -> None:
    name = path.name
    if verdict == "encrypted":
        raise EncryptedDocumentError(
            f"{name} is password-protected or encrypted; it cannot be extracted"
        )
    if verdict == "pre-word97":
        raise UnsupportedDocumentError(
            f"{name} is a Word 6/95 (pre-97) binary; only Word 97-2003 .doc is supported"
        )
    if verdict in {"not-word", "not-ole"}:
        raise UnsupportedDocumentError(f"{name} is not a Word 97-2003 .doc document")


# ── Conversion ─────────────────────────────────────────────────────────


def extract_legacy_doc(path: str | Path, *, source_url: str, config: Config) -> SourceDoc:
    """Convert a Word 97-2003 ``.doc`` to DOCX and extract its native text."""
    path = Path(path)
    _reject(inspect_word_binary(path), path)
    soffice = find_libreoffice()
    if not soffice:
        raise LibreOfficeUnavailableError(
            "LibreOffice unavailable: no soffice/libreoffice executable on PATH"
        )
    from obsidian_llm_wiki.ingest.extractors import docx as docx_extractor

    if not docx_extractor._DEPS_AVAILABLE:
        raise OfficeRouteUnavailableError(
            "python-docx unavailable: install obsidian-llm-wiki[docx]"
        )

    with tempfile.TemporaryDirectory(prefix="olw-office-") as raw_workdir:
        workdir = Path(raw_workdir)
        profile, inbox, outbox, scratch = (
            workdir / name for name in ("profile", "in", "out", "tmp")
        )
        for directory in (profile, inbox, outbox, scratch):
            directory.mkdir(mode=0o700)
        # Never hand LibreOffice the caller's file: it would place lock files
        # beside it.  The private copy keeps the original stem so a title
        # fallback can be recognized by the caller.
        private_copy = inbox / f"{_safe_stem(path.stem)}.doc"
        shutil.copyfile(path, private_copy)
        command = [
            soffice,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--invisible",
            "--nologo",
            "--nodefault",
            "--norestore",
            "--nolockcheck",
            f"--infilter={_INPUT_FILTER}",
            "--convert-to",
            _OUTPUT_FILTER,
            "--outdir",
            str(outbox),
            str(private_copy),
        ]
        returncode, _stdout, stderr = _run_soffice(
            command, env=_isolated_env(scratch), config=config
        )
        detail = stderr.decode("utf-8", errors="replace").strip() or "no stderr"
        if returncode != 0:
            raise OfficeConversionError(
                f"LibreOffice exited {returncode} converting {path.name}: {detail}"
            )
        output = outbox / f"{private_copy.stem}.docx"
        if not output.is_file() or output.stat().st_size == 0:
            raise OfficeConversionError(
                f"LibreOffice exited 0 but produced no DOCX output for {path.name}: {detail}"
            )
        _validate_docx_output(output, config)
        diagnostics = _unextracted_content(output)
        source = _extract_converted_docx(str(output))

    source.url = source_url
    source.provenance = SourceProvenance(
        extractor_chain=(EXTRACTOR_STAGE,),
        document_format="doc",
        diagnostics=diagnostics,
    )
    return source


def _extract_converted_docx(path: str) -> SourceDoc:
    from obsidian_llm_wiki.ingest.extractors.docx import extract_docx

    return extract_docx(path)


def _safe_stem(stem: str) -> str:
    cleaned = "".join(char for char in stem if char.isalnum() or char in "._- ").strip(" .")
    return cleaned[:100] or "document"


def _isolated_env(scratch: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _STRIPPED_ENV}
    for key in ("TMPDIR", "TMP", "TEMP"):
        env[key] = str(scratch)
    return env


def _run_soffice(
    command: list[str], *, env: dict[str, str], config: Config
) -> tuple[int, bytes, bytes]:
    """Run LibreOffice in its own session with bounded output and runtime."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise LibreOfficeUnavailableError(
            f"LibreOffice unavailable: could not start {command[0]}"
        ) from exc
    stdout = _BoundedPipe(process.stdout, config.max_parser_stdout_bytes)
    stderr = _BoundedPipe(process.stderr, config.max_parser_stderr_bytes)
    stdout.start()
    stderr.start()
    timeout = config.parser_timeout_seconds
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_group(process)
        process.wait()
        raise OfficeConversionError(f"LibreOffice timed out after {timeout}s") from exc
    finally:
        # Reap helpers (oosplash/soffice.bin) that could outlive the launcher
        # and keep the pipes open.
        _kill_group(process)
        stdout.join()
        stderr.join()
    if stdout.overflowed:
        raise OfficeConversionError(
            f"LibreOffice stdout exceeded {config.max_parser_stdout_bytes} bytes"
        )
    return returncode, stdout.value, stderr.value


def _kill_group(process: subprocess.Popen) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def _validate_docx_output(output: Path, config: Config) -> None:
    if output.stat().st_size > config.max_document_bytes:
        raise OfficeConversionError(f"Converted DOCX exceeds {config.max_document_bytes} bytes")
    try:
        with zipfile.ZipFile(output) as archive:
            info = archive.getinfo("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise OfficeConversionError("LibreOffice output is not a valid DOCX package") from exc
    if info.file_size > config.max_document_bytes * _MAX_XML_EXPANSION:
        raise OfficeConversionError("Converted DOCX body exceeds the decompression bound")


# ── Content not represented in extracted text ─────────────────────────

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
_PIC = "{http://schemas.openxmlformats.org/drawingml/2006/picture}pic"
_V_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"


def _unextracted_content(output: Path) -> tuple[str, ...]:
    """Report embedded content that plain paragraph/table text cannot carry."""
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    counts: dict[str, int] = dict.fromkeys(
        ("image", "embedded OLE object", "text box", "footnote", "endnote", "comment"), 0
    )
    headers = 0
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        body = etree.fromstring(archive.read("word/document.xml"), parser)
        for element in body.iter(_PIC, _V_IMAGEDATA, f"{{{_W}}}object", f"{{{_W}}}txbxContent"):
            if any(ancestor.tag == _MC_FALLBACK for ancestor in element.iterancestors()):
                continue  # VML fallback duplicates the DrawingML choice
            if element.tag == f"{{{_W}}}object":
                counts["embedded OLE object"] += 1
            elif element.tag == f"{{{_W}}}txbxContent":
                counts["text box"] += 1
            elif not any(a.tag == f"{{{_W}}}object" for a in element.iterancestors()):
                counts["image"] += 1  # an OLE preview image is not a separate picture
        for part, tag, kind in (
            ("word/footnotes.xml", "footnote", "footnote"),
            ("word/endnotes.xml", "endnote", "endnote"),
            ("word/comments.xml", "comment", "comment"),
        ):
            if part in names:
                root = etree.fromstring(archive.read(part), parser)
                counts[kind] += sum(
                    1 for note in root.iter(f"{{{_W}}}{tag}") if note.get(f"{{{_W}}}type") is None
                )
        for name in names:
            if name.startswith(("word/header", "word/footer")) and name.endswith(".xml"):
                root = etree.fromstring(archive.read(name), parser)
                if any((t.text or "").strip() for t in root.iter(f"{{{_W}}}t")):
                    headers += 1
    diagnostics = [
        f"{EXTRACTOR_STAGE}: {count} {kind}(s) not extracted; content not verified"
        for kind, count in counts.items()
        if count
    ]
    if headers:
        diagnostics.append(
            f"{EXTRACTOR_STAGE}: {headers} header/footer part(s) with text not extracted"
        )
    return tuple(diagnostics)
