"""Real LibreOffice contract for legacy Word (.doc) extraction.

Skipped explicitly when LibreOffice or python-docx is not installed.  The
EN/CN control is authored here with python-docx, exported to Word 97 by the
installed LibreOffice, and must come back through ``dispatch_document`` with
every authored string byte-for-byte intact (no punctuation normalization).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

from obsidian_llm_wiki.config import Config  # noqa: E402
from obsidian_llm_wiki.ingest.office import find_libreoffice  # noqa: E402

SOFFICE = find_libreoffice()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(SOFFICE is None, reason="LibreOffice (soffice) is not installed"),
]

TITLE = "EN/CN Fidelity Control — 中英文对照"
PARAGRAPHS = [
    "The editor’s note says “keep curly quotes” – and dashes — intact…",
    "中文段落：“引号”与‘单引号’必须保留，全角标点（，。；：）不可替换。",
]
LIST_LINES = [
    "- Top-level bullet one",
    "  - Nested bullet: third item’s child",
    "  - 嵌套项目：第二层",
    "1. First ordered step",
    "2. Second ordered step",
]
TABLE_ROWS = ["Region | 地区", "North “A” | 华北", "O’Brien & Co. | 上海·浦东"]
CLOSING = "End of control — 结束。"


def _list_definition(document, formats: list[str]) -> int:
    root = document.part.numbering_part.element
    abstract_id = 1 + max(
        int(a.get(qn("w:abstractNumId"))) for a in root.findall(qn("w:abstractNum"))
    )
    num_id = 1 + max(int(n.get(qn("w:numId"))) for n in root.findall(qn("w:num")))
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    for level, fmt in enumerate(formats):
        lvl = OxmlElement("w:lvl")
        lvl.set(qn("w:ilvl"), str(level))
        text = "\u2022" if fmt == "bullet" else f"%{level + 1}."
        for tag, val in (("w:start", "1"), ("w:numFmt", fmt), ("w:lvlText", text)):
            element = OxmlElement(tag)
            element.set(qn("w:val"), val)
            lvl.append(element)
        abstract.append(lvl)
    root.findall(qn("w:num"))[0].addprevious(abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    ref = OxmlElement("w:abstractNumId")
    ref.set(qn("w:val"), str(abstract_id))
    num.append(ref)
    root.append(num)
    return num_id


def _numbered(paragraph, num_id: int, level: int) -> None:
    num_pr = OxmlElement("w:numPr")
    for tag, val in (("w:ilvl", level), ("w:numId", num_id)):
        element = OxmlElement(tag)
        element.set(qn("w:val"), str(val))
        num_pr.append(element)
    paragraph._p.get_or_add_pPr().append(num_pr)


def _author_word97(tmp_path: Path) -> Path:
    document = docx.Document()
    document.core_properties.title = ""
    bullets = _list_definition(document, ["bullet", "bullet"])
    ordered = _list_definition(document, ["decimal", "lowerLetter"])
    document.add_heading(TITLE, level=1)
    for text in PARAGRAPHS:
        document.add_paragraph(text)
    for line in LIST_LINES:
        level = 1 if line.startswith("  ") else 0
        num_id = bullets if line.lstrip().startswith("-") else ordered
        _numbered(document.add_paragraph(line.lstrip().split(" ", 1)[1]), num_id, level)
    table = document.add_table(rows=len(TABLE_ROWS), cols=2)
    for row, line in enumerate(TABLE_ROWS):
        left, right = line.split(" | ")
        table.cell(row, 0).text = left
        table.cell(row, 1).text = right
    # Word always closes the story with a paragraph after a table.  Without it,
    # LibreOffice's Word 97 export omits the final paragraph mark and its own
    # re-import merges the cells into one paragraph (text kept, cells lost).
    document.add_paragraph(CLOSING)
    authored = tmp_path / "authored" / "en-cn-control.docx"
    authored.parent.mkdir()
    document.save(authored)

    outdir = tmp_path / "doc"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    result = subprocess.run(
        [
            SOFFICE,
            f"-env:UserInstallation={(tmp_path / 'profile').as_uri()}",
            "--headless",
            "--norestore",
            "--convert-to",
            "doc:MS Word 97",
            "--outdir",
            str(outdir),
            str(authored),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    output = outdir / "en-cn-control.doc"
    assert result.returncode == 0 and output.is_file(), result.stderr
    return output


def test_en_cn_word97_control_round_trips_exactly(tmp_path: Path) -> None:
    from obsidian_llm_wiki.ingest.documents import dispatch_document

    word97 = _author_word97(tmp_path)
    before = hashlib.sha256(word97.read_bytes()).hexdigest()

    source = dispatch_document(word97, config=Config(parser_timeout_seconds=120))

    lines = source.content.splitlines()
    assert source.title == TITLE
    assert f"# {TITLE}" in lines
    for expected in (*PARAGRAPHS, *LIST_LINES, *TABLE_ROWS, CLOSING):
        assert expected in lines, (expected, source.content)
    assert "'" not in source.content  # no U+2019 -> ASCII apostrophe substitution
    assert source.provenance.extractor_chain == ("libreoffice-docx",)
    assert hashlib.sha256(word97.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in word97.parent.iterdir()) == ["en-cn-control.doc"]


def test_corrupt_ole_is_rejected_instead_of_imported_as_text(tmp_path: Path) -> None:
    """Without a forced Word filter LibreOffice imports these bytes as plain text."""
    from obsidian_llm_wiki.ingest.documents import DocumentError, dispatch_document

    corrupt = tmp_path / "corrupt.doc"
    corrupt.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(range(256)) * 16)

    with pytest.raises(DocumentError, match="LibreOffice exited"):
        dispatch_document(corrupt, config=Config(parser_timeout_seconds=120))
