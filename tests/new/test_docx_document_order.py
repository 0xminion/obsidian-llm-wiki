"""Fidelity contracts for the native DOCX extractor.

Fixtures are authored in-test with python-docx so every expected string is
known independently of the extractor under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402


def _numbered(paragraph, num_id: int, level: int) -> None:
    """Attach direct list numbering the way Word/LibreOffice write it."""
    p_pr = paragraph._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), str(level))
    num_id_el = OxmlElement("w:numId")
    num_id_el.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num_id_el)
    p_pr.append(num_pr)


def _extract(path: Path):
    from obsidian_llm_wiki.ingest.extractors.docx import extract_docx

    return extract_docx(str(path))


def test_tables_stay_between_their_surrounding_paragraphs(tmp_path: Path) -> None:
    document = docx.Document()
    document.add_paragraph("Here is a table:")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Team"
    table.cell(0, 1).text = "City"
    table.cell(1, 0).text = "Mets"
    table.cell(1, 1).text = "New York"
    document.add_paragraph("This concludes our test.")
    path = tmp_path / "ordered.docx"
    document.save(path)

    content = _extract(path).content

    before = content.index("Here is a table:")
    row = content.index("Mets | New York")
    after = content.index("This concludes our test.")
    assert before < row < after


def test_nested_bullets_and_numbering_keep_list_markers(tmp_path: Path) -> None:
    document = docx.Document()
    # numId 1 is a bullet list and numId 5 a decimal list in the default template.
    _numbered(document.add_paragraph("Third item of bulleted list."), 1, 0)
    _numbered(document.add_paragraph("First item of third item’s nested list"), 1, 1)
    _numbered(document.add_paragraph("First item of numbered list."), 5, 0)
    _numbered(document.add_paragraph("Second item of numbered list."), 5, 0)
    path = tmp_path / "lists.docx"
    document.save(path)

    lines = _extract(path).content.splitlines()

    assert "- Third item of bulleted list." in lines
    assert "  - First item of third item’s nested list" in lines
    assert "1. First item of numbered list." in lines
    assert "2. Second item of numbered list." in lines


def test_horizontally_merged_cells_are_not_duplicated(tmp_path: Path) -> None:
    document = docx.Document()
    table = document.add_table(rows=2, cols=3)
    merged = table.cell(0, 1).merge(table.cell(0, 2))
    merged.text = "New York"
    table.cell(1, 0).text = "Baseball"
    table.cell(1, 1).text = "Mets"
    table.cell(1, 2).text = "Yankees"
    path = tmp_path / "merged.docx"
    document.save(path)

    content = _extract(path).content

    assert content.count("New York") == 1
    assert "Baseball | Mets | Yankees" in content


def test_unicode_punctuation_and_cjk_are_preserved_exactly(tmp_path: Path) -> None:
    text = "It’s “quoted” — 中文：表格与嵌套列表。"
    document = docx.Document()
    document.add_paragraph(text)
    path = tmp_path / "unicode.docx"
    document.save(path)

    assert text in _extract(path).content.splitlines()
