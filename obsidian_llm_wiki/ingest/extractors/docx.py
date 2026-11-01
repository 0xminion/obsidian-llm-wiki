"""DOCX extractor — extracts text from Word .docx files using python-docx.

Body paragraphs and tables are emitted in document order.  List paragraphs
keep a Markdown marker (``-`` for bullets, ``N.`` for ordered levels) indented
by nesting level, and merged table cells are emitted once.  Text inside
paragraphs is copied verbatim, so typographic punctuation and CJK survive.

Not extracted: headers/footers, footnotes/endnotes, comments, text boxes and
embedded objects/images.

Dependency (optional): ``python-docx``.
Install with: ``pip install obsidian-llm-wiki[docx]``
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.ingest.extractors import register_extractor

logger = logging.getLogger("obswiki.ingest.extractors.docx")

# ── Dependency check ────────────────────────────────────────────────────

try:
    import docx  # python-docx  # type: ignore[import-untyped]
    from docx.oxml.ns import qn  # type: ignore[import-untyped]
    from docx.text.paragraph import Paragraph  # type: ignore[import-untyped]

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ── Matching ────────────────────────────────────────────────────────────


def _is_docx(parsed, raw: str) -> bool:
    """Match .docx file paths."""
    return raw.lower().endswith(".docx")


# ── Structure helpers ───────────────────────────────────────────────────

_LIST_STYLE_LEVEL = re.compile(r"^list (?:bullet|number|continue)\s+(\d)$", re.IGNORECASE)


def _iter_blocks(container) -> Iterator:
    """Yield body ``w:p``/``w:tbl`` elements in order, entering content controls."""
    for child in container.iterchildren():
        if child.tag in (qn("w:p"), qn("w:tbl")):
            yield child
        elif child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _iter_blocks(content)


def _val(element, tag: str) -> str | None:
    found = element.find(qn(tag)) if element is not None else None
    return found.get(qn("w:val")) if found is not None else None


def _number(value: int, fmt: str) -> str:
    if fmt in {"lowerLetter", "upperLetter"}:
        letters = ""
        while value > 0:
            value, rest = divmod(value - 1, 26)
            letters = chr(ord("a") + rest) + letters
        return letters.upper() if fmt == "upperLetter" else letters
    if fmt in {"lowerRoman", "upperRoman"}:
        numerals = (
            (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
        )  # fmt: skip
        roman = ""
        for amount, numeral in numerals:
            count, value = divmod(value, amount)
            roman += numeral * count
        return roman.upper() if fmt == "upperRoman" else roman
    return str(value)


class _ListNumbering:
    """Resolve paragraph numbering to Markdown list markers."""

    def __init__(self, document) -> None:
        self._levels: dict[str, dict[int, tuple[str, int]]] = {}
        self._counters: dict[str, dict[int, int]] = {}
        try:
            root = document.part.numbering_part.element
        except (KeyError, NotImplementedError, AttributeError):
            return
        abstract = {a.get(qn("w:abstractNumId")): a for a in root.findall(qn("w:abstractNum"))}
        for num in root.findall(qn("w:num")):
            levels: dict[int, tuple[str, int]] = {}
            definition = abstract.get(_val(num, "w:abstractNumId"))
            if definition is not None:
                for lvl in definition.findall(qn("w:lvl")):
                    levels[int(lvl.get(qn("w:ilvl"), "0"))] = self._format(lvl)
            for override in num.findall(qn("w:lvlOverride")):
                level = int(override.get(qn("w:ilvl"), "0"))
                lvl = override.find(qn("w:lvl"))
                if lvl is not None:
                    levels[level] = self._format(lvl)
                start = _val(override, "w:startOverride")
                if start is not None and level in levels:
                    levels[level] = (levels[level][0], int(start))
            self._levels[num.get(qn("w:numId"))] = levels

    @staticmethod
    def _format(lvl) -> tuple[str, int]:
        return _val(lvl, "w:numFmt") or "decimal", int(_val(lvl, "w:start") or "1")

    def marker(self, paragraph) -> tuple[int, str] | None:
        num_id, level = self._num_pr(paragraph)
        if num_id is None or num_id == "0":
            return None
        fmt, start = self._levels.get(num_id, {}).get(level, ("bullet", 1))
        if fmt == "none":
            return None
        if fmt == "bullet":
            return level, "-"
        counters = self._counters.setdefault(num_id, {})
        counters[level] = counters.get(level, start - 1) + 1
        for deeper in [key for key in counters if key > level]:
            del counters[deeper]
        return level, f"{_number(counters[level], fmt)}."

    @staticmethod
    def _num_pr(paragraph) -> tuple[str | None, int]:
        p_pr = paragraph._p.find(qn("w:pPr"))
        num_pr = p_pr.find(qn("w:numPr")) if p_pr is not None else None
        num_id = _val(num_pr, "w:numId")
        ilvl = _val(num_pr, "w:ilvl")
        style, depth = paragraph.style, 0
        while num_id is None and style is not None and depth < 16:
            style_num_pr = style.element.find(f"{qn('w:pPr')}/{qn('w:numPr')}")
            if style_num_pr is not None:
                num_id = _val(style_num_pr, "w:numId")
                ilvl = ilvl or _val(style_num_pr, "w:ilvl")
                match = _LIST_STYLE_LEVEL.match(style.name or "")
                if ilvl is None and match:
                    ilvl = str(int(match.group(1)) - 1)
            style, depth = style.base_style, depth + 1
        return num_id, int(ilvl or 0)


def _paragraph_line(paragraph, numbering: _ListNumbering) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    if "heading 1" in style_name:
        return f"# {text}"
    if "heading 2" in style_name:
        return f"## {text}"
    if "heading 3" in style_name:
        return f"### {text}"
    if "heading" in style_name:
        return f"#### {text}"
    marker = numbering.marker(paragraph)
    if marker:
        level, symbol = marker
        return f"{'  ' * level}{symbol} {text}"
    return text


def _table_lines(table, parent) -> list[str]:
    lines: list[str] = []
    for row in table.iterchildren(qn("w:tr")):
        cells: list[str] = []
        for cell in row.iterchildren(qn("w:tc")):
            # One w:tc per merged span: horizontal merges are not repeated.
            paragraphs = (Paragraph(p, parent).text.strip() for p in cell.iter(qn("w:p")))
            text = "\n".join(p for p in paragraphs if p)
            if text:
                cells.append(text)
        if cells:
            lines.append(" | ".join(cells))
    return lines


# ── Registration (only if deps available) ────────────────────────────────

if _DEPS_AVAILABLE:

    @register_extractor(_is_docx)
    def extract_docx(raw_path: str) -> SourceDoc:
        """Extract text from a Word .docx file.

        Paragraphs and tables are emitted in body order.  Title is derived
        from core properties or the first non-empty paragraph.
        """
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"DOCX file not found: {raw_path}")

        document = docx.Document(str(path))  # type: ignore[union-attr]

        # ── Title from core properties ──────────────────────────────────
        title = ""
        core_props = document.core_properties
        if core_props and core_props.title:
            title = core_props.title.strip()
        if not title:
            # Use first non-empty paragraph.
            for para in document.paragraphs:
                if para.text.strip():
                    title = para.text.strip()[:120]
                    break
        if not title:
            title = path.stem

        # ── Paragraphs and tables in document order ──────────────────────
        numbering = _ListNumbering(document)
        parts: list[str] = []
        for block in _iter_blocks(document.element.body):
            if block.tag == qn("w:tbl"):
                parts.extend(_table_lines(block, document))
            else:
                line = _paragraph_line(Paragraph(block, document), numbering)
                if line:
                    parts.append(line)

        content = "\n\n".join(parts)
        if not content.strip():
            raise RuntimeError(f"DOCX has no extractable text: {raw_path}")

        return SourceDoc(
            title=title,
            content=content,
            url=str(path),
        )
