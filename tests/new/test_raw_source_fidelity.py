"""Original-text newline/UTF-8 controls; no private or external source corpus."""
from __future__ import annotations

import hashlib

import pytest

from obsidian_llm_wiki.ingest.sources import load_source_file
from obsidian_llm_wiki.render.frontmatter import parse_frontmatter, safe_read_file


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\n\r\n"])
def test_disk_loader_preserves_interior_and_existing_outer_strip(tmp_path, ending):
    body = "雪 café  " + ending + "\tinterior\rbare CR" + ending + "last"
    raw = " \t\n" + body + "  \r\n\t"
    path = tmp_path / "raw.md"
    path.write_bytes(raw.encode("utf-8"))
    before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
    assert safe_read_file(path) == raw
    assert load_source_file(path).content == body
    assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) == before


@pytest.mark.parametrize("opening,closing", [("\n", "\n"), ("\r\n", "\r\n"),
                                                ("\n", "\r\n"), ("\r\n", "\n")])
def test_frontmatter_delimiters_do_not_translate_body(tmp_path, opening, closing):
    body = " \t雪  \r\nsecond\nthird\rfourth\r\n  "
    raw = "---" + opening + "title: 雪" + closing + "---" + closing + closing + body
    meta, parsed = parse_frontmatter(raw)
    assert meta == {"title": "雪"}
    assert parsed == body
    path = tmp_path / "raw.md"
    path.write_bytes(raw.encode())
    doc = load_source_file(path)
    assert doc.title == "雪" and doc.content == body.strip()
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(raw.encode()).digest()


@pytest.mark.parametrize("suffix", ["", "\n", "\r\n"])
def test_empty_frontmatter_body_and_unterminated_delimiter(suffix):
    assert parse_frontmatter("---\r\ntitle: 雪\r\n---" + suffix) == ({"title": "雪"}, "")
    malformed = "---\r\ntitle: 雪\r\nnot closed" + suffix
    assert parse_frontmatter(malformed) == ({}, malformed)


def test_frontmatter_does_not_start_stripping_spaces_or_bare_carriage_returns():
    for body in (" \nline", "\t\r\nline", "\rline"):
        assert parse_frontmatter("---\ntitle: x\n---\n" + body) == ({"title": "x"}, body)
