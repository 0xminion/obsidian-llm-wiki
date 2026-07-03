"""Tests for pipeline.okf_markdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import okf_markdown as om

# ── parse_frontmatter ──────────────────────────────────────────────────


def test_parse_frontmatter_basic():
    raw = "---\ntitle: Hello\ntags:\n- foo\n- bar\n---\nBody here.\n"
    meta, body = om.parse_frontmatter(raw)
    assert meta == {"title": "Hello", "tags": ["foo", "bar"]}
    assert body == "Body here.\n"


def test_parse_frontmatter_no_frontmatter():
    raw = "No frontmatter here.\n"
    meta, body = om.parse_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_parse_frontmatter_empty_body():
    raw = "---\ntitle: Hi\n---\n"
    meta, body = om.parse_frontmatter(raw)
    assert meta == {"title": "Hi"}
    assert body == ""


def test_parse_frontmatter_invalid_yaml_returns_empty():
    raw = "---\ntitle: [unclosed\n---\nbody\n"
    meta, body = om.parse_frontmatter(raw)
    assert meta == {}
    # On invalid YAML we return the original raw text untouched.
    assert body == raw


# ── build_frontmatter + roundtrip ──────────────────────────────────────


def test_build_frontmatter_roundtrip():
    original = {"title": "Hi", "tags": ["a", "b"], "count": 3}
    rendered = om.build_frontmatter(original)
    assert rendered.startswith("---\n")
    assert rendered.endswith("---")
    meta, _body = om.parse_frontmatter(rendered + "\nbody\n")
    assert meta == original


def test_build_frontmatter_preserves_order():
    fm = {"zeta": 1, "alpha": 2, "middle": 3}
    rendered = om.build_frontmatter(fm)
    # sort_keys=False means keys keep insertion order.
    keys_block = rendered.split("---\n")[1]
    first_key = keys_block.split(":")[0]
    assert first_key == "zeta"


# ── extract_links ──────────────────────────────────────────────────────


def test_extract_links():
    body = "See [Foo](http://x/y) and [Bar](/bar.md) and image ![](img.png)."
    links = om.extract_links(body)
    assert ("Foo", "http://x/y") in links
    assert ("Bar", "/bar.md") in links
    # Image link excluded
    assert not any(text == "" for text, _ in links)


# ── make_absolute_link / make_relative_link ────────────────────────────


def test_make_absolute_link_default_display():
    assert om.make_absolute_link("foo") == "[foo](/foo.md)"


def test_make_absolute_link_custom_display():
    assert om.make_absolute_link("foo", display_text="Foo") == "[Foo](/foo.md)"


def test_make_absolute_link_no_double_md():
    assert om.make_absolute_link("foo.md") == "[foo.md](/foo.md)"


def test_make_relative_link():
    link = om.make_relative_link("concepts/a.md", "concepts/b.md")
    assert link == "[concepts/b.md](b.md)"


def test_make_relative_link_custom_display():
    link = om.make_relative_link("concepts/a.md", "concepts/b.md",
                                  display_text="B")
    assert link == "[B](b.md)"


# ── slugify ────────────────────────────────────────────────────────────


def test_slugify_basic():
    assert om.slugify("Hello World") == "hello-world"


def test_slugify_punctuation():
    assert om.slugify("Foo, Bar! & Baz?") == "foo-bar-baz"


def test_slugify_collapse_hyphens():
    assert om.slugify("foo   ---   bar") == "foo-bar"


def test_slugify_unicode():
    # Unicode letters preserved
    assert om.slugify("Café Résumé") == "café-résumé"


def test_slugify_apostrophe():
    assert om.slugify("don't stop") == "dont-stop"


# ── safe_read_file / atomic_write ──────────────────────────────────────


def test_safe_read_file_missing(tmp_path: Path):
    assert om.safe_read_file(tmp_path / "nope.md") == ""


def test_atomic_write_and_read(tmp_path: Path):
    target = tmp_path / "out.md"
    om.atomic_write(target, "hello world\n")
    assert target.read_text(encoding="utf-8") == "hello world\n"
    # safe_read_file reads it back too
    assert om.safe_read_file(target) == "hello world\n"


def test_atomic_write_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "out.md"
    om.atomic_write(target, "content")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "content"


def test_atomic_write_overwrites(tmp_path: Path):
    target = tmp_path / "out.md"
    om.atomic_write(target, "v1")
    om.atomic_write(target, "v2")
    assert target.read_text(encoding="utf-8") == "v2"


# ── pytest config ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
