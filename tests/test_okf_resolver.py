"""Tests for pipeline.okf_resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import okf_resolver as okfr

# ── build_concept_registry ─────────────────────────────────────────────


def test_build_concept_registry_multiple_subdirs(tmp_path: Path):
    """Registry should map slugs and full concept ids across subdirs."""
    (tmp_path / "concepts" / "alpha").mkdir(parents=True)
    (tmp_path / "concepts" / "beta").mkdir(parents=True)
    (tmp_path / "sources").mkdir(parents=True)

    (tmp_path / "concepts" / "alpha" / "foo.md").write_text("# Foo\n", encoding="utf-8")
    (tmp_path / "concepts" / "beta" / "bar.md").write_text("# Bar\n", encoding="utf-8")
    (tmp_path / "sources" / "web.md").write_text("# Web\n", encoding="utf-8")

    registry = okfr.build_concept_registry(tmp_path)

    # slug -> concept_id
    assert registry["foo"] == "concepts/alpha/foo"
    assert registry["bar"] == "concepts/beta/bar"
    assert registry["web"] == "sources/web"

    # full concept_id -> concept_id (idempotent)
    assert registry["concepts/alpha/foo"] == "concepts/alpha/foo"
    assert registry["concepts/beta/bar"] == "concepts/beta/bar"
    assert registry["sources/web"] == "sources/web"


def test_build_concept_registry_skips_index_and_log(tmp_path: Path):
    """index.md, log.md, and viz.html should be excluded from the registry."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "foo.md").write_text("# Foo\n", encoding="utf-8")
    (tmp_path / "index.md").write_text("# Index\n", encoding="utf-8")
    (tmp_path / "log.md").write_text("# Log\n", encoding="utf-8")

    registry = okfr.build_concept_registry(tmp_path)

    assert "foo" in registry
    assert "index" not in registry
    assert "log" not in registry


# ── resolve_links ──────────────────────────────────────────────────────


def test_resolve_links_bare_slug_to_absolute(tmp_path: Path):
    """A bare slug link should be rewritten to an absolute bundle-relative path."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "foo.md").write_text(
        "---\ntitle: Foo\n---\nSee [bar](bar) for more.\n", encoding="utf-8"
    )
    (tmp_path / "concepts" / "bar.md").write_text(
        "---\ntitle: Bar\n---\n# Bar\n", encoding="utf-8"
    )

    count = okfr.resolve_links(tmp_path)
    assert count == 1

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    assert "[bar](/concepts/bar.md)" in content


def test_resolve_links_relative_md_to_absolute(tmp_path: Path):
    """A relative .md path should be resolved to an absolute bundle-relative path."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "alpha").mkdir(parents=True)

    # foo.md links to alpha/baz.md with a relative path
    (tmp_path / "concepts" / "foo.md").write_text(
        "---\ntitle: Foo\n---\nSee [baz](alpha/baz.md).\n", encoding="utf-8"
    )
    (tmp_path / "concepts" / "alpha" / "baz.md").write_text(
        "---\ntitle: Baz\n---\n# Baz\n", encoding="utf-8"
    )

    count = okfr.resolve_links(tmp_path)
    assert count == 1

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    assert "[baz](/concepts/alpha/baz.md)" in content


def test_resolve_links_external_url_unchanged(tmp_path: Path):
    """External URLs (http, https, mailto) and anchors must not be rewritten."""
    (tmp_path / "concepts").mkdir(parents=True)
    body = (
        "---\ntitle: Foo\n---\n"
        "[web](https://example.com) "
        "[mail](mailto:a@b.com) "
        "[anchor](#section)\n"
    )
    (tmp_path / "concepts" / "foo.md").write_text(body, encoding="utf-8")

    count = okfr.resolve_links(tmp_path)
    assert count == 0

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    assert "https://example.com" in content
    assert "mailto:a@b.com" in content
    assert "#section" in content


def test_resolve_links_absolute_links_unchanged(tmp_path: Path):
    """Links already starting with / are left as-is."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "foo.md").write_text(
        "---\ntitle: Foo\n---\n[bar](/concepts/bar.md)\n", encoding="utf-8"
    )
    (tmp_path / "concepts" / "bar.md").write_text(
        "---\ntitle: Bar\n---\n# Bar\n", encoding="utf-8"
    )

    count = okfr.resolve_links(tmp_path)
    assert count == 0

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    assert "[bar](/concepts/bar.md)" in content


def test_resolve_links_unknown_slug_left_as_is(tmp_path: Path):
    """Unknown bare slugs are tolerated and left unchanged (OKF spec)."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "foo.md").write_text(
        "---\ntitle: Foo\n---\n[missing](nonexistent)\n", encoding="utf-8"
    )

    count = okfr.resolve_links(tmp_path)
    assert count == 0

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    assert "[missing](nonexistent)" in content


def test_resolve_links_mixed_links_in_one_file(tmp_path: Path):
    """A file with a mix of link types should only rewrite resolvable ones."""
    (tmp_path / "concepts").mkdir(parents=True)
    (tmp_path / "concepts" / "foo.md").write_text(
        "---\ntitle: Foo\n---\n"
        "See [bar](bar) and [ext](https://x.io) and [abs](/concepts/bar.md) "
        "and [rel](bar.md) and [unknown](zzz).\n",
        encoding="utf-8",
    )
    (tmp_path / "concepts" / "bar.md").write_text(
        "---\ntitle: Bar\n---\n# Bar\n", encoding="utf-8"
    )

    count = okfr.resolve_links(tmp_path)
    assert count == 1

    content = (tmp_path / "concepts" / "foo.md").read_text(encoding="utf-8")
    # Bare slug resolved
    assert "[bar](/concepts/bar.md)" in content
    # Relative .md resolved (same dir → /concepts/bar.md)
    assert "[rel](/concepts/bar.md)" in content
    # External left as-is
    assert "https://x.io" in content
    # Already-absolute left as-is
    assert "[abs](/concepts/bar.md)" in content
    # Unknown slug left as-is
    assert "[unknown](zzz)" in content


# ── pytest config ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
