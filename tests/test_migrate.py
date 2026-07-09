"""Tests for pipeline.migrate."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.migrate import (
    _convert_inline_citations,
    _infer_type_from_path,
    convert_wikilink_to_okf,
    extract_wikilinks,
    migrate_vault,
    rewrite_wikilinks,
)
from pipeline.okf_markdown import parse_frontmatter

# ── extract_wikilinks (moved from okf_markdown) ─────────────────────────


def test_extract_wikilinks_plain():
    body = "Link to [[foo]] and [[bar]]."
    links = extract_wikilinks(body)
    assert ("foo", None) in links
    assert ("bar", None) in links


def test_extract_wikilinks_alias():
    body = "See [[foo|display]] here."
    links = extract_wikilinks(body)
    assert links == [("foo", "display")]


# ── convert_wikilink_to_okf (moved from okf_markdown) ───────────────────


def test_convert_wikilink_plain():
    assert convert_wikilink_to_okf("foo") == "[foo](/concepts/foo.md)"


def test_convert_wikilink_alias():
    assert convert_wikilink_to_okf("foo", alias="Foo Bar") == \
        "[Foo Bar](/concepts/foo.md)"


def test_convert_wikilink_custom_directory():
    assert convert_wikilink_to_okf("foo", directory="notes") == \
        "[foo](/notes/foo.md)"


# ── rewrite_wikilinks (moved from okf_markdown) ─────────────────────────


def test_rewrite_wikilinks_plain():
    body = "See [[foo]] and [[bar]] in the text."
    out = rewrite_wikilinks(body)
    assert "[foo](/concepts/foo.md)" in out
    assert "[bar](/concepts/bar.md)" in out
    assert "[[" not in out


def test_rewrite_wikilinks_alias():
    body = "See [[foo|My Foo]] here."
    out = rewrite_wikilinks(body)
    assert out == "See [My Foo](/concepts/foo.md) here."

# ── Helpers ─────────────────────────────────────────────────────────────


def _write(path: Path, content: str) -> None:
    """Write ``content`` to ``path``, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_legacy_vault(vault_dir: Path) -> Path:
    """Create a mock legacy Obsidian vault under ``vault_dir``.

    The vault contains ``04-Wiki/`` with ``concepts``, ``entries``,
    ``sources`` and ``references`` subdirectories, each populated with
    legacy-style concept files that exhibit the issues the migration is
    expected to fix:

    * Missing ``type`` field in frontmatter.
    * ``[[wikilink]]`` / ``[[slug|alias]]`` syntax in the body.
    * Obsidian ``aliases`` key in frontmatter.
    * ``orphaned`` / ``orphaned_reason`` keys in frontmatter.
    * Inline ``^[citation]`` footnotes.
    * A root ``index.md`` and ``MOC.md`` that should be deleted.
    """
    wiki = vault_dir / "04-Wiki"
    wiki.mkdir(parents=True, exist_ok=True)

    # ── concepts/alpha.md ── missing type, wikilinks, aliases ──────────
    _write(
        wiki / "concepts" / "alpha.md",
        "---\n"
        "title: Alpha\n"
        "aliases:\n"
        "- Alpha Concept\n"
        "- A\n"
        "tags:\n"
        "- test\n"
        "---\n\n"
        "# Alpha\n\n"
        "See [[beta]] for more.\n"
        "Also [[gamma|Gamma]].\n",
    )

    # ── concepts/beta.md ── has type, has orphaned + inline citation ────
    _write(
        wiki / "concepts" / "beta.md",
        "---\n"
        "type: Concept\n"
        "title: Beta\n"
        "orphaned: true\n"
        "orphaned_reason: No inbound links\n"
        "---\n\n"
        "# Beta\n\n"
        "Reference to ^[Smith, 2020] here.\n"
        "Another ^[Jones, 2021] note.\n",
    )

    # ── entries/note1.md ── missing type, wikilink ─────────────────────
    _write(
        wiki / "entries" / "note1.md",
        "---\n"
        "title: Note One\n"
        "aliases:\n"
        "- N1\n"
        "---\n\n"
        "# Note One\n\n"
        "Link to [[alpha]].\n",
    )

    # ── sources/source1.md ── missing type ─────────────────────────────
    _write(
        wiki / "sources" / "source1.md",
        "---\n"
        "title: Source One\n"
        "---\n\n"
        "# Source One\n\n"
        "Body text.\n",
    )

    # ── references/ref1.md ── missing type ──────────────────────────────
    _write(
        wiki / "references" / "ref1.md",
        "---\n"
        "title: Reference One\n"
        "---\n\n"
        "# Reference One\n\n"
        "Body.\n",
    )

    # ── Legacy root index.md and MOC.md (should be deleted) ───────────
    _write(wiki / "index.md", "# Old Index\n\n- [Alpha](concepts/alpha.md)\n")
    _write(wiki / "MOC.md", "# Old MOC\n\n- [[alpha]]\n")

    return wiki


# ── _infer_type_from_path ──────────────────────────────────────────────


class TestInferTypeFromPath:
    """_infer_type_from_path maps directory names to OKFConceptType values."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("04-Wiki/concepts/foo.md", "Concept"),
            ("04-Wiki/entries/foo.md", "Entry"),
            ("04-Wiki/mocs/foo.md", "Map of Content"),
            ("04-Wiki/sources/foo.md", "Source"),
            ("04-Wiki/references/foo.md", "Reference"),
            ("04-Wiki/unknown/foo.md", "Concept"),  # unknown → default
        ],
    )
    def test_infer(self, path: str, expected: str):
        assert _infer_type_from_path(path) == expected


# ── _convert_inline_citations ──────────────────────────────────────────


class TestConvertInlineCitations:
    """_convert_inline_citations extracts ^[...] footnotes to a section."""

    def test_no_citations(self):
        body = "Just normal text with no citations.\n"
        new_body, citations = _convert_inline_citations(body)
        assert new_body == body
        assert citations == []

    def test_single_citation(self):
        body = "See ^[Smith, 2020] for details.\n"
        new_body, citations = _convert_inline_citations(body)
        assert citations == ["Smith, 2020"]
        assert "[^1]" in new_body
        assert "^[Smith, 2020]" not in new_body
        assert "# Citations" in new_body
        assert "[^1]: Smith, 2020" in new_body

    def test_multiple_citations(self):
        body = "First ^[A] then ^[B] then ^[C].\n"
        new_body, citations = _convert_inline_citations(body)
        assert citations == ["A", "B", "C"]
        assert new_body.count("[^") == 6  # 3 inline + 3 in definitions
        assert "[^1]: A" in new_body
        assert "[^2]: B" in new_body
        assert "[^3]: C" in new_body

    def test_citations_section_at_bottom(self):
        body = "Text ^[X].\n"
        new_body, _ = _convert_inline_citations(body)
        # The # Citations heading must come after the original text.
        assert new_body.index("# Citations") > new_body.index("Text")


# ── migrate_vault ───────────────────────────────────────────────────────


class TestMigrateVault:
    """Integration tests for migrate_vault on a mock legacy vault."""

    def test_migrate_adds_type_field_if_missing(self, tmp_path: Path):
        """Concept files without a ``type`` field get one inferred."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(alpha)
        assert fm.get("type") == "Concept"

        note1 = (wiki / "entries" / "note1.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(note1)
        assert fm.get("type") == "Entry"

        source1 = (wiki / "sources" / "source1.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(source1)
        assert fm.get("type") == "Source"

        ref1 = (wiki / "references" / "ref1.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(ref1)
        assert fm.get("type") == "Reference"

    def test_migrate_preserves_existing_type(self, tmp_path: Path):
        """A file that already has a ``type`` field keeps it unchanged."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        beta = (wiki / "concepts" / "beta.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(beta)
        assert fm.get("type") == "Concept"

    def test_migrate_converts_wikilinks_to_markdown_links(self, tmp_path: Path):
        """All ``[[wikilinks]]`` are rewritten to standard markdown links."""
        wiki = _make_legacy_vault(tmp_path)
        result = migrate_vault(tmp_path)

        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        assert "[[" not in alpha
        assert "[beta](/concepts/beta.md)" in alpha
        assert "[Gamma](/concepts/gamma.md)" in alpha

        note1 = (wiki / "entries" / "note1.md").read_text(encoding="utf-8")
        assert "[[" not in note1
        assert "[alpha](/entries/alpha.md)" in note1

        assert result["wikilinks_converted"] > 0

    def test_migrate_removes_aliases_from_frontmatter(self, tmp_path: Path):
        """The ``aliases`` key is stripped from frontmatter."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        alpha = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(alpha)
        assert "aliases" not in fm

        note1 = (wiki / "entries" / "note1.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(note1)
        assert "aliases" not in fm

    def test_migrate_removes_orphaned_keys(self, tmp_path: Path):
        """``orphaned`` and ``orphaned_reason`` are removed from frontmatter."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        beta = (wiki / "concepts" / "beta.md").read_text(encoding="utf-8")
        fm, _body = parse_frontmatter(beta)
        assert "orphaned" not in fm
        assert "orphaned_reason" not in fm

    def test_migrate_converts_inline_citations_to_section(self, tmp_path: Path):
        """``^[...]`` footnotes become a ``# Citations`` section."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        beta = (wiki / "concepts" / "beta.md").read_text(encoding="utf-8")
        assert "^[Smith, 2020]" not in beta
        assert "^[Jones, 2021]" not in beta
        assert "# Citations" in beta
        assert "[^1]: Smith, 2020" in beta
        assert "[^2]: Jones, 2021" in beta
        assert "[^1]" in beta
        assert "[^2]" in beta

    def test_migrate_deletes_old_index_and_moc(self, tmp_path: Path):
        """Root ``index.md`` and ``MOC.md`` are deleted."""
        wiki = _make_legacy_vault(tmp_path)
        assert (wiki / "index.md").exists()
        assert (wiki / "MOC.md").exists()

        result = migrate_vault(tmp_path)
        assert not (wiki / "MOC.md").exists()
        # The old index.md is deleted, then a new bundle index.md is created.
        assert (wiki / "index.md").exists()  # regenerated by generate_bundle_index
        assert result["files_deleted"] == 2

    def test_migrate_generates_per_directory_index(self, tmp_path: Path):
        """Each subdirectory with concept files gets an ``index.md``."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        for sub in ("concepts", "entries", "sources", "references"):
            idx = wiki / sub / "index.md"
            assert idx.exists(), f"{sub}/index.md was not generated"
            content = idx.read_text(encoding="utf-8")
            assert content.startswith(f"# {sub}")

    def test_migrate_generates_root_bundle_index(self, tmp_path: Path):
        """A bundle-root ``index.md`` with ``okf_version`` is created."""
        wiki = _make_legacy_vault(tmp_path)
        migrate_vault(tmp_path)

        root_index = wiki / "index.md"
        content = root_index.read_text(encoding="utf-8")
        assert "okf_version" in content
        assert "# Knowledge Bundle" in content

    def test_migrate_generates_log_with_migration_entry(self, tmp_path: Path):
        """``log.md`` is created containing a migration log entry."""
        wiki = _make_legacy_vault(tmp_path)
        result = migrate_vault(tmp_path)

        log_path = wiki / "log.md"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert content.startswith("# Change Log")
        assert "**migrated**" in content
        assert result["log_created"] is True

    def test_migrate_returns_correct_keys(self, tmp_path: Path):
        """The result dict has all required keys."""
        _make_legacy_vault(tmp_path)
        result = migrate_vault(tmp_path)
        expected_keys = {
            "files_migrated",
            "wikilinks_converted",
            "files_deleted",
            "indexes_generated",
            "log_created",
            "errors",
        }
        assert set(result.keys()) == expected_keys
        assert isinstance(result["errors"], list)

    def test_dry_run_does_not_write_files(self, tmp_path: Path):
        """In dry-run mode no files are modified or created."""
        wiki = _make_legacy_vault(tmp_path)

        # Snapshot original contents.
        alpha_before = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        old_index = (wiki / "index.md").read_text(encoding="utf-8")

        result = migrate_vault(tmp_path, dry_run=True)

        # Files should be unchanged.
        alpha_after = (wiki / "concepts" / "alpha.md").read_text(encoding="utf-8")
        assert alpha_after == alpha_before

        # Old index.md should still exist (not deleted in dry-run).
        assert (wiki / "index.md").exists()
        assert (wiki / "index.md").read_text(encoding="utf-8") == old_index
        assert (wiki / "MOC.md").exists()

        # Per-directory index.md should NOT have been created.
        assert not (wiki / "concepts" / "index.md").exists()
        assert not (wiki / "log.md").exists()

        # But the result should still report what would change.
        assert result["files_deleted"] == 2
        assert result["indexes_generated"] > 0
        assert result["log_created"] is True

    def test_migrate_no_errors_on_clean_vault(self, tmp_path: Path):
        """Migrating a valid legacy vault produces no errors."""
        _make_legacy_vault(tmp_path)
        result = migrate_vault(tmp_path)
        assert result["errors"] == []


# ── pytest entrypoint ───────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
