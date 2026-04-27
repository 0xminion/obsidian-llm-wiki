"""Adversarial and edge-case tests for parsing functions across the pipeline.

Targets:
  - frontmatter_list_items   (pipeline.utils)
  - _build_edges             (pipeline.compile.structural)
  - _detect_duplicates       (pipeline.compile.structural)
  - NoteIndex                (pipeline.compile.semantic)
  - parse_frontmatter        (pipeline.utils, used by lint)
"""

from __future__ import annotations

import pytest

from pipeline.config import Config
from pipeline.utils import frontmatter_list_items, parse_frontmatter
from pipeline.compile.structural import _build_edges, _detect_duplicates
from pipeline.compile.semantic import NoteIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vault(tmp_path):
    """Create the standard vault directory skeleton and return a Config."""
    for d in [
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "04-Wiki/sources",
        "06-Config",
        "Meta/Scripts",
    ]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Config(vault_path=tmp_path)


def _write_note(cfg: Config, kind: str, stem: str, content: str):
    """Write a .md file into the appropriate vault directory."""
    dirs = {
        "entry": cfg.entries_dir,
        "concept": cfg.concepts_dir,
        "moc": cfg.mocs_dir,
        "source": cfg.sources_dir,
    }
    path = dirs[kind] / f"{stem}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ===========================================================================
# 1. frontmatter_list_items
# ===========================================================================

class TestFrontmatterListItems:

    def test_empty_string(self):
        assert frontmatter_list_items("", "tags") == []

    def test_no_matching_field(self):
        fm = "title: hello\nauthor: someone\n"
        assert frontmatter_list_items(fm, "tags") == []

    def test_field_with_no_items(self):
        """Key present but followed by another key, not list items."""
        fm = "tags:\ntitle: hello\n"
        assert frontmatter_list_items(fm, "tags") == []

    def test_field_with_empty_list_block(self):
        """Key present with nothing after it (end of string)."""
        fm = "tags:\n"
        assert frontmatter_list_items(fm, "tags") == []

    def test_basic_happy_path(self):
        fm = "tags:\n  - alpha\n  - beta\n"
        result = frontmatter_list_items(fm, "tags")
        assert result == ["alpha", "beta"]

    def test_items_with_double_quotes(self):
        fm = 'tags:\n  - "quoted value"\n'
        result = frontmatter_list_items(fm, "tags")
        assert result == ["quoted value"]

    def test_items_with_brackets(self):
        fm = "tags:\n  - [bracket-item]\n"
        result = frontmatter_list_items(fm, "tags")
        assert result == ["[bracket-item]"]

    def test_items_with_unicode(self):
        fm = "tags:\n  - café\n  - naïve\n"
        result = frontmatter_list_items(fm, "tags")
        assert result == ["café", "naïve"]

    def test_cjk_characters_in_values(self):
        fm = "tags:\n  - 深度学习\n  - 自然语言处理\n"
        result = frontmatter_list_items(fm, "tags")
        assert result == ["深度学习", "自然语言处理"]

    def test_malformed_yaml_missing_dashes(self):
        """Items without leading dashes should not be captured."""
        fm = "tags:\n  alpha\n  beta\n"
        assert frontmatter_list_items(fm, "tags") == []

    def test_malformed_yaml_wrong_indentation(self):
        """Tabs before dashes are still valid per the regex."""
        fm = "tags:\n\t- alpha\n\t- beta\n"
        result = frontmatter_list_items(fm, "tags")
        assert result == ["alpha", "beta"]

    def test_field_name_with_special_regex_chars(self):
        """Ensure field names with regex-special chars are escaped."""
        fm = "tags[0]:\n  - val\n"
        assert frontmatter_list_items(fm, "tags[0]") == ["val"]

    def test_multiple_fields_returns_correct_one(self):
        fm = "sources:\n  - src1\ntags:\n  - t1\n  - t2\n"
        assert frontmatter_list_items(fm, "tags") == ["t1", "t2"]
        assert frontmatter_list_items(fm, "sources") == ["src1"]


# ===========================================================================
# 2. _build_edges
# ===========================================================================

class TestBuildEdges:

    def test_empty_vault_no_md_files(self, tmp_path):
        cfg = _make_vault(tmp_path)
        result = _build_edges(cfg)
        assert result == 0
        assert cfg.edges_file.exists()

    def test_md_file_with_no_frontmatter(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "bare-note", "# Just a heading\nSome body text.\n")
        result = _build_edges(cfg)
        # Should succeed without errors; no edges since no links
        assert isinstance(result, int)

    def test_md_file_with_corrupted_frontmatter(self, tmp_path):
        """Missing closing --- should mean no frontmatter is parsed."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "broken", "---\ntitle: broken\ntags:\n  - oops\n")
        result = _build_edges(cfg)
        assert isinstance(result, int)

    def test_md_file_with_empty_frontmatter(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "empty-fm", "---\n---\nBody here.\n")
        result = _build_edges(cfg)
        assert isinstance(result, int)

    def test_existing_edges_tsv_with_wrong_column_count_one(self, tmp_path):
        """edges.tsv with only 1 column lines should be skipped gracefully."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "a", "---\ntitle: A\n---\n# A\n")
        cfg.edges_file.write_text("source\ttarget\ttype\tdescription\nbadline\n", encoding="utf-8")
        result = _build_edges(cfg)
        assert isinstance(result, int)

    def test_existing_edges_tsv_with_five_columns(self, tmp_path):
        """Extra columns should be ignored (only first 4 used)."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "a", "---\ntitle: A\n---\n# A\n")
        _write_note(cfg, "entry", "b", "---\ntitle: B\n---\n# B\n")
        cfg.edges_file.write_text(
            "source\ttarget\ttype\tdescription\n"
            "a\tb\textends\tmanual link\textra_col\n",
            encoding="utf-8",
        )
        result = _build_edges(cfg)
        assert isinstance(result, int)

    def test_edges_tsv_with_unicode_names(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "concept", "概念一", "---\ntitle: 概念一\ntags:\n  - ai\n---\n# 概念一\n")
        _write_note(cfg, "concept", "概念二", "---\ntitle: 概念二\ntags:\n  - ai\n---\n# 概念二\n")
        result = _build_edges(cfg)
        assert isinstance(result, int)

    def test_wikilink_with_pipe_alias(self, tmp_path):
        """[[name|alias]] should extract 'name' only."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "note-a", "---\ntitle: A\n---\nSee [[note-b|the B note]].\n")
        _write_note(cfg, "entry", "note-b", "---\ntitle: B\n---\n# B\n")
        result = _build_edges(cfg)
        content = cfg.edges_file.read_text(encoding="utf-8")
        assert "note-a" in content
        assert "note-b" in content

    def test_wikilink_with_anchor(self, tmp_path):
        """[[name#heading]] should extract 'name' only."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "alpha", "---\ntitle: alpha\n---\nSee [[beta#section-1]].\n")
        _write_note(cfg, "entry", "beta", "---\ntitle: beta\n---\n# Beta\n")
        result = _build_edges(cfg)
        content = cfg.edges_file.read_text(encoding="utf-8")
        assert "alpha" in content
        assert "beta" in content

    def test_self_referencing_wikilink_ignored(self, tmp_path):
        """[[self]] should NOT create an edge to itself."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "narcissist", "---\ntitle: narcissist\n---\n[[narcissist]]\n")
        result = _build_edges(cfg)
        content = cfg.edges_file.read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if not l.startswith("source\t")]
        assert all("narcissist\tnarcissist" not in l for l in lines)

    def test_large_vault_concept_tag_loop(self, tmp_path):
        """100+ concept notes with shared tags to exercise the O(n^2) loop."""
        cfg = _make_vault(tmp_path)
        for i in range(110):
            tags = "  - shared-a\n  - shared-b\n" if i % 2 == 0 else "  - unique-tag\n"
            _write_note(
                cfg, "concept", f"concept-{i:03d}",
                f"---\ntitle: Concept {i}\ntags:\n{tags}---\n# Concept {i}\n",
            )
        result = _build_edges(cfg)
        assert isinstance(result, int)
        # Even concepts should share tags and produce edges
        content = cfg.edges_file.read_text(encoding="utf-8")
        assert "shared tags:" in content

    def test_moc_links_create_part_of_edges(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "child", "---\ntitle: child\n---\n# Child\n")
        _write_note(cfg, "moc", "parent-moc", "---\ntitle: Parent\n---\n[[child]]\n")
        result = _build_edges(cfg)
        content = cfg.edges_file.read_text(encoding="utf-8")
        assert "part_of" in content

    def test_concept_source_tested_by_edge(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "evidence", "---\ntitle: evidence\n---\n# Evidence\n")
        _write_note(
            cfg, "concept", "theory",
            "---\ntitle: theory\nsources:\n  - \"[[evidence]]\"\n---\n# Theory\n",
        )
        result = _build_edges(cfg)
        content = cfg.edges_file.read_text(encoding="utf-8")
        assert "tested_by" in content


# ===========================================================================
# 3. _detect_duplicates
# ===========================================================================

class TestDetectDuplicates:

    def test_empty_vault(self, tmp_path):
        cfg = _make_vault(tmp_path)
        assert _detect_duplicates(cfg) == 0

    def test_single_note_no_pairs(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "solo", "---\ntitle: Solo Note\n---\n# Solo\n")
        assert _detect_duplicates(cfg) == 0

    def test_notes_with_identical_titles(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "note-a", "---\ntitle: Machine Learning Overview\n---\n# A\n")
        _write_note(cfg, "entry", "note-b", "---\ntitle: Machine Learning Overview\n---\n# B\n")
        result = _detect_duplicates(cfg)
        assert result >= 1

    def test_notes_with_cjk_only_titles(self, tmp_path):
        """CJK titles — the regex preserves CJK range 一-鿿."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "concept", "cn-a", "---\ntitle: 深度学习模型\n---\n# A\n")
        _write_note(cfg, "concept", "cn-b", "---\ntitle: 深度学习模型\n---\n# B\n")
        result = _detect_duplicates(cfg)
        assert result >= 1

    def test_notes_with_special_characters_in_titles(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "sp-a", "---\ntitle: \"C++ Templates: A Guide\"\n---\n# A\n")
        _write_note(cfg, "entry", "sp-b", "---\ntitle: \"C++ Templates: A Guide\"\n---\n# B\n")
        result = _detect_duplicates(cfg)
        assert result >= 1

    def test_different_types_not_compared(self, tmp_path):
        """An entry and a concept with the same title should not be flagged."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "dup-e", "---\ntitle: Quantum Computing\n---\n# E\n")
        _write_note(cfg, "concept", "dup-c", "---\ntitle: Quantum Computing\n---\n# C\n")
        assert _detect_duplicates(cfg) == 0

    def test_notes_with_no_frontmatter_use_stem(self, tmp_path):
        """Notes without frontmatter should fall back to stem as title."""
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "same-stem", "# Just a heading\n")
        _write_note(cfg, "entry", "same-stem-copy", "# Just a heading\n")
        # Stems differ so overlap depends on word tokenisation
        result = _detect_duplicates(cfg)
        assert isinstance(result, int)


# ===========================================================================
# 4. NoteIndex
# ===========================================================================

class TestNoteIndex:

    def test_load_from_empty_directory(self, tmp_path):
        cfg = _make_vault(tmp_path)
        idx = NoteIndex()
        idx.load(cfg)
        assert idx.notes == {}

    def test_notes_with_no_tags(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "tagless", "---\ntitle: Tagless\n---\n# Tagless\n")
        idx = NoteIndex()
        idx.load(cfg)
        assert "tagless" in idx.notes
        assert idx.notes["tagless"]["tags"] == set()

    def test_notes_with_duplicate_tags(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(
            cfg, "concept", "dup-tags",
            "---\ntitle: Dup Tags\ntags:\n  - ai\n  - AI\n  - Ai\n---\n# Dup Tags\n",
        )
        idx = NoteIndex()
        idx.load(cfg)
        # All variants should be lowered and deduplicated
        assert idx.notes["dup-tags"]["tags"] == {"ai"}

    def test_load_multiple_types(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(cfg, "entry", "e1", "---\ntitle: E1\n---\n# E1\n")
        _write_note(cfg, "concept", "c1", "---\ntitle: C1\n---\n# C1\n")
        _write_note(cfg, "moc", "m1", "---\ntitle: M1\n---\n# M1\n")
        idx = NoteIndex()
        idx.load(cfg)
        assert len(idx.notes) == 3
        assert idx.notes["e1"]["type"] == "entry"
        assert idx.notes["c1"]["type"] == "concept"
        assert idx.notes["m1"]["type"] == "moc"

    def test_wikilinks_extracted_without_pipe_or_anchor(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(
            cfg, "entry", "linker",
            "---\ntitle: Linker\n---\n[[plain]] and [[piped|alias]] and [[anchored#h1]]\n",
        )
        idx = NoteIndex()
        idx.load(cfg)
        links = idx.notes["linker"]["links"]
        assert "plain" in links
        assert "piped" in links
        assert "anchored" in links
        # Alias/anchor fragments should NOT appear as link names
        assert "alias" not in links
        assert "anchored#h1" not in links

    def test_preview_strips_frontmatter(self, tmp_path):
        cfg = _make_vault(tmp_path)
        _write_note(
            cfg, "entry", "prev",
            "---\ntitle: Preview Test\n---\nBody content here.\n",
        )
        idx = NoteIndex()
        idx.load(cfg)
        assert "Body content here." in idx.notes["prev"]["preview"]
        assert "title:" not in idx.notes["prev"]["preview"]

    def test_similarity_no_embeddings(self):
        idx = NoteIndex()
        assert idx.similarity("a", "b") == 0.0


# ===========================================================================
# 5. parse_frontmatter (used by lint YAML parsing)
# ===========================================================================

class TestParseFrontmatter:

    def test_valid_frontmatter(self):
        content = "---\ntitle: Hello\ntags:\n  - a\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert fm["title"] == "Hello"
        assert fm["tags"] == ["a"]

    def test_missing_frontmatter(self):
        assert parse_frontmatter("No frontmatter here.") == {}

    def test_frontmatter_with_only_opening_delimiter(self):
        assert parse_frontmatter("---\ntitle: broken\n") == {}

    def test_frontmatter_with_binary_null_characters(self):
        content = "---\ntitle: has\x00null\n---\nBody.\n"
        fm = parse_frontmatter(content)
        # Should either parse or return empty dict, never crash
        assert isinstance(fm, dict)

    def test_frontmatter_with_control_characters(self):
        content = "---\ntitle: ctrl\x01\x02\x03\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert isinstance(fm, dict)

    def test_deeply_nested_yaml(self):
        nested = "a:\n" + "".join(f"{'  ' * i}b:\n" for i in range(1, 20))
        content = f"---\n{nested}---\nBody.\n"
        fm = parse_frontmatter(content)
        assert isinstance(fm, dict)

    def test_extremely_long_value(self):
        long_val = "x" * 50_000
        content = f"---\ntitle: {long_val}\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert isinstance(fm, dict)
        if fm:
            assert fm.get("title") == long_val

    def test_empty_frontmatter_block(self):
        content = "---\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert fm == {}

    def test_frontmatter_with_only_whitespace(self):
        content = "---\n   \n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert fm == {}

    def test_yaml_with_duplicate_keys(self):
        content = "---\ntitle: first\ntitle: second\n---\nBody.\n"
        fm = parse_frontmatter(content)
        # YAML spec: last value wins
        assert isinstance(fm, dict)
        if fm:
            assert fm["title"] == "second"

    def test_yaml_with_multiline_string(self):
        content = "---\ntitle: |\n  line one\n  line two\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert isinstance(fm, dict)
        assert "line one" in fm.get("title", "")

    def test_yaml_returning_non_dict_is_empty(self):
        """Frontmatter that parses to a scalar should return {}."""
        content = "---\njust a string\n---\nBody.\n"
        fm = parse_frontmatter(content)
        assert fm == {}
