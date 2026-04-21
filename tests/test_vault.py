"""Tests for pipeline/vault.py — vault file operations."""

import pytest
from pathlib import Path

from pipeline.config import Config
from pipeline.models import Edge, EdgeType, ExtractedSource, Language, Plan, SourceType, Template
from pipeline.vault import (
    title_to_filename,
    check_collision,
    resolve_collision,
    write_source,
    write_entry,
    write_concept,
    update_moc,
    write_edge,
    read_edges,
    register_url,
    url_exists,
    reindex,
    archive_inbox,
    _normalize_url,
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Create a Config pointing to a tmp vault."""
    return Config(vault_path=tmp_path)


# ─── title_to_filename ────────────────────────────────────────────────────────

class TestTitleToFilename:
    def test_english_lowercased(self):
        assert title_to_filename("Hello World") == "hello-world"

    def test_english_kebab(self):
        assert title_to_filename("My Great Article!") == "my-great-article"

    def test_english_colon_replaced(self):
        assert title_to_filename("Python: A Guide") == "python-a-guide"

    def test_english_apostrophe_stripped(self):
        assert title_to_filename("It's a Test") == "its-a-test"

    def test_chinese_kept(self):
        result = title_to_filename("市场预测模型")
        assert "市场预测模型" in result

    def test_chinese_colon_to_hyphen(self):
        result = title_to_filename("深度学习：入门指南")
        assert "-" in result
        assert "深度学习" in result

    def test_chinese_punctuation_stripped(self):
        result = title_to_filename("「机器学习」基础")
        assert "「" not in result
        assert "」" not in result

    def test_truncation(self):
        long_title = "A" * 200
        result = title_to_filename(long_title)
        assert len(result) <= 120

    def test_empty(self):
        result = title_to_filename("")
        assert result == ""

    def test_no_url_slug(self):
        # Should never produce a URL slug
        result = title_to_filename("Simple Title")
        assert "http" not in result
        assert "/" not in result


# ─── Collision Detection ──────────────────────────────────────────────────────

class TestCollision:
    def test_check_collision_safe(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        assert check_collision(cfg.entries_dir, "new-note") is True

    def test_check_collision_exists(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        (cfg.entries_dir / "existing.md").write_text("test")
        assert check_collision(cfg.entries_dir, "existing") is False

    def test_resolve_collision_no_conflict(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        result = resolve_collision(cfg.entries_dir, "unique-note")
        assert result == "unique-note"

    def test_resolve_collision_with_conflict(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        (cfg.entries_dir / "note.md").write_text("1")
        (cfg.entries_dir / "note-1.md").write_text("2")
        result = resolve_collision(cfg.entries_dir, "note")
        assert result == "note-2"


# ─── write_source ─────────────────────────────────────────────────────────────

class TestWriteSource:
    def test_creates_file(self, cfg: Config):
        source = ExtractedSource(
            url="https://example.com/article",
            title="Test Article",
            content="Some content here.",
            type=SourceType.WEB,
            author="Author Name",
        )
        path = write_source(cfg, source)
        assert path.exists()
        assert path.suffix == ".md"
        assert path.parent == cfg.sources_dir

    def test_frontmatter_fields(self, cfg: Config):
        source = ExtractedSource(
            url="https://example.com/article",
            title="Test Article",
            content="Body text.",
            type=SourceType.WEB,
            author="Test Author",
        )
        path = write_source(cfg, source)
        text = path.read_text()
        assert "title: Test Article" in text
        assert "source_url: https://example.com/article" in text
        assert "source_type: web" in text
        assert "author: Test Author" in text
        assert "status: raw" in text

    def test_no_collision_overwrite(self, cfg: Config):
        source = ExtractedSource(
            url="https://example.com/a",
            title="Same Title",
            content="First",
            type=SourceType.WEB,
        )
        p1 = write_source(cfg, source)
        source2 = ExtractedSource(
            url="https://example.com/b",
            title="Same Title",
            content="Second",
            type=SourceType.WEB,
        )
        p2 = write_source(cfg, source2)
        assert p1 != p2
        assert p1.exists()
        assert p2.exists()

    def test_chinese_title(self, cfg: Config):
        source = ExtractedSource(
            url="https://example.com/zh",
            title="深度学习入门",
            content="内容",
            type=SourceType.WEB,
        )
        path = write_source(cfg, source)
        assert path.exists()
        assert "深度学习" in path.stem


# ─── write_entry ──────────────────────────────────────────────────────────────

class TestWriteEntry:
    def test_creates_entry(self, cfg: Config):
        plan = Plan(hash="abc123", title="My Entry", tags=["test"])
        content = "## Summary\n\nThis is the summary.\n"
        path = write_entry(cfg, plan, content)
        assert path.exists()
        text = path.read_text()
        assert "title: My Entry" in text
        assert "status: draft" in text

    def test_quoted_wikilink_in_yaml(self, cfg: Config):
        plan = Plan(hash="abc123", title="Test Entry", tags=[])
        content = "Body"
        path = write_entry(cfg, plan, content)
        text = path.read_text()
        # source wikilink should be quoted in YAML
        assert '"[[' in text

    def test_template_in_frontmatter(self, cfg: Config):
        plan = Plan(
            hash="abc123",
            title="Tech Entry",
            template=Template.TECHNICAL,
            tags=["python"],
        )
        content = "## Summary\n\nTech summary.\n"
        path = write_entry(cfg, plan, content)
        text = path.read_text()
        assert "template: technical" in text

    def test_tags_in_frontmatter(self, cfg: Config):
        plan = Plan(
            hash="abc123",
            title="Tagged Entry",
            tags=["forecasting", "prediction-markets"],
        )
        content = "Body"
        path = write_entry(cfg, plan, content)
        text = path.read_text()
        assert "- forecasting" in text
        assert "- prediction-markets" in text


# ─── write_concept ────────────────────────────────────────────────────────────

class TestWriteConcept:
    def test_creates_concept(self, cfg: Config):
        path = write_concept(cfg, "Prediction Markets", "Markets for predictions.", ["source-1"])
        assert path.exists()
        text = path.read_text()
        assert "title: Prediction Markets" in text
        assert "type: concept" in text
        assert "- source-1" in text

    def test_no_h1_if_content_has_heading(self, cfg: Config):
        content = "# Prediction Markets\n\nDetailed explanation."
        path = write_concept(cfg, "Prediction Markets", content, [])
        text = path.read_text()
        # Should not duplicate the H1
        assert text.count("# Prediction Markets") == 1

    def test_adds_h1_if_missing(self, cfg: Config):
        content = "Just body text, no heading."
        path = write_concept(cfg, "My Concept", content, [])
        text = path.read_text()
        assert "# My Concept" in text


# ─── update_moc ───────────────────────────────────────────────────────────────

class TestUpdateMoc:
    def test_creates_new_moc(self, cfg: Config):
        update_moc(cfg, "Prediction Markets", "entry-1", "An entry about PM")
        moc_path = cfg.mocs_dir / "prediction-markets.md"
        assert moc_path.exists()
        text = moc_path.read_text()
        assert "[[entry-1]]" in text
        assert "## Overview / 概述" in text

    def test_appends_to_existing_moc(self, cfg: Config):
        update_moc(cfg, "PM", "entry-1", "First entry")
        update_moc(cfg, "PM", "entry-2", "Second entry")
        moc_path = cfg.mocs_dir / "pm.md"
        text = moc_path.read_text()
        assert "[[entry-1]]" in text
        assert "[[entry-2]]" in text

    def test_no_duplicate_entries(self, cfg: Config):
        update_moc(cfg, "PM", "entry-1", "First entry")
        update_moc(cfg, "PM", "entry-1", "First entry")
        moc_path = cfg.mocs_dir / "pm.md"
        text = moc_path.read_text()
        assert text.count("[[entry-1]]") == 1


# ─── Edge Read/Write ──────────────────────────────────────────────────────────

class TestEdges:
    def test_roundtrip(self, cfg: Config):
        edge = Edge(
            source="Note A",
            target="Note B",
            type=EdgeType.CONTRADICTS,
            description="They disagree on X",
        )
        write_edge(cfg, edge)
        edges = read_edges(cfg)
        assert len(edges) == 1
        assert edges[0].source == "Note A"
        assert edges[0].target == "Note B"
        assert edges[0].type == EdgeType.CONTRADICTS
        assert edges[0].description == "They disagree on X"

    def test_no_duplicates(self, cfg: Config):
        edge = Edge(source="A", target="B", type=EdgeType.EXTENDS)
        write_edge(cfg, edge)
        write_edge(cfg, edge)
        edges = read_edges(cfg)
        assert len(edges) == 1

    def test_multiple_edges(self, cfg: Config):
        write_edge(cfg, Edge(source="A", target="B", type=EdgeType.EXTENDS))
        write_edge(cfg, Edge(source="A", target="C", type=EdgeType.SUPPORTS))
        write_edge(cfg, Edge(source="B", target="C", type=EdgeType.RELATES_TO))
        edges = read_edges(cfg)
        assert len(edges) == 3

    def test_empty_file(self, cfg: Config):
        edges = read_edges(cfg)
        assert edges == []

    def test_header_skipped(self, cfg: Config):
        write_edge(cfg, Edge(source="X", target="Y", type=EdgeType.DEPENDS_ON))
        # Header line should not be parsed as edge
        edges = read_edges(cfg)
        assert all(e.source != "source" for e in edges)


# ─── URL Deduplication ────────────────────────────────────────────────────────

class TestUrlDedup:
    def test_register_and_exists(self, cfg: Config):
        assert url_exists(cfg, "https://example.com/article") is False
        register_url(cfg, "https://example.com/article", "my-entry")
        assert url_exists(cfg, "https://example.com/article") is True

    def test_normalize_strips_protocol(self, cfg: Config):
        register_url(cfg, "https://example.com/page", "entry")
        assert url_exists(cfg, "http://example.com/page") is True

    def test_normalize_strips_trailing_slash(self, cfg: Config):
        register_url(cfg, "https://example.com/page/", "entry")
        assert url_exists(cfg, "https://example.com/page") is True

    def test_no_duplicate_registration(self, cfg: Config):
        register_url(cfg, "https://example.com/a", "entry-1")
        register_url(cfg, "https://example.com/a", "entry-2")
        text = cfg.url_index.read_text()
        assert text.count("example.com/a") == 1

    def test_case_insensitive(self, cfg: Config):
        register_url(cfg, "https://Example.COM/Page", "entry")
        assert url_exists(cfg, "https://example.com/page") is True


# ─── Reindex ──────────────────────────────────────────────────────────────────

class TestReindex:
    def test_empty_vault(self, cfg: Config):
        content = reindex(cfg)
        assert "# Wiki Index" in content
        assert "0 entries, 0 concepts, 0 MoCs" in content

    def test_indexes_entries(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        (cfg.entries_dir / "test-entry.md").write_text(
            "---\ntitle: Test Entry\n---\n\n# Test Entry\n\n## Summary\n\nA test summary.\n"
        )
        content = reindex(cfg)
        assert "[[test-entry]]" in content
        assert "A test summary." in content
        assert "1 entries" in content

    def test_indexes_concepts(self, cfg: Config):
        cfg.concepts_dir.mkdir(parents=True)
        (cfg.concepts_dir / "my-concept.md").write_text(
            "---\ntitle: My Concept\ntype: concept\n---\n\n# My Concept\n\nConcept description here.\n"
        )
        content = reindex(cfg)
        assert "[[my-concept]]" in content
        assert "1 concepts" in content

    def test_indexes_mocs(self, cfg: Config):
        cfg.mocs_dir.mkdir(parents=True)
        (cfg.mocs_dir / "my-moc.md").write_text(
            "---\ntitle: My MoC\n---\n\n# My MoC\n\n## Overview / 概述\n\nOverview text.\n"
        )
        content = reindex(cfg)
        assert "[[my-moc]]" in content
        assert "1 MoCs" in content

    def test_writes_to_disk(self, cfg: Config):
        reindex(cfg)
        assert cfg.wiki_index.exists()
        text = cfg.wiki_index.read_text()
        assert "# Wiki Index" in text

    def test_sections_order(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        cfg.concepts_dir.mkdir(parents=True)
        (cfg.entries_dir / "e.md").write_text("---\ntitle: E\n---\n# E\n## Summary\nSum.\n")
        (cfg.concepts_dir / "c.md").write_text("---\ntitle: C\ntype: concept\n---\n# C\nBody.\n")
        content = reindex(cfg)
        entries_pos = content.index("## Entries")
        concepts_pos = content.index("## Concepts")
        assert entries_pos < concepts_pos

    def test_chinese_summary(self, cfg: Config):
        cfg.entries_dir.mkdir(parents=True)
        (cfg.entries_dir / "zh-entry.md").write_text(
            "---\ntitle: 中文条目\n---\n\n# 中文条目\n\n## 摘要\n\n这是一个测试摘要。\n"
        )
        content = reindex(cfg)
        assert "这是一个测试摘要" in content


# ─── archive_inbox ────────────────────────────────────────────────────────────

class TestArchiveInbox:
    def test_archives_matching_files(self, cfg: Config):
        import hashlib
        cfg.inbox_dir.mkdir(parents=True)
        url = "https://example.com/article"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        (cfg.inbox_dir / "article.url").write_text(f"[InternetShortcut]\nURL={url}\n")

        count = archive_inbox(cfg, {url_hash})
        assert count == 1
        assert not (cfg.inbox_dir / "article.url").exists()
        assert (cfg.archive_dir / "article.url").exists()

    def test_skips_non_matching(self, cfg: Config):
        cfg.inbox_dir.mkdir(parents=True)
        (cfg.inbox_dir / "article.url").write_text("[InternetShortcut]\nURL=https://example.com/a\n")

        count = archive_inbox(cfg, {"nonexistent-hash"})
        assert count == 0
        assert (cfg.inbox_dir / "article.url").exists()

    def test_empty_inbox(self, cfg: Config):
        count = archive_inbox(cfg, {"any-hash"})
        assert count == 0

    def test_multiple_files(self, cfg: Config):
        import hashlib
        cfg.inbox_dir.mkdir(parents=True)

        urls = ["https://example.com/a", "https://example.com/b"]
        hashes = set()
        for i, url in enumerate(urls):
            (cfg.inbox_dir / f"file{i}.url").write_text(f"[InternetShortcut]\nURL={url}\n")
            hashes.add(hashlib.md5(url.encode()).hexdigest()[:12])

        count = archive_inbox(cfg, hashes)
        assert count == 2

    def test_skips_non_url_files(self, cfg: Config):
        cfg.inbox_dir.mkdir(parents=True)
        (cfg.inbox_dir / "notes.txt").write_text("not a url file")

        import hashlib
        url = "https://example.com/article"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        (cfg.inbox_dir / "good.url").write_text(f"[InternetShortcut]\nURL={url}\n")

        count = archive_inbox(cfg, {url_hash})
        assert count == 1
        assert (cfg.inbox_dir / "notes.txt").exists()  # non-.url untouched


# ─── Edge Cache Tests ────────────────────────────────────────────────────────

from pipeline.vault import clear_edge_cache

class TestEdgeCache:
    def test_duplicate_edge_skipped(self, cfg: Config):
        clear_edge_cache()
        edge = Edge(source="X", target="Y", type=EdgeType.SUPPORTS, description="first")
        write_edge(cfg, edge)
        write_edge(cfg, edge)  # same edge again
        edges = read_edges(cfg)
        assert len(edges) == 1

    def test_clear_cache_reloads(self, cfg: Config):
        clear_edge_cache()
        edge = Edge(source="A", target="B", type=EdgeType.SUPPORTS, description="")
        write_edge(cfg, edge)

        # Manually append a duplicate directly to the file (bypassing cache)
        with cfg.edges_file.open("a") as f:
            f.write("A\tB\tsupports\tdupe\n")

        # Cache still thinks (A,B,supports) exists — would skip
        edge2 = Edge(source="A", target="B", type=EdgeType.SUPPORTS, description="new")
        write_edge(cfg, edge2)  # skipped because cache says duplicate

        # After clear, re-reads from file
        clear_edge_cache()
        edge3 = Edge(source="C", target="D", type=EdgeType.EXTENDS, description="")
        write_edge(cfg, edge3)

        edges = read_edges(cfg)
        assert len(edges) == 3  # original + manual dupe + new edge

    def test_different_types_not_duplicate(self, cfg: Config):
        clear_edge_cache()
        e1 = Edge(source="A", target="B", type=EdgeType.SUPPORTS, description="")
        e2 = Edge(source="A", target="B", type=EdgeType.EXTENDS, description="")
        write_edge(cfg, e1)
        write_edge(cfg, e2)
        edges = read_edges(cfg)
        assert len(edges) == 2
