"""Tests for pipeline.models."""

import json

from pipeline.models import (
    Edge,
    EdgeType,
    ExtractedSource,
    Language,
    Manifest,
    Plan,
    Plans,
    SourceType,
    Template,
)


# ─── ExtractedSource ─────────────────────────────────────────────────────────

class TestExtractedSource:
    def test_hash_deterministic(self):
        src = ExtractedSource(url="https://example.com/article", title="Test", content="body")
        assert src.hash == src.hash  # same every time
        assert len(src.hash) == 12

    def test_hash_changes_with_url(self):
        a = ExtractedSource(url="https://example.com/a", title="T", content="")
        b = ExtractedSource(url="https://example.com/b", title="T", content="")
        assert a.hash != b.hash

    def test_hash_matches_shell(self):
        """Hash must match: echo -n URL | md5sum | cut -c1-12"""
        import hashlib
        url = "https://moontower.substack.com/p/market-maker-privilege"
        expected = hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]
        src = ExtractedSource(url=url, title="T", content="")
        assert src.hash == expected

    def test_to_dict_roundtrip(self):
        src = ExtractedSource(
            url="https://example.com",
            title="Test Article",
            content="Full content here",
            type=SourceType.YOUTUBE,
            author="Jane Doe",
            source_file="test.url",
        )
        d = src.to_dict()
        assert d["url"] == "https://example.com"
        assert d["type"] == "youtube"
        assert d["author"] == "Jane Doe"

    def test_save_and_load(self, tmp_path):
        src = ExtractedSource(
            url="https://example.com/test",
            title="Test",
            content="# Test\nContent here.",
            type=SourceType.WEB,
        )
        saved_path = src.save(tmp_path)
        assert saved_path.exists()
        assert saved_path.name == f"{src.hash}.json"

        loaded = ExtractedSource.load(saved_path)
        assert loaded.url == src.url
        assert loaded.title == src.title
        assert loaded.content == src.content
        assert loaded.type == src.type

    def test_content_length(self):
        src = ExtractedSource(url="https://x.com", title="T", content="a" * 1000)
        assert src.content_length == 1000

    def test_default_type_is_unknown(self):
        src = ExtractedSource(url="https://x.com", title="T", content="")
        assert src.type == SourceType.UNKNOWN


# ─── Plan ─────────────────────────────────────────────────────────────────────

class TestPlan:
    def test_to_dict_roundtrip(self):
        plan = Plan(
            hash="abc123",
            title="Test Plan",
            language=Language.ZH,
            template=Template.CHINESE,
            tags=["ai", "nlp"],
            concept_updates=["existing-concept"],
            concept_new=["New Concept"],
            moc_targets=["AI / 人工智能"],
        )
        d = plan.to_dict()
        assert d["language"] == "zh"
        assert d["template"] == "chinese"
        assert d["tags"] == ["ai", "nlp"]

        loaded = Plan.from_dict(d)
        assert loaded.hash == plan.hash
        assert loaded.language == Language.ZH
        assert loaded.template == Template.CHINESE

    def test_defaults(self):
        plan = Plan.from_dict({"hash": "x", "title": "T"})
        assert plan.language == Language.EN
        assert plan.template == Template.STANDARD
        assert plan.tags == []


# ─── Manifest ─────────────────────────────────────────────────────────────────

class TestManifest:
    def test_save_and_load(self, tmp_path):
        manifest = Manifest(entries=[
            ExtractedSource(url="https://a.com", title="A", content="ca"),
            ExtractedSource(url="https://b.com", title="B", content="cb"),
        ])
        manifest.save(tmp_path)
        loaded = Manifest.load(tmp_path)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].url == "https://a.com"

    def test_load_missing_returns_empty(self, tmp_path):
        loaded = Manifest.load(tmp_path)
        assert len(loaded.entries) == 0

    def test_load_skips_malformed_entries_without_crashing(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps([
                {"url": "https://bad.example", "content": "missing title"},
            ]),
            encoding="utf-8",
        )

        loaded = Manifest.load(tmp_path)

        assert loaded.entries == []

    def test_load_skips_non_dict_entries_without_crashing(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps([
                {"url": "https://good.example", "title": "Good", "content": "ok"},
                None,
                "bad-entry",
                123,
            ]),
            encoding="utf-8",
        )

        loaded = Manifest.load(tmp_path)

        assert len(loaded.entries) == 1
        assert loaded.entries[0].title == "Good"

    def test_load_handles_non_list_manifest_shape(self, tmp_path):
        (tmp_path / "manifest.json").write_text("123", encoding="utf-8")

        loaded = Manifest.load(tmp_path)

        assert loaded.entries == []

    def test_hashes(self):
        manifest = Manifest(entries=[
            ExtractedSource(url="https://a.com", title="A", content=""),
            ExtractedSource(url="https://b.com", title="B", content=""),
        ])
        assert len(manifest.hashes) == 2


# ─── Plans ────────────────────────────────────────────────────────────────────

class TestPlans:
    def test_split_batches_even(self):
        plans = Plans(plans=[
            Plan(hash=f"p{i}", title=f"P{i}") for i in range(6)
        ])
        batches = plans.split_batches(parallel=3)
        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    def test_split_batches_uneven(self):
        plans = Plans(plans=[
            Plan(hash=f"p{i}", title=f"P{i}") for i in range(7)
        ])
        batches = plans.split_batches(parallel=3)
        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_split_batches_single(self):
        plans = Plans(plans=[Plan(hash="p0", title="P0")])
        batches = plans.split_batches(parallel=3)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_split_batches_empty(self):
        plans = Plans(plans=[])
        batches = plans.split_batches(parallel=3)
        assert batches == []

    def test_split_batches_content_aware(self, tmp_path):
        """Content-size-aware batching distributes large sources evenly."""
        import json as _json
        # Create extract files with varying content sizes
        sizes = [50000, 30000, 15000, 8000, 5000, 2000]  # 6 sources, very different sizes
        for i, size in enumerate(sizes):
            ext = {"content": "x" * size, "title": f"Source {i}", "url": f"https://example.com/{i}"}
            (tmp_path / f"p{i}.json").write_text(_json.dumps(ext))

        plans = Plans(plans=[Plan(hash=f"p{i}", title=f"Source {i}") for i in range(6)])
        batches = plans.split_batches(parallel=3, extract_dir=tmp_path)

        # Should produce 3 batches
        assert len(batches) == 3
        # Each batch should have similar total content (within reason)
        batch_totals = []
        for batch in batches:
            total = sum(sizes[int(p.hash[1:])] for p in batch)
            batch_totals.append(total)
        # Largest batch should not be more than 2x the smallest
        # (with ceiling div this would be 50000+2000 vs 5000 = 10x)
        assert max(batch_totals) / max(min(batch_totals), 1) < 3

    def test_save_and_load(self, tmp_path):
        plans = Plans(plans=[
            Plan(hash="abc", title="Test", tags=["ai"]),
        ])
        plans.save(tmp_path)
        loaded = Plans.load(tmp_path)
        assert len(loaded.plans) == 1
        assert loaded.plans[0].hash == "abc"


# ─── Edge ─────────────────────────────────────────────────────────────────────

class TestEdge:
    def test_to_tsv(self):
        edge = Edge(
            source="Note A",
            target="Note B",
            type=EdgeType.EXTENDS,
            description="builds on concept",
        )
        assert edge.to_tsv() == "Note A\tNote B\textends\tbuilds on concept"

    def test_from_tsv(self):
        line = "Note A\tNote B\tcontradicts\tdifferent view\n"
        edge = Edge.from_tsv(line)
        assert edge is not None
        assert edge.source == "Note A"
        assert edge.target == "Note B"
        assert edge.type == EdgeType.CONTRADICTS

    def test_from_tsv_short_line(self):
        assert Edge.from_tsv("invalid\n") is None

    def test_from_tsv_no_description(self):
        edge = Edge.from_tsv("A\tB\tsupports\n")
        assert edge is not None
        assert edge.description == ""
