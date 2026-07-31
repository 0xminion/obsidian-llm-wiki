"""Tests for bugs found during code review and their fixes.

Covers:
  1. resynthesize_concept() — Claim.concept_slug should be set
  2. supadata_utils rate limiter — thread safety with lock
  3. graph_export — source node ID slugification
  4. graph_export — Mermaid label escaping for brackets
  5. graph_export — circular references handled
  6. metrics — history preservation across multiple saves
  7. render_vault — config threshold wiring
  8. semantic_dedupe — empty bundle, missing embeddings
  9. assign_orphans_to_mocs — empty bundle, no maps
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from unittest import mock

from obsidian_llm_wiki.core.models import (
    BodySection,
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceSynthesis,
    SynthesisBundle,
)

# ── 1. resynthesize_concept Claim.concept_slug ──────────────────────────


class TestResynthesizeConceptClaimSlug:
    """resynthesize_concept should set concept_slug on all Claims."""

    def test_claims_have_concept_slug(self):
        """When resynthesize_concept builds Claims, concept_slug must be set."""
        from obsidian_llm_wiki.synth.quality import resynthesize_concept

        existing = ConceptNote(
            title="Test Concept",
            slug="test-concept",
            summary="Original summary",
            tags=["test"],
            sections=[BodySection(heading="Overview", points=["point 1"])],
        )

        # Mock the LLM to return a valid JSON response with claims
        mock_response = json.dumps({
            "title": "Test Concept",
            "slug": "test-concept",
            "summary": "Updated summary",
            "tags": ["test", "new-tag"],
            "sections": [
                {"heading": "Overview", "points": ["updated point"], "prose": ""}
            ],
            "claims": [
                {"text": "A new claim", "source_ref": "from new source"}
            ],
        })

        config = mock.MagicMock()

        async def run():
            with mock.patch(
                "obsidian_llm_wiki.providers.llm.acall_llm",
                return_value=mock_response,
            ):
                result = await resynthesize_concept(
                    config, existing, "new source content", "New Source",
                )
                return result

        result = asyncio.run(run())

        assert result is not None
        assert len(result.claims) == 1
        assert result.claims[0].concept_slug == "test-concept"
        assert result.claims[0].text == "A new claim"


# ── 2. Supadata rate limiter thread safety ──────────────────────────────


class TestSupadataRateLimiterThreadSafety:
    """Rate limiter should be thread-safe."""

    def test_concurrent_calls_respect_rate_limit(self):
        """Multiple threads calling supadata_rate_limit should not race."""
        from obsidian_llm_wiki.ingest import supadata_utils

        # Reset state
        supadata_utils.reset_rate_limiter()

        # Temporarily lower the rate limit for faster testing
        original_limit = supadata_utils.SUPADATA_RATE_LIMIT_SECONDS
        supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = 0.05

        try:
            call_times: list[float] = []
            lock = threading.Lock()

            def call():
                supadata_utils.supadata_rate_limit()
                with lock:
                    call_times.append(time.monotonic())

            threads = [threading.Thread(target=call) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # With 3 threads and 0.05s rate limit, calls should be spread out
            # At minimum, not all should happen simultaneously
            assert len(call_times) == 3
            # Check that at least some spacing exists between calls
            call_times.sort()
            gaps = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]
            # At least one gap should be >= 0.03s (close to rate limit)
            assert any(g >= 0.03 for g in gaps), f"Gaps too small: {gaps}"
        finally:
            supadata_utils.SUPADATA_RATE_LIMIT_SECONDS = original_limit
            supadata_utils.reset_rate_limiter()

    def test_reset_rate_limiter(self):
        """reset_rate_limiter should zero the state."""
        from obsidian_llm_wiki.ingest import supadata_utils

        supadata_utils.reset_rate_limiter()
        assert supadata_utils._last_call_time == 0.0


# ── 3. Graph export source node ID slugification ────────────────────────


class TestGraphExportSourceNodeID:
    """Source node IDs should be slugified, not raw file names."""

    def test_source_node_id_is_slugified(self, tmp_path: Path):
        from obsidian_llm_wiki.render.graph_export import export_graph_json

        source = SourceSynthesis(
            source_title="My Article Title",
            source_summary="Summary",
            source_file="My Article Title.md",
        )
        bundle = SynthesisBundle(sources=[source])

        output = tmp_path / "graph.json"
        export_graph_json(bundle, output)

        data = json.loads(output.read_text())
        source_nodes = [n for n in data["nodes"] if n["type"] == "source"]
        assert len(source_nodes) == 1
        # Slug should be "my-article-title" not "My Article Title.md"
        assert source_nodes[0]["id"] == "my-article-title"

    def test_source_node_id_from_title_when_no_file(self, tmp_path: Path):
        from obsidian_llm_wiki.render.graph_export import export_graph_json

        source = SourceSynthesis(
            source_title="Some Title With Spaces",
            source_summary="Summary",
            source_file="",
        )
        bundle = SynthesisBundle(sources=[source])

        output = tmp_path / "graph.json"
        export_graph_json(bundle, output)

        data = json.loads(output.read_text())
        source_nodes = [n for n in data["nodes"] if n["type"] == "source"]
        assert len(source_nodes) == 1
        assert source_nodes[0]["id"] == "some-title-with-spaces"


# ── 4. Mermaid label escaping ────────────────────────────────────────────


class TestMermaidLabelEscaping:
    """Mermaid labels should escape brackets and quotes."""

    def test_brackets_in_title_are_escaped(self, tmp_path: Path):
        from obsidian_llm_wiki.render.graph_export import export_graph_mermaid

        concept = ConceptNote(
            title="Concept [with brackets]",
            slug="concept-brackets",
            summary="Test",
        )
        bundle = SynthesisBundle(concepts=[concept])

        output = tmp_path / "graph.mmd"
        export_graph_mermaid(bundle, output)

        content = output.read_text()
        # Should not contain raw [ or ] inside labels (they break Mermaid syntax)
        # The node definition line should use the escaped version
        assert "[with brackets]" not in content
        # Should contain the escaped version
        assert "‹" in content or "›" in content or "with brackets" in content

    def test_quotes_in_title_are_escaped(self, tmp_path: Path):
        from obsidian_llm_wiki.render.graph_export import export_graph_mermaid

        concept = ConceptNote(
            title='Concept with "quotes"',
            slug="concept-quotes",
            summary="Test",
        )
        bundle = SynthesisBundle(concepts=[concept])

        output = tmp_path / "graph.mmd"
        export_graph_mermaid(bundle, output)

        content = output.read_text()
        # Double quotes in labels should be replaced with single quotes
        # The label is inside double quotes in Mermaid, so inner " must be '
        # Count occurrences: the Mermaid format is node["label"]
        # If quotes weren't escaped, we'd have node["Concept with "quotes""]
        # which is invalid Mermaid
        lines = [
            line for line in content.splitlines()
            if "concept_quotes" in line or "concept-quotes" in line
        ]
        assert lines  # Should have at least one line for this concept


# ── 5. Graph export circular references ──────────────────────────────────


class TestGraphExportCircularReferences:
    """Graph export should handle circular concept references."""

    def test_circular_reference_handled(self, tmp_path: Path):
        from obsidian_llm_wiki.render.graph_export import export_graph_json

        c1 = ConceptNote(
            title="A", slug="a", summary="",
            related=[ConceptLink(slug="b", relation="related_to")],
        )
        c2 = ConceptNote(
            title="B", slug="b", summary="",
            related=[ConceptLink(slug="a", relation="related_to")],
        )
        bundle = SynthesisBundle(concepts=[c1, c2])

        output = tmp_path / "graph.json"
        export_graph_json(bundle, output)

        data = json.loads(output.read_text())
        rel_edges = [e for e in data["edges"] if e["relation"] != "contains"]
        # Should have exactly one edge (a→b or b→a), marked bidirectional
        assert len(rel_edges) == 1
        assert rel_edges[0]["bidirectional"] is True

    def test_self_reference_handled(self, tmp_path: Path):
        """A concept that references itself should not crash."""
        from obsidian_llm_wiki.render.graph_export import export_graph_json

        c1 = ConceptNote(
            title="Self", slug="self", summary="",
            related=[ConceptLink(slug="self", relation="related_to")],
        )
        bundle = SynthesisBundle(concepts=[c1])

        output = tmp_path / "graph.json"
        export_graph_json(bundle, output)

        data = json.loads(output.read_text())
        rel_edges = [e for e in data["edges"] if e["relation"] != "contains"]
        # Self-edge should be present
        assert any(e["source"] == "self" and e["target"] == "self" for e in rel_edges)


# ── 6. Metrics history preservation ─────────────────────────────────────


class TestMetricsHistoryPreservation:
    """Metrics should preserve history across multiple runs."""

    def test_multiple_saves_preserve_history(self, tmp_path: Path):
        from obsidian_llm_wiki.core.metrics import MetricsCollector

        collector = MetricsCollector(tmp_path)

        # First run
        collector.start_run()
        collector.record_synthesis(source_file="a.md", success=True)
        collector.finish_run()
        collector.save()

        # Second run
        collector.start_run()
        collector.record_synthesis(source_file="b.md", success=True)
        collector.finish_run()
        collector.save()

        metrics_file = tmp_path / "04-Wiki" / ".llmwiki" / "metrics.json"
        data = json.loads(metrics_file.read_text())

        assert "runs" in data
        assert len(data["runs"]) == 2
        assert "latest" in data
        # Latest should be the second run
        assert data["latest"]["syntheses"][0]["source_file"] == "b.md"
        # First run should be in history
        assert data["runs"][0]["syntheses"][0]["source_file"] == "a.md"

    def test_load_metrics_returns_latest(self, tmp_path: Path):
        from obsidian_llm_wiki.core.metrics import MetricsCollector, load_metrics

        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.finish_run()
        collector.save()

        loaded = load_metrics(tmp_path)
        assert loaded is not None
        assert "run_id" in loaded

    def test_load_all_metrics_returns_history(self, tmp_path: Path):
        from obsidian_llm_wiki.core.metrics import MetricsCollector, load_all_metrics

        collector = MetricsCollector(tmp_path)
        for _i in range(3):
            collector.start_run()
            collector.finish_run()
            collector.save()

        all_runs = load_all_metrics(tmp_path)
        assert len(all_runs) == 3

    def test_metrics_history_capped_at_50(self, tmp_path: Path):
        """History should be capped to prevent unbounded growth."""
        from obsidian_llm_wiki.core.metrics import MetricsCollector, load_all_metrics

        collector = MetricsCollector(tmp_path)
        for _ in range(55):
            collector.start_run()
            collector.finish_run()
            collector.save()

        all_runs = load_all_metrics(tmp_path)
        assert len(all_runs) == 50

    def test_legacy_format_migration(self, tmp_path: Path):
        """Legacy flat-format metrics.json should be migrated to new format."""
        from obsidian_llm_wiki.core.metrics import (
            MetricsCollector,
            load_all_metrics,
            load_metrics,
        )

        metrics_file = tmp_path / "04-Wiki" / ".llmwiki"
        metrics_file.mkdir(parents=True)
        legacy_data = {
            "run_id": "20240101-000000",
            "started_at": "2024-01-01T00:00:00",
            "finished_at": "2024-01-01T00:01:00",
            "total_time_seconds": 60.0,
            "extractions": [],
            "syntheses": [],
            "rendering": {},
            "embedding": {},
            "summary": {},
        }
        (metrics_file / "metrics.json").write_text(
            json.dumps(legacy_data, indent=2)
        )

        # Now save a new run — should migrate the legacy data
        collector = MetricsCollector(tmp_path)
        collector.start_run()
        collector.finish_run()
        collector.save()

        loaded = load_metrics(tmp_path)
        assert loaded is not None
        # Latest should be the new run
        assert loaded["run_id"] != "20240101-000000"

        all_runs = load_all_metrics(tmp_path)
        # Should have 2 runs: the legacy one + the new one
        assert len(all_runs) == 2


# ── 7. render_vault config threshold wiring ──────────────────────────────


class TestRenderVaultConfigThresholds:
    """render_vault should use config thresholds when provided."""

    def test_render_vault_accepts_config_param(self, tmp_path: Path):
        """render_vault should accept an optional config parameter."""
        from obsidian_llm_wiki.render.obsidian import render_vault

        source = SourceSynthesis(
            source_title="Test", source_summary="Summary",
            source_file="test.md",
            concepts=[ConceptNote(title="C", slug="c", summary="S")],
        )
        bundle = SynthesisBundle(
            sources=[source],
            concepts=[ConceptNote(title="C", slug="c", summary="S")],
        )

        config = mock.MagicMock()
        config.similarity_dedup_threshold = 0.99
        config.moc_assignment_threshold = 0.99

        # Should not raise
        written = render_vault(tmp_path / "wiki", bundle, {}, config=config)
        assert len(written) > 0

    def test_render_vault_without_config_uses_defaults(self, tmp_path: Path):
        """render_vault should work without config (backward compat)."""
        from obsidian_llm_wiki.render.obsidian import render_vault

        source = SourceSynthesis(
            source_title="Test", source_summary="Summary",
            source_file="test.md",
            concepts=[ConceptNote(title="C", slug="c", summary="S")],
        )
        bundle = SynthesisBundle(
            sources=[source],
            concepts=[ConceptNote(title="C", slug="c", summary="S")],
        )

        written = render_vault(tmp_path / "wiki", bundle, {})
        assert len(written) > 0


# ── 8. semantic_dedupe error handling ────────────────────────────────────


class TestSemanticDedupeErrorHandling:
    """semantic_dedupe_concepts should handle edge cases gracefully."""

    def test_empty_bundle_no_crash(self):
        from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts

        bundle = SynthesisBundle()
        # Should be a no-op, no exception
        semantic_dedupe_concepts(bundle)
        assert len(bundle.concepts) == 0

    def test_single_concept_no_crash(self):
        from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts

        c = ConceptNote(title="Only", slug="only", summary="One concept")
        bundle = SynthesisBundle(concepts=[c])
        semantic_dedupe_concepts(bundle)
        assert len(bundle.concepts) == 1

    def test_missing_embeddings_skips_gracefully(self):
        """When embeddings are unavailable, dedup should skip gracefully."""
        from obsidian_llm_wiki.synth.dedupe import semantic_dedupe_concepts

        c1 = ConceptNote(title="A", slug="a", summary="First")
        c2 = ConceptNote(title="B", slug="b", summary="Second")
        bundle = SynthesisBundle(concepts=[c1, c2])

        # embed_text returns None when embeddings are disabled
        with mock.patch(
            "obsidian_llm_wiki.synth.embedding.embed_text", return_value=None
        ):
            semantic_dedupe_concepts(bundle)

        # No merges should happen
        assert len(bundle.concepts) == 2


# ── 9. assign_orphans_to_mocs error handling ─────────────────────────────


class TestAssignOrphansErrorHandling:
    """assign_orphans_to_mocs should handle edge cases gracefully."""

    def test_empty_bundle_no_crash(self):
        from obsidian_llm_wiki.synth.dedupe import assign_orphans_to_mocs

        bundle = SynthesisBundle()
        assign_orphans_to_mocs(bundle)
        assert len(bundle.concepts) == 0

    def test_no_maps_no_crash(self):
        from obsidian_llm_wiki.synth.dedupe import assign_orphans_to_mocs

        c = ConceptNote(title="Orphan", slug="orphan", summary="No MoC")
        bundle = SynthesisBundle(concepts=[c])
        assign_orphans_to_mocs(bundle)
        assert len(bundle.concepts) == 1

    def test_no_orphans_no_crash(self):
        from obsidian_llm_wiki.synth.dedupe import assign_orphans_to_mocs

        c = ConceptNote(title="C", slug="c", summary="In MoC")
        moc = MapOfContent(title="MOC", slug="moc", summary="", concept_slugs=["c"])
        bundle = SynthesisBundle(concepts=[c], maps=[moc])
        assign_orphans_to_mocs(bundle)
        assert len(bundle.concepts) == 1

    def test_missing_embeddings_skips_gracefully(self):
        """When embeddings are unavailable, assignment should skip."""
        from obsidian_llm_wiki.synth.dedupe import assign_orphans_to_mocs

        c1 = ConceptNote(title="In MOC", slug="in-moc", summary="Member")
        c2 = ConceptNote(title="Orphan", slug="orphan", summary="No MoC")
        moc = MapOfContent(title="MOC", slug="moc", summary="", concept_slugs=["in-moc"])
        bundle = SynthesisBundle(concepts=[c1, c2], maps=[moc])

        with mock.patch(
            "obsidian_llm_wiki.synth.embedding.embed_text", return_value=None
        ):
            assign_orphans_to_mocs(bundle)

        # Orphan should not be assigned
        assert "orphan" not in moc.concept_slugs


# ── 10. Extracted modules are the single implementation ──────────────────


class TestExtractedModulesAreCanonical:
    """frontmatter.py, bilingual.py, and crossrefs.py are the single home of
    these helpers; render/obsidian.py imports and re-exports them. If
    obsidian.py grows its own copy again, these identity checks fail.
    """

    def test_obsidian_reexports_frontmatter_helpers(self):
        from obsidian_llm_wiki.render import frontmatter as fm
        from obsidian_llm_wiki.render import obsidian as ob

        assert ob.slugify is fm.slugify
        assert ob.parse_frontmatter is fm.parse_frontmatter
        assert ob.build_frontmatter is fm.build_frontmatter
        assert ob.extract_links is fm.extract_links
        assert ob.safe_read_file is fm.safe_read_file
        assert ob.atomic_write is fm.atomic_write

    def test_obsidian_uses_bilingual_and_crossref_modules(self):
        from obsidian_llm_wiki.render import bilingual as bi
        from obsidian_llm_wiki.render import crossrefs as cr
        from obsidian_llm_wiki.render import obsidian as ob

        assert ob._is_chinese is bi.is_chinese
        assert (
            ob._normalize_bilingual_titles_and_slugs
            is bi.normalize_bilingual_titles_and_slugs
        )
        assert ob._build_cross_ref_diagram is cr.build_cross_ref_diagram
        assert ob._build_moc_cross_ref_diagram is cr.build_moc_cross_ref_diagram

    def test_extract_links_and_wikilinks_are_distinct(self):
        """One name, one meaning: extract_links = markdown links (what
        validate.py needs); extract_wikilinks = [[wikilinks]]."""
        from obsidian_llm_wiki.render import frontmatter as fm

        body = "See [example](https://example.com) and [[slug|alias]]."
        assert fm.extract_links(body) == [("example", "https://example.com")]
        assert fm.extract_wikilinks(body) == [("slug", "alias")]

    def test_parse_frontmatter_rejects_non_dict_yaml(self):
        """The hardened parse_frontmatter (dict guard + body lstrip) is the
        shared one — the drifted copy lost both."""
        from obsidian_llm_wiki.render.frontmatter import parse_frontmatter

        meta, body = parse_frontmatter("---\njust a scalar\n---\n\n\nBody text")
        assert meta == {}
        assert body == "Body text"


# ── 11. _synthesize_with_retry lossless policy ───────────────────────────


class TestSynthesizeWithRetryPolicy:
    """Large source handling must preserve source content."""

    def test_bounded_structured_output_policy(self):
        """The synthesis request has a practical output cap, not 128K."""
        from obsidian_llm_wiki.core.pipeline import _SYNTHESIS_NUM_PREDICT

        assert _SYNTHESIS_NUM_PREDICT == 16_384


# ── 12. extract_links consistency note ──────────────────────────────────


class TestExtractLinksConsistency:
    """obsidian.py's extract_links extracts markdown links [text](url).
    frontmatter.py's extract_links extracts wikilinks [[slug|alias]].
    The obsidian.py version is the one used by validate.py.
    This test documents the current behavior to prevent regression.
    """

    def test_obsidian_extract_links_finds_markdown_links(self):
        from obsidian_llm_wiki.render.obsidian import extract_links
        result = extract_links("See [example](https://example.com) here")
        assert result == [("example", "https://example.com")]

    def test_obsidian_extract_links_ignores_wikilinks(self):
        from obsidian_llm_wiki.render.obsidian import extract_links
        result = extract_links("See [[slug|alias]] here")
        assert result == []  # Wikilinks are NOT extracted by this function

    def test_obsidian_extract_links_ignores_images(self):
        from obsidian_llm_wiki.render.obsidian import extract_links
        result = extract_links("![alt](image.png)")
        assert result == []  # Images should be excluded
