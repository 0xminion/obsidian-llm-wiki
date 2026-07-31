"""Regression tests for the deferred review items.

Covers: semantic dedup moved from the render layer into the pipeline,
lossless synthesis retries, shared embedding cache between the dedup passes,
and the graph-export index lookups.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import httpx
import pytest

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)


def _concept(slug: str, confidence: float = 0.8, related: list[str] | None = None) -> ConceptNote:
    return ConceptNote(
        title=slug.replace("-", " ").title(),
        slug=slug,
        summary=f"Summary of {slug}.",
        tags=[slug],
        confidence=confidence,
        related=[ConceptLink(slug=r, relation="related_to") for r in (related or [])],
    )


def _bundle(concepts: list[ConceptNote], maps: list[MapOfContent] | None = None) -> SynthesisBundle:
    synthesis = SourceSynthesis(
        source_title="S", source_summary="s", source_file="s.md",
        concepts=list(concepts),
    )
    return SynthesisBundle(
        sources=[synthesis], concepts=list(concepts), maps=list(maps or []),
    )


# ── Dedup runs in the pipeline stage, not inside render_vault ─────────────


def test_render_vault_does_not_run_semantic_dedup(tmp_path, monkeypatch):
    """render_vault must be a pure renderer: bundle-mutating dedup belongs in
    the synthesis stage where the pipeline can see its failures and write
    consistent state."""
    from obsidian_llm_wiki.render.obsidian import render_vault
    from obsidian_llm_wiki.synth import dedupe

    called = {"dedup": False, "orphans": False}
    monkeypatch.setattr(
        dedupe, "semantic_dedupe_concepts",
        lambda *a, **k: called.__setitem__("dedup", True),
    )
    monkeypatch.setattr(
        dedupe, "assign_orphans_to_mocs",
        lambda *a, **k: called.__setitem__("orphans", True),
    )

    bundle = _bundle([_concept("alpha")])
    render_vault(tmp_path / "wiki", bundle, {})

    assert called == {"dedup": False, "orphans": False}


def test_pipeline_runs_dedup_before_state_write(tmp_path, monkeypatch):
    """run_pipeline must invoke the dedup passes itself (with the shared
    embeddings cache), so slug mutations land before render AND state."""
    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.synth import dedupe

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    order: list[str] = []

    monkeypatch.setattr(
        dedupe, "semantic_dedupe_concepts",
        lambda bundle, threshold, embeddings_cache=None, **_kwargs: order.append("dedup"),
    )
    monkeypatch.setattr(
        dedupe, "assign_orphans_to_mocs",
        lambda bundle, threshold, embeddings_cache=None, **_kwargs: order.append("orphans"),
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.render.obsidian.render_vault",
        lambda *a, **k: order.append("render") or [],
    )
    monkeypatch.setattr(pl, "render_vault", lambda *a, **k: order.append("render") or [])

    async def fake_retry(config, filename, source, existing, *_args, metrics=None, **_kwargs):
        return SourceSynthesis(
            source_title="T", source_summary="s", source_file=filename,
            concepts=[_concept("alpha")],
        )

    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)

    source = SourceDoc(title="T", content="x " * 200, source_file="t.md")
    result = asyncio.run(pl.run_pipeline(tmp_path, {"t.md": source}))

    assert result.compiled == 1
    assert order == ["dedup", "orphans", "render"]


def test_pipeline_dedup_failure_is_survivable_and_visible(tmp_path, monkeypatch, caplog):
    """A dedup crash must not kill the build, and must log at WARNING —
    the old render-layer wrapper swallowed it at debug level forever."""
    import logging

    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.synth import dedupe

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("embedding service wedged")

    monkeypatch.setattr(dedupe, "semantic_dedupe_concepts", boom)
    monkeypatch.setattr(pl, "render_vault", lambda *a, **k: [])

    async def fake_retry(config, filename, source, existing, *_args, metrics=None, **_kwargs):
        return SourceSynthesis(
            source_title="T", source_summary="s", source_file=filename,
            concepts=[_concept("alpha")],
        )

    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)

    source = SourceDoc(title="T", content="x " * 200, source_file="t.md")
    with caplog.at_level(logging.WARNING, logger="obswiki.core.pipeline"):
        result = asyncio.run(pl.run_pipeline(tmp_path, {"t.md": source}))

    assert result.compiled == 1
    assert any("Semantic dedup failed" in r.message for r in caplog.records)


# ── Complete-source retry policy ──────────────────────────────────────────


def _retry_config():
    from obsidian_llm_wiki.config import Config
    return Config()


def test_connection_error_does_not_retry_or_discard_content():
    """A provider failure re-raises immediately and never truncates content."""
    from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

    calls: list[int] = []

    async def dead_server(config, filename, src, existing, **_kwargs):
        calls.append(len(src.content))
        raise httpx.ConnectError("connection refused")

    source = SourceDoc(title="T", content="x" * 60_000, source_file="t.md")
    with mock.patch(
        "obsidian_llm_wiki.core.pipeline._synthesize_source",
        side_effect=dead_server,
    ), pytest.raises(httpx.ConnectError):
        asyncio.run(_synthesize_with_retry(_retry_config(), "t.md", source, []))

    assert calls == [60_000]


def test_parser_retry_keeps_complete_source():
    """A parser miss gets one retry of the full source, never a prefix."""
    from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

    calls: list[int] = []

    async def parse_miss_once(config, filename, src, existing, **_kwargs):
        calls.append(len(src.content))
        if len(calls) == 1:
            return None
        return SourceSynthesis(
            source_title="T", source_summary="s", source_file=filename,
        )

    source = SourceDoc(title="T", content="x" * 60_000, source_file="t.md")
    with mock.patch(
        "obsidian_llm_wiki.core.pipeline._synthesize_source",
        side_effect=parse_miss_once,
    ):
        result = asyncio.run(_synthesize_with_retry(_retry_config(), "t.md", source, []))

    assert result is not None
    assert calls == [60_000, 60_000]


# ── Shared embedding cache between dedup passes ────────────────────────────


def test_dedup_and_orphan_assignment_share_embeddings(monkeypatch):
    """Passing one embeddings_cache through both passes must not re-embed a
    concept that was already embedded — each embed is a network round-trip."""
    from obsidian_llm_wiki.synth import dedupe

    embed_calls: list[str] = []

    def counting_embed(text):
        embed_calls.append(text)
        return [float(len(text))]

    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.embed_text", counting_embed,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.cosine_similarity", lambda a, b: 0.0,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.language.detect_language", lambda _t: "en",
    )

    member = _concept("member-concept")
    orphan = _concept("orphan-concept")
    bundle = _bundle(
        [member, orphan],
        maps=[MapOfContent(title="Map", summary="m", slug="map",
                           concept_slugs=["member-concept"])],
    )

    cache: dict[str, list[float]] = {}
    dedupe.semantic_dedupe_concepts(bundle, threshold=0.85, embeddings_cache=cache)
    calls_after_dedup = len(embed_calls)
    dedupe.assign_orphans_to_mocs(bundle, threshold=0.55, embeddings_cache=cache)

    # Dedup embedded both concepts; orphan assignment needs the same two
    # (orphan + the MoC member) and must find both in the cache.
    assert calls_after_dedup == 2
    assert len(embed_calls) == 2, (
        f"orphan assignment re-embedded: {embed_calls[calls_after_dedup:]}"
    )


# ── Graph export: correctness preserved with index lookups ────────────────


def test_graph_bidirectional_edge_detected_once():
    from obsidian_llm_wiki.render.graph_export import _build_graph_dict

    a = _concept("alpha", related=["beta"])
    b = _concept("beta", related=["alpha"])
    bundle = _bundle([a, b])

    graph = _build_graph_dict(bundle)
    concept_edges = [e for e in graph["edges"] if e["relation"] == "related_to"]

    assert len(concept_edges) == 1
    assert concept_edges[0]["bidirectional"] is True


def test_graph_source_id_strips_only_trailing_md():
    from obsidian_llm_wiki.render.graph_export import _slugify_source_id

    assert _slugify_source_id("article.md") == "article"
    assert _slugify_source_id("notes.mdx") == "notesmdx"  # dot removed, name intact
    assert _slugify_source_id("My Article.md") == "my-article"
