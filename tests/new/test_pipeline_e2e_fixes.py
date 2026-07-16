"""Regression tests for the end-to-end pipeline review (synth → render → cli).

Each test pins a bug found reviewing the restore-pipeline branch beyond the
ingest stage. Every test fails against the pre-fix code.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest

from obsidian_llm_wiki.core.models import (
    ConceptLink,
    ConceptNote,
    MapOfContent,
    SourceDoc,
    SourceSynthesis,
    SynthesisBundle,
)

_BODY = "A meaningful body sentence. " * 30


def _concept(slug: str, confidence: float = 0.8, related: list[str] | None = None) -> ConceptNote:
    return ConceptNote(
        title=slug.replace("-", " ").title(),
        slug=slug,
        summary=f"Summary of {slug}.",
        tags=[slug],
        confidence=confidence,
        related=[ConceptLink(slug=r, relation="related_to") for r in (related or [])],
    )


# ── Ollama num_ctx must live in the options object ────────────────────────


def test_ollama_num_ctx_goes_into_options():
    """Ollama only reads num_ctx from `options`; a top-level field is ignored.

    Before the fix, kwargs['num_ctx'] was merged into the top level of the
    /api/chat body, so LLM_CONTEXT_WINDOW never took effect and long prompts
    were silently truncated at the model's default context.
    """
    from obsidian_llm_wiki.config import LLMProviderConfig
    from obsidian_llm_wiki.providers.llm import OllamaClient

    config = LLMProviderConfig(
        provider="ollama",
        host="http://localhost:11434",
        model="test-model",
        context_window=256_000,
    )
    client = OllamaClient(config)

    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": "ok"}}

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def post(self, url, json):  # noqa: A002 — httpx kwarg name
            captured.update(json)
            return _FakeResponse()

    with mock.patch("obsidian_llm_wiki.providers.llm.httpx.Client", _FakeClient):
        client.chat("system prompt", "user prompt")

    assert "num_ctx" not in captured, "num_ctx at top level is ignored by Ollama"
    assert captured.get("options", {}).get("num_ctx") == 256_000


def test_ollama_explicit_num_ctx_kwarg_still_wins():
    from obsidian_llm_wiki.config import LLMProviderConfig
    from obsidian_llm_wiki.providers.llm import OllamaClient

    config = LLMProviderConfig(
        provider="ollama", host="http://x", model="m", context_window=256_000,
    )
    client = OllamaClient(config)
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"message": {"content": "ok"}}

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def post(self, url, json):  # noqa: A002
            captured.update(json)
            return _FakeResponse()

    with mock.patch("obsidian_llm_wiki.providers.llm.httpx.Client", _FakeClient):
        client.chat("s", "u", num_ctx=8192)

    assert captured["options"]["num_ctx"] == 8192


# ── The ops CLI commands must actually be registered ──────────────────────


def test_metrics_and_recompile_commands_are_registered():
    """cli/ops.py was never imported, so its @app.command() never ran."""
    from obsidian_llm_wiki.cli import app

    names = set()
    for command in app.registered_commands:
        names.add(command.name or command.callback.__name__)

    assert "metrics" in names
    assert "recompile" in names


# ── Chunked Pass 1 must fail loudly on partial chunk failure ──────────────


def _quality_config(**overrides):
    from obsidian_llm_wiki.config import load_config

    cfg = load_config(env_file=None, VAULT_PATH="/tmp/does-not-matter")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_partial_chunk_failure_raises_instead_of_returning_partial(monkeypatch):
    """A source whose chunks partially fail must NOT produce a synthesis.

    Before the fix, failed chunks were skipped and the partial skeleton
    returned as success — the pipeline cached it and hash-stamped the source,
    permanently losing the failed chunks' content with no recorded error.
    """
    from obsidian_llm_wiki.synth import quality

    cfg = _quality_config(chunk_size=100, compile_concurrency=4)

    good_response = json.dumps({
        "source_title": "T",
        "concepts": [{"title": "Alpha", "slug": "alpha", "summary": "s"}],
        "maps": [],
    })

    calls = {"n": 0}

    async def flaky_llm(prompt, messages, config):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("Ollama died mid-run")
        return good_response

    monkeypatch.setattr(quality, "acall_llm", None, raising=False)
    with mock.patch("obsidian_llm_wiki.providers.llm.acall_llm", flaky_llm):
        source = SourceDoc(title="T", content="word " * 200, source_file="t.md")
        with pytest.raises(RuntimeError, match="incomplete"):
            asyncio.run(
                quality.quality_synthesize_source(cfg, "t.md", source, []),
            )


def test_unparseable_chunk_also_fails_the_source(monkeypatch):
    """A chunk whose response can't be parsed is missing content too."""
    from obsidian_llm_wiki.synth import quality

    cfg = _quality_config(chunk_size=100, compile_concurrency=4)

    good_response = json.dumps({
        "source_title": "T",
        "concepts": [{"title": "Alpha", "slug": "alpha", "summary": "s"}],
        "maps": [],
    })
    calls = {"n": 0}

    async def sometimes_garbage(prompt, messages, config):
        calls["n"] += 1
        return "complete garbage, no JSON here" if calls["n"] == 2 else good_response

    with mock.patch("obsidian_llm_wiki.providers.llm.acall_llm", sometimes_garbage):
        source = SourceDoc(title="T", content="word " * 200, source_file="t.md")
        with pytest.raises(RuntimeError, match="incomplete"):
            asyncio.run(
                quality.quality_synthesize_source(cfg, "t.md", source, []),
            )


def test_chunk_gate_is_config_chunk_size(monkeypatch):
    """Chunking must trigger above config.chunk_size, not a hardcoded 40K.

    Before the fix a hardcoded 40_000 gate meant CHUNK_SIZE had no effect on
    *when* chunking happens — sources between chunk_size and 40K silently
    took the single-call path, contradicting the documented behavior.
    """
    from obsidian_llm_wiki.synth import quality

    cfg = _quality_config(chunk_size=100, compile_concurrency=2)

    prompts_seen: list[str] = []

    async def counting_llm(prompt, messages, config, **_kwargs):
        prompts_seen.append(prompt)
        return json.dumps({
            "source_title": "T",
            "concepts": [{"title": "Alpha", "slug": "alpha", "summary": "s"}],
            "maps": [],
        })

    # 300 chars > chunk_size 100 (and far below the old 40K gate) → must chunk.
    with mock.patch("obsidian_llm_wiki.providers.llm.acall_llm", counting_llm):
        source = SourceDoc(title="T", content="word " * 60, source_file="t.md")
        result = asyncio.run(
            quality.quality_synthesize_source(cfg, "t.md", source, []),
        )

    assert result is not None
    # More than one Pass-1 call proves the chunked path ran. (Pass 2 calls are
    # per-concept; the merged skeleton has one concept, so: N chunks + 1.)
    assert len(prompts_seen) > 2


# ── Semantic dedup: merge chains and source-local slugs ───────────────────


def _bundle_with_similarity(monkeypatch, concepts, sims):
    """Build a bundle and force embed/similarity results for dedup tests."""
    from obsidian_llm_wiki.synth import dedupe

    synthesis = SourceSynthesis(
        source_title="S", source_summary="s", source_file="s.md", concepts=list(concepts),
    )
    bundle = SynthesisBundle(
        sources=[synthesis],
        concepts=list(concepts),
        maps=[MapOfContent(title="Map", summary="m", slug="map",
                           concept_slugs=[c.slug for c in concepts])],
    )

    fake_embeddings = {c.slug: [float(i + 1)] for i, c in enumerate(concepts)}
    slug_by_text = {f"{c.title}. {c.summary or ''}": c.slug for c in concepts}

    def fake_embed(text):
        return fake_embeddings.get(slug_by_text.get(text, ""), [1.0])

    def fake_cosine(a, b):
        slug_a = next(s for s, e in fake_embeddings.items() if e == a)
        slug_b = next(s for s, e in fake_embeddings.items() if e == b)
        return sims.get((slug_a, slug_b), sims.get((slug_b, slug_a), 0.0))

    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.embed_text", fake_embed,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.cosine_similarity", fake_cosine,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.language.detect_language", lambda _t: "en",
    )
    return bundle, dedupe


def test_merge_victim_is_never_reused_as_survivor(monkeypatch):
    """After X merges into Y, a later pair must not merge content INTO X.

    Before the fix, the inner loop kept using slug_a after it became a merge
    victim: with X~Y and X~W, W was merged into the already-deleted X and W's
    content vanished while MoC entries pointed at nonexistent slugs.
    """
    x = _concept("x-concept", confidence=0.8)
    y = _concept("y-concept", confidence=0.9)
    w = _concept("w-concept", confidence=0.7)
    w.sections = []
    w.claims = []

    bundle, dedupe = _bundle_with_similarity(
        monkeypatch, [x, y, w],
        sims={("x-concept", "y-concept"): 0.95, ("x-concept", "w-concept"): 0.95},
    )

    dedupe.semantic_dedupe_concepts(bundle, threshold=0.85)

    surviving = {c.slug for c in bundle.concepts}
    # Deterministic traversal merges W into X, then X into Y. The flattened
    # merge map must leave one live survivor with no dangling MoC references.
    assert surviving == {"y-concept"}
    # Every MoC slug must reference a live concept.
    for moc in bundle.maps:
        for slug in moc.concept_slugs:
            assert slug in surviving, f"MoC references deleted slug {slug}"


def test_merge_map_chains_are_flattened(monkeypatch):
    """A→B then B→C must remap A's references to C, not to the deleted B."""
    a = _concept("a-concept", confidence=0.5)
    b = _concept("b-concept", confidence=0.7)
    c = _concept("c-concept", confidence=0.9)

    bundle, dedupe = _bundle_with_similarity(
        monkeypatch, [a, b, c],
        sims={("a-concept", "b-concept"): 0.95, ("b-concept", "c-concept"): 0.95},
    )

    dedupe.semantic_dedupe_concepts(bundle, threshold=0.85)

    surviving = {con.slug for con in bundle.concepts}
    assert surviving == {"c-concept"}
    for moc in bundle.maps:
        assert set(moc.concept_slugs) <= surviving


def test_source_local_concept_slugs_are_remapped(monkeypatch):
    """Entry pages and state.json read bundle.sources[*].concepts — dedup must
    remap those too, or entries link [[victim]] pages that no longer exist."""
    x = _concept("x-concept", confidence=0.5)
    y = _concept("y-concept", confidence=0.9)

    bundle, dedupe = _bundle_with_similarity(
        monkeypatch, [x, y],
        sims={("x-concept", "y-concept"): 0.95},
    )

    dedupe.semantic_dedupe_concepts(bundle, threshold=0.85)

    surviving = {c.slug for c in bundle.concepts}
    assert surviving == {"y-concept"}
    for synthesis in bundle.sources:
        source_slugs = [c.slug for c in synthesis.concepts]
        assert "x-concept" not in source_slugs
        # No duplicates after remap collapses two concepts into one.
        assert len(source_slugs) == len(set(source_slugs))


# ── Incremental resynthesis: grouping and persistence ─────────────────────


def test_resynthesis_groups_multiple_sources_per_slug(monkeypatch):
    """k new sources referencing one concept must produce ONE resynthesis call
    that sees all k sources — not k racing calls where the last one wins."""
    from obsidian_llm_wiki.core import pipeline as pl

    shared = _concept("shared-concept")
    cached_synth = SourceSynthesis(
        source_title="Old", source_summary="s", source_file="old.md",
        concepts=[_concept("shared-concept")],
    )
    new_a = SourceSynthesis(
        source_title="A", source_summary="s", source_file="a.md",
        concepts=[_concept("shared-concept")],
    )
    new_b = SourceSynthesis(
        source_title="B", source_summary="s", source_file="b.md",
        concepts=[_concept("shared-concept")],
    )
    bundle = SynthesisBundle(
        sources=[cached_synth, new_a, new_b], concepts=[shared], maps=[],
    )
    sources = {
        "a.md": SourceDoc(title="A", content="alpha content", source_file="a.md"),
        "b.md": SourceDoc(title="B", content="beta content", source_file="b.md"),
    }

    resynth_calls: list[tuple[str, str, str]] = []

    async def fake_resynthesize(config, concept, content, title):
        resynth_calls.append((concept.slug, content, title))
        updated = _concept(concept.slug)
        updated.summary = "integrated summary"
        return updated

    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.quality.resynthesize_concept",
        fake_resynthesize,
    )

    cfg = _quality_config(compile_concurrency=4)
    result = asyncio.run(
        pl._resynthesize_referenced_concepts(
            cfg, bundle, ["a.md", "b.md"],
            {"old.md": cached_synth, "a.md": new_a, "b.md": new_b},
            sources,
        ),
    )

    assert len(resynth_calls) == 1, "one call per slug, not per (slug, source)"
    slug, content, title = resynth_calls[0]
    assert slug == "shared-concept"
    assert "alpha content" in content and "beta content" in content
    # The updated concept is returned for persistence and applied to the bundle.
    assert "shared-concept" in result
    assert bundle.concepts[0].summary == "integrated summary"


def test_resynthesis_overlay_round_trip(tmp_path):
    """Resynthesized concepts must survive a save/load cycle so later builds
    can re-apply them over the mechanically merged bundle."""
    from obsidian_llm_wiki.core.cache import (
        load_resynthesis_overlay,
        save_resynthesis_overlay,
    )

    concept = _concept("persisted-concept")
    concept.summary = "coherently integrated"
    save_resynthesis_overlay(tmp_path, {"persisted-concept": concept})

    loaded = load_resynthesis_overlay(tmp_path)

    assert set(loaded) == {"persisted-concept"}
    assert loaded["persisted-concept"].summary == "coherently integrated"
    assert loaded["persisted-concept"].slug == "persisted-concept"


def test_resynthesis_overlay_missing_or_corrupt(tmp_path):
    from obsidian_llm_wiki.core.cache import load_resynthesis_overlay

    assert load_resynthesis_overlay(tmp_path) == {}
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cache" / "_resynthesis_overlay.json").write_text("{not json")
    assert load_resynthesis_overlay(tmp_path) == {}


# ── recompile_single_source must load like the build does ─────────────────


def test_recompile_strips_frontmatter_and_matches_build_hash(tmp_path, monkeypatch):
    """Recompile must synthesize the body (not raw frontmatter) and store the
    same hash the next `olw build` will compute, or the recompile is wasted."""
    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.core.state import hash_content, read_state
    from obsidian_llm_wiki.ingest.sources import load_source_file

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    body = "The actual article body. " * 30
    raw = f"---\ntype: source\ntitle: Real Title\nurl: https://x\n---\n\n{body}"
    sources_dir = tmp_path / "04-Wiki" / "sources"
    sources_dir.mkdir(parents=True)
    (sources_dir / "article.md").write_text(raw, encoding="utf-8")

    seen: dict = {}

    async def fake_retry(config, filename, source, existing, *_args, metrics=None, **_kwargs):
        seen["title"] = source.title
        seen["content"] = source.content
        synth = SourceSynthesis(
            source_title=source.title, source_summary="s", source_file=filename,
            concepts=[_concept("a-concept")],
        )
        return synth

    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)

    result = asyncio.run(
        pl.recompile_single_source(tmp_path, "article.md"),
    )

    assert result.compiled == 1
    assert seen["title"] == "Real Title", "title must come from frontmatter"
    assert "type: source" not in seen["content"], "frontmatter must be stripped"

    # The stored hash must equal what the build path would compute.
    from obsidian_llm_wiki.config import load_config
    config = load_config(env_file=None, VAULT_PATH=str(tmp_path))
    state = read_state(config.state_file)
    build_doc = load_source_file(sources_dir / "article.md")
    assert state.sources["article.md"].hash == hash_content(build_doc.content)


# ── Metrics: one failed source = its attempts, not attempts + 1 ───────────


def test_pipeline_does_not_double_record_synthesis_failures(tmp_path, monkeypatch):
    """run_pipeline's results loop must not add extra failure records on top
    of the per-attempt records _synthesize_with_retry already made."""
    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.core.metrics import load_metrics

    monkeypatch.setenv("VAULT_PATH", str(tmp_path))

    async def fake_retry(config, filename, source, existing, *_args, metrics=None, **_kwargs):
        # Simulate the real helper: it records once per attempt (here: one),
        # then re-raises on the final level.
        if metrics:
            metrics.record_synthesis(
                source_file=filename, success=False, error_type="TimeoutError",
            )
        raise TimeoutError("permanent failure")

    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)

    source = SourceDoc(title="T", content="x " * 200, source_file="t.md")
    result = asyncio.run(
        pl.run_pipeline(tmp_path, {"t.md": source}),
    )

    assert any("synth:t.md" in e for e in result.errors)
    metrics_data = load_metrics(tmp_path)
    synth_records = metrics_data.get("syntheses", [])
    failures = [r for r in synth_records if not r.get("success")]
    assert len(failures) == 1, (
        f"expected exactly the helper's 1 per-attempt record, got {len(failures)}"
    )


def test_pipeline_records_and_reuses_cross_lingual_embedding_links(tmp_path, monkeypatch):
    """One embedding pass must drive both rendering and truthful metrics."""
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.core.metrics import load_metrics

    english = _concept("language-model")
    english.title = "Language model"
    chinese = _concept("da-yuyan-moxing")
    chinese.title = "大语言模型"
    moc = MapOfContent(
        title="Language models", slug="language-models", summary="", concept_slugs=[english.slug],
    )

    async def fake_retry(_config, filename, _source, _existing, *_args, **_kwargs):
        return SourceSynthesis(
            source_title="Bilingual source",
            source_summary="",
            source_file=filename,
            concepts=[english, chinese],
            maps=[moc],
        )

    captured: dict = {}

    def fake_render(_dir, _bundle, _sources, **kwargs):
        captured["cross_lingual_links"] = kwargs["cross_lingual_links"]
        return []

    links = {
        english.slug: [(chinese.slug, 0.91, chinese.title)],
        chinese.slug: [(english.slug, 0.91, english.title)],
    }
    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)
    monkeypatch.setattr(pl, "render_vault", fake_render)
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.find_cross_lingual_links",
        lambda _concepts, **_kwargs: links,
    )

    result = asyncio.run(
        pl.run_pipeline(
            tmp_path,
            {"bilingual.md": SourceDoc(title="Bilingual source", content="x " * 100)},
            Config(vault_path=str(tmp_path)),
        )
    )

    assert result.compiled == 1
    assert captured["cross_lingual_links"] == links
    metrics_data = load_metrics(tmp_path)
    assert metrics_data is not None
    assert metrics_data["rendering"]["cross_lingual_links"] == 1


def test_pipeline_embedding_metric_uses_pruned_cache_count(tmp_path, monkeypatch):
    """A dedup victim must not inflate the final embedding metric."""
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline as pl
    from obsidian_llm_wiki.core.metrics import MetricsCollector

    survivor = _concept("survivor")
    victim = _concept("victim")

    async def fake_retry(_config, filename, _source, _existing, *_args, **_kwargs):
        return SourceSynthesis(
            source_title="Source",
            source_summary="",
            source_file=filename,
            concepts=[survivor, victim],
        )

    def fake_dedup(bundle, **_kwargs):
        bundle.concepts = [survivor]

    persisted: dict[str, list[float]] = {}
    metrics: dict[str, int] = {}
    monkeypatch.setattr(pl, "_synthesize_with_retry", fake_retry)
    monkeypatch.setattr(pl, "render_vault", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.dedupe.semantic_dedupe_concepts", fake_dedup,
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.load_embeddings_cache",
        lambda _path, **_kwargs: {"survivor": [1.0, 0.0], "victim": [0.0, 1.0]},
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.save_embeddings_cache",
        lambda _path, cache, **_kwargs: persisted.update(cache),
    )
    monkeypatch.setattr(
        "obsidian_llm_wiki.synth.embedding.find_cross_lingual_links",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        MetricsCollector,
        "record_embedding",
        lambda _self, **kwargs: metrics.update(kwargs),
    )

    result = asyncio.run(
        pl.run_pipeline(
            tmp_path,
            {"source.md": SourceDoc(title="Source", content="x " * 100)},
            Config(vault_path=str(tmp_path), embeddings_enabled=True),
        )
    )

    assert result.compiled == 1
    assert persisted == {"survivor": [1.0, 0.0]}
    assert metrics["concepts_embedded"] == 1


# ── Health report wikilink normalization ──────────────────────────────────


def test_wikilink_target_normalization():
    from obsidian_llm_wiki.cli.health import _normalize_wikilink_target

    assert _normalize_wikilink_target("bitcoin#Consensus") == "bitcoin"
    assert _normalize_wikilink_target("concepts/bitcoin") == "bitcoin"
    assert _normalize_wikilink_target("concepts/bitcoin#History") == "bitcoin"
    assert _normalize_wikilink_target("bitcoin.md") == "bitcoin"
    assert _normalize_wikilink_target("bitcoin#^block-ref") == "bitcoin"
    assert _normalize_wikilink_target("bitcoin") == "bitcoin"


def test_health_report_accepts_valid_anchor_and_path_links(tmp_path):
    """[[slug#heading]] and [[dir/slug]] resolve fine in Obsidian and must not
    be reported as broken; genuinely missing targets still must be."""
    from obsidian_llm_wiki.cli.health import _generate_health_report

    bundle = tmp_path / "04-Wiki"
    concepts = bundle / "concepts"
    concepts.mkdir(parents=True)
    (concepts / "bitcoin.md").write_text(
        "---\ntype: Concept\nconfidence: 0.9\n---\nBody.", encoding="utf-8",
    )
    (bundle / "note.md").write_text(
        "---\ntype: Concept\nconfidence: 0.9\n---\n"
        "See [[bitcoin#Consensus]] and [[concepts/bitcoin]] and [[missing-page]].",
        encoding="utf-8",
    )

    report = _generate_health_report(bundle)

    assert "[[bitcoin#Consensus]]" not in report
    assert "[[concepts/bitcoin]]" not in report
    assert "missing-page" in report
