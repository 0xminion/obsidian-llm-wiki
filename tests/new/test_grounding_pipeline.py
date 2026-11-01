"""Admission of grounded syntheses through the real pipeline lifecycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.cache import load_synthesis, save_synthesis
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.state import read_state
from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
from tests.new.immutable_semantic_helpers import (
    file_state,
    published_meta,
    published_registry,
    published_root,
    seed_legacy_cache,
)

LLM = "obsidian_llm_wiki.providers.llm.acall_llm"

SOURCE = (
    "# Trial report\n\nThe **new solver** reduced routing errors by 12% in field trials.\n"
    "It did not reduce latency.\n\nThe team says further validation is needed.\n"
)


def _vault(tmp_path: Path, extra: str = "") -> tuple[Path, object]:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=gen-model\n{extra}"
    )
    return vault, load_config(env_file=str(vault / ".env"))


def _producer(prompt: str) -> str:
    return grounded_response(
        title="Trial report",
        summary="A new solver reduced routing errors by 12% but not latency.",
        key_points=["Routing errors fell by 12% in field trials."],
        concepts=[
            {
                "title": "New solver",
                "slug": "new-solver",
                "summary": "A solver that reduced routing errors by 12%.",
                "sections": [{"heading": "Effect", "points": ["Routing errors fell by 12%."]}],
                "claims": [
                    {
                        "text": "The new solver reduced routing errors by 12%.",
                        "evidence_span_ids": cite(prompt, "routing errors by 12%"),
                    }
                ],
            }
        ],
    )


def _sources(content: str = SOURCE) -> dict[str, SourceDoc]:
    return {"trial.md": SourceDoc(title="Trial report", content=content, source_file="trial.md")}


def _run(vault, config, provider, **kwargs):
    with patch(LLM, side_effect=provider.__call__):
        return asyncio.run(
            run_pipeline(vault, _sources(kwargs.pop("content", SOURCE)), config, **kwargs)
        )


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(root.rglob("*.md"))
    }


def test_review_is_required_by_default_and_receipt_is_persisted(tmp_path):
    vault, config = _vault(tmp_path)
    assert config.grounding_review == "required"
    provider = GroundedProvider(_producer)
    result = _run(vault, config, provider)
    assert result.errors == [] and result.compiled == 1
    assert len(provider.producer_calls) == 1 and len(provider.review_calls) == 1

    cached = load_synthesis(published_meta(config), "trial.md")
    receipt = cached.review_receipt
    assert receipt["status"] == "approved" and receipt["source_file"] == "trial.md"
    evidence = cached.concepts[0].claims[0].evidence
    assert evidence.quote == "The **new solver** reduced routing errors by 12% in field trials."
    assert SOURCE[evidence.start_offset : evidence.end_offset] == evidence.quote
    with published_registry(config) as registry:
        stored = registry.load_syntheses({"trial.md"})["trial.md"]
    assert stored.review_receipt == receipt
    assert stored.concepts[0].claims[0].evidence.span_ids == evidence.span_ids

    note = (published_root(config) / "concepts" / "new-solver.md").read_text()
    assert evidence.span_ids[0] in note


def test_unchanged_rerun_makes_zero_calls_and_zero_markdown_writes(tmp_path):
    vault, config = _vault(tmp_path)
    _run(vault, config, GroundedProvider(_producer))
    before = _snapshot(vault)
    provider = GroundedProvider(_producer)
    result = _run(vault, config, provider)
    assert result.errors == [] and result.compiled == 0 and result.skipped == 1
    assert provider.calls == []
    assert _snapshot(vault) == before


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient"])
def test_rejected_review_blocks_cache_state_and_notes_after_bounded_retry(tmp_path, verdict):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer, verdicts={"reduced routing errors by 12%.": verdict})
    result = _run(vault, config, provider)
    assert result.compiled == 0
    assert any("grounding review rejected" in error and verdict in error for error in result.errors)
    assert len(provider.producer_calls) == 2  # whole-source retry, then permanent failure
    assert load_synthesis(published_meta(config), "trial.md") is None
    assert "trial.md" not in read_state(published_meta(config) / "state.json").sources
    assert not (published_root(config) / "concepts" / "new-solver.md").exists()
    with published_registry(config) as registry:
        failures = registry.connection.execute(
            "SELECT error FROM dirty_queue WHERE entity_id='trial.md'"
        ).fetchall()
    assert failures and "grounding review rejected" in failures[0][0]


def test_reviewer_provider_failure_is_not_silently_accepted(tmp_path):
    vault, config = _vault(tmp_path)

    def broken_review(answer):
        raise ConnectionError("review endpoint unavailable")

    provider = GroundedProvider(_producer, review_hook=broken_review)
    result = _run(vault, config, provider)
    assert result.compiled == 0
    assert any("review endpoint unavailable" in error for error in result.errors)
    assert load_synthesis(published_meta(config), "trial.md") is None


def test_reviewer_missing_units_fail_closed(tmp_path):
    vault, config = _vault(tmp_path)

    def drop(answer):
        answer["reviews"].pop(next(iter(answer["reviews"])))
        return json.dumps(answer)

    result = _run(vault, config, GroundedProvider(_producer, review_hook=drop))
    assert result.compiled == 0
    assert any("missing units" in error for error in result.errors)


def test_two_pass_runs_grounded_draft_refinement_under_required_review(tmp_path):
    # Supersedes the former explicit refusal: two_pass is now span-grounded and
    # reviewed (see test_grounding_feedback_twopass.py for the full contract).
    vault, config = _vault(tmp_path, "SYNTHESIS_MODE=two_pass\n")
    provider = GroundedProvider(_producer)
    result = _run(vault, config, provider)
    assert result.errors == [] and result.compiled == 1
    assert [call["task"] for call in provider.calls] == ["ingest", "expand", "review"]
    assert load_synthesis(published_meta(config), "trial.md").review_receipt["status"] == "approved"


def test_explicit_off_mode_is_unchecked_and_cache_incompatible_with_required(tmp_path):
    vault, config = _vault(tmp_path, "GROUNDING_REVIEW=off\n")
    provider = GroundedProvider(_producer)
    assert _run(vault, config, provider).compiled == 1
    assert provider.review_calls == []
    assert (
        load_synthesis(published_meta(config), "trial.md").review_receipt["status"] == "unchecked"
    )

    (vault / ".env").write_text((vault / ".env").read_text().replace("=off", "=required"))
    required = load_config(env_file=str(vault / ".env"))
    provider = GroundedProvider(_producer)
    result = _run(vault, required, provider)
    assert result.compiled == 1 and len(provider.review_calls) == 1
    assert (
        load_synthesis(published_meta(required), "trial.md").review_receipt["status"] == "approved"
    )


def test_same_version_cache_without_receipt_is_not_reused(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    with patch(LLM, side_effect=provider.__call__):
        asyncio.run(seed_legacy_cache(config, _sources()))
    legacy_before = file_state(config.concepts_dir)
    cached = load_synthesis(published_meta(config), "trial.md")
    cached.review_receipt = None
    with VaultRegistry(config.llmwiki_dir) as registry:
        registry.update_synthesis("trial.md", cached, registry_versions(config))
    save_synthesis(cached, config.llmwiki_dir, "trial.md")
    provider = GroundedProvider(_producer)
    result = _run(vault, config, provider, refresh_render=True)
    assert result.compiled == 1 and len(provider.review_calls) == 1
    assert file_state(config.concepts_dir) == legacy_before


def registry_versions(config):
    from obsidian_llm_wiki.core.pipeline import _artifact_versions_for_config

    return _artifact_versions_for_config(config)


def test_legacy_unreviewed_cache_is_invalidated_by_version_upgrade(tmp_path):
    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    with patch(LLM, side_effect=provider.__call__):
        asyncio.run(seed_legacy_cache(config, _sources()))
    legacy_before = file_state(config.concepts_dir)
    with VaultRegistry(config.llmwiki_dir) as registry:
        registry.connection.execute(
            "UPDATE synthesis SET prompt_version='synthesis-v4:legacy', "
            "schema_version='synthesis-schema-v3'"
        )
        registry.connection.commit()
    provider = GroundedProvider(_producer)
    result = _run(vault, config, provider)
    assert result.compiled == 1 and len(provider.review_calls) == 1
    assert file_state(config.concepts_dir) == legacy_before


def test_interrupted_review_leaves_no_success_and_next_run_recovers(tmp_path):
    vault, config = _vault(tmp_path)

    def interrupt(answer):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(vault, config, GroundedProvider(_producer, review_hook=interrupt))
    assert load_synthesis(published_meta(config), "trial.md") is None
    assert "trial.md" not in read_state(published_meta(config) / "state.json").sources
    result = _run(vault, config, GroundedProvider(_producer))
    assert result.errors == [] and result.compiled == 1


CHUNKED = "\n\n".join(
    f"Section {n}: measurement {n} recorded {n * 3} faults in plant {n}." for n in range(1, 13)
)


def _chunk_producer(prompt: str) -> str:
    numbers = [
        int(text.split(":")[0].split()[-1])
        for _id, text in __import__("tests.new.grounding_doubles", fromlist=["x"]).offered_spans(
            prompt
        )
    ]
    first = numbers[0]
    return grounded_response(
        summary=f"Plant {first} recorded {first * 3} faults.",
        concepts=[
            {
                "title": f"Plant {first}",
                "slug": f"plant-{first}",
                "summary": f"Plant {first} recorded {first * 3} faults.",
                "sections": [{"heading": "Faults", "points": [f"{first * 3} faults."]}],
                "claims": [
                    {
                        "text": f"Plant {first} recorded {first * 3} faults.",
                        "evidence_span_ids": cite(prompt, f"Section {first}:"),
                    }
                ],
            }
        ],
    )


def test_chunked_source_reviews_every_window_before_merge(tmp_path):
    vault, config = _vault(tmp_path, "CHUNK_SIZE=200\n")
    provider = GroundedProvider(_chunk_producer)
    result = _run(vault, config, provider, content=CHUNKED)
    assert result.errors == [] and result.compiled == 1
    windows = len(provider.producer_calls)
    assert windows > 1 and len(provider.review_calls) == windows
    cached = load_synthesis(published_meta(config), "trial.md")
    assert len(cached.concepts) == windows
    for concept in cached.concepts:
        evidence = concept.claims[0].evidence
        assert CHUNKED[evidence.start_offset : evidence.end_offset] == evidence.quote


def test_chunk_citing_a_span_from_another_window_fails(tmp_path):
    vault, config = _vault(tmp_path, "CHUNK_SIZE=200\n")
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog

    foreign = build_span_catalog(CHUNKED, "trial.md").spans[-1].id

    def producer(prompt):
        data = json.loads(_chunk_producer(prompt))
        data["concepts"][0]["claims"][0]["evidence_span_range"] = {
            "start_id": foreign,
            "end_id": foreign,
        }
        return json.dumps(data)

    provider = GroundedProvider(producer)
    result = _run(vault, config, provider, content=CHUNKED)
    assert result.compiled == 0
    assert any("not offered" in error for error in result.errors)
    assert load_synthesis(published_meta(config), "trial.md") is None


def test_one_rejected_window_rejects_the_whole_source(tmp_path):
    vault, config = _vault(tmp_path, "CHUNK_SIZE=200\n")
    provider = GroundedProvider(_chunk_producer, verdicts={"Plant 1 recorded": "contradicted"})
    result = _run(vault, config, provider, content=CHUNKED)
    assert result.compiled == 0
    assert any("contradicted" in error for error in result.errors)
    assert load_synthesis(published_meta(config), "trial.md") is None


def test_unreviewed_resynthesis_overlay_is_not_rendered_under_required_review(tmp_path):
    from obsidian_llm_wiki.core.cache import save_resynthesis_overlay
    from obsidian_llm_wiki.core.models import BodySection, ConceptNote

    vault, config = _vault(tmp_path)
    provider = GroundedProvider(_producer)
    with patch(LLM, side_effect=provider.__call__):
        asyncio.run(seed_legacy_cache(config, _sources()))
    save_resynthesis_overlay(
        config.llmwiki_dir,
        {
            "new-solver": ConceptNote(
                title="New solver",
                slug="new-solver",
                summary="UNREVIEWED overlay text.",
                sections=[BodySection(heading="X", prose="UNREVIEWED overlay body.")],
            )
        },
    )
    legacy_before = file_state(config.llmwiki_dir)
    _run(vault, config, GroundedProvider(_producer), refresh_render=True)
    assert file_state(config.llmwiki_dir) == legacy_before
    note = (published_root(config) / "concepts" / "new-solver.md").read_text()
    assert "UNREVIEWED" not in note


def test_manual_recompile_path_uses_the_same_review_gate(tmp_path):
    from obsidian_llm_wiki.core.pipeline import recompile_single_source

    vault, config = _vault(tmp_path)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "trial.md").write_text(SOURCE)
    provider = GroundedProvider(
        _producer,
        verdicts={"reduced routing errors by 12%.": "insufficient"},
    )
    # The complete pipeline returns a visible failed CompileResult; the CLI
    # maps it to exit 1. Rejection still cannot admit cache/state or pages.
    with patch(LLM, side_effect=provider.__call__):
        result = asyncio.run(recompile_single_source(vault, "trial.md", config))
    assert result.compiled == 0
    assert any("insufficient" in error for error in result.errors)
    assert not (published_root(config) / "concepts" / "new-solver.md").exists()
    assert provider.review_calls
    assert load_synthesis(published_meta(config), "trial.md") is None
    assert "trial.md" not in read_state(published_meta(config) / "state.json").sources


BILINGUAL = (
    "新路由器将丢包率降低了百分之十二。路由器依赖调度器分配带宽。\n\n"
    "The scheduler allocates bandwidth for the router.\n"
)


def _bilingual_producer(prompt: str) -> str:
    return grounded_response(
        summary="新路由器降低了丢包率，并依赖调度器。",
        concepts=[
            {
                "title": "新路由器",
                "slug": "xin-lu-you-qi",
                "summary": "新路由器将丢包率降低了百分之十二。",
                "sections": [{"heading": "效果", "points": ["丢包率降低了百分之十二。"]}],
                "claims": [
                    {
                        "text": "新路由器将丢包率降低了百分之十二。",
                        "evidence_span_ids": cite(prompt, "降低了百分之十二"),
                    }
                ],
                "related": [{"slug": "scheduler", "relation": "depends_on"}],
            },
            {
                "title": "Scheduler",
                "slug": "scheduler",
                "summary": "The scheduler allocates bandwidth for the router.",
                "sections": [{"heading": "Role", "prose": "It allocates bandwidth."}],
            },
        ],
        maps=[
            {
                "title": "路由",
                "slug": "routing",
                "summary": "路由器依赖调度器分配带宽。",
                "concept_slugs": ["xin-lu-you-qi", "scheduler"],
            }
        ],
        language="zh",
    )


def test_relations_maps_and_bilingual_identity_normalization_stay_no_op(tmp_path):
    vault, config = _vault(tmp_path)
    first = GroundedProvider(_bilingual_producer)
    result = _run(vault, config, first, content=BILINGUAL)
    assert result.errors == [] and result.compiled == 1
    kinds = {
        unit["kind"]
        for unit in load_synthesis(published_meta(config), "trial.md")
        .review_receipt["units"]
        .values()
    }
    assert {"relation", "map_summary", "claim", "body_prose"} <= kinds
    before = _snapshot(vault)
    again = GroundedProvider(_bilingual_producer)
    result = _run(vault, config, again, content=BILINGUAL)
    assert result.errors == [] and result.compiled == 0
    assert again.calls == [] and _snapshot(vault) == before
