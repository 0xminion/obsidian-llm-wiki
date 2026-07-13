"""Integration coverage for production schema policy and ingest model wiring."""

from __future__ import annotations

import json

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.schema import SchemaPolicy
from obsidian_llm_wiki.providers import llm


def _single_response() -> str:
    return json.dumps(
        {
            "source_title": "Source",
            "source_summary": "Summary",
            "concepts": [
                {"title": "Concept", "slug": "concept", "summary": "Summary"}
            ],
        }
    )


def _skeleton_response() -> str:
    return json.dumps(
        {
            "source_title": "Source",
            "source_summary": "Summary",
            "concepts": [
                {
                    "title": "Concept",
                    "slug": "concept",
                    "summary": "Summary",
                    "rationale": "This matters because the source explains it.",
                }
            ],
        }
    )


def _expanded_response() -> str:
    return json.dumps(
        {
            "title": "Concept",
            "slug": "concept",
            "summary": "Expanded summary",
            "sections": [{"heading": "Evidence", "points": ["Source-backed detail."]}],
        }
    )


@pytest.mark.asyncio
async def test_single_pass_loads_policy_once_and_applies_per_source_granularity(
    tmp_path, monkeypatch
):
    """Single-pass ingest sends shared policy plus source-specific detail to each call."""
    policy = SchemaPolicy(
        required_sections=("Evidence",),
        instructions=("Prefer mechanisms.",),
    )
    loads: list[object] = []
    calls: list[tuple[str, dict]] = []

    def load_policy(vault):
        loads.append(vault)
        return policy

    async def fake_acall(prompt, _messages, _config, **kwargs):
        calls.append((prompt, kwargs))
        return _single_response()

    monkeypatch.setattr(pipeline, "load_schema_policy", load_policy, raising=False)
    monkeypatch.setattr(llm, "acall_llm", fake_acall)
    config = Config(vault_path=str(tmp_path), min_source_chars=1, retry_count=1)
    sources = {
        "social.md": SourceDoc(title="Social", content="x" * 20_001, source_type="tweet"),
        "paper.md": SourceDoc(title="Paper", content="x" * 4_001, source_type="paper"),
    }

    result = await pipeline.run_pipeline(tmp_path, sources, config, force=True)

    assert result.errors == []
    assert loads == [tmp_path.resolve()]
    assert len(calls) == 2
    assert all(kwargs["task"] in ("ingest", "expand") for _prompt, kwargs in calls)
    prompts = {
        "concise" if "Requested synthesis granularity: concise" in prompt else "detailed": prompt
        for prompt, _ in calls
    }
    assert set(prompts) == {"concise", "detailed"}
    assert all("Required concept sections: Evidence" in prompt for prompt, _ in calls)
    assert all("Preference: Prefer mechanisms." in prompt for prompt, _ in calls)


@pytest.mark.asyncio
async def test_two_pass_applies_policy_granularity_and_ingest_model_to_every_llm_call(
    tmp_path, monkeypatch
):
    """Two-pass extraction and expansion both retain the production policy context."""
    policy = SchemaPolicy(
        required_sections=("Evidence",),
        instructions=("Prefer mechanisms.",),
    )
    loads: list[object] = []
    calls: list[tuple[str, dict]] = []
    responses = iter([_skeleton_response(), _expanded_response()])

    def load_policy(vault):
        loads.append(vault)
        return policy

    async def fake_acall(prompt, _messages, _config, **kwargs):
        calls.append((prompt, kwargs))
        return next(responses)

    monkeypatch.setattr(pipeline, "load_schema_policy", load_policy, raising=False)
    monkeypatch.setattr(llm, "acall_llm", fake_acall)
    config = Config(
        vault_path=str(tmp_path),
        min_source_chars=1,
        retry_count=1,
        synthesis_mode="two_pass",
    )
    sources = {"social.md": SourceDoc(title="Social", content="x" * 4_001, source_type="tweet")}

    result = await pipeline.run_pipeline(tmp_path, sources, config, force=True)

    assert result.errors == []
    assert loads == [tmp_path.resolve()]
    assert len(calls) == 2
    assert all(kwargs["task"] in ("ingest", "expand") for _prompt, kwargs in calls)
    assert all("Required concept sections: Evidence" in prompt for prompt, _ in calls)
    assert all("Requested synthesis granularity: concise" in prompt for prompt, _ in calls)
