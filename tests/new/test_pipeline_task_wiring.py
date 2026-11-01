"""Integration coverage for production schema policy and ingest model wiring."""

from __future__ import annotations

import json

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.schema import SchemaPolicy, load_schema_policy
from obsidian_llm_wiki.providers import llm
from tests.new.grounding_doubles import legacy_fixture_llm
from tests.new.immutable_lifecycle_helpers import assert_private_policy_load
from tests.new.immutable_semantic_helpers import assert_preserved, file_state


def _single_response() -> str:
    return json.dumps(
        {
            "source_title": "Source",
            "source_summary": "Chunking bounds request size.",
            "concepts": [
                {
                    "title": "Chunking", "slug": "concept",
                    "summary": "Chunking bounds request size.",
                    "sections": [{"heading": "Evidence", "prose": "Chunking bounds request size."}],
                }
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
    policy_path = tmp_path / ".llmwiki/schema.yaml"
    policy_path.parent.mkdir()
    policy_path.write_text("required_sections: [Evidence]\ninstructions: [Prefer mechanisms.]\n")
    before = file_state(policy_path.parent)
    loads: list[object] = []
    calls: list[tuple[str, dict]] = []

    def load_policy(vault):
        loads.append(vault)
        loaded = load_schema_policy(vault)
        assert loaded == policy
        return loaded

    async def fake_acall(prompt, _messages, _config, **kwargs):
        calls.append((prompt, kwargs))
        return _single_response()

    monkeypatch.setattr(pipeline, "load_schema_policy", load_policy, raising=False)
    monkeypatch.setattr(llm, "acall_llm", legacy_fixture_llm(fake_acall))
    config = Config(vault_path=str(tmp_path), min_source_chars=1, retry_count=1)
    sources = {
        "social.md": SourceDoc(
            title="Social", content="Chunking bounds request size. " * 800, source_type="tweet",
        ),
        "paper.md": SourceDoc(
            title="Paper", content="Chunking bounds request size. " * 200, source_type="paper",
        ),
    }

    result = await pipeline.run_pipeline(tmp_path, sources, config, force=True)

    assert result.errors == []
    assert_private_policy_load(config, loads)
    assert_preserved(policy_path.parent, before)
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
    policy_path = tmp_path / ".llmwiki/schema.yaml"
    policy_path.parent.mkdir()
    policy_path.write_text("required_sections: [Evidence]\ninstructions: [Prefer mechanisms.]\n")
    before = file_state(policy_path.parent)
    loads: list[object] = []
    calls: list[tuple[str, dict]] = []
    responses = iter([_skeleton_response(), _expanded_response()])
    models = []
    real_acall = llm.acall_llm

    class ScriptedClient:
        def chat(self, _system, _user, **kwargs):
            models.append(kwargs["model"])
            response = next(responses, None)
            assert response is not None, "unexpected additional model request"
            return response

    monkeypatch.setattr(llm, "create_llm_client", lambda _config: ScriptedClient())

    def load_policy(vault):
        loads.append(vault)
        loaded = load_schema_policy(vault)
        assert loaded == policy
        return loaded

    async def fake_acall(prompt, _messages, _config, **kwargs):
        calls.append((prompt, kwargs))
        return await real_acall(prompt, _messages, _config, **kwargs)

    monkeypatch.setattr(pipeline, "load_schema_policy", load_policy, raising=False)
    monkeypatch.setattr(llm, "acall_llm", fake_acall)
    config = Config(
        vault_path=str(tmp_path),
        min_source_chars=1,
        retry_count=1,
        synthesis_mode="two_pass",
        # Two-pass has no span-ID anchoring yet: explicit unchecked mode only.
        grounding_review="off",
    )
    config.llm.model = "default-not-ingest"
    config.llm.ingest_model = "fixture-ingest"
    # Standard two-pass mode: expand falls back to default, not ingest.
    config.llm.expand_model = None
    sources = {"social.md": SourceDoc(title="Social", content="x" * 4_001, source_type="tweet")}

    result = await pipeline.run_pipeline(tmp_path, sources, config, force=True)

    assert result.errors == []
    assert_private_policy_load(config, loads)
    assert_preserved(policy_path.parent, before)
    assert len(calls) == 2
    assert all(kwargs["task"] in ("ingest", "expand") for _prompt, kwargs in calls)
    assert all("Required concept sections: Evidence" in prompt for prompt, _ in calls)
    assert all("Requested synthesis granularity: concise" in prompt for prompt, _ in calls)
    assert [kwargs["task"] for _, kwargs in calls] == ["ingest", "expand"]
    assert models == ["fixture-ingest", "default-not-ingest"]
