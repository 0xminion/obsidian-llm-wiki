"""End-to-end lifecycle coverage for changed-source contradiction review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.contradictions import ContradictionStore
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.providers import llm

runner = CliRunner()


def _response(claim: str) -> str:
    return json.dumps(
        {
            "source_title": "Deployment guidance",
            "source_summary": "A source about deployment timing.",
            "concepts": [
                {
                    "title": "Deployment window",
                    "slug": "deployment-window",
                    "summary": "Guidance for a deployment window.",
                    "claims": [
                        {
                            "text": claim,
                            "concept_slug": "deployment-window",
                            "source_ref": "section-1",
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_changed_source_creates_reviewable_record_visible_to_health_and_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed keyed claim is durable evidence, not an automatic resolution."""
    vault = tmp_path / "vault"
    config = Config(vault_path=str(vault), min_source_chars=1, retry_count=1)
    source_name = "deployment.md"

    async def synthesize_initial(*_args, **_kwargs) -> str:
        return _response("Deploy on Friday.")

    monkeypatch.setattr(llm, "acall_llm", synthesize_initial)
    initial = await run_pipeline(
        vault,
        {source_name: SourceDoc(title="Deployment", content="Original source.")},
        config,
    )
    assert initial.errors == []

    async def synthesize_changed(*_args, **_kwargs) -> str:
        return _response("Do not deploy on Friday.")

    monkeypatch.setattr(llm, "acall_llm", synthesize_changed)
    changed = await run_pipeline(
        vault,
        {source_name: SourceDoc(title="Deployment", content="Revised source.")},
        config,
    )
    assert changed.errors == []

    store_path = vault / "04-Wiki" / ".llmwiki" / "contradictions.json"
    store = ContradictionStore(store_path)
    records = store.records()
    assert len(records) == 1
    record = records[0]
    assert record.status == "detected"
    assert record.sources[0].source_path == source_name
    assert {revision.content_hash for revision in record.sources} == {
        store.source_revisions(source_name)[0].content_hash,
        store.source_revisions(source_name)[1].content_hash,
    }
    assert record.evidence == ("previous: Deploy on Friday.", "current: Do not deploy on Friday.")

    health = runner.invoke(app, ["health", str(vault), "--json"])
    assert health.exit_code == 0, health.output
    health_payload = json.loads(health.output)
    assert health_payload["contradictions"] == [
        {
            "evidence": ["previous: Deploy on Friday.", "current: Do not deploy on Friday."],
            "id": record.id,
            "sources": [
                {
                    "content_hash": record.sources[0].content_hash,
                    "revision": record.sources[0].revision,
                    "source_path": source_name,
                },
                {
                    "content_hash": record.sources[1].content_hash,
                    "revision": record.sources[1].revision,
                    "source_path": source_name,
                },
            ],
            "status": "detected",
            "summary": record.summary,
        }
    ]
    assert health_payload["contradiction_summary"] == {"detected": 1}

    before_fix = store_path.read_text(encoding="utf-8")
    dry_run = runner.invoke(app, ["fix", str(vault), "--dry-run", "--json"])
    assert dry_run.exit_code == 0, dry_run.output
    fix_payload = json.loads(dry_run.output)
    assert fix_payload["contradiction_review_actions"] == [
        {
            "action": "review_contradiction",
            "path": ".llmwiki/contradictions.json",
            "payload": {"record_id": record.id, "status": "detected"},
            "requires_review": True,
        }
    ]
    assert store_path.read_text(encoding="utf-8") == before_fix
    assert ContradictionStore(store_path).get(record.id).status == "detected"
