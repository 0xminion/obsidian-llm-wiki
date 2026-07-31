from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.core.registry import VaultRegistry


@pytest.mark.asyncio
async def test_synthesis_failures_persist_retry_diagnostics(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "Vault"
    vault.mkdir()
    config = Config(vault_path=str(vault), embeddings_enabled=False)
    monkeypatch.setattr(
        "obsidian_llm_wiki.core.pipeline._synthesize_with_retry",
        AsyncMock(return_value=None),
    )

    result = await run_pipeline(
        vault,
        {
            "failed.md": SourceDoc(
                "Failed Source",
                "Enough source content for synthesis to run and then fail. " * 3,
            )
        },
        config,
        force=True,
    )

    with VaultRegistry(config.llmwiki_dir) as registry:
        retries = [
            record
            for record in registry.list_dirty(state="pending")
            if record.entity_type == "source"
            and record.entity_id == "failed.md"
            and record.reason == "synthesis_retry"
        ]

    assert result.compiled == 0
    assert len(retries) == 1
    assert retries[0].attempts == 1
    assert "permanent failure" in retries[0].error
