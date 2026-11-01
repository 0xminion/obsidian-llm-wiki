"""Single-source retry must validate the complete inventory before reading."""
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.ingest import sources


@pytest.mark.asyncio
async def test_retry_checks_all_paths_before_reading_requested_body(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path / "vault"), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "alpha.md").write_text("Valid source evidence. " * 20)
    outside = tmp_path / "outside.md"
    outside.write_text("Outside evidence must not be read.")
    (config.sources_dir / "z-escape.md").symlink_to(outside)
    reads = []
    original = sources.load_source_file

    def observed(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(sources, "load_source_file", observed)
    publish = AsyncMock()
    monkeypatch.setattr(pipeline, "run_pipeline", publish)
    try:
        result = await pipeline.recompile_single_source(config.vault, "alpha.md", config)
    except ValueError:
        # The read-before-validation invariant matters independently of how
        # the CLI presents the containment error.
        result = None
    assert reads == []
    publish.assert_not_called()
    if result is not None:
        assert result.errors
