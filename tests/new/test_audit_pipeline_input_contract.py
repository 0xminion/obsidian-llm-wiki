"""Pipeline acceptance boundaries apply equally to files and URL ingestion."""
from unittest.mock import AsyncMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance
from obsidian_llm_wiki.core.pipeline import _synthesize_source, run_pipeline
from tests.new.test_incremental_cache import FAKE_SYNTH_A


@pytest.mark.asyncio
async def test_explicit_translation_overrides_detected_language(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), output_language="zh", embeddings_enabled=False)
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    synthesis = await _synthesize_source(
        config, "alpha.md", SourceDoc("Alpha", "This experiment measured cell growth." * 8), []
    )
    assert synthesis.language == "zh"
    assert "Chinese" in llm.call_args.args[0] or "中文" in llm.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("known_truncated", [False, True])
async def test_truncated_source_never_publishes_complete_synthesis(
    tmp_path, monkeypatch, known_truncated
):
    config = Config(vault_path=str(tmp_path), max_source_chars=200, embeddings_enabled=False)
    llm = AsyncMock(return_value=FAKE_SYNTH_A)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    source = SourceDoc(
        "Alpha", "A measured source observation. " * (4 if known_truncated else 20),
        provenance=SourceProvenance(
            diagnostics=("content truncated to safety cap",) if known_truncated else ()
        ),
    )
    result = await run_pipeline(tmp_path, {"alpha.md": source}, config)
    assert result.compiled == 0
    assert any("incomplete" in error for error in result.errors)
    llm.assert_not_called()


@pytest.mark.asyncio
async def test_generic_synthesis_is_retried_not_published(tmp_path, monkeypatch):
    import json

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    payload = json.dumps({
        "source_title": "Alpha", "source_summary": "TODO", "concepts": [], "maps": [],
    })
    llm = AsyncMock(return_value=payload)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    result = await run_pipeline(
        tmp_path, {"alpha.md": SourceDoc("Alpha", "A measured source observation. " * 8)}, config
    )
    assert result.compiled == 0
    assert any("placeholder" in error for error in result.errors)
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_unmatched_quote_is_retried_before_success(tmp_path, monkeypatch):
    import json

    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    source = SourceDoc(
        "Alpha", "Treatment increased growth to fifteen units. Controls stayed at ten."
    )
    payload = {
        "source_title": "Alpha", "source_summary": "Treatment increased growth.",
        "concepts": [{
            "title": "Growth", "slug": "growth", "summary": "Treatment changed growth.",
            "sections": [{"heading": "Evidence", "prose": "Controls stayed at ten."}],
            "claims": [{"text": "Treatment increased growth.", "quote": "Invented quotation."}],
        }], "maps": [],
    }
    rejected = json.dumps(payload)
    payload["concepts"][0]["claims"][0]["quote"] = "Treatment increased growth to fifteen units."
    llm = AsyncMock(side_effect=[rejected, json.dumps(payload)])
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", llm)
    result = await run_pipeline(tmp_path, {"alpha.md": source}, config)
    assert result.compiled == 1
    assert not result.errors
    assert llm.call_count == 2
    assert result.concepts[0].claims[0].evidence.verification.value == "verified"
