"""Critical regression test — multi-source cache reuse + orphan on delete.

This test proves the core architectural fix: ingesting a second source
preserves the first source's concepts (via synthesis cache), and deleting
a source orphans its exclusively-owned concepts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.config import load_config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from obsidian_llm_wiki.render.obsidian import parse_frontmatter, safe_read_file

# ── Fake LLM responses ──────────────────────────────────────────────────

FAKE_SYNTH_A = json.dumps({
    "source_title": "Source Alpha",
    "source_summary": "Alpha source about concepts A1 and A2.",
    "source_tags": ["alpha"],
    "concepts": [
        {"title": "Concept A1", "slug": "concept-a1", "summary": "A1 summary.",
         "sections": [{"heading": "Core", "points": ["A1 detail one", "A1 detail two"]}]},
        {"title": "Concept Shared", "slug": "concept-shared", "summary": "Shared concept.",
         "sections": [{"heading": "Core", "points": ["Shared from A"]}]},
    ],
    "maps": [],
})

FAKE_SYNTH_B = json.dumps({
    "source_title": "Source Beta",
    "source_summary": "Beta source about concepts B1 and the shared concept.",
    "source_tags": ["beta"],
    "concepts": [
        {"title": "Concept B1", "slug": "concept-b1", "summary": "B1 summary.",
         "sections": [{"heading": "Core", "points": ["B1 detail one", "B1 detail two"]}]},
        {"title": "Concept Shared", "slug": "concept-shared", "summary": "Shared concept.",
         "sections": [{"heading": "Core", "points": ["Shared from B"]}]},
    ],
    "maps": [],
})


@pytest.mark.asyncio
async def test_incremental_cache_preserves_existing_concepts(tmp_path: Path):
    """Second ingest preserves first source's concepts via cache.

    1. Ingest source A → concepts A1, Shared rendered.
    2. Ingest source B (A still on disk) → concepts A1, B1, Shared rendered.
       Source A is NOT re-synthesised (cache hit).
    """
    vault = tmp_path / "TestVault"
    vault.mkdir()
    (vault / ".env").write_text(f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=test\n")

    config = load_config(env_file=str(vault / ".env"))

    source_a_content = "This is source alpha with sufficient content for the length gate. " * 3
    source_b_content = "This is source beta with sufficient content for the length gate. " * 3

    # ── Run 1: ingest source A only ────────────────────────────────────
    sources_a = {"source-alpha.md": SourceDoc(title="Source Alpha", content=source_a_content)}

    call_count = 0

    async def _mock_acall_a(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FAKE_SYNTH_A

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall_a,
    ):
        result1 = await run_pipeline(vault, sources_a, config, force=True)

    assert result1.compiled == 1
    assert len(result1.concepts) == 2  # A1 + Shared
    assert call_count == 1

    # Verify A1 is in the vault.
    assert (config.concepts_dir / "concept-a1.md").exists()
    assert (config.concepts_dir / "concept-shared.md").exists()

    # ── Run 2: ingest source B (source A still present) ────────────────
    # The pipeline should reuse A's cached synthesis and only call the LLM for B.
    sources_ab = {
        "source-alpha.md": SourceDoc(title="Source Alpha", content=source_a_content),
        "source-beta.md": SourceDoc(title="Source Beta", content=source_b_content),
    }

    async def _mock_acall_b(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FAKE_SYNTH_B

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall_b,
    ):
        result2 = await run_pipeline(vault, sources_ab, config, force=False)

    # Source A should be skipped (cache hit), B should be compiled.
    assert result2.compiled == 1
    assert result2.skipped == 1
    assert call_count == 2  # Only 2 total LLM calls (A in run 1, B in run 2)

    # Both A's and B's concepts should be in the vault.
    assert (config.concepts_dir / "concept-a1.md").exists()
    assert (config.concepts_dir / "concept-b1.md").exists()
    assert (config.concepts_dir / "concept-shared.md").exists()

    # The rendered bundle should have all 3 unique concepts.
    assert len(result2.concepts) == 3  # A1, B1, Shared (merged)


@pytest.mark.asyncio
async def test_delete_source_orphans_exclusive_concepts(tmp_path: Path):
    """Deleting a source orphans its exclusively-owned concepts.

    1. Ingest sources A + B (sharing 'concept-shared').
    2. Run with only B (A deleted from corpus).
    3. A's exclusive concept (A1) should be orphaned.
    4. Shared concept should NOT be orphaned.
    """
    vault = tmp_path / "TestVault2"
    vault.mkdir()
    (vault / ".env").write_text(f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=test\n")

    config = load_config(env_file=str(vault / ".env"))

    source_a_content = "Alpha source content long enough to pass the gate. " * 3
    source_b_content = "Beta source content long enough to pass the gate. " * 3

    # ── Run 1: ingest both A and B ─────────────────────────────────────
    sources_ab = {
        "source-alpha.md": SourceDoc(title="Source Alpha", content=source_a_content),
        "source-beta.md": SourceDoc(title="Source Beta", content=source_b_content),
    }

    responses = [FAKE_SYNTH_A, FAKE_SYNTH_B]
    idx = 0

    async def _mock_acall(*args, **kwargs):
        nonlocal idx
        resp = responses[idx]
        idx += 1
        return resp

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=_mock_acall,
    ):
        result1 = await run_pipeline(vault, sources_ab, config, force=True)

    assert result1.compiled == 2
    assert (config.concepts_dir / "concept-a1.md").exists()
    assert (config.concepts_dir / "concept-b1.md").exists()
    assert (config.concepts_dir / "concept-shared.md").exists()

    # ── Run 2: only B (A deleted) ──────────────────────────────────────
    sources_b_only = {
        "source-beta.md": SourceDoc(title="Source Beta", content=source_b_content),
    }

    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value=FAKE_SYNTH_B,
    ):
        result2 = await run_pipeline(vault, sources_b_only, config, force=False)

    # A should be detected as deleted.
    assert result2.deleted == 1

    # A's exclusive concept (A1) should be orphaned.
    a1_raw = safe_read_file(config.concepts_dir / "concept-a1.md")
    a1_meta, _ = parse_frontmatter(a1_raw)
    assert a1_meta.get("orphaned") is True

    # Shared concept should NOT be orphaned (B still owns it).
    shared_raw = safe_read_file(config.concepts_dir / "concept-shared.md")
    shared_meta, _ = parse_frontmatter(shared_raw)
    assert "orphaned" not in shared_meta

    # B's exclusive concept (B1) should still be alive.
    b1_raw = safe_read_file(config.concepts_dir / "concept-b1.md")
    b1_meta, _ = parse_frontmatter(b1_raw)
    assert "orphaned" not in b1_meta
