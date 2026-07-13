"""Golden end-to-end test — full pipeline with mocked LLM.

This is the test the old repo was missing.  It feeds a fake source through
the complete pipeline (synthesis → merge → render) with a mocked LLM
response, and asserts the exact output structure.  This proves the product
works as intended — not just that modules pass their own unit tests.
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

# ── Fake LLM response matching the SynthesisBundle schema ───────────────

FAKE_LLM_RESPONSE = json.dumps({
    "source_title": "Attention Is All You Need",
    "source_summary": "Introduces the Transformer architecture, replacing recurrence with self-attention.",  # noqa: E501
    "source_tags": ["deep-learning", "attention", "transformers"],
    "key_points": [
        "Self-attention mechanism eliminates recurrence entirely",
        "Parallelizable training leads to significant speedup",
        "Achieves state-of-the-art on translation tasks",
    ],
    "open_questions": [
        "How does attention scale to very long sequences?",
    ],
    "language": "en",
    "concepts": [
        {
            "title": "Self-Attention",
            "slug": "self-attention",
            "summary": "Attention mechanism relating different positions of a single sequence.",
            "tags": ["attention", "neural-network"],
            "aliases": ["scaled dot-product attention"],
            "sections": [
                {
                    "heading": "Core concept",
                    "points": [
                        "Computes attention weights using query, key, value vectors",
                        "Scaled by 1/sqrt(d_k) for stable gradients",
                    ],
                },
                {
                    "heading": "Context",
                    "prose": "Self-attention allows each position to attend to all positions in the sequence, enabling long-range dependencies without recurrence.",  # noqa: E501
                },
            ],
            "claims": [
                {"text": "Self-attention complexity is O(n^2) in sequence length", "source_ref": "section 3.2"},  # noqa: E501
            ],
            "related": [
                {"slug": "multi-head-attention", "relation": "component_of", "display": "Multi-Head Attention"},  # noqa: E501
                {"slug": "positional-encoding", "relation": "depends_on"},
            ],
            "confidence": 0.95,
            "provenance": "extracted",
            "is_new": True,
        },
        {
            "title": "Multi-Head Attention",
            "slug": "multi-head-attention",
            "summary": "Runs multiple attention mechanisms in parallel for richer representations.",
            "tags": ["attention", "neural-network"],
            "sections": [
                {
                    "heading": "Core concept",
                    "points": [
                        "Multiple attention heads capture different relationship types",
                        "Outputs concatenated and linearly projected",
                    ],
                },
            ],
            "related": [
                {"slug": "self-attention", "relation": "variant_of"},
            ],
            "confidence": 0.9,
            "provenance": "extracted",
            "is_new": True,
        },
        {
            "title": "Positional Encoding",
            "slug": "positional-encoding",
            "summary": "Injects position information into the input embeddings using sinusoidal functions.",  # noqa: E501
            "tags": ["embeddings", "transformers"],
            "sections": [
                {
                    "heading": "Core concept",
                    "points": [
                        "Uses sine and cosine functions of different frequencies",
                        "Allows model to learn relative positions",
                    ],
                },
            ],
            "confidence": 0.85,
            "provenance": "extracted",
            "is_new": True,
        },
    ],
    "maps": [
        {
            "title": "Attention Mechanisms",
            "slug": "attention-mechanisms",
            "summary": "Overview of attention-based mechanisms in the Transformer architecture.",
            "tags": ["attention"],
            "concept_slugs": ["self-attention", "multi-head-attention"],
        },
    ],
})


# ── Golden test ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_golden_pipeline_end_to_end(tmp_path: Path):
    """Full pipeline: source → LLM synthesis → merge → render → vault files.

    This is the single test that proves the product works.  If this passes,
    the core flow (ingest → synthesise → render) is correct.
    """
    # ── Setup: create vault + .env ─────────────────────────────────────
    vault = tmp_path / "TestVault"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\n"
        f"LLM_PROVIDER=ollama\n"
        f"LLM_HOST=http://localhost:99999\n"  # won't be called — mocked
        f"LLM_MODEL=test-model\n"
    )

    config = load_config(env_file=str(vault / ".env"))

    # ── Source ──────────────────────────────────────────────────────────
    sources = {
        "attention-is-all-you-need.md": SourceDoc(
            title="Attention Is All You Need",
            content="We propose a new simple network architecture, the Transformer, "
                    "based solely on attention mechanisms, dispensing with recurrence "
                    "and convolutions entirely.",
            url="https://arxiv.org/abs/1706.03762",
        ),
    }

    # ── Mock the LLM call ───────────────────────────────────────────────
    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value=FAKE_LLM_RESPONSE,
    ):
        result = await run_pipeline(vault, sources, config, force=True)

    # ── Assert pipeline result ──────────────────────────────────────────
    assert result.compiled == 1
    assert len(result.concepts) == 3
    assert len(result.errors) == 0

    # ── Assert vault structure ──────────────────────────────────────────
    bundle = vault / "04-Wiki"
    assert (bundle / "sources" / "attention-is-all-you-need.md").exists()
    assert (bundle / "entries" / "attention-is-all-you-need.md").exists()
    assert (bundle / "concepts" / "self-attention.md").exists()
    assert (bundle / "concepts" / "multi-head-attention.md").exists()
    assert (bundle / "concepts" / "positional-encoding.md").exists()
    assert (bundle / "mocs" / "attention-mechanisms.md").exists()
    assert (bundle / "concepts" / "index.md").exists()
    assert (bundle / "index.md").exists()

    # ── Assert concept page content ─────────────────────────────────────
    sa_page = safe_read_file(bundle / "concepts" / "self-attention.md")
    meta, body = parse_frontmatter(sa_page)

    assert meta["type"] == "Concept"
    assert meta["title"] == "Self-Attention"
    assert "attention" in meta["tags"]
    assert "neural-network" in meta["tags"]
    assert meta["aliases"] == ["scaled dot-product attention"]
    assert {relation["target"] for relation in meta["relations"]} == {
        "multi-head-attention",
        "positional-encoding",
    }

    assert "# Self-Attention" in body
    assert "## Core concept" in body
    assert "- Computes attention weights using query, key, value vectors" in body
    assert "## Context" in body
    assert "Self-attention allows each position" in body
    assert "## Claims" in body
    assert "Self-attention complexity is O(n^2)" in body
    assert "## Related Concepts" not in body
    assert "## Cross-References / 关联图谱" in body
    assert "[[multi-head-attention|" in body
    assert "[[positional-encoding|" in body

    # ── Assert entry page content ───────────────────────────────────────
    entry_page = safe_read_file(bundle / "entries" / "attention-is-all-you-need.md")
    meta, body = parse_frontmatter(entry_page)

    assert meta["type"] == "Entry"
    assert meta["title"] == "Attention Is All You Need"
    assert "deep-learning" in meta["tags"]

    assert "## Key Findings" in body
    assert "Self-attention mechanism eliminates recurrence" in body
    assert "## Linked Concepts" in body
    assert "[[self-attention]]" in body
    assert "[[multi-head-attention]]" in body
    assert "## Open Questions" in body

    # ── Assert MOC page content ─────────────────────────────────────────
    moc_page = safe_read_file(bundle / "mocs" / "attention-mechanisms.md")
    meta, body = parse_frontmatter(moc_page)

    assert meta["type"] == "Map of Content"
    assert "# Attention Mechanisms" in body
    assert "## Concepts" in body
    assert "[[self-attention]]" in body
    assert "[[multi-head-attention]]" in body

    # ── Assert state was persisted ──────────────────────────────────────
    import json as _json
    state_data = _json.loads(safe_read_file(config.state_file))
    assert "attention-is-all-you-need.md" in state_data["sources"]
    src_state = state_data["sources"]["attention-is-all-you-need.md"]
    assert len(src_state["concepts"]) == 3
    assert "self-attention" in src_state["concepts"]


@pytest.mark.asyncio
async def test_golden_pipeline_incremental_skip(tmp_path: Path):
    """Second run with unchanged source should skip synthesis."""
    vault = tmp_path / "TestVault2"
    vault.mkdir()
    (vault / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_PROVIDER=ollama\nLLM_MODEL=test\n"
    )
    config = load_config(env_file=str(vault / ".env"))

    sources = {
        "test.md": SourceDoc(
            title="Test",
            content="This is a test document with enough content to pass the length gate. "
            * 3,
        ),
    }

    # First run.
    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        return_value=json.dumps({
            "source_title": "Test",
            "source_summary": "Summary",
            "concepts": [{"title": "C", "slug": "c", "summary": "S"}],
        }),
    ):
        result1 = await run_pipeline(vault, sources, config, force=True)
    assert result1.compiled == 1

    # Second run — should skip (source unchanged).
    call_count = 0

    async def _mock_acall(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return json.dumps({"source_title": "Test", "source_summary": "S"})

    with patch("obsidian_llm_wiki.providers.llm.acall_llm", side_effect=_mock_acall):
        result2 = await run_pipeline(vault, sources, config, force=False)

    assert result2.compiled == 0
    assert result2.skipped == 1
    assert call_count == 0  # LLM was NOT called on second run
