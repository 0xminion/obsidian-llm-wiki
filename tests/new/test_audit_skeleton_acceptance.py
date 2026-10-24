"""Every two-pass chunk must have a valid summary before first-summary merge."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.synth.quality import chunk_content, quality_synthesize_source
from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

FIRST = "Queue leases limit task ownership. An expired lease permits another worker to retry."
SECOND = "Retry backoff delays repeated failures so workers do not overwhelm the provider."


def _source_and_config():
    source = SourceDoc("Worker recovery", FIRST + "\n\n" + SECOND)
    config = Config(chunk_size=len(FIRST), compile_concurrency=1)
    assert chunk_content(source.content, config.chunk_size) == [FIRST, SECOND]
    return source, config


def _skeleton(summary, concepts=()):
    return json.dumps({"source_summary": summary, "concepts": list(concepts)})


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [
    "TODO",
    "This document provides useful information.",
    "本文介绍了相关内容。",
    "I cannot access the article. Please provide the full text so I can summarize it.",
])
async def test_later_semantically_failed_skeleton_chunk_cannot_hide_in_merge(summary):
    source, config = _source_and_config()
    with (
        patch("obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock,
              side_effect=[_skeleton(FIRST), _skeleton(summary)]) as llm,
        pytest.raises(RuntimeError, match=r"chunk 2/2: source_summary"),
    ):
        await quality_synthesize_source(config, "workers.md", source, [])
    assert llm.await_count == 2


@pytest.mark.asyncio
async def test_bodyless_skeletons_are_valid_and_receive_real_expansion():
    source, config = _source_and_config()
    concept = {"title": "Queue leases", "slug": "queue-leases", "summary": FIRST}
    expanded = json.dumps({
        **concept, "sections": [{"heading": "Recovery", "prose": FIRST}],
    })
    with patch("obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock,
               side_effect=[_skeleton(FIRST, [concept]), _skeleton(SECOND), expanded]) as llm:
        result = await quality_synthesize_source(config, "workers.md", source, [])
    assert llm.await_count == 3
    assert result is not None
    assert result.concepts[0].sections[0].prose == FIRST
    assert validate_synthesis(result, source) is result


@pytest.mark.asyncio
async def test_chunk_quote_exemption_uses_that_chunk_not_other_source_passages():
    # The first chunk studies a provider's message. Its occurrence must not
    # excuse the provider refusing to summarize an unrelated second chunk.
    quoted = "I cannot access the article."
    first = f'The interface returned "{quoted}" when the browser was disabled.'
    source = SourceDoc("Recovery", first + "\n\n" + SECOND)
    config = Config(chunk_size=len(first), compile_concurrency=1)
    assert chunk_content(source.content, config.chunk_size) == [first, SECOND]
    with (
        patch("obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock,
              side_effect=[_skeleton(first), _skeleton(quoted)]),
        pytest.raises(RuntimeError, match=r"chunk 2/2: source_summary"),
    ):
        await quality_synthesize_source(config, "recovery.md", source, [])


@pytest.mark.asyncio
async def test_source_appropriate_disclaimer_quote_remains_valid_in_its_chunk():
    quote = "I cannot access the article."
    source = SourceDoc("Browser study", FIRST + "\n\n" + quote)
    config = Config(chunk_size=len(FIRST), compile_concurrency=1)
    assert chunk_content(source.content, config.chunk_size) == [FIRST, quote]
    with patch("obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock,
               side_effect=[_skeleton(FIRST), _skeleton(quote)]):
        result = await quality_synthesize_source(config, "browser.md", source, [])
    assert result is not None
    assert result.source_summary == FIRST


@pytest.mark.asyncio
async def test_single_chunk_skeleton_cannot_start_expansion_with_failed_summary():
    with (
        patch("obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock,
              return_value=_skeleton("TODO")) as llm,
        pytest.raises(SynthesisValidationError, match="source_summary"),
    ):
        await quality_synthesize_source(Config(), "workers.md", SourceDoc("Workers", FIRST), [])
    assert llm.await_count == 1
