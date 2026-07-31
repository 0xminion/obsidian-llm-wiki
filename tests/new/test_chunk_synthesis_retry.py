"""Chunk synthesis retry contracts."""

import asyncio

from obsidian_llm_wiki.core import pipeline


def test_chunk_retry_returns_later_valid_result():
    attempts = 0

    async def synthesize():
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else "valid"

    assert asyncio.run(pipeline._retry_chunk_synthesis(synthesize, "source.md", 1, 3)) == "valid"
    assert attempts == 2


def test_chunk_retry_never_accepts_missing_chunk():
    attempts = 0

    async def synthesize():
        nonlocal attempts
        attempts += 1
        return None

    assert asyncio.run(pipeline._retry_chunk_synthesis(synthesize, "source.md", 1, 3)) is None
    assert attempts == pipeline._SYNTHESIS_PARSE_ATTEMPTS
