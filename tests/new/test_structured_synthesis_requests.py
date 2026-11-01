"""Regression tests for bounded, globally limited synthesis requests."""

from __future__ import annotations

import asyncio


def test_structured_synthesis_uses_schema_and_bounded_output(monkeypatch):
    """Synthesis requests the closed schema, not prompt-only or plain JSON mode."""
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.pipeline import _call_structured_synthesis

    captured: dict[str, object] = {}

    async def fake_call(*args, **kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", fake_call)

    result = asyncio.run(
        _call_structured_synthesis(
            "prompt", [{"role": "user", "content": "prompt"}], Config(),
        )
    )

    assert result == "{}"
    schema = captured["format"]
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "source_title", "source_summary", "source_tags", "key_points",
        "open_questions", "language", "concepts", "maps",
    }
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert captured["options"] == {"num_predict": 16_384, "temperature": 0}


def test_structured_synthesis_obeys_global_request_limit(monkeypatch):
    """Chunk and source calls share the same capacity limiter."""
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.pipeline import _call_structured_synthesis

    active = 0
    peak = 0

    async def fake_call(*_args, **_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "{}"

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", fake_call)

    async def run() -> None:
        limiter = asyncio.Semaphore(2)
        await asyncio.gather(*(
            _call_structured_synthesis(
                "prompt", [{"role": "user", "content": "prompt"}], Config(),
                llm_semaphore=limiter,
            )
            for _ in range(6)
        ))

    asyncio.run(run())

    assert peak == 2
