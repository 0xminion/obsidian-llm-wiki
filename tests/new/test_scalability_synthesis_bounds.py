from __future__ import annotations

import asyncio
import json

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.request_limiter import AsyncRequestLimiter
from obsidian_llm_wiki.synth.evidence_windows import build_evidence_window
from obsidian_llm_wiki.synth.quality import _bounded_map, quality_synthesize_source


def test_evidence_window_selects_relevant_paragraphs_with_a_hard_budget() -> None:
    irrelevant = [
        f"Paragraph {index} discusses unrelated deployment housekeeping and routine logs."
        for index in range(30)
    ]
    relevant = (
        "Bounded candidate retrieval preserves prompt quality because only the most "
        "semantically relevant concepts enter synthesis context."
    )
    content = "\n\n".join([*irrelevant[:15], relevant, *irrelevant[15:]])

    window = build_evidence_window(
        content,
        terms=["bounded candidate retrieval", "prompt quality", "synthesis context"],
        char_budget=700,
        context_paragraphs=1,
    )

    assert relevant in window.text
    assert len(window.text) <= 700
    assert window.omitted_chars > 0
    assert window.selected_paragraphs == tuple(sorted(window.selected_paragraphs))
    assert "omitted" in window.text


def test_shared_request_limiter_caps_nested_concurrency() -> None:
    async def run() -> tuple[int, int]:
        limiter = AsyncRequestLimiter(2)
        active = 0
        maximum = 0
        lock = asyncio.Lock()

        async def request() -> None:
            nonlocal active, maximum
            async with limiter.slot():
                async with lock:
                    active += 1
                    maximum = max(maximum, active)
                await asyncio.sleep(0.01)
                async with lock:
                    active -= 1

        async def source_job() -> None:
            await asyncio.gather(*(request() for _ in range(3)))

        await asyncio.gather(*(source_job() for _ in range(4)))
        return maximum, limiter.requests_started

    maximum, requests_started = asyncio.run(run())
    assert maximum == 2
    assert requests_started == 12


def test_bounded_map_keeps_only_worker_budget_items_active() -> None:
    async def run():
        active = 0
        maximum = 0

        async def work(value: int) -> int:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.001)
            active -= 1
            return value * 2

        results = await _bounded_map(list(range(100)), work, max_workers=3)
        return maximum, results

    maximum, results = asyncio.run(run())
    assert maximum == 3
    assert results == [value * 2 for value in range(100)]


def test_two_pass_synthesis_uses_the_pipeline_shared_limiter(monkeypatch) -> None:
    active = 0
    maximum = 0

    async def fake_llm(_prompt, _messages, _config, *, task, **_kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.005)
        active -= 1
        if task == "expand":
            return json.dumps(
                {
                    "title": "Bounded Retrieval",
                    "summary": "Bounded semantic candidates.",
                    "aliases": [],
                    "sections": [{"heading": "Core", "prose": "Substantive body."}],
                    "claims": [],
                    "related": [],
                }
            )
        return json.dumps(
            {
                "source_title": "Source",
                "source_summary": "Summary",
                "concepts": [
                    {
                        "title": "Bounded Retrieval",
                        "slug": "bounded-retrieval",
                        "summary": "Candidate-bounded retrieval.",
                        "sections": [{"heading": "Seed", "prose": "Seed body."}],
                    }
                ],
                "maps": [],
            }
        )

    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", fake_llm)

    async def run() -> tuple[int, int]:
        limiter = AsyncRequestLimiter(1)
        config = Config(
            synthesis_mode="two_pass",
            compile_concurrency=4,
            chunk_size=10_000,
            concept_min_body_chars=0,
        )
        sources = [
            SourceDoc(title=f"Source {index}", content="Substantive source content " * 20)
            for index in range(2)
        ]
        results = await asyncio.gather(
            *(
                quality_synthesize_source(
                    config,
                    f"source-{index}.md",
                    source,
                    [],
                    request_limiter=limiter,
                )
                for index, source in enumerate(sources)
            )
        )
        assert all(result is not None for result in results)
        return limiter.max_active, limiter.requests_started

    limiter_maximum, requests_started = asyncio.run(run())
    assert maximum == 1
    assert limiter_maximum == 1
    assert requests_started == 4
