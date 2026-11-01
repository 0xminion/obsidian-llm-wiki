"""Failed expansion cannot make a two-pass source look complete."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.synth.quality import quality_synthesize_source
from tests.new.grounding_doubles import legacy_fixture_llm, producer_call_count, producer_prompts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_expansion",
    [
        '{"title":"Result","summary":"The trial found',
        '{"title":"Result","sections":null}',
        RuntimeError("provider timeout"),
    ],
)
async def test_failed_expansion_is_not_completed_even_with_nonempty_skeleton(failed_expansion):
    skeleton = json.dumps(
        {
            "source_title": "Trial",
            "source_summary": "No benefit was observed.",
            "concepts": [
                {
                    "title": "Result",
                    "slug": "result",
                    "summary": "No benefit.",
                    "sections": [{"heading": "Result", "prose": "No benefit was observed."}],
                }
            ],
        }
    )
    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=[skeleton, failed_expansion],
    ) as llm:
        result = await quality_synthesize_source(
            Config(),
            "trial.md",
            SourceDoc(title="Trial", content="No benefit was observed."),
            [],
        )
    assert result is None
    assert llm.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    ["{}", '{"source_summary":"No benefit', '{"source_summary":"No benefit.","concepts":null}'],
)
async def test_parse_failure_retries_complete_source(invalid):
    from obsidian_llm_wiki.core.pipeline import _synthesize_with_retry

    content = "No benefit was observed. The sample was small."
    valid = json.dumps({"source_title": "Trial", "source_summary": content, "concepts": []})
    responses = iter([invalid, valid])
    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm",
        new_callable=AsyncMock,
        side_effect=legacy_fixture_llm(lambda *_args, **_kwargs: next(responses)),
    ) as llm:
        result = await _synthesize_with_retry(
            Config(min_source_chars=1),
            "trial.md",
            SourceDoc(title="Trial", content=content),
            [],
        )
    assert result is not None
    assert result.source_summary == content
    assert producer_call_count(llm) == 2
    # Grounded prompts render the complete source as catalog spans.
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog

    spans = build_span_catalog(content, "trial.md").spans
    assert all(
        all(f"⟦{span.id}⟧ {span.text}" in prompt for span in spans)
        for prompt in producer_prompts(llm)
    )
