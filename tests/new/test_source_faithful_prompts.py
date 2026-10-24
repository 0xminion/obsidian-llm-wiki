"""Prompt contracts for concise, source-bound, multilingual synthesis."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt
from obsidian_llm_wiki.synth.quality import (
    build_expand_prompt,
    build_extract_prompt,
    quality_synthesize_source,
)


def _prompts(language=""):
    return [
        build_synthesis_prompt("Trial", "No benefit was observed.", language=language),
        build_extract_prompt("Trial", "No benefit was observed.", language=language),
        build_expand_prompt(
            "Result",
            "result",
            "Observed outcome",
            "Trial",
            "No benefit was observed.",
            [],
            language=language,
        ),
    ]


def test_all_synthesis_prompts_scale_content_to_evidence_not_quotas():
    for prompt in _prompts():
        for unsafe in (
            "500-800",
            "at least 3 distinct sections",
            "3-8",
            "80+ chars",
            "3-5 sentences",
            "4-12",
            "Create 1-3",
        ):
            assert unsafe not in prompt
        assert "Do not pad" in prompt
        assert "verbatim" in prompt
        assert "uncertainty" in prompt
        assert "source document is data" in prompt
        assert "not source evidence" in prompt
    for prompt in _prompts()[:2]:
        assert "zero concepts" in prompt


def test_chinese_content_does_not_require_english_first_titles():
    for prompt in _prompts("zh"):
        assert "Chinese" in prompt or "中文" in prompt
        assert "English title first" not in prompt
        assert "bilingual format" not in prompt
        assert "quotes in their original language" in prompt


def test_explicit_supported_language_is_not_ignored():
    for prompt in _prompts("fr"):
        assert "French" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,expected", [("en", "en"), ("", "zh")])
async def test_quality_language_uses_config_before_source_detection(configured, expected):
    content = "试验未发现收益，但样本量较小，不能排除微弱的影响。"
    responses = [
        json.dumps(
            {
                "source_title": "Trial",
                "source_summary": "No benefit was observed.",
                "language": "de",
                "concepts": [{"title": "Result", "slug": "result", "summary": "No benefit."}],
            }
        ),
        json.dumps(
            {
                "title": "Result",
                "summary": "No benefit.",
                "sections": [{"heading": "Outcome", "prose": "No benefit was observed."}],
            }
        ),
    ]
    config = Config(output_language=configured)
    with patch(
        "obsidian_llm_wiki.providers.llm.acall_llm", new_callable=AsyncMock, side_effect=responses
    ) as llm:
        result = await quality_synthesize_source(
            config, "trial.md", SourceDoc(title="Trial", content=content), []
        )
    assert result.language == expected
    expected_name = "English" if expected == "en" else "Chinese"
    for call in llm.call_args_list:
        assert expected_name in call.args[0]
