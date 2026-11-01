"""Producer prompt coverage policy and compatibility, not model-quality tests."""

import pytest

from obsidian_llm_wiki.core.span_catalog import build_span_catalog
from obsidian_llm_wiki.synth.language import get_language_instruction
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt


@pytest.mark.parametrize("language,source", [
    ("en", "The method requires verification. It may improve efficiency.\n\n"
           "The extent of improvement remains uncertain."),
    ("zh", "该方法需要验证。它可能提高效率。\n\n改进程度仍不确定。"),
])
@pytest.mark.parametrize("chunk_size", [10000, 25])
def test_coverage_rules_in_every_real_catalog_window(language, source, chunk_size):
    windows = build_span_catalog(source, "source.md").windows(chunk_size)
    assert bool(windows)
    if chunk_size == 25:
        assert len(windows) > 1
    for window in windows:
        prompt = build_synthesis_prompt("Source", window.text, language=language,
                                        span_window=window, source_file="source.md")
        normalized = " ".join(prompt.split())
        for rule in (
            "Source-summary coverage and qualification fidelity:",
            "source_summary and key_points",
            "thesis, conditions, contrasts, and limitations",
            "necessary versus sufficient",
            "graded questions into binary ones",
            "mandatory conditions into optional or comparative advice",
            "A quote alone does not provide prose coverage",
        ):
            assert normalized.count(rule) == 1
        assert get_language_instruction(language) in prompt
        assert "With no requested language, preserve the source's language" in prompt
        assert "checked against the source by a separate review" in prompt
        assert 'Cite evidence ONLY as "evidence_span_ranges"' in prompt
        assert '1..3 ordered disjoint ranges' in prompt


def test_old_producer_prompt_version_is_not_compatible(monkeypatch):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline

    current = pipeline._artifact_versions_for_config(Config())
    with monkeypatch.context() as old:
        old.setattr(pipeline, "_PROMPT_VERSION", "synthesis-v8-span-count-feedback-refine")
        previous = pipeline._artifact_versions_for_config(Config())
    assert current.prompt != previous.prompt
    assert current.schema == previous.schema
