"""Tests for deterministic, overrideable source synthesis granularity."""

from __future__ import annotations

from obsidian_llm_wiki.core.schema import (
    LONG_SOURCE_CHARS,
    SHORT_SOURCE_CHARS,
    Granularity,
    select_synthesis_granularity,
)


def test_granularity_uses_explicit_length_thresholds_for_generic_sources():
    """Generic source detail rises deterministically at public boundaries."""
    assert select_synthesis_granularity("x" * SHORT_SOURCE_CHARS) is Granularity.CONCISE
    assert select_synthesis_granularity("x" * (SHORT_SOURCE_CHARS + 1)) is Granularity.STANDARD
    assert select_synthesis_granularity("x" * LONG_SOURCE_CHARS) is Granularity.STANDARD
    assert select_synthesis_granularity("x" * (LONG_SOURCE_CHARS + 1)) is Granularity.DETAILED


def test_granularity_accounts_for_high_density_source_types():
    """Papers and transcripts receive more detail than equivalently short articles."""
    content = "x" * (SHORT_SOURCE_CHARS + 1)

    assert select_synthesis_granularity(content, source_type="article") is Granularity.STANDARD
    assert (
        select_synthesis_granularity(content, source_type="scientific-paper")
        is Granularity.DETAILED
    )
    assert select_synthesis_granularity(content, source_type="transcript") is Granularity.DETAILED


def test_granularity_keeps_social_sources_concise_unless_user_overrides():
    """Source type defaults are deterministic but a valid override wins."""
    content = "x" * (LONG_SOURCE_CHARS + 1)

    assert select_synthesis_granularity(content, source_type="tweet") is Granularity.CONCISE
    assert (
        select_synthesis_granularity(content, source_type="tweet", override="detailed")
        is Granularity.DETAILED
    )
    assert (
        select_synthesis_granularity(content, source_type="tweet", override="invalid")
        is Granularity.CONCISE
    )


def test_granularity_accepts_a_length_for_callers_that_do_not_keep_source_text():
    """The selector works from a character count without changing its result."""
    assert (
        select_synthesis_granularity(LONG_SOURCE_CHARS + 1, source_type="document")
        is Granularity.DETAILED
    )
