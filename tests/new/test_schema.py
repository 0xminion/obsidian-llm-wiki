"""Tests for vault-local, bounded synthesis schema policy."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.core.schema import (
    DEFAULT_SCHEMA_POLICY,
    SchemaPolicy,
    load_schema_policy,
)
from obsidian_llm_wiki.synth.prompts import build_synthesis_prompt


def test_load_schema_policy_returns_bounded_defaults_when_file_is_absent(tmp_path: Path):
    """A vault without policy configuration keeps the built-in prompt contract."""
    policy = load_schema_policy(tmp_path)

    assert policy == DEFAULT_SCHEMA_POLICY
    assert policy.required_sections == ()
    assert policy.allowed_tags == ()
    assert policy.instructions == ()
    assert policy.granularity_override is None


def test_load_schema_policy_reads_vault_local_file_and_sanitizes_values(tmp_path: Path):
    """Policy is local to 04-Wiki and accepts only bounded scalar guidance."""
    policy_path = tmp_path / "04-Wiki" / ".llmwiki" / "schema.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        "\n".join(
            [
                "required_sections:",
                "  - Core idea",
                "  - Evidence",
                "  - Core idea",
                "allowed_tags:",
                "  - AI Research",
                "  - '#Knowledge'",
                "  - AI Research",
                "instructions:",
                "  - Prefer concrete mechanisms. Never emit markdown.",
                f"  - {'x' * 900}",
                "granularity: detailed",
                "unknown_setting: should-not-propagate",
                "",
            ]
        ),
        encoding="utf-8",
    )

    policy = load_schema_policy(tmp_path)

    assert policy.required_sections == ("Core idea", "Evidence")
    assert policy.allowed_tags == ("ai-research", "knowledge")
    assert policy.instructions == ("Prefer concrete mechanisms. Never emit markdown.",)
    assert policy.granularity_override == "detailed"


def test_schema_policy_rejects_invalid_or_oversized_untrusted_values(tmp_path: Path):
    """Malformed YAML values cannot become arbitrary or unbounded prompt text."""
    policy_path = tmp_path / ".llmwiki" / "schema.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        "\n".join(
            [
                "required_sections: not-a-list",
                "allowed_tags: [ok, 3, {bad: value}]",
                "instructions:",
                f"  - {'z' * 513}",
                "  - valid guidance",
                "  - [not, scalar]",
                "granularity: dangerous-value",
                "",
            ]
        ),
        encoding="utf-8",
    )

    policy = load_schema_policy(tmp_path)

    assert policy.required_sections == ()
    assert policy.allowed_tags == ("ok", "3")
    assert policy.instructions == ("valid guidance",)
    assert policy.granularity_override is None


def test_prompt_includes_policy_only_in_explicit_user_controlled_guidance_block():
    """The stable JSON contract is unchanged and policy is visibly delimited."""
    policy = SchemaPolicy(
        required_sections=("Core idea",),
        allowed_tags=("ai-research",),
        instructions=("Prioritize primary sources.",),
    )

    default_prompt = build_synthesis_prompt("T", "source text")
    configured_prompt = build_synthesis_prompt(
        "T",
        "source text",
        schema_policy=policy,
        granularity="detailed",
    )

    assert "USER-CONTROLLED SCHEMA GUIDANCE" not in default_prompt
    assert "USER-CONTROLLED SCHEMA GUIDANCE" in configured_prompt
    assert "Required concept sections: Core idea" in configured_prompt
    assert "Allowed tags: ai-research" in configured_prompt
    assert "Prioritize primary sources." in configured_prompt
    assert "Requested synthesis granularity: detailed" in configured_prompt
    assert configured_prompt.count("--- SOURCE DOCUMENT ---") == 1
