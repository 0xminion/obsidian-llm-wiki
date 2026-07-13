"""Behavior tests for vault-local query profiles."""

from __future__ import annotations


def test_builtin_profiles_are_available_without_a_profile_file(tmp_path):
    from obsidian_llm_wiki.query.profiles import QueryProfileStore

    store = QueryProfileStore(tmp_path / ".llmwiki" / "query-profiles.json")

    assert {profile.name for profile in store.list()} >= {
        "default",
        "research",
        "exact-facts",
        "commitments",
    }
    assert "retrieved" in store.load("research").instructions.casefold()


def test_user_profile_persists_and_is_selected_over_builtins(tmp_path):
    from obsidian_llm_wiki.query.profiles import QueryProfile, QueryProfileStore

    store = QueryProfileStore(tmp_path / ".llmwiki" / "query-profiles.json")
    saved = store.save(
        QueryProfile(
            name="briefing",
            instructions="Give a concise executive briefing with explicit uncertainty.",
        )
    )

    reloaded = QueryProfileStore(store.path).load("briefing")

    assert reloaded == saved
    assert reloaded.instructions.startswith("Give a concise")
    assert "briefing" in store.path.read_text(encoding="utf-8")


def test_profile_instructions_are_bounded_and_unknown_profiles_are_not_selected(tmp_path):
    from obsidian_llm_wiki.query.profiles import (
        MAX_PROFILE_INSTRUCTIONS,
        QueryProfile,
        QueryProfileStore,
    )

    store = QueryProfileStore(tmp_path / ".llmwiki" / "query-profiles.json")
    saved = store.save(QueryProfile(name="long", instructions="x" * (MAX_PROFILE_INSTRUCTIONS + 5)))

    assert len(saved.instructions) == MAX_PROFILE_INSTRUCTIONS
    assert store.load("does-not-exist") is None
