"""Vault-local, bounded instruction profiles for wiki queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BUILTIN_PROFILES",
    "MAX_PROFILE_INSTRUCTIONS",
    "QueryProfile",
    "QueryProfileStore",
]

MAX_PROFILE_INSTRUCTIONS = 2_000
_PROFILE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


@dataclass(frozen=True, slots=True)
class QueryProfile:
    """A named, query-only set of instructions."""

    name: str
    instructions: str


BUILTIN_PROFILES = (
    QueryProfile(
        name="default",
        instructions=(
            "Answer only from the retrieved wiki pages. Be clear about uncertainty and "
            "cite retrieved pages with their exact [[vault-relative/path.md]] paths."
        ),
    ),
    QueryProfile(
        name="research",
        instructions=(
            "Synthesize the retrieved evidence carefully, distinguish facts from inference, "
            "note disagreements or gaps, and cite retrieved pages with exact paths."
        ),
    ),
    QueryProfile(
        name="exact-facts",
        instructions=(
            "State only precise facts supported by the retrieved pages. Do not speculate, "
            "and cite every factual claim with an exact retrieved page path."
        ),
    ),
    QueryProfile(
        name="commitments",
        instructions=(
            "Focus on decisions, promises, owners, dates, and open commitments found in the "
            "retrieved pages. Do not infer commitments and cite exact retrieved page paths."
        ),
    ),
)


class QueryProfileStore:
    """JSON-backed custom profiles, layered over safe built-in defaults."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def list(self) -> tuple[QueryProfile, ...]:
        """Return built-ins plus valid custom profiles, sorted by profile name."""
        profiles = {profile.name: profile for profile in BUILTIN_PROFILES}
        profiles.update({profile.name: profile for profile in self._custom_profiles()})
        return tuple(profiles[name] for name in sorted(profiles))

    def load(self, name: str) -> QueryProfile | None:
        """Return one selected profile or ``None`` for an unknown name."""
        normalized = name.strip().casefold()
        return next((profile for profile in self.list() if profile.name == normalized), None)

    def save(self, profile: QueryProfile) -> QueryProfile:
        """Persist one custom profile without changing built-ins or other profiles."""
        normalized = _normalize_profile(profile)
        profiles = {item.name: item for item in self._custom_profiles()}
        profiles[normalized.name] = normalized
        payload = {
            "version": 1,
            "profiles": [
                {"name": item.name, "instructions": item.instructions}
                for item in sorted(profiles.values(), key=lambda item: item.name)
            ],
        }
        from obsidian_llm_wiki.render.frontmatter import atomic_write

        atomic_write(self.path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return normalized

    def _custom_profiles(self) -> tuple[QueryProfile, ...]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ()
        raw_profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if not isinstance(raw_profiles, list):
            return ()
        profiles: list[QueryProfile] = []
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            instructions = item.get("instructions")
            if not isinstance(name, str) or not isinstance(instructions, str):
                continue
            try:
                profiles.append(_normalize_profile(QueryProfile(name, instructions)))
            except ValueError:
                continue
        return tuple(profiles)


def _normalize_profile(profile: QueryProfile) -> QueryProfile:
    name = profile.name.strip().casefold()
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError("Profile names must be 1-64 lowercase letters, numbers, _ or -")
    instructions = profile.instructions.strip()[:MAX_PROFILE_INSTRUCTIONS]
    if not instructions:
        raise ValueError("Profile instructions cannot be empty")
    return QueryProfile(name=name, instructions=instructions)
