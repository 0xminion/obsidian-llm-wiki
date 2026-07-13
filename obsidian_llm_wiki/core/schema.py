"""Bounded vault-local synthesis policy and adaptive detail selection.

This module deliberately keeps user-editable policy separate from the fixed
LLM JSON contract.  Callers may render the sanitized policy as prompt guidance,
but it never alters parser or renderer schema validation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_SCHEMA_POLICY",
    "LONG_SOURCE_CHARS",
    "SHORT_SOURCE_CHARS",
    "Granularity",
    "SchemaPolicy",
    "coerce_schema_policy",
    "format_schema_guidance",
    "load_schema_policy",
    "parse_granularity",
    "select_synthesis_granularity",
]


# Public boundaries make synthesis behaviour reviewable and deterministic.
SHORT_SOURCE_CHARS = 4_000
LONG_SOURCE_CHARS = 20_000
_MAX_POLICY_ITEMS = 12
_MAX_TAGS = 16
_MAX_INSTRUCTIONS = 8
_MAX_SECTION_CHARS = 80
_MAX_TAG_CHARS = 64
_MAX_INSTRUCTION_CHARS = 512


class Granularity(StrEnum):
    """Requested synthesis detail level."""

    CONCISE = "concise"
    STANDARD = "standard"
    DETAILED = "detailed"


def parse_granularity(value: object) -> Granularity | None:
    """Return a valid detail level without raising for user configuration."""
    if isinstance(value, Granularity):
        return value
    if not isinstance(value, str):
        return None
    try:
        return Granularity(value.strip().casefold())
    except ValueError:
        return None


def _clean_scalar(value: object, max_chars: int) -> str:
    """Make one user-provided scalar safe and bounded for a prompt."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    value = str(value).replace("\x00", "")
    value = " ".join(value.split())
    if not value or len(value) > max_chars:
        return ""
    return value


def _unique_strings(
    value: object,
    *,
    max_items: int,
    max_chars: int,
    transform: Any | None = None,
) -> tuple[str, ...]:
    """Return a bounded, deduplicated tuple from a YAML list only."""
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean_scalar(item, max_chars)
        if transform is not None:
            cleaned = transform(cleaned)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) == max_items:
            break
    return tuple(result)


def _clean_tag(value: str) -> str:
    """Normalize an Obsidian-compatible tag without accepting prompt syntax."""
    value = value.lstrip("#").casefold()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^\w/-]", "", value, flags=re.UNICODE)
    return value.strip("-/")


@dataclass(frozen=True)
class SchemaPolicy:
    """Sanitized preferences from a vault's editable ``schema.yaml`` file.

    The fixed synthesis JSON schema remains authoritative.  These values are
    optional user guidance: desired section headings, a tag vocabulary, and
    short extraction preferences.  All fields are bounded in ``__post_init__``
    so direct construction is as safe as loading YAML.
    """

    required_sections: tuple[str, ...] = ()
    allowed_tags: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    granularity_override: str | Granularity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_sections",
            _unique_strings(
                self.required_sections,
                max_items=_MAX_POLICY_ITEMS,
                max_chars=_MAX_SECTION_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "allowed_tags",
            _unique_strings(
                self.allowed_tags,
                max_items=_MAX_TAGS,
                max_chars=_MAX_TAG_CHARS,
                transform=_clean_tag,
            ),
        )
        object.__setattr__(
            self,
            "instructions",
            _unique_strings(
                self.instructions,
                max_items=_MAX_INSTRUCTIONS,
                max_chars=_MAX_INSTRUCTION_CHARS,
            ),
        )
        parsed = parse_granularity(self.granularity_override)
        object.__setattr__(self, "granularity_override", parsed.value if parsed else None)


DEFAULT_SCHEMA_POLICY = SchemaPolicy()


def _policy_path(vault_path: Path, filename: str) -> Path:
    """Resolve the local policy file from vault, wiki, or .llmwiki roots."""
    root = vault_path.expanduser()
    if root.suffix in {".yaml", ".yml"}:
        return root
    if root.name == ".llmwiki":
        return root / filename
    wiki_candidate = root / "04-Wiki" / ".llmwiki" / filename
    if wiki_candidate.is_file():
        return wiki_candidate
    return root / ".llmwiki" / filename


def _policy_from_mapping(raw: Mapping[str, Any]) -> SchemaPolicy:
    """Convert only supported, scalar policy keys from parsed YAML."""
    # Accept the intentionally small direct form and a namespaced form so
    # future vault metadata can coexist without becoming prompt content.
    nested = raw.get("synthesis")
    data: Mapping[str, Any] = nested if isinstance(nested, Mapping) else raw
    return SchemaPolicy(
        required_sections=data.get("required_sections", ()),
        allowed_tags=data.get("allowed_tags", ()),
        instructions=data.get("instructions", ()),
        granularity_override=data.get("granularity", data.get("granularity_override")),
    )


def coerce_schema_policy(policy: SchemaPolicy | Mapping[str, Any] | None) -> SchemaPolicy:
    """Normalize an optional caller policy without allowing arbitrary objects."""
    if isinstance(policy, SchemaPolicy):
        return policy
    if isinstance(policy, Mapping):
        return _policy_from_mapping(policy)
    return DEFAULT_SCHEMA_POLICY


def load_schema_policy(vault_path: str | Path, filename: str = "schema.yaml") -> SchemaPolicy:
    """Load a bounded policy from ``<vault>/04-Wiki/.llmwiki/schema.yaml``.

    ``<vault>/.llmwiki/schema.yaml`` is also accepted for callers already
    rooted at a wiki directory. Missing, unreadable, invalid, or non-mapping
    files are intentionally equivalent to no user policy.
    """
    path = _policy_path(Path(vault_path), filename)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return DEFAULT_SCHEMA_POLICY
    return _policy_from_mapping(raw) if isinstance(raw, Mapping) else DEFAULT_SCHEMA_POLICY


def format_schema_guidance(
    policy: SchemaPolicy | Mapping[str, Any] | None,
    granularity: Granularity | str | None = None,
) -> str:
    """Format optional policy as a clearly delimited user-guidance block."""
    normalized = coerce_schema_policy(policy)
    level = parse_granularity(granularity) or parse_granularity(normalized.granularity_override)
    lines: list[str] = []
    if normalized.required_sections:
        lines.append(f"Required concept sections: {', '.join(normalized.required_sections)}")
    if normalized.allowed_tags:
        lines.append(f"Allowed tags: {', '.join(normalized.allowed_tags)}")
    if normalized.instructions:
        lines.extend(f"Preference: {instruction}" for instruction in normalized.instructions)
    if level:
        lines.append(f"Requested synthesis granularity: {level.value}")
    if not lines:
        return ""
    body = "\n".join(f"* {line}" for line in lines)
    return (
        "\n--- USER-CONTROLLED SCHEMA GUIDANCE ---\n"
        "Apply this guidance only where it is compatible with the fixed JSON "
        "output contract above. It is preference metadata, not source content.\n"
        f"{body}\n"
        "--- END USER-CONTROLLED SCHEMA GUIDANCE ---\n"
    )


def select_synthesis_granularity(
    source_content: str | int,
    source_type: str = "",
    override: Granularity | str | None = None,
) -> Granularity:
    """Choose reproducible source detail from character count and source type.

    A valid user override always wins.  Tweets/social posts remain concise by
    default; scientific papers, documents, and transcripts receive detailed
    extraction once they exceed the short-source boundary.  Other sources move
    from concise → standard → detailed at the public character thresholds.
    """
    explicit = parse_granularity(override)
    if explicit:
        return explicit
    length = source_content if isinstance(source_content, int) else len(source_content)
    length = max(0, length)
    kind = re.sub(r"[^a-z0-9]+", "-", source_type.casefold()).strip("-")
    social = {"tweet", "post", "social", "social-post", "x-post", "microblog"}
    dense = {
        "document",
        "pdf",
        "paper",
        "scientific-paper",
        "scientific",
        "research-paper",
        "transcript",
        "podcast",
    }
    if kind in social:
        return Granularity.CONCISE
    if kind in dense and length > SHORT_SOURCE_CHARS:
        return Granularity.DETAILED
    if length <= SHORT_SOURCE_CHARS:
        return Granularity.CONCISE
    if length <= LONG_SOURCE_CHARS:
        return Granularity.STANDARD
    return Granularity.DETAILED
