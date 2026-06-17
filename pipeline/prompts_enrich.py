"""LLM prompt construction + response parsing for the OKF enrichment agent.

The enrichment agent crawls a set of seed URLs, asks the LLM to decide
for each fetched page whether to *enrich* an existing concept, *mint* a
new reference/concept, or *skip* it, and then follows outbound links
within the allowed host for further crawling.

This module provides:

* :func:`build_enrich_prompt` — builds the system/user prompt that asks
  the LLM to emit a JSON array of decisions.
* :func:`parse_enrich_response` — parses that JSON array into
  :class:`EnrichDecision` objects.
* :class:`EnrichDecision` — dataclass representing one LLM decision.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "EnrichDecision",
    "build_enrich_prompt",
    "parse_enrich_response",
]

logger = logging.getLogger("llmwiki.enrich.prompts")


# ── Decision dataclass ──────────────────────────────────────────────────────


@dataclass
class EnrichDecision:
    """A single LLM enrichment decision for one fetched page.

    Fields:
        action: One of ``"enrich"``, ``"mint"``, ``"skip"``.
        concept_id: The concept ID to enrich (action=``enrich``) or the
            proposed slug for a new concept (action=``mint``). Ignored for
            ``skip``.
        title: Display title for a minted reference.
        summary: One-line summary for the reference page.
        body: Full markdown body to write/append.
        addition: When ``action == "enrich"``, the text to append to the
            existing concept's Citations section.
        tags: Suggested tags for a minted reference.
        follow_links: Additional URLs to crawl (outbound links the LLM
            deems worth following).
    """

    action: str = "skip"
    concept_id: str = ""
    title: str = ""
    summary: str = ""
    body: str = ""
    addition: str = ""
    tags: list[str] = field(default_factory=list)
    follow_links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Prompt construction ─────────────────────────────────────────────────────


def build_enrich_prompt(
    url: str,
    title: str,
    content: str,
    existing_concepts: list[str] | dict[str, str] | None = None,
) -> str:
    """Build the prompt that asks the LLM to decide enrich / mint / skip.

    Args:
        url: The source URL the page was fetched from.
        title: Extracted page title.
        content: Extracted page content (truncated to a reasonable size
            by the caller if needed).
        existing_concepts: Either a list of concept IDs / slugs, or a dict
            of ``slug -> concept_id`` mappings. Used to let the LLM pick
            the right concept to enrich.

    Returns:
        A prompt string suitable for use as a system or user message.
    """
    # Normalise existing_concepts into a plain list of slugs/ids.
    if existing_concepts is None:
        concept_lines = "(none yet)"
    elif isinstance(existing_concepts, dict):
        if existing_concepts:
            concept_lines = "\n".join(
                f"  - {slug} → {cid}" for slug, cid in existing_concepts.items()
            )
        else:
            concept_lines = "(none yet)"
    else:
        concept_lines = (
            "\n".join(f"  - {c}" for c in existing_concepts) if existing_concepts else "(none yet)"
        )

    # Truncate content to keep the prompt within typical context limits.
    max_content = 8000
    if len(content) > max_content:
        content = content[:max_content] + "\n…[truncated]…"

    return f"""You are the enrichment agent for an OKF (Open Knowledge Format) wiki.

A web page was fetched and its content is provided below.  Decide what to
do with it.  You have three options:

  1. **enrich** — append a citation / addition to an EXISTING concept page.
  2. **mint**  — create a brand-new Reference page in references/.
  3. **skip**  — the page is not relevant; ignore it.

Source URL: {url}
Page title: {title}

Existing concepts (slug → concept_id):
{concept_lines}

--- BEGIN PAGE CONTENT ---
{content}
--- END PAGE CONTENT ---

Respond with a JSON array of one or more decision objects.  Each object
MUST have this shape:

{{
  "action": "enrich" | "mint" | "skip",
  "concept_id": "<concept id to enrich, or proposed slug for mint>",
  "title": "<display title for mint>",
  "summary": "<one-line summary for mint>",
  "body": "<full markdown body for mint, or empty>",
  "addition": "<markdown text to append to the concept's Citations section for enrich, or empty>",
  "tags": ["tag1", "tag2"],
  "follow_links": ["https://…", "https://…"]
}}

Rules:
* Return ONLY the JSON array — no prose before or after, no code fences.
* For "enrich", set "concept_id" to an existing concept id and put the
  citation text in "addition".
* For "mint", set "concept_id" to a proposed slug and put the full page
  body in "body".
* For "skip", all other fields can be empty.
* "follow_links" lists outbound URLs from this page that are worth
  crawling next (same host only).
"""


# ── Response parsing ────────────────────────────────────────────────────────


def parse_enrich_response(response: str) -> list[EnrichDecision]:
    """Parse the LLM's JSON response into a list of :class:`EnrichDecision`.

    Tolerates:
      * Leading/trailing prose (extracts the first ``[`` … ``]`` block).
      * Code fences (```json … ```).
      * A single object wrapped in braces instead of an array.

    Returns an empty list if no valid JSON array or object can be found.
    """
    if not response or not response.strip():
        return []

    text = response.strip()

    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # If the response contains prose around the JSON, try to extract the
    # JSON array or object.
    if not text.startswith("[") and not text.startswith("{"):
        # Find the first '[' or '{' that begins a JSON block.
        start = text.find("[")
        obj_start = text.find("{")
        if start == -1 or (obj_start != -1 and obj_start < start):
            start = obj_start
        if start == -1:
            logger.warning("No JSON block found in enrich response")
            return []
        text = text[start:]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try progressively trimming from the end until it parses.
        for end in range(len(text), 0, -1):
            try:
                data = json.loads(text[:end])
                break
            except json.JSONDecodeError:
                continue
        else:
            logger.warning("Failed to parse enrich response as JSON")
            return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []

    decisions: list[EnrichDecision] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "skip")).strip().lower()
        if action not in ("enrich", "mint", "skip"):
            action = "skip"
        decisions.append(
            EnrichDecision(
                action=action,
                concept_id=str(item.get("concept_id", "")).strip(),
                title=str(item.get("title", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                body=str(item.get("body", "") or ""),
                addition=str(item.get("addition", "") or ""),
                tags=list(item.get("tags", []) or []),
                follow_links=list(item.get("follow_links", []) or []),
            )
        )
    return decisions
