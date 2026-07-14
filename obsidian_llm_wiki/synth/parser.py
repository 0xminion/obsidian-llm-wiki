"""Parse and validate LLM synthesis JSON into SynthesisBundle.

The LLM produces a JSON array of source-synthesis objects (one per source).
This module extracts, validates, and converts them into typed dataclasses.

Tolerant of:
  * Leading/trailing prose around the JSON
  * Markdown code fences (```json … ```)
  * A single object instead of an array
  * Missing optional fields
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from obsidian_llm_wiki.core.models import (
    SourceSynthesis,
    SynthesisBundle,
    source_synthesis_from_dict,
)

__all__ = [
    "parse_synthesis_response",
    "parse_single_source_synthesis",
]

logger = logging.getLogger("obswiki.synth.parser")
_MAX_RESPONSE_CHARS = 1_000_000


def parse_synthesis_response(response: str) -> SynthesisBundle:
    """Parse a multi-source LLM synthesis response into a SynthesisBundle.

    The response is expected to be a JSON array of source-synthesis objects,
    or a single source-synthesis object.  Tolerates surrounding prose and
    code fences.

    Returns a SynthesisBundle (possibly with errors populated if parsing
    failed for some items).
    """
    if not response or not response.strip():
        return SynthesisBundle(errors=["empty response"])

    data = _extract_json(response)
    if data is None:
        return SynthesisBundle(errors=["no valid JSON found in response"])

    # Normalise to a list.
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return SynthesisBundle(
            errors=[f"expected JSON array or object, got {type(data).__name__}"]
        )

    sources: list[SourceSynthesis] = []
    errors: list[str] = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"source {i}: expected object, got {type(item).__name__}")
            continue
        try:
            sources.append(source_synthesis_from_dict(item))
        except Exception as exc:
            errors.append(f"source {i}: {exc}")

    return SynthesisBundle(sources=sources, errors=errors)


def parse_single_source_synthesis(response: str) -> SourceSynthesis | None:
    """Parse a single-source LLM synthesis response.

    Returns None if no valid JSON could be extracted.
    """
    if not response or not response.strip():
        return None

    data = _extract_json(response)
    if data is None:
        return None

    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None

    # Sanitize LaTeX artifacts: LLMs sometimes produce $\rightarrow$ or
    # $\leftarrow$ in text fields. JSON parsing interprets \r as carriage
    # return, eating the 'r' and leaving 'ightarrow'. Fix by restoring
    # the intended LaTeX arrows as Unicode arrows before parsing.
    data = _sanitize_latex_artifacts(data)

    return source_synthesis_from_dict(data)


def _sanitize_latex_artifacts(data: Any) -> Any:
    """Fix LaTeX arrow artifacts that get mangled by JSON escape parsing.

    LLMs produce $\\rightarrow$ and $\\leftarrow$ in text. JSON's \\r
    escape eats the 'r', producing '$ightarrow$'. We replace these with
    Unicode arrows (→ ←) that render correctly in Obsidian markdown.
    """
    arrow_map = {
        "$ightarrow$": "→",
        "$eftarrow$": "←",
        "\\rightarrow": "→",
        "\\leftarrow": "←",
        "\\Rightarrow": "⇒",
        "\\Leftarrow": "⇐",
        "\\leftrightarrow": "↔",
        "\\Leftrightarrow": "⇔",
        "$\\rightarrow$": "→",
        "$\\leftarrow$": "←",
    }
    if isinstance(data, dict):
        return {k: _sanitize_latex_artifacts(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_latex_artifacts(v) for v in data]
    if isinstance(data, str):
        result = data
        for latex, arrow in arrow_map.items():
            result = result.replace(latex, arrow)
        return result
    return data


# ── JSON extraction ─────────────────────────────────────────────────────


def _extract_json(text: str) -> list[Any] | dict[str, Any] | None:
    """Extract the first valid JSON array or object from ``text``.

    Handles code fences and surrounding prose without repeatedly parsing every
    shorter suffix of malformed model output.
    """
    text = text.strip()
    if len(text) > _MAX_RESPONSE_CHARS:
        logger.warning("Refusing oversized synthesis response (%d chars)", len(text))
        return None

    # Strip markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Fast path: already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first '[' or '{' that begins a JSON block.
    start = _find_json_start(text)
    if start == -1:
        return None

    text = text[start:]

    # Decode one JSON value from the first object/array. ``raw_decode`` accepts
    # explanatory trailing prose while keeping malformed-input work bounded.
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
        return value
    except json.JSONDecodeError:
        logger.warning("Could not parse JSON from response (first 200 chars): %s", text[:200])
        return None


def _find_json_start(text: str) -> int:
    """Find the index of the first ``[`` or ``{`` that starts a JSON block."""
    bracket = text.find("[")
    brace = text.find("{")
    if bracket == -1:
        return brace
    if brace == -1:
        return bracket
    return min(bracket, brace)
