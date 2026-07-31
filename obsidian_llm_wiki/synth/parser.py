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

# Valid JSON escape characters (after backslash): " \ / b f n r t u
# Everything else is invalid and causes json.loads to fail.
# LaTeX commands like \epsilon, \alpha, \rightarrow produce invalid escapes.
_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _fix_invalid_escapes(text: str) -> str:
    """Replace invalid JSON escape sequences with safe alternatives.

    LLMs sometimes include LaTeX (``$\\epsilon$``, ``\\rightarrow``) in text
    fields. JSON only allows ``\\n \\t \\r \\b \\f \\\\ \\" \\/ \\uXXXX`` —
    all other ``\\X`` sequences are invalid and break ``json.loads``.

    This function finds backslash + letter sequences that are NOT valid JSON
    escapes and replaces the backslash with a doubled backslash (``\\\\``)
    so the literal text survives JSON round-tripping.
    """
    # We need to handle escapes only inside JSON strings (between quotes),
    # but a full string-aware parser is expensive. Instead, we use a
    # regex that matches backslash followed by a character that is NOT a
    # valid JSON escape, and doubles the backslash.
    #
    # This is safe because:
    # - Valid escapes (\n, \t, etc.) are NOT matched (they're in the exclusion set)
    # - \\ is already doubled and won't be matched (second \ is not in the set)
    # - \uXXXX is not matched because 'u' is in the valid set
    # - LaTeX like \epsilon → \\epsilon (literal backslash in JSON string)
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in _VALID_JSON_ESCAPES:
                # Valid escape — keep both chars
                result.append(char)
                result.append(next_char)
                i += 2
                continue
            # Invalid escape — double the backslash
            result.append("\\\\")
            result.append(next_char)
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


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

    # Fix invalid JSON escape sequences from LaTeX in LLM output.
    # LLMs produce $\epsilon$, $\alpha$, \rightarrow, etc. in text fields.
    # JSON only allows \n \t \r \b \f \\ \" \/ \uXXXX — all other \X sequences
    # are invalid and cause json.loads to fail. Replace them before parsing.
    text = _fix_invalid_escapes(text)

    # Fast path: already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Local models commonly emit LaTeX commands such as ``\epsilon`` without
    # JSON-escaping the backslash, or stop while closing the final object.
    # Repair only these mechanical serialization failures before extracting.
    repaired = _repair_json_response(text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Find the first '[' or '{' that begins a JSON block.
    start = _find_json_start(text)
    if start == -1:
        return None

    text = text[start:]

    # Decode one JSON value from the first object/array.  ``raw_decode`` accepts
    # explanatory trailing prose while keeping malformed-input work bounded.
    try:
        value, _end = json.JSONDecoder().raw_decode(text)
        return value
    except json.JSONDecodeError:
        pass

    # ── Repair common LLM JSON issues ──────────────────────────────────

    # 1. Trailing commas before } or ] (common LLM output error).
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    if repaired != text:
        try:
            value, _end = json.JSONDecoder().raw_decode(repaired)
            return value
        except json.JSONDecodeError:
            pass

    # 2. Truncated JSON — the LLM hit the output token limit mid-object.
    #    Attempt to close incomplete JSON by counting open brackets/braces
    #    and appending the needed closers.  This is a best-effort recovery
    #    that salvage partially-truncated synthesis output.
    repaired = _repair_truncated_json(text)
    if repaired and repaired != text:
        try:
            value = json.loads(repaired)
            logger.info(
                "Recovered truncated JSON (%d → %d chars)", len(text), len(repaired),
            )
            return value
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from response (first 200 chars): %s", text[:200])
    return None


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to close truncated JSON by appending missing brackets/braces.

    The LLM sometimes hits the output token limit mid-object, producing
    valid JSON up to a point and then nothing.  This function counts
    open delimiters (ignoring those inside strings) and appends the
    needed closers.  It also strips trailing partial key-value pairs
    that were cut mid-string.
    """
    if not text or text[0] not in "{[":
        return None

    # First pass: count depth and detect if we're inside a string
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False
    last_complete_pos = 0  # position after the last , or complete value

    for pos, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth_brace += 1
        elif char == "}":
            depth_brace -= 1
        elif char == "[":
            depth_bracket += 1
        elif char == "]":
            depth_bracket -= 1
        elif char == ",":
            # Record position after the comma — a valid break point
            last_complete_pos = pos

    if depth_brace == 0 and depth_bracket == 0:
        return None  # Not truncated

    # If we're inside a string, truncate to the last comma (which is a
    # safe break point).  Otherwise, try to use the last comma position.
    candidate = text[:last_complete_pos].rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]

    # Re-count depth on the stripped version, tracking open order
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape = False
    open_stack: list[str] = []  # track order of opens for correct closing
    for char in candidate:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth_brace += 1
            open_stack.append("}")
        elif char == "}":
            depth_brace -= 1
            if open_stack and open_stack[-1] == "}":
                open_stack.pop()
        elif char == "[":
            depth_bracket += 1
            open_stack.append("]")
        elif char == "]":
            depth_bracket -= 1
            if open_stack and open_stack[-1] == "]":
                open_stack.pop()

    # Append closers in reverse order of opening (stack order)
    closers = "".join(reversed(open_stack))
    if closers:
        return candidate + closers
    return None


def _repair_json_response(text: str) -> str:
    """Repair invalid escapes and missing final JSON delimiters conservatively."""
    # JSON allows only these escapes. Preserve valid sequences and double every
    # other backslash so LaTex-like ``\epsilon`` remains literal text.
    repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack and char == stack[-1]:
            stack.pop()

    if in_string:
        repaired += '"'
    return repaired + "".join(reversed(stack))


def _find_json_start(text: str) -> int:
    """Find the index of the first ``[`` or ``{`` that starts a JSON block."""
    bracket = text.find("[")
    brace = text.find("{")
    if bracket == -1:
        return brace
    if brace == -1:
        return bracket
    return min(bracket, brace)
