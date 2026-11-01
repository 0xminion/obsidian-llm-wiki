"""Pure offset-based context offers; a digest is NOT proof of semantic use.

Citation segmentation is a caller concern. The application adapter validates its
catalog separately; offset-catalog callers can use this same constructor without
pretending that a catalog span is a paragraph. All coordinates address the exact
UTF-8 encoding of the supplied, unnormalized source body (not file frontmatter).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# Legacy callers retain their v2 default until their receipt epoch is integrated.
CONTEXT_POLICY_VERSION = "review-context-v2"
FRAGMENT_CONTEXT_POLICY_VERSION = "review-context-v4"
LEGACY_CONTEXT_POLICY_VERSIONS = ("review-context-v1", "review-context-v2")
KNOWN_FRAGMENT_CONTEXT_POLICY_VERSIONS = ("review-context-v3", "review-context-v4")
KNOWN_CONTEXT_POLICY_VERSIONS = (
    *LEGACY_CONTEXT_POLICY_VERSIONS,
    *KNOWN_FRAGMENT_CONTEXT_POLICY_VERSIONS,
)
MAX_CONTEXT_CHARS_PER_SIDE = 384
MAX_CONTEXT_BYTES_PER_SIDE = 1536
MAX_CITED_FRAGMENTS = 3
MAX_CITED_SPAN_IDS = 6
MAX_CITED_FRAGMENT_CHARS = 4000
MAX_FRAGMENT_CONTEXT_RECORDS = 6
MAX_FRAGMENT_CONTEXT_CHARS = 2304
MAX_FRAGMENT_CONTEXT_BYTES = 9216
CONTEXT_PURPOSES = ("referent", "attribution", "local_qualifier")
_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")


def context_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _quotation_introduction_start(
    content: str,
    left: int,
    citation_start: int,
    citation_end: int,
) -> int | None:
    """Recognize one complete plain introduction line, not a speaker or a fact.

    Only a colon-ended line, one blank-line separator and the first explicit
    top-level blockquote line qualify. Do not search past another paragraph or
    recover a line whose start lies outside the bounded visible prefix.
    """
    breaks = list(_PARAGRAPH_BREAK.finditer(content, left, citation_start))
    if not breaks:
        return None
    separator = breaks[-1]
    quote_start = separator.end()
    # A citation later in the quote does not authorize searching back through it.
    if "\n" in content[quote_start:citation_start]:
        return None
    if not re.match(r" {0,3}>[ \t]+[^ \t\r\n>]", content[quote_start:citation_end]):
        return None
    start = breaks[-2].end() if len(breaks) > 1 else left
    if start == left and left != 0:
        return None  # No fully visible paragraph boundary: possibly truncated.
    line = content[start : separator.start()]
    if not re.fullmatch(r" {0,3}\S[^\r\n]*[:：][ \t]*", line):
        return None
    # Refuse obvious Markdown block/container syntax, not arbitrary prose search.
    if re.match(r"\s*(?:[#>*+`~|\[\]-]|\d+[.)][ \t])", line):
        return None
    return start


def build_context_binding(
    *,
    content: str,
    source_file: str,
    source_sha256: str,
    catalog_version: str,
    unit_key: str,
    cited_span_ids: list[str],
    citation_start: int,
    citation_end: int,
    visible_intervals: list[list[int]],
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Offer bounded raw local context; v2 also recognizes one quote introduction.

    Never jump an unseen gap, normalize text, or admit an entire neighbor merely
    because it is one span. Missing/truncated antecedents are not reconstructed.
    Explicit known-v1 replay retains its original paragraph-only selection.
    Explicitly named offset catalogs may use this pure constructor after checking
    their own ID/anchor integrity. The runtime adapter only admits its own catalog.
    """
    policy = CONTEXT_POLICY_VERSION if policy_version is None else policy_version
    if policy not in LEGACY_CONTEXT_POLICY_VERSIONS:
        raise ValueError("unknown context policy")
    if not isinstance(content, str) or not isinstance(source_file, str):
        raise ValueError("context requires raw source text and source identity")
    if source_sha256 != hashlib.sha256(content.encode("utf-8")).hexdigest():
        raise ValueError("context source hash does not match source bytes")
    if not all(isinstance(value, str) and value for value in (catalog_version, unit_key)):
        raise ValueError("context requires catalog and unit identities")
    if (
        not isinstance(cited_span_ids, list)
        or not cited_span_ids
        or not all(isinstance(value, str) and value for value in cited_span_ids)
        or len(set(cited_span_ids)) != len(cited_span_ids)
    ):
        raise ValueError("context requires distinct citation IDs")
    if (
        type(citation_start) is not int
        or type(citation_end) is not int
        or not 0 <= citation_start < citation_end <= len(content)
    ):
        raise ValueError("context citation offsets are invalid")
    if not isinstance(visible_intervals, list) or not 0 < len(visible_intervals) <= 1024:
        raise ValueError("context visibility intervals are invalid or unrepresentable")
    normalized: list[list[int]] = []
    for pair in visible_intervals:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            or not 0 <= pair[0] < pair[1] <= len(content)
            or (normalized and pair[0] < normalized[-1][1])
        ):
            raise ValueError("context visibility offsets are invalid")
        if normalized and pair[0] == normalized[-1][1]:
            normalized[-1][1] = pair[1]
        else:
            normalized.append(list(pair))
    visible_intervals = normalized
    containing = [
        (start, end)
        for start, end in visible_intervals
        if start <= citation_start < citation_end <= end
    ]
    if len(containing) != 1:
        raise ValueError("context citation crosses unseen gaps or is unrepresentable")
    binding = {
        "policy": policy,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "catalog_version": catalog_version,
        "unit_key": unit_key,
        "cited_span_ids": list(cited_span_ids),
        "citation_start": citation_start,
        "citation_end": citation_end,
        "visible_intervals": [list(pair) for pair in visible_intervals],
    }
    visible_start, visible_end = containing[0]
    left = max(visible_start, citation_start - MAX_CONTEXT_CHARS_PER_SIDE)
    right = min(visible_end, citation_end + MAX_CONTEXT_CHARS_PER_SIDE)
    breaks = list(_PARAGRAPH_BREAK.finditer(content, left, citation_start))
    if breaks:
        introduction = (
            _quotation_introduction_start(content, left, citation_start, citation_end)
            if policy == "review-context-v2"
            else None
        )
        left = introduction if introduction is not None else breaks[-1].end()
    following = _PARAGRAPH_BREAK.search(content, citation_end, right)
    if following:
        right = following.start()
    offered = []
    for start, end in ((left, citation_start), (citation_end, right)):
        text = content[start:end]
        if not text.strip():
            continue
        record = {
            "char_start": start,
            "char_end": end,
            "byte_start": len(content[:start].encode("utf-8")),
            "byte_end": len(content[:end].encode("utf-8")),
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        record["context_id"] = "ctx-" + context_digest([binding, record])
        offered.append(record)
    return {**binding, "offered_context": offered}


def validate_context_binding(binding: Any, **expected: Any) -> str:
    """Check exact deterministic offers against independent source/unit/anchor inputs.

    The caller supplies the expected source revision and citation identity; never
    derive those from the untrusted binding. Visibility can be provided by the
    live request or replayed from the receipt; it does not authorize distant text.
    """
    fields = {
        "policy",
        "source_file",
        "source_sha256",
        "catalog_version",
        "unit_key",
        "cited_span_ids",
        "citation_start",
        "citation_end",
        "visible_intervals",
        "offered_context",
    }
    if not isinstance(binding, dict) or set(binding) != fields:
        return "context binding has invalid fields"
    if binding["policy"] not in LEGACY_CONTEXT_POLICY_VERSIONS:
        return "unknown context policy"
    records = binding["offered_context"]
    if not isinstance(records, list) or len(records) > 2:
        return "context records exceed policy bounds"
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "context_id",
            "char_start",
            "char_end",
            "byte_start",
            "byte_end",
            "text",
            "text_sha256",
        }:
            return "context record has invalid fields"
        text = record["text"]
        if (
            not isinstance(text, str)
            or len(text) > MAX_CONTEXT_CHARS_PER_SIDE
            or len(text.encode("utf-8")) > MAX_CONTEXT_BYTES_PER_SIDE
        ):
            return "context text exceeds policy bounds"
        if any(
            type(record[key]) is not int
            for key in ("char_start", "char_end", "byte_start", "byte_end")
        ):
            return "context offsets must be integers"
    try:
        visibility = expected.pop("visible_intervals", binding["visible_intervals"])
        canonical = build_context_binding(
            **expected,
            visible_intervals=visibility,
            policy_version=binding["policy"],
        )
    except (ValueError, TypeError, KeyError, StopIteration):
        return "context inputs are invalid or unrepresentable"

    if context_digest(binding) != context_digest(canonical):
        return "context binding differs from exact source/unit/citation/local policy"
    return ""


def build_fragment_context_binding(
    *,
    content: str,
    source_file: str,
    source_sha256: str,
    catalog_version: str,
    unit_key: str,
    cited_fragments: list[dict],
    visible_intervals: list[list[int]],
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Offer versioned context for separate citations, never a bounding union.

    The adapter must first validate every fragment against its own catalog. This
    pure layer validates raw offsets/visibility, not catalog membership or meaning.
    Every ID binds the complete ordered citation identity, not only its neighbor.
    V4 may offer a linked earlier introduction, without claiming speaker identity.
    When no record slot is safely available, that optional offer is omitted.
    """
    policy = FRAGMENT_CONTEXT_POLICY_VERSION if policy_version is None else policy_version
    if policy not in KNOWN_FRAGMENT_CONTEXT_POLICY_VERSIONS:
        raise ValueError("unknown fragment context policy")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError("context requires a source identity")
    if (
        not isinstance(cited_fragments, list)
        or not 1 <= len(cited_fragments) <= MAX_CITED_FRAGMENTS
    ):
        raise ValueError("context fragment count exceeds policy bounds")
    spans: set[str] = set()
    previous_end = 0
    cited_chars = 0
    anchors = []
    normalized = None
    for fragment in cited_fragments:
        if not isinstance(fragment, dict) or set(fragment) != {
            "span_ids",
            "start_offset",
            "end_offset",
        }:
            raise ValueError("context fragment has invalid fields")
        # Reuse the stable raw-input checks, not its citation representation or IDs.
        checked = build_context_binding(
            content=content,
            source_file=source_file,
            source_sha256=source_sha256,
            catalog_version=catalog_version,
            unit_key=unit_key,
            cited_span_ids=fragment["span_ids"],
            citation_start=fragment["start_offset"],
            citation_end=fragment["end_offset"],
            visible_intervals=visible_intervals,
            policy_version="review-context-v1",
        )
        start, end = fragment["start_offset"], fragment["end_offset"]
        if start < previous_end or spans.intersection(fragment["span_ids"]):
            raise ValueError("context fragments must be ordered, disjoint and have distinct IDs")
        previous_end = end
        cited_chars += end - start
        spans.update(fragment["span_ids"])
        if len(spans) > MAX_CITED_SPAN_IDS or cited_chars > MAX_CITED_FRAGMENT_CHARS:
            raise ValueError("context citation totals exceed policy bounds")
        normalized = checked["visible_intervals"]
        anchors.append(
            {
                "span_ids": list(fragment["span_ids"]),
                "start_offset": start,
                "end_offset": end,
                "byte_start": len(content[:start].encode("utf-8")),
                "byte_end": len(content[:end].encode("utf-8")),
                "text_sha256": hashlib.sha256(content[start:end].encode("utf-8")).hexdigest(),
            }
        )
    identity = {
        "policy": policy,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "catalog_version": catalog_version,
        "unit_key": unit_key,
        "cited_fragments": anchors,
        "visible_intervals": normalized,
    }
    binding = {
        **identity,
        "cited_fragments": [
            {
                **anchor,
                "fragment_index": index,
                "fragment_id": "frag-" + context_digest([identity, index]),
            }
            for index, anchor in enumerate(anchors)
        ],
    }
    offered: list[dict[str, Any]] = []
    for index, anchor in enumerate(binding["cited_fragments"]):
        start, end = anchor["start_offset"], anchor["end_offset"]
        visible_start, visible_end = next((a, b) for a, b in normalized if a <= start < end <= b)
        # A selected citation is a hard boundary, never context for another fragment.
        left = max(
            visible_start,
            start - MAX_CONTEXT_CHARS_PER_SIDE,
            anchors[index - 1]["end_offset"] if index else 0,
            offered[-1]["char_end"] if offered else 0,
        )
        right = min(
            visible_end,
            end + MAX_CONTEXT_CHARS_PER_SIDE,
            anchors[index + 1]["start_offset"] if index + 1 < len(anchors) else len(content),
        )
        ranges = _fragment_context_ranges(content, left, start, end, right)
        if (
            policy == "review-context-v4"
            and len(offered) + 1 + 2 * (len(anchors) - index) <= MAX_FRAGMENT_CONTEXT_RECORDS
        ):
            scan_left = max(
                visible_start,
                start - 1536,
                anchors[index - 1]["end_offset"] if index else 0,
                offered[-1]["char_end"] if offered else 0,
            )
            intro = _linked_intro_range(content, scan_left, start, end)
            if intro is not None and intro[1] <= ranges[0][0]:
                ranges.insert(0, (*intro, "linked_quote_introduction", "before"))
        for a, b, relation, side in ranges:
            text = content[a:b]
            if not text.strip():
                continue
            record = {
                "fragment_index": index,
                "fragment_id": anchor["fragment_id"],
                "relation": relation,
                "side": side,
                "char_start": a,
                "char_end": b,
                "byte_start": len(content[:a].encode("utf-8")),
                "byte_end": len(content[:b].encode("utf-8")),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            record["context_id"] = "ctx-" + context_digest([binding, record])
            offered.append(record)
    return {**binding, "offered_context": offered}


def _linked_intro_range(content: str, left: int, start: int, end: int) -> tuple[int, int] | None:
    """Recognize exactly intro / quote / commentary / current quote structure.

    This is a structural offer only. Natural-language actor changes cannot be
    decided here. The scan is contiguous, visible, and bounded to 1536 Unicode
    characters (at most 6144 UTF-8 bytes); truncated blocks never qualify.
    """
    # Without the source start we cannot rule out an already-open Markdown fence.
    # This intentionally omits otherwise plausible links in long/partial views.
    if left != 0 or re.search(r"(?m)^ {0,3}(?:`{3,}|~{3,})", content[left:start]):
        return None
    breaks = list(_PARAGRAPH_BREAK.finditer(content, left, start))
    if len(breaks) < 3:
        return None
    a, b, c = breaks[-3:]
    intro_start = breaks[-4].end() if len(breaks) >= 4 else left
    if intro_start == left and left != 0:
        return None
    intro = content[intro_start : a.start()]
    prior_quote = content[a.end() : b.start()]
    commentary = content[b.end() : c.start()]
    current = content[c.end() : end]
    plain = r" {0,3}\S[^\r\n]*"
    block = r"\s*(?:[#>*+`~|\[\]-]|\d+[.)][ \t])"
    if not re.fullmatch(plain, intro) or not re.search(r"[:：][ \t]*$", intro):
        return None
    if re.match(block, intro) or len(intro) > MAX_CONTEXT_CHARS_PER_SIDE:
        return None
    if len(intro.encode("utf-8")) > MAX_CONTEXT_BYTES_PER_SIDE:
        return None
    if not re.fullmatch(plain, commentary) or re.match(block, commentary):
        return None
    if re.search(r"[:：][ \t]*$", commentary):
        return None
    quote_line = r" {0,3}>[ \t]+[^ \t\r\n>][^\r\n]*"
    if not all(re.fullmatch(quote_line, line) for line in prior_quote.splitlines()):
        return None
    for line in prior_quote.splitlines():
        inner = re.sub(r"^ {0,3}>[ \t]+", "", line)
        if re.match(block, inner) or re.search(r"[:：][ \t]*$", inner):
            return None
    if not prior_quote or "\n" in content[c.end() : start] or "\r" in content[c.end() : start]:
        return None
    if not re.match(r" {0,3}>[ \t]+[^ \t\r\n>]", current):
        return None
    return intro_start, a.start()


def _adjacent_frame(
    content: str,
    left: int,
    start: int,
    end: int,
    breaks: list[re.Match],
) -> tuple[int, str] | None:
    """Recognize a complete nearest line, not a speaker or a factual premise.

    One blank separator and the first line of the next block are the entire
    structural link. Noncolon prose is an offer, not asserted attribution. Only
    ATX levels 2-6 are local headings; a document-title heading is never recovered.
    """
    separator = breaks[-1]
    block_start = separator.end()
    if "\n" in content[block_start:start] or "\r" in content[block_start:start]:
        return None
    frame_start = breaks[-2].end() if len(breaks) > 1 else left
    if frame_start == left and left != 0:
        return None  # Cannot prove the complete line's boundary from visible text.
    line = content[frame_start : separator.start()]
    if not re.fullmatch(r" {0,3}\S[^\r\n]*", line):
        return None
    heading = bool(re.fullmatch(r" {0,3}#{2,6}[ \t]+\S[^\r\n]*", line))
    block_syntax = r"\s*(?:[#>*+`~|\[\]-]|\d+[.)][ \t])"
    if not heading and re.match(block_syntax, line):
        return None
    destination = content[block_start:end]
    quote = bool(re.match(r" {0,3}>[ \t]+[^ \t\r\n>]", destination))
    if not quote and (
        not re.match(r" {0,3}\S", destination) or re.match(block_syntax, destination)
    ):
        return None
    relation = "adjacent_quote_introduction" if quote and not heading else "local_frame"
    return frame_start, relation


def _fragment_context_ranges(
    content: str,
    left: int,
    start: int,
    end: int,
    right: int,
) -> list[tuple[int, int, str, str]]:
    """Return at most one raw local range on each side of this fragment."""
    relation = "same_paragraph"
    breaks = list(_PARAGRAPH_BREAK.finditer(content, left, start))
    if breaks:
        frame = _adjacent_frame(content, left, start, end, breaks)
        left, relation = frame if frame is not None else (breaks[-1].end(), relation)
    following = _PARAGRAPH_BREAK.search(content, end, right)
    if following:
        right = following.start()
    return [(left, start, relation, "before"), (end, right, "same_paragraph", "after")]


def validate_fragment_context_binding(binding: Any, **expected: Any) -> str:
    """Rebuild v3 from independent raw inputs; checksums alone certify nothing.

    Supply all constructor inputs, including the actual visibility intervals.
    Neither citation identities nor visibility are inferred from the binding.
    """
    if not isinstance(binding, dict) or set(binding) != {
        "policy",
        "source_file",
        "source_sha256",
        "catalog_version",
        "unit_key",
        "cited_fragments",
        "visible_intervals",
        "offered_context",
    }:
        return "context fragment binding has invalid fields"
    if binding["policy"] not in KNOWN_FRAGMENT_CONTEXT_POLICY_VERSIONS:
        return "unknown fragment context policy"
    records = binding["offered_context"]
    if not isinstance(records, list) or len(records) > MAX_FRAGMENT_CONTEXT_RECORDS:
        return "context records exceed policy bounds"
    chars = byte_count = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "fragment_index",
            "fragment_id",
            "relation",
            "side",
            "context_id",
            "char_start",
            "char_end",
            "byte_start",
            "byte_end",
            "text",
            "text_sha256",
        }:
            return "context record has invalid fields"
        text = record["text"]
        if not isinstance(text, str):
            return "context text must be raw text"
        try:
            size = len(text.encode("utf-8"))
        except UnicodeError:
            return "context text must be valid UTF-8"
        chars += len(text)
        byte_count += size
        if len(text) > MAX_CONTEXT_CHARS_PER_SIDE or size > MAX_CONTEXT_BYTES_PER_SIDE:
            return "context text exceeds policy bounds"
        if any(
            type(record[key]) is not int
            for key in ("fragment_index", "char_start", "char_end", "byte_start", "byte_end")
        ):
            return "context offsets and fragment index must be integers"
    if chars > MAX_FRAGMENT_CONTEXT_CHARS or byte_count > MAX_FRAGMENT_CONTEXT_BYTES:
        return "context totals exceed policy bounds"
    try:
        canonical = build_fragment_context_binding(**expected, policy_version=binding["policy"])
        if context_digest(binding) != context_digest(canonical):
            return "context binding differs from exact source/unit/fragments/local policy"
    except (ValueError, TypeError, KeyError, StopIteration):
        return "context inputs are invalid or unrepresentable"
    return ""


def validate_model_context_uses(uses: Any, binding: dict | None) -> str:
    """Validate the model's declaration, NOT whether the model actually used text."""
    if not isinstance(uses, list):
        return "model_context_uses must be a list"
    offered = {record["context_id"] for record in binding["offered_context"]} if binding else set()
    if len(uses) > len(offered) * len(CONTEXT_PURPOSES):
        return "too many model context uses"
    seen = set()
    for use in uses:
        if not isinstance(use, dict) or set(use) != {"context_id", "purpose"}:
            return "model context use has invalid fields"
        identifier, purpose = use["context_id"], use["purpose"]
        if not isinstance(identifier, str) or identifier not in offered:
            return "model context use was not offered to this unit"
        if not isinstance(purpose, str) or purpose not in CONTEXT_PURPOSES:
            return "model context use has forbidden purpose"
        pair = (identifier, purpose)
        if pair in seen:
            return "duplicate model context use"
        seen.add(pair)
    return ""
