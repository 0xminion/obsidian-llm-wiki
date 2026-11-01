"""Self-authored pure-context contracts; no provider or semantic accuracy claims."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from obsidian_llm_wiki.synth import review_context as rc


def fragment_inputs(content=None, quotes=None, visible=None):
    if content is None:
        content = "Mira: It stopped. In this trial.\n\nUNSEEN\n\n柳青：它保持关闭。仅限校准。"
        quotes = ["It stopped.", "它保持关闭。"]
    assert quotes
    fragments = []
    for index, quote in enumerate(quotes):
        start = content.index(quote)
        fragments.append({
            "span_ids": [f"fragment-{index}"],
            "start_offset": start,
            "end_offset": start + len(quote),
        })
    if visible is None:
        hidden = content.find("UNSEEN")
        visible = (
            [[0, hidden], [hidden + len("UNSEEN"), len(content)]]
            if hidden >= 0 else [[0, len(content)]]
        )
    return {
        "content": content,
        "source_file": "fragment-notes.md",
        "source_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "catalog_version": "explicit-offset-v1",
        "unit_key": "claim-with-independent-anchors",
        "cited_fragments": fragments,
        "visible_intervals": visible,
    }


def build(inputs, *, policy_version=None):
    constructor = getattr(rc, "build_fragment_context_binding", None)
    assert callable(constructor), "fragment-linked context constructor is missing"
    return constructor(**inputs, policy_version=policy_version)


def test_noncontiguous_fragments_offer_separate_visible_context_and_reject_second_tamper():
    inputs = fragment_inputs()
    binding = build(inputs, policy_version="review-context-v3")
    assert binding["policy"] == "review-context-v3"
    assert "citation_start" not in binding and "citation_end" not in binding
    records = binding["offered_context"]
    assert [(r["fragment_index"], r["text"]) for r in records] == [
        (0, "Mira: "), (0, " In this trial."), (1, "柳青："), (1, "仅限校准。"),
    ]
    for record in records:
        fragment = binding["cited_fragments"][record["fragment_index"]]
        assert record["fragment_id"] == fragment["fragment_id"]
        assert record["relation"] == "same_paragraph"
        assert "UNSEEN" not in record["text"]
        assert any(a <= record["char_start"] < record["char_end"] <= b
                   for a, b in inputs["visible_intervals"])
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""
    for tamper in ("context", "citation"):
        changed = deepcopy(binding)
        if tamper == "context":
            changed["offered_context"][2]["text"] = "Someone else: "
        else:
            changed["cited_fragments"][1]["span_ids"] = ["foreign-span"]
        assert rc.validate_fragment_context_binding(changed, **inputs)
    assert binding == build(inputs, policy_version="review-context-v3")


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\n \t\n"])
@pytest.mark.parametrize("frame,quote,relation", [
    ("Mira, discussing the valve, added:", "> It stays closed.", "adjacent_quote_introduction"),
    ("Mira was speaking about the valve.", "> It stays closed.", "adjacent_quote_introduction"),
    ("柳青讨论的是柳泵。", "> 它必须关闭。", "adjacent_quote_introduction"),
    ("### Calibration only", "It stays closed.", "local_frame"),
    ("A local caution follows.", "It stays closed.", "local_frame"),
])
def test_complete_nearest_frame_has_explicit_structural_link(separator, frame, quote, relation):
    content = "Not this earlier block.\n\n" + frame + separator + quote
    inputs = fragment_inputs(content, [quote])
    binding = build(inputs, policy_version="review-context-v3")
    assert [(r["text"], r["relation"], r["side"]) for r in binding["offered_context"]] == [
        (frame + separator, relation, "before"),
    ]
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""
    record = binding["offered_context"][0]
    assert record["fragment_id"] == binding["cited_fragments"][0]["fragment_id"]
    assert content[record["char_start"]:record["char_end"]] == record["text"]
    assert content.encode()[record["byte_start"]:record["byte_end"]] == record["text"].encode()


@pytest.mark.parametrize("policy", ["review-context-v1", "review-context-v2"])
def test_historical_serialization_matches_frozen_original_builder(policy):
    import json
    from pathlib import Path

    fixtures = json.loads((Path(__file__).with_name("fragment_context_legacy.json")).read_text())
    for case in fixtures:
        inputs = case["inputs"]
        expected = case["bindings"][policy]
        actual = rc.build_context_binding(**inputs, policy_version=policy)
        serialized = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert serialized == expected["json"]
        assert rc.context_digest(actual) == expected["sha256"]
        assert rc.validate_context_binding(actual, **inputs) == ""
        changed = deepcopy(actual)
        changed["source_sha256"] = "0" * 64
        assert rc.validate_context_binding(changed, **inputs)
