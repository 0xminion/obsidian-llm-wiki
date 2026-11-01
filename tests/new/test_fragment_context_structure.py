"""Bounded structural offers, not attribution inference or fact entailment."""
from __future__ import annotations

from copy import deepcopy

import pytest

from obsidian_llm_wiki.synth import review_context as rc
from tests.new.test_fragment_context_v3 import build, fragment_inputs


@pytest.mark.parametrize("before", [
    "# Whole document title\n\n",
    "####### Not a supported heading\n\n",
    "A wrapped\nframe line.\n\n",
    "    Indented code\n\n",
    "\tTabbed code\n\n",
    "- List entry\n\n",
    "1. Ordered list entry\n\n",
    "> Parent quote\n\n",
    "```\n\n",
    "[ref]: destination\n\n",
    "Frame.\n\n\n",
    "Frame.\n\n\n\n",
    "Frame.\r\n\r\n\r\n",
])
def test_unsupported_frame_shapes_are_not_recovered(before):
    quote = "> It stays closed."
    inputs = fragment_inputs(before + quote, [quote])
    binding = build(inputs)
    assert binding["offered_context"] == []
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""


@pytest.mark.parametrize("destination", [
    ">> Nested quote", "> > Nested quote", "- List item", "1. Ordered item",
    "    Indented code", "\tTabbed code", "# Next heading", "```code",
])
def test_frame_does_not_infer_a_link_to_unrecognized_container(destination):
    inputs = fragment_inputs("Mira spoke.\n\n" + destination, [destination])
    assert build(inputs)["offered_context"] == []


def test_frame_is_nearest_only_not_a_search_for_speaker_or_distant_heading():
    content = "## Distant frame\n\nMira spoke.\n\nIntervening paragraph.\n\n> It stays closed."
    inputs = fragment_inputs(content, ["> It stays closed."])
    records = build(inputs)["offered_context"]
    assert [r["text"] for r in records] == ["Intervening paragraph.\n\n"]
    assert records[0]["relation"] == "adjacent_quote_introduction"
    # Structural adjacency is not a claim that this line supplies a speaker.
    assert not any("speaker" in key or "resolved" in key for key in records[0])


def test_later_quote_line_does_not_search_back_for_introduction():
    inputs = fragment_inputs("Mira spoke.\n\n> First line.\n> It stays closed.",
                             ["> It stays closed."])
    records = build(inputs)["offered_context"]
    assert [r["text"] for r in records] == ["> First line.\n"]
    assert records[0]["relation"] == "same_paragraph"


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n"])
def test_every_partial_frame_or_separator_visibility_cut_refuses_crossing(separator):
    frame = "柳青说的是柳泵。"
    quote = "> 它必须关闭。"
    content = frame + separator + quote
    inputs = fragment_inputs(content, [quote])
    complete = build(inputs)
    assert [r["text"] for r in complete["offered_context"]] == [frame + separator]
    for cut in range(1, len(frame + separator) + 1):
        clipped = {**inputs, "visible_intervals": [[cut, len(content)]]}
        records = build(clipped)["offered_context"]
        assert records == [], (cut, records)
        assert rc.validate_fragment_context_binding(complete, **clipped)
    for cut in range(len(frame + separator)):
        intervals = [[cut + 1, len(content)]]
        if cut:
            intervals.insert(0, [0, cut])
        hidden = {**inputs, "visible_intervals": intervals}
        assert build(hidden)["offered_context"] == [], cut


@pytest.mark.parametrize("length", [383, 384, 385, 600])
@pytest.mark.parametrize("character", ["中", "🪶"])
def test_structural_frame_is_complete_or_absent_at_bound(length, character):
    frame_and_separator = character * (length - 2) + "\n\n"
    quote = "> It stays closed."
    inputs = fragment_inputs(frame_and_separator + quote, [quote])
    records = build(inputs)["offered_context"]
    assert [r["text"] for r in records] == ([frame_and_separator] if length <= 384 else [])


def test_hidden_island_never_becomes_a_bounding_citation_or_context_union():
    content = "Mira spoke.\n\n> First.\n\nUNSEEN\n\n柳青说。\r\n\r\n> 第二段。"
    inputs = fragment_inputs(content, ["> First.", "> 第二段。"])
    binding = build(inputs)
    assert [r["text"] for r in binding["offered_context"]] == [
        "Mira spoke.\n\n", "柳青说。\r\n\r\n",
    ]
    assert [r["fragment_index"] for r in binding["offered_context"]] == [0, 1]
    assert "citation_start" not in binding and "citation_end" not in binding
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""


def test_no_aliasing_of_inputs_bindings_and_repeated_calls():
    inputs = fragment_inputs()
    original = deepcopy(inputs)
    binding = build(inputs)
    assert inputs == original
    binding["cited_fragments"][0]["span_ids"].append("mutation")
    binding["visible_intervals"][0][0] += 1
    assert inputs == original
    assert rc.validate_fragment_context_binding(binding, **inputs)
    assert build(inputs) == build(original)


@pytest.mark.parametrize("change", ["bool-fragment-start", "policy", "malformed-record",
                                    "foreign-binding-field", "unknown-fragment-field", "surrogate"])
def test_malformed_bindings_return_diagnostic_not_an_exception(change):
    inputs = fragment_inputs()
    binding = build(inputs)
    if change == "bool-fragment-start":
        binding["cited_fragments"][0]["start_offset"] = False
    elif change == "policy":
        binding["policy"] = "review-context-v999"
    elif change == "malformed-record":
        binding["offered_context"][1] = None
    elif change == "foreign-binding-field":
        binding["citation_start"] = 0
    elif change == "unknown-fragment-field":
        binding["cited_fragments"][1]["quote"] = "untrusted"
    else:
        binding["offered_context"][1]["text"] = "\ud800"
    assert rc.validate_fragment_context_binding(binding, **inputs)
