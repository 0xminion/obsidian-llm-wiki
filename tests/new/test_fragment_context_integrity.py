"""Adversarial structural controls for pure context v3, independent of catalogs."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from obsidian_llm_wiki.synth import review_context as rc
from tests.new.test_fragment_context_v3 import build, fragment_inputs


def reseal(binding):
    """Deliberately recompute every public digest after hostile fixture changes."""
    identity = {k: deepcopy(v) for k, v in binding.items() if k != "offered_context"}
    for anchor in identity["cited_fragments"]:
        anchor.pop("fragment_index", None)
        anchor.pop("fragment_id", None)
    for index, anchor in enumerate(binding["cited_fragments"]):
        anchor["fragment_id"] = "frag-" + rc.context_digest([identity, index])
    base = {k: v for k, v in binding.items() if k != "offered_context"}
    for record in binding["offered_context"]:
        record["text_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
        record["fragment_id"] = binding["cited_fragments"][record["fragment_index"]]["fragment_id"]
        record.pop("context_id", None)
        record["context_id"] = "ctx-" + rc.context_digest([base, record])
    return binding


def test_legacy_api_cannot_silently_label_single_fragment_algorithm_v3():
    content = "Mira: It stopped."
    inputs = {
        "content": content, "source_file": "old.md",
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "catalog_version": "old-catalog", "unit_key": "old-unit", "cited_span_ids": ["s"],
        "citation_start": 6, "citation_end": len(content),
        "visible_intervals": [[0, len(content)]],
    }
    assert "review-context-v3" in rc.KNOWN_CONTEXT_POLICY_VERSIONS
    with pytest.raises(ValueError, match="context policy"):
        rc.build_context_binding(**inputs, policy_version="review-context-v3")
    binding = rc.build_context_binding(**inputs, policy_version="review-context-v2")
    binding["policy"] = "review-context-v3"
    assert rc.validate_context_binding(binding, **inputs)


@pytest.mark.parametrize("field,value", [
    ("content", None), ("source_file", ""), ("source_file", 3),
    ("source_sha256", "0" * 64), ("source_sha256", None),
    ("catalog_version", ""), ("catalog_version", False), ("unit_key", ""), ("unit_key", []),
    ("cited_fragments", None), ("cited_fragments", []), ("cited_fragments", "bad"),
    ("cited_fragments", [{}]), ("cited_fragments", [None]),
    ("visible_intervals", None), ("visible_intervals", []),
    ("visible_intervals", [[False, 10]]), ("visible_intervals", [[0, 10.0]]),
    ("visible_intervals", [[0, 10], [9, 11]]),
    ("visible_intervals", [[2, 3], [0, 1]]),
    ("visible_intervals", [[0, 100000]]), ("visible_intervals", [[0, 0]]),
    ("visible_intervals", [[-1, 3]]), ("visible_intervals", [(0, 10)]),
])
def test_invalid_input_types_identities_and_visibility_fail_closed(field, value):
    inputs = fragment_inputs()
    inputs[field] = value
    with pytest.raises(ValueError, match="context"):
        build(inputs)


@pytest.mark.parametrize("field,value", [
    ("start_offset", True), ("end_offset", True), ("start_offset", 1.0),
    ("end_offset", "19"), ("start_offset", -1), ("end_offset", 100000),
    ("span_ids", []), ("span_ids", ["s", "s"]), ("span_ids", [False]),
    ("span_ids", "s"), ("span_ids", [""]), ("span_ids", None),
])
def test_malformed_second_citation_is_not_dropped(field, value):
    inputs = fragment_inputs()
    inputs["cited_fragments"][1][field] = value
    with pytest.raises(ValueError, match="context"):
        build(inputs)


@pytest.mark.parametrize("attack", ["reverse", "duplicate", "overlap", "foreign-field",
                                    "repeated-id", "empty", "inverted"])
def test_fragments_are_strict_ordered_disjoint_distinct_exact_shapes(attack):
    inputs = fragment_inputs()
    fragments = inputs["cited_fragments"]
    if attack == "reverse":
        fragments.reverse()
    elif attack == "duplicate":
        fragments.append(deepcopy(fragments[1]))
    elif attack == "overlap":
        fragments[1]["start_offset"] = fragments[0]["end_offset"] - 1
    elif attack == "foreign-field":
        fragments[1]["quote"] = "model supplied quote"
    elif attack == "repeated-id":
        fragments[1]["span_ids"] = fragments[0]["span_ids"][:]
    elif attack == "empty":
        fragments[1]["end_offset"] = fragments[1]["start_offset"]
    else:
        fragments[1]["end_offset"] = fragments[1]["start_offset"] - 1
    with pytest.raises(ValueError, match="context"):
        build(inputs)


@pytest.mark.parametrize("which", ["count", "ids", "chars"])
def test_citation_totals_have_positive_boundary_and_overflow_controls(which):
    if which == "count":
        valid = fragment_inputs("A B C D", ["A", "B", "C"])
        invalid = fragment_inputs("A B C D", ["A", "B", "C", "D"])
    elif which == "ids":
        valid = fragment_inputs()
        for i, frag in enumerate(valid["cited_fragments"]):
            frag["span_ids"] = [f"span-{i}-{j}" for j in range(3)]
        invalid = deepcopy(valid)
        invalid["cited_fragments"][1]["span_ids"].append("overflow")
    else:
        content = "A" * 2000 + "\n\n" + "中" * 2001
        valid = fragment_inputs(content, ["A" * 2000, "中" * 2000])
        invalid = fragment_inputs(content, ["A" * 2000, "中" * 2001])
    assert rc.validate_fragment_context_binding(build(valid), **valid) == ""
    with pytest.raises(ValueError, match="context"):
        build(invalid)


def test_three_fragment_six_side_totals_are_strict_and_bytes_are_not_characters():
    quotes = [f"[citation-{i}]" for i in range(3)]
    content = "\r\n\r\n".join("🪶" * 384 + q + "🙂" * 384 for q in quotes)
    inputs = fragment_inputs(content, quotes)
    binding = build(inputs)
    records = binding["offered_context"]
    assert len(records) == 6
    assert sum(len(r["text"]) for r in records) == 2304
    assert sum(len(r["text"].encode()) for r in records) == 9216
    for record in records:
        assert len(record["text"]) == 384
        assert len(record["text"].encode()) == 1536
        assert content.encode()[record["byte_start"]:record["byte_end"]] == record["text"].encode()
    for anchor in binding["cited_fragments"]:
        assert anchor["byte_start"] == len(content[:anchor["start_offset"]].encode())
        assert anchor["byte_end"] == len(content[:anchor["end_offset"]].encode())
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""
    changed = deepcopy(binding)
    changed["offered_context"].append(deepcopy(records[0]))
    assert rc.validate_fragment_context_binding(changed, **inputs)


@pytest.mark.parametrize("gap", ["", " ", "中" * 20, "🪶" * 700])
def test_selected_fragments_and_offered_context_never_overlap(gap):
    inputs = fragment_inputs("Mira: Alpha." + gap + "Beta. Tail.", ["Alpha.", "Beta."])
    binding = build(inputs)
    records = binding["offered_context"]
    assert [r["char_start"] for r in records] == sorted(r["char_start"] for r in records)
    assert all(a["char_end"] <= b["char_start"] for a, b in zip(records, records[1:], strict=False))
    for record in records:
        for fragment in inputs["cited_fragments"]:
            assert (record["char_end"] <= fragment["start_offset"]
                    or fragment["end_offset"] <= record["char_start"])
    assert rc.validate_fragment_context_binding(binding, **inputs) == ""


@pytest.mark.parametrize("field,value", [
    ("source_file", "another.md"), ("catalog_version", "other-catalog"), ("unit_key", "other-unit"),
])
def test_full_identity_binds_every_context_id_and_rejects_cross_unit_replay(field, value):
    inputs = fragment_inputs()
    binding = build(inputs)
    changed_inputs = {**inputs, field: value}
    changed_binding = build(changed_inputs)
    assert rc.validate_fragment_context_binding(binding, **changed_inputs)
    assert {r["context_id"] for r in binding["offered_context"]}.isdisjoint(
        r["context_id"] for r in changed_binding["offered_context"])


@pytest.mark.parametrize("change", ["second-id", "second-start", "second-end", "revision"])
def test_every_context_id_binds_all_citations_not_only_its_own_fragment(change):
    inputs = fragment_inputs()
    changed = deepcopy(inputs)
    if change == "second-id":
        changed["cited_fragments"][1]["span_ids"] = ["replacement"]
    elif change == "second-start":
        changed["cited_fragments"][1]["start_offset"] += 1
    elif change == "second-end":
        changed["cited_fragments"][1]["end_offset"] -= 1
    else:
        changed["content"] = changed["content"].replace("UNSEEN", "HIDDEN")
        changed["source_sha256"] = hashlib.sha256(changed["content"].encode()).hexdigest()
    first, second = build(inputs), build(changed)
    assert rc.validate_fragment_context_binding(first, **changed)
    assert first["offered_context"][0]["text"] == second["offered_context"][0]["text"]
    assert {r["context_id"] for r in first["offered_context"]}.isdisjoint(
        r["context_id"] for r in second["offered_context"])


@pytest.mark.parametrize("change", [
    "second-context", "second-citation", "byte-as-char", "fragment-byte-as-char", "second-link",
    "relation", "hidden-context", "missing-record", "extra-record", "missing-fragment",
    "bool-offset", "bool-fragment-index", "visibility",
])
def test_resealed_payload_tampering_still_fails_exact_reconstruction(change):
    inputs = fragment_inputs()
    binding = build(inputs)
    assert reseal(deepcopy(binding)) == binding
    changed = deepcopy(binding)
    record = changed["offered_context"][2]
    if change == "second-context":
        record["text"] = "Different speaker: "
    elif change == "second-citation":
        changed["cited_fragments"][1]["span_ids"] = ["other-span"]
    elif change == "byte-as-char":
        record["byte_end"] = record["char_end"]
    elif change == "fragment-byte-as-char":
        changed["cited_fragments"][1]["byte_start"] += 1
    elif change == "second-link":
        record["fragment_index"] = 0
    elif change == "relation":
        record["relation"] = "proven_speaker"
    elif change == "hidden-context":
        a = inputs["content"].index("UNSEEN")
        record.update(char_start=a, char_end=a + 6, byte_start=a, byte_end=a + 6, text="UNSEEN")
    elif change == "missing-record":
        changed["offered_context"].pop(2)
    elif change == "extra-record":
        changed["offered_context"].append(deepcopy(record))
    elif change == "missing-fragment":
        changed["cited_fragments"].pop()
        changed["offered_context"] = changed["offered_context"][:2]
    elif change == "bool-offset":
        changed["offered_context"][0]["char_start"] = False
    elif change == "bool-fragment-index":
        changed["offered_context"][0]["fragment_index"] = False
    else:
        changed["visible_intervals"] = [[0, len(inputs["content"])]]
    assert rc.validate_fragment_context_binding(reseal(changed), **inputs)


def test_visibility_must_be_independently_supplied_and_cannot_bridge_hidden_citation_gap():
    inputs = fragment_inputs()
    binding = build(inputs)
    without_visibility = {k: v for k, v in inputs.items() if k != "visible_intervals"}
    assert rc.validate_fragment_context_binding(binding, **without_visibility)
    without_source = {k: v for k, v in inputs.items() if k != "content"}
    assert rc.validate_fragment_context_binding(binding, **without_source)
    citation = inputs["cited_fragments"][1]
    cut = citation["start_offset"] + 1
    inputs["visible_intervals"] = [[0, cut], [cut + 1, len(inputs["content"])]]
    with pytest.raises(ValueError, match="unseen gaps"):
        build(inputs)


@pytest.mark.parametrize("policy", ["review-context-v1", "review-context-v2"])
@pytest.mark.parametrize("attack", [
    "source", "unit", "revision", "gap", "bytes", "truncation", "boolean", "duplicate", "overlap",
])
def test_known_legacy_replay_rejects_resealed_integrity_attacks(policy, attack):
    content = "中" * 500 + "It stopped." + "🪶" * 500
    inputs = {
        "content": content, "source_file": "old.md",
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "catalog_version": "old-offsets", "unit_key": "old-claim", "cited_span_ids": ["old-id"],
        "citation_start": 500, "citation_end": 511, "visible_intervals": [[0, len(content)]],
    }
    binding = rc.build_context_binding(**inputs, policy_version=policy)
    assert rc.validate_context_binding(binding, **inputs) == ""
    changed = deepcopy(binding)
    if attack == "source":
        changed["source_file"] = "foreign.md"
    elif attack == "unit":
        changed["unit_key"] = "foreign-unit"
    elif attack == "revision":
        changed["source_sha256"] = "0" * 64
    elif attack == "gap":
        changed["visible_intervals"] = [[0, 505], [506, len(content)]]
    elif attack == "bytes":
        changed["offered_context"][1]["byte_start"] = 511
    elif attack == "truncation":
        changed["offered_context"][1]["text"] = changed["offered_context"][1]["text"][:-1]
    elif attack == "boolean":
        changed["offered_context"][0]["char_start"] = True
    elif attack == "duplicate":
        changed["offered_context"][1] = deepcopy(changed["offered_context"][0])
    else:
        changed["offered_context"][1]["char_start"] = 499
    base = {k: v for k, v in changed.items() if k != "offered_context"}
    for record in changed["offered_context"]:
        record.pop("context_id")
        record["text_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
        record["context_id"] = "ctx-" + rc.context_digest([base, record])
    assert rc.validate_context_binding(changed, **inputs)


def test_touching_visibility_is_coalesced_but_real_gaps_are_not():
    inputs = fragment_inputs("Mira: It stopped.", ["It stopped."])
    full = build(inputs)
    inputs["visible_intervals"] = [[0, 9], [9, len(inputs["content"])]]
    assert full == build(inputs)
    assert rc.validate_fragment_context_binding(full, **inputs) == ""


@pytest.mark.parametrize("purpose", ["referent", "attribution", "local_qualifier"])
def test_model_declarations_use_existing_vocabulary_and_offered_fragment_ids_only(purpose):
    binding = build(fragment_inputs())
    assert rc.validate_model_context_uses([], binding) == ""
    identifier = binding["offered_context"][2]["context_id"]
    use = {"context_id": identifier, "purpose": purpose}
    assert rc.validate_model_context_uses([use], binding) == ""
    assert rc.validate_model_context_uses([use, use], binding)
    assert rc.validate_model_context_uses([{**use, "context_id": "foreign"}], binding)
    assert rc.validate_model_context_uses([{**use, "context_id": ""}], binding)
    assert rc.validate_model_context_uses([{"purpose": purpose}], binding)
    assert rc.validate_model_context_uses([{**use, "purpose": "independent_fact"}], binding)
    assert rc.validate_model_context_uses([{**use, "purpose": "framing"}], binding)
