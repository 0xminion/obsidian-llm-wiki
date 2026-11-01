"""Offline, generic versioned context contracts."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from obsidian_llm_wiki.synth import review_context as rc


def inputs(text, quote):
    start = text.index(quote)
    return {
        "content": text,
        "source_file": "generic.md",
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "catalog_version": "offset-v1",
        "unit_key": "unit",
        "cited_fragments": [
            {"span_ids": ["s1"], "start_offset": start, "end_offset": start + len(quote)}
        ],
        "visible_intervals": [[0, len(text)]],
    }


@pytest.mark.parametrize(
    "intro,commentary,quote,sep",
    [
        ("Dr. Ada Green said:", "Green continued.", "The valve closed.", "\n\n"),
        ("研究员林晓说道：", "林晓接着说。", "设备已停止。", "\r\n\r\n"),
    ],
)
def test_linked_literal_and_v3(intro, commentary, quote, sep):
    assert Path(rc.__file__).resolve().parents[2] == Path(__file__).resolve().parents[2]
    text = sep.join([intro, "> " + "Earlier words. " * 30, commentary, "> " + quote])
    args = inputs(text, quote)
    old = rc.build_fragment_context_binding(**args, policy_version="review-context-v3")
    new = rc.build_fragment_context_binding(**args)
    assert new["policy"] == "review-context-v4"
    assert not any(r["text"] == intro for r in old["offered_context"])
    record = next(r for r in new["offered_context"] if r["text"] == intro)
    assert text[record["char_start"] : record["char_end"]] == intro
    assert text.encode()[record["byte_start"] : record["byte_end"]] == intro.encode()
    assert rc.validate_fragment_context_binding(old, **args) == ""
    assert rc.validate_fragment_context_binding(new, **args) == ""


@pytest.mark.parametrize(
    "middle",
    [
        "> Earlier.\n\nAnother speaker:",
        "> Earlier.\n\n## New section",
        "> Earlier.\n\n```code```",
        "> Earlier.\n\n- List",
        "> Earlier.\n\n| table |",
        "> > Nested quote.\n\nCommentary.",
        "> Earlier.\n\nOther.\n\nCommentary.",
        "> ```python\n> print(1)\n> ```\n\nCommentary.",
        "> Another speaker:\n> Earlier.\n\nCommentary.",
        "> - List\n\nCommentary.",
        "> ## Heading\n\nCommentary.",
    ],
)
def test_structural_barriers(middle):
    intro = "Professor Dana Vale said:"
    text = intro + "\n\n" + middle + "\n\n> The motor stopped."
    binding = rc.build_fragment_context_binding(**inputs(text, "The motor stopped."))
    assert all(r["text"] != intro for r in binding["offered_context"])


def test_visibility_hole_and_scan_budget():
    intro = "Professor Dana Vale said:"
    text = intro + "\n\n> Earlier.\n\nVale continued.\n\n> The motor stopped."
    args = inputs(text, "The motor stopped.")
    hole = text.index("Earlier")
    args["visible_intervals"] = [[0, hole], [hole + 3, len(text)]]
    assert all(
        r["text"] != intro for r in rc.build_fragment_context_binding(**args)["offered_context"]
    )
    text = intro + "\n\n> " + "A" * 1600 + "\n\nVale continued.\n\n> The motor stopped."
    assert all(
        r["text"] != intro
        for r in rc.build_fragment_context_binding(**inputs(text, "The motor stopped."))[
            "offered_context"
        ]
    )


def test_multiple_fragments_and_corruption():
    text = "First item. Surrounding words. Second item. More words."
    args = inputs(text, "First item.")
    pos = text.index("Second item.")
    args["cited_fragments"].append(
        {"span_ids": ["s2"], "start_offset": pos, "end_offset": pos + len("Second item.")}
    )
    binding = rc.build_fragment_context_binding(**args)
    assert len(binding["offered_context"]) <= 6
    for r in binding["offered_context"]:
        assert all(
            r["char_end"] <= f["start_offset"] or r["char_start"] >= f["end_offset"]
            for f in args["cited_fragments"]
        )
        assert len(r["text"]) <= 384 and len(r["text"].encode()) <= 1536
    assert sum(len(r["text"]) for r in binding["offered_context"]) <= 2304
    assert sum(len(r["text"].encode()) for r in binding["offered_context"]) <= 9216
    assert rc.validate_fragment_context_binding(binding, **args) == ""
    for field, value in [
        ("char_start", 999),
        ("text", "forged"),
        ("context_id", "ctx-fake"),
        ("relation", "forged"),
    ]:
        changed = deepcopy(binding)
        changed["offered_context"][0][field] = value
        item = changed["offered_context"][0]
        item["text_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
        assert rc.validate_fragment_context_binding(changed, **args)
    changed = deepcopy(binding)
    changed["policy"] = "review-context-v99"
    assert rc.validate_fragment_context_binding(changed, **args)
    use = [{"context_id": binding["offered_context"][0]["context_id"], "purpose": "attribution"}]
    assert rc.validate_model_context_uses(use, binding) == ""
    use[0]["context_id"] = "ctx-foreign"
    assert rc.validate_model_context_uses(use, binding)


def test_open_fence_before_intro_is_not_crossed():
    intro = "Professor Vale said:"
    text = "```markdown\n\n" + intro + "\n\n> Earlier.\n\nVale continued.\n\n> The motor stopped."
    binding = rc.build_fragment_context_binding(**inputs(text, "The motor stopped."))
    assert all(r["text"] != intro for r in binding["offered_context"])
