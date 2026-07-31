"""Tests for obsidian_llm_wiki.synth.parser — JSON validation."""

from __future__ import annotations

import json

from obsidian_llm_wiki.synth.parser import (
    parse_single_source_synthesis,
    parse_synthesis_response,
)

# ── parse_single_source_synthesis ────────────────────────────────────────


def test_parse_single_clean_json():
    response = json.dumps({
        "source_title": "Paper",
        "source_summary": "A summary",
        "concepts": [
            {"title": "C", "slug": "c", "summary": "S", "tags": ["t1"]},
        ],
    })
    synth = parse_single_source_synthesis(response)
    assert synth is not None
    assert synth.source_title == "Paper"
    assert len(synth.concepts) == 1


def test_parse_single_with_prose_around():
    response = 'Here is the synthesis:\n```json\n{"source_title": "P", "source_summary": "S"}\n```\nDone.'
    synth = parse_single_source_synthesis(response)
    assert synth is not None
    assert synth.source_title == "P"


def test_parse_single_repairs_invalid_latex_escape():
    """A literal LaTeX command is not a legal JSON escape sequence."""
    response = r'{"source_title":"P","source_summary":"Uses $\epsilon$ risk."}'

    synth = parse_single_source_synthesis(response)

    assert synth is not None
    assert synth.source_summary == r"Uses $\epsilon$ risk."


def test_parse_single_repairs_trailing_comma():
    response = '{"source_title":"P","source_summary":"S",}'

    synth = parse_single_source_synthesis(response)

    assert synth is not None
    assert synth.source_title == "P"


def test_parse_single_empty():
    assert parse_single_source_synthesis("") is None
    assert parse_single_source_synthesis("   ") is None


def test_parse_single_no_json():
    assert parse_single_source_synthesis("just text, no json") is None


def test_parse_single_array_takes_first():
    response = json.dumps([
        {"source_title": "First", "source_summary": "S1"},
        {"source_title": "Second", "source_summary": "S2"},
    ])
    synth = parse_single_source_synthesis(response)
    assert synth is not None
    assert synth.source_title == "First"


def test_parse_single_repairs_invalid_latex_double_escape():
    response = (
        '{"source_title":"Paper","source_summary":"Uses \\\\epsilon notation",'
        '"concepts":[]}'
    )

    synth = parse_single_source_synthesis(response)

    assert synth is not None
    assert "epsilon" in synth.source_summary


def test_parse_single_repairs_truncated_nested_json():
    response = (
        '{"source_title":"Paper","source_summary":"Summary",'
        '"concepts":[{"title":"Concept","slug":"concept","summary":"Body"}'
    )

    synth = parse_single_source_synthesis(response)

    assert synth is not None
    assert synth.concepts[0].slug == "concept"


# ── parse_synthesis_response ─────────────────────────────────────────────


def test_parse_bundle_multiple_sources():
    response = json.dumps([
        {"source_title": "A", "source_summary": "SA", "concepts": [
            {"title": "CA", "slug": "ca", "summary": "s"},
        ]},
        {"source_title": "B", "source_summary": "SB", "concepts": [
            {"title": "CB", "slug": "cb", "summary": "s"},
        ]},
    ])
    bundle = parse_synthesis_response(response)
    assert len(bundle.sources) == 2
    assert len(bundle.concepts) == 0  # not merged yet — merge is in dedupe
    assert bundle.errors == []


def test_parse_bundle_single_object():
    response = json.dumps({"source_title": "A", "source_summary": "S"})
    bundle = parse_synthesis_response(response)
    assert len(bundle.sources) == 1
    assert bundle.sources[0].source_title == "A"


def test_parse_bundle_empty():
    bundle = parse_synthesis_response("")
    assert bundle.sources == []
    assert len(bundle.errors) == 1


def test_parse_bundle_no_json():
    bundle = parse_synthesis_response("no json here")
    assert bundle.sources == []
    assert len(bundle.errors) == 1


def test_parse_bundle_with_code_fence():
    response = '```json\n[{"source_title": "A", "source_summary": "S"}]\n```'
    bundle = parse_synthesis_response(response)
    assert len(bundle.sources) == 1


def test_parse_bundle_invalid_items_recorded_as_errors():
    response = json.dumps([
        {"source_title": "Good", "source_summary": "S"},
        "not an object",
        {"source_title": "Also Good", "source_summary": "S2"},
    ])
    bundle = parse_synthesis_response(response)
    assert len(bundle.sources) == 2
    assert len(bundle.errors) == 1
