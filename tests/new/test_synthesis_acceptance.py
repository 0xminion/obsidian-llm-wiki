"""Fail-closed boundaries for model-generated synthesis (no live LLM calls)."""

import json

import pytest

from obsidian_llm_wiki.synth.parser import (
    parse_single_source_synthesis,
    parse_synthesis_response,
)


@pytest.mark.parametrize(
    "response",
    [
        '{"source_title":"Trial","source_summary":"The trial proved',
        '{"source_title":"Trial","source_summary":"No effect."',
        '{"source_title":"Trial","source_summary":"No effect.","concepts":[',
        '[{"source_title":"Trial","source_summary":"No effect."},',
        'Here is the result: {"source_title":"Trial","source_summary":"No effect.",'
        '"concepts":[{"title":"Effect"}',
    ],
)
def test_truncated_synthesis_is_not_fabricated_into_success(response):
    assert parse_single_source_synthesis(response) is None
    bundle = parse_synthesis_response(response)
    assert bundle.sources == []
    assert bundle.errors


@pytest.mark.parametrize("payload", [{}, {"source_title": "Trial"}, {"source_summary": " \n"}])
def test_empty_payload_is_not_a_source_synthesis(payload):
    assert parse_single_source_synthesis(json.dumps(payload)) is None
    bundle = parse_synthesis_response(json.dumps(payload))
    assert not bundle.sources
    assert bundle.errors


@pytest.mark.parametrize(
    "field", ["concepts", "maps", "key_points", "source_tags", "open_questions"]
)
@pytest.mark.parametrize("value", [None, 7, "not an array", {}, [None]])
def test_malformed_collections_reject_without_type_error(field, value):
    payload = {"source_title": "Trial", "source_summary": "No effect.", field: value}
    assert parse_single_source_synthesis(json.dumps(payload)) is None
    bundle = parse_synthesis_response(json.dumps(payload))
    assert not bundle.sources
    assert bundle.errors


@pytest.mark.parametrize("field", ["sections", "claims", "related", "tags", "aliases"])
@pytest.mark.parametrize("value", [None, 7, "not an array", {}, [None]])
def test_malformed_concept_collections_reject_without_type_error(field, value):
    payload = {
        "source_title": "Trial",
        "source_summary": "No effect.",
        "concepts": [
            {"title": "Effect", "slug": "effect", "summary": "No effect.", field: value},
        ],
    }
    assert parse_single_source_synthesis(json.dumps(payload)) is None


def test_trailing_comma_repair_preserves_punctuation_inside_text():
    response = '{"source_title":"Syntax","source_summary":"The token ,} closes a mapping.",}'
    result = parse_single_source_synthesis(response)
    assert result.source_summary == "The token ,} closes a mapping."


def test_single_source_does_not_silently_discard_additional_sources():
    response = json.dumps(
        [
            {"source_title": "One", "source_summary": "One finding."},
            {"source_title": "Two", "source_summary": "Another finding."},
        ]
    )
    assert parse_single_source_synthesis(response) is None


def test_deeply_nested_output_is_rejected_without_recursion_error():
    assert parse_single_source_synthesis("[" * 20_000 + "]" * 20_000) is None


@pytest.mark.parametrize("field", ["sections", "claims", "related", "aliases"])
@pytest.mark.parametrize("bad", [None, 3, "not an array", {}, [None]])
def test_expansion_parser_rejects_malformed_collections(field, bad):
    from obsidian_llm_wiki.synth.quality import _parse_concept_json

    assert _parse_concept_json(json.dumps({"title": "Result", field: bad})) is None


def test_advertised_relationships_survive_model_parsing():
    from obsidian_llm_wiki.core.models import normalize_relation
    from obsidian_llm_wiki.synth.prompts import SYNTHESIS_SCHEMA

    values = SYNTHESIS_SCHEMA["concepts"][0]["related"][0]["relation"].split("|")
    assert {value.strip() for value in values} == {
        normalize_relation(value.strip()) for value in values
    }


@pytest.mark.parametrize("confidence", [None, "high", True, -0.1, 1.1, "NaN", "Infinity"])
def test_invalid_confidence_is_not_promoted_to_certain(confidence):
    payload = {
        "source_summary": "No benefit was observed.",
        "concepts": [{"title": "Result", "summary": "No benefit.", "confidence": confidence}],
    }
    assert parse_single_source_synthesis(json.dumps(payload)) is None
