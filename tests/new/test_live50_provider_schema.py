"""Production schema regressions at the serialized HTTP boundary (no network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core import pipeline
from obsidian_llm_wiki.providers import llm


@pytest.fixture
def production_wire(monkeypatch):
    """Keep the real pipeline/client; replace only httpx's transport."""
    client_type = httpx.Client

    def send(content="{}", provider="openai"):
        requests = []

        def handle(request):
            requests.append(request)
            if provider == "ollama":
                payload = {"message": {"content": content}, "done": True, "done_reason": "stop"}
            else:
                payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            return httpx.Response(200, json=payload)

        monkeypatch.setattr(
            llm.httpx, "Client",
            lambda **kwargs: client_type(
                transport=httpx.MockTransport(handle), trust_env=False, **kwargs,
            ),
        )
        config = Config(
            llm=LLMProviderConfig(
                provider=provider, host="https://fixture.invalid/v1", model="fixture-model",
            ),
            retry_count=1,
        )
        messages = [
            {"role": "system", "content": pipeline._SYSTEM_PROMPT},
            {"role": "user", "content": "Source-backed synthesis fixture."},
        ]
        actual = asyncio.run(pipeline._call_structured_synthesis("unused", messages, config))
        assert actual == content
        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        return request, json.loads(request.content), messages

    return send


def test_production_wire_root_is_closed_and_requires_every_field(production_wire):
    request, body, messages = production_wire()
    assert str(request.url) == "https://fixture.invalid/v1/chat/completions"
    assert body == {
        "model": "fixture-model",
        "messages": messages,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": pipeline._SYNTHESIS_RESPONSE_SCHEMA},
        },
        "temperature": 0,
        "max_tokens": 16_384,
    }
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False
    assert set(schema["required"]) == set(schema["properties"])


def _assert_prompt_contract(schema, prompt_model, path="$", object_paths=None):
    """Compare the wire contract to the actual prompt's nested data model."""
    from obsidian_llm_wiki.core.models import VALID_RELATIONS, ProvenanceState

    if object_paths is None:
        object_paths = []
    if isinstance(prompt_model, dict):
        assert schema["type"] == "object", path
        assert schema.get("additionalProperties") is False, path
        properties = schema.get("properties", {})
        assert set(properties) == set(prompt_model), path
        assert set(schema.get("required", [])) == set(properties), path
        object_paths.append(path)
        for name, model in prompt_model.items():
            _assert_prompt_contract(properties[name], model, f"{path}.{name}", object_paths)
    elif isinstance(prompt_model, list):
        assert schema["type"] == "array", path
        # Arrays represent optional information; requiring keys must not require facts.
        assert schema.get("minItems", 0) == 0, path
        _assert_prompt_contract(schema["items"], prompt_model[0], f"{path}[]", object_paths)
    else:
        kind = "number" if path.endswith(".confidence") else (
            "boolean" if path.endswith(".is_new") else "string"
        )
        assert schema["type"] == kind, path
        if path.endswith(".relation"):
            assert set(schema["enum"]) == VALID_RELATIONS
        elif path.endswith(".provenance"):
            assert set(schema["enum"]) == {item.value for item in ProvenanceState}
        elif path.endswith(".confidence"):
            assert schema["minimum"] == 0
            assert schema["maximum"] == 1
        else:
            assert schema.get("minLength", 0) == 0, path
    return object_paths


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_production_wire_recursively_matches_prompt_objects(production_wire, provider):
    from obsidian_llm_wiki.synth.prompts import SYNTHESIS_SCHEMA

    _, body, _ = production_wire(provider=provider)
    schema = (body["format"] if provider == "ollama" else
              body["response_format"]["json_schema"]["schema"])
    paths = _assert_prompt_contract(schema, SYNTHESIS_SCHEMA)
    assert set(paths) == {
        "$", "$.concepts[]", "$.concepts[].sections[]", "$.concepts[].claims[]",
        "$.concepts[].related[]", "$.maps[]",
    }


def test_new_wire_contract_invalidates_v2_artifact_versions():
    versions = pipeline._artifact_versions_for_config(Config())
    assert versions.schema == "synthesis-schema-v3"


@pytest.fixture
def synthesis_payload():
    """Complete offline model-output fixture, not a claimed provider response."""
    return {
        "source_title": "Request accounting",
        "source_summary": "A durable ledger records outbound requests against a call budget.",
        "source_tags": [], "key_points": [], "open_questions": [], "language": "en",
        "concepts": [{
            "title": "Request ledger", "slug": "request-ledger", "canonical_id": "",
            "summary": "The ledger durably stores requests.", "tags": [], "aliases": [],
            "sections": [{
                "heading": "Storage", "points": ["The ledger durably stores requests."],
                "prose": "",
            }],
            "claims": [{
                "text": "The ledger stores recorded requests durably.",
                "quote": "Recorded requests are stored in a durable ledger.", "source_ref": "",
            }],
            "related": [{"slug": "request-budget", "relation": "related_to", "display": ""}],
            "confidence": 1.0, "provenance": "extracted", "is_new": True,
        }, {
            "title": "Request budget", "slug": "request-budget", "canonical_id": "",
            "summary": "The budget caps outbound calls.", "tags": [], "aliases": [],
            "sections": [{
                "heading": "Limit", "points": [],
                "prose": "The request budget caps outbound calls.",
            }],
            "claims": [], "related": [], "confidence": 0.9,
            "provenance": "extracted", "is_new": True,
        }],
        "maps": [{
            "title": "Request accounting", "slug": "request-accounting",
            "summary": "The ledger tracks the calls counted against the request budget.",
            "tags": [], "concept_slugs": ["request-ledger", "request-budget"],
        }],
    }


@pytest.fixture
def synthesis_source():
    from obsidian_llm_wiki.core.models import SourceDoc

    return SourceDoc(
        title="Request accounting",
        content=("Recorded requests are stored in a durable ledger. "
                 "The request budget caps outbound calls. "
                 "The ledger tracks the calls counted against the request budget."),
    )


def test_full_nested_response_preserves_optional_empty_values(
    production_wire, synthesis_payload, synthesis_source,
):
    from obsidian_llm_wiki.core.models import EvidenceSpan
    from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis
    from obsidian_llm_wiki.synth.validation import validate_synthesis

    text = json.dumps(synthesis_payload)
    production_wire(content=text)
    result = validate_synthesis(parse_single_source_synthesis(text), synthesis_source)
    ledger, budget = result.concepts
    assert ledger.canonical_id == ""
    assert ledger.sections[0].prose == ""
    assert ledger.related[0].display == ""
    assert ledger.claims[0].source_ref == ""
    assert isinstance(ledger.claims[0].evidence, EvidenceSpan)
    assert ledger.claims[0].evidence.verification == "verified"
    assert budget.sections[0].points == []
    assert budget.claims == budget.related == budget.aliases == budget.tags == []
    assert result.maps[0].concept_slugs == ["request-ledger", "request-budget"]


def test_zero_concepts_and_optional_information_remain_valid(
    production_wire, synthesis_payload, synthesis_source,
):
    from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis
    from obsidian_llm_wiki.synth.validation import validate_synthesis

    synthesis_payload.update(concepts=[], maps=[], language="")
    text = json.dumps(synthesis_payload)
    production_wire(content=text)
    result = validate_synthesis(parse_single_source_synthesis(text), synthesis_source)
    assert result.concepts == result.maps == result.key_points == result.open_questions == []
    assert result.source_tags == []
    assert result.language == ""


@pytest.mark.parametrize("defect", ["unmatched_quote", "empty_body", "placeholder", "nonanswer"])
def test_structured_shape_does_not_bypass_semantic_acceptance(
    production_wire, synthesis_payload, synthesis_source, defect,
):
    from obsidian_llm_wiki.synth.parser import parse_single_source_synthesis
    from obsidian_llm_wiki.synth.validation import SynthesisValidationError, validate_synthesis

    if defect == "unmatched_quote":
        synthesis_payload["concepts"][0]["claims"][0]["quote"] = "An invented supporting quote."
        diagnostic = "unmatched quote"
    elif defect == "empty_body":
        synthesis_payload["concepts"][0]["sections"] = []
        diagnostic = "no completed body"
    elif defect == "placeholder":
        synthesis_payload["source_summary"] = "TODO"
        diagnostic = "placeholder"
    else:
        synthesis_payload["source_summary"] = "I cannot access the source document."
        diagnostic = "provider inability/refusal"
    text = json.dumps(synthesis_payload)
    production_wire(content=text)
    parsed = parse_single_source_synthesis(text)
    assert parsed is not None  # Structurally parseable is not semantically accepted.
    with pytest.raises(SynthesisValidationError, match=diagnostic):
        validate_synthesis(parsed, synthesis_source)
