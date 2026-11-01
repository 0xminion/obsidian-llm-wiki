"""Full application-schema wire checks; HTTP responses are authored offline fixtures."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.providers import llm
from obsidian_llm_wiki.synth.response_schema import SYNTHESIS_RESPONSE_SCHEMA


@pytest.fixture
def wire(monkeypatch, request):
    """Exercise real httpx serialization; intercept only the HTTP transport."""
    client_type = httpx.Client
    bodies = []

    def install(content="{}", provider="openai", status=200):
        def handle(outbound):
            assert outbound.method == "POST"
            assert outbound.url.host == "fixture.invalid"
            assert outbound.url.path == (
                "/api/chat" if provider == "ollama" else "/v1/chat/completions"
            )
            assert "authorization" not in outbound.headers
            body = json.loads(outbound.content)
            bodies.append(body)
            receipts = os.environ.get("STRICT_WIRE_RECEIPTS")
            if receipts:
                target = Path(receipts) / request.node.name / f"{len(bodies)}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(outbound.content)
            payload = (
                {"done": True, "done_reason": "stop", "message": {"content": content}}
                if provider == "ollama" else
                {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            )
            return httpx.Response(status, json=payload)

        monkeypatch.setattr(
            llm.httpx, "Client",
            lambda **kwargs: client_type(
                transport=httpx.MockTransport(handle), trust_env=False, **kwargs,
            ),
        )
        return bodies

    return install


def test_explicit_strict_schema_opt_in_is_serialized_for_full_producer_schema(wire):
    bodies = wire()
    schema = copy.deepcopy(SYNTHESIS_RESPONSE_SCHEMA)
    before = copy.deepcopy(schema)
    llm.OpenAICompatibleClient(LLMProviderConfig(host="https://fixture.invalid")).chat(
        "system", "user", format=schema, strict_schema=True,
    )
    assert bodies[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": before, "strict": True},
    }
    assert "strict_schema" not in bodies[0]
    assert schema == before


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"label": {"type": "string"}}},
    SYNTHESIS_RESPONSE_SCHEMA,
])
def test_generic_schema_callers_are_not_silently_forced_strict(wire, schema):
    bodies = wire()
    llm.OpenAICompatibleClient(LLMProviderConfig(host="https://fixture.invalid")).chat(
        "system", "user", format=schema,
    )
    assert bodies[0]["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "response", "schema": schema},
    }


def config_for(provider):
    return Config(
        llm=LLMProviderConfig(provider=provider, host="https://fixture.invalid"),
        retry_count=1,
    )


def assert_application_schema(body, provider, schema):
    assert "strict_schema" not in body
    if provider == "ollama":
        assert body["format"] == schema
        assert "response_format" not in body
    else:
        assert "format" not in body
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_producer_call_site_selects_strict_without_changing_native_schema(wire, provider):
    from obsidian_llm_wiki.core.pipeline import _call_structured_synthesis

    bodies = wire(provider=provider)
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert asyncio.run(_call_structured_synthesis("unused", messages, config_for(provider))) == "{}"
    assert len(bodies) == 1
    assert bodies[0]["messages"] == messages
    assert_application_schema(bodies[0], provider, SYNTHESIS_RESPONSE_SCHEMA)


def test_native_ollama_consumes_strict_opt_in_without_transmitting_it(wire):
    bodies = wire(provider="ollama")
    options = {"temperature": 0, "num_predict": 512}
    original = copy.deepcopy(options)
    client = llm.OllamaClient(config_for("ollama").llm)
    client.chat("system", "user", format=SYNTHESIS_RESPONSE_SCHEMA,
                strict_schema=True, options=options)
    assert_application_schema(bodies[0], "ollama", SYNTHESIS_RESPONSE_SCHEMA)
    assert bodies[0]["options"]["num_predict"] == 512
    assert options == original


def review_fixture():
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog
    from obsidian_llm_wiki.synth import review

    content = "The ledger is an append-only record.\nIt stores completed requests.\n"
    catalog = build_span_catalog(content, "ledger.md")
    window = catalog.windows(10_000)[0]
    evidence = catalog.cite([catalog.spans[-1].id], offered=window)
    units = [
        review.ReviewUnit(
            key="c" * 32, kind="claim", path="concepts[0].claims[0]",
            text="The ledger stores completed requests.", subject="Ledger",
            cited_span_ids=evidence.span_ids, cited_text=evidence.quote,
        ),
        review.ReviewUnit(
            key="s" * 32, kind="source_summary", path="source_summary[0]",
            text="The ledger stores completed requests.",
        ),
    ]
    batch = [(f"u{i + 1}", unit) for i, unit in enumerate(units)]
    bindings = {unit.key: review.context_binding_for_unit(unit, catalog, window) for unit in units}
    assert bindings[units[0].key]["offered_context"]
    answers = {uid: {
        "text_sha256": unit.digest, "verdict": "supported",
        "support_span_ids": list(evidence.span_ids), "rationale": "Offline test fixture",
        "model_context_uses": [],
    } for uid, unit in batch}
    return catalog, window, units, review._schema(batch, bindings), {"reviews": answers}


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_reviewer_call_site_transmits_full_bound_schema(wire, provider):
    from obsidian_llm_wiki.synth import review

    catalog, window, units, schema, response = review_fixture()
    bodies = wire(json.dumps(response), provider=provider)
    outcomes = asyncio.run(review._review_units(units, catalog, window, config_for(provider)))
    assert set(outcomes) == {unit.key for unit in units}
    assert len(bodies) == 1
    assert_application_schema(bodies[0], provider, schema)


def identity_fixture():
    from obsidian_llm_wiki.core.identity_context import source_context
    from obsidian_llm_wiki.core.identity_plan import digest
    from obsidian_llm_wiki.core.models import ConceptNote, SourceSynthesis
    from obsidian_llm_wiki.core.span_catalog import build_span_catalog

    text = "Ledger stores completed requests."
    catalog = build_span_catalog(text, "ledger.md")
    synthesis = SourceSynthesis(
        "Ledger", text, source_file=catalog.source_file,
        concepts=[ConceptNote("Ledger", "ledger", text)],
        # Minimal authored context fixture, never a claimed model approval.
        review_receipt={"status": "approved", "source_sha256": catalog.revision},
    )
    owner = copy.deepcopy(synthesis)
    owner.source_file = "owner.md"
    candidate_context = source_context(owner, 0, text, side="candidate")
    subject_context = source_context(synthesis, 0, text)
    request = {
        "concept_index": 0, "subject_title": "Ledger", "subject_summary": text,
        "subject_sha256": digest({"subject": text}), "candidate_id": "fixture-ledger",
        "candidate_title": "Ledger", "candidate_summary": text,
        "candidate_sha256": digest({"candidate": text}), "anchor_source": "owner.md",
        "candidate_context": {"complete": True, "owners": [candidate_context]},
    }
    answer = {key: request[key] for key in (
        "candidate_id", "candidate_sha256", "subject_sha256",
    )}
    answer.update(
        verdict="same", rationale="Offline test fixture",
        support_span_ids=[subject_context["spans"][0]["support_id"],
                          candidate_context["spans"][0]["support_id"]],
    )
    return synthesis, request, catalog, {"assessments": {"a1": answer}}


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_identity_call_site_transmits_full_bound_schema(wire, provider):
    from obsidian_llm_wiki.synth import identity

    synthesis, request, catalog, response = identity_fixture()
    bodies = wire(json.dumps(response), provider=provider)
    records = asyncio.run(identity.assess_identity(
        synthesis, [request], catalog, config_for(provider),
    ))
    assert len(records) == 1
    assert records[0]["verdict"] == "same"
    assert len(bodies) == 1
    assert_application_schema(bodies[0], provider, identity._schema([("a1", request)]))


@pytest.mark.parametrize("override", [
    None,
    {},
    {"type": "text"},
    {"type": "json_object"},
    {"type": "json_schema", "json_schema": {
        "name": "caller-owned", "schema": SYNTHESIS_RESPONSE_SCHEMA, "strict": False,
    }},
    {"type": "json_schema", "json_schema": {
        "name": "caller-owned", "schema": SYNTHESIS_RESPONSE_SCHEMA, "strict": True,
    }},
])
def test_explicit_response_format_wins_even_when_falsey_or_non_strict(wire, override):
    bodies = wire()
    before = copy.deepcopy(override)
    llm.OpenAICompatibleClient(config_for("openai").llm).chat(
        "system", "user", format=SYNTHESIS_RESPONSE_SCHEMA,
        strict_schema=True, response_format=override,
    )
    assert bodies[0]["response_format"] == before
    assert "format" not in bodies[0] and "strict_schema" not in bodies[0]
    assert override == before


@pytest.mark.parametrize("output_format", [None, "json"])
@pytest.mark.parametrize("opt_in", [False, True])
def test_opt_in_does_not_change_non_schema_requests(wire, output_format, opt_in):
    bodies = wire()
    llm.OpenAICompatibleClient(config_for("openai").llm).chat(
        "system", "user", format=output_format, strict_schema=opt_in,
    )
    assert "strict_schema" not in bodies[0]
    if output_format is None:
        assert "response_format" not in bodies[0]
    else:
        assert bodies[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("structured", [False, True])
def test_strict_request_retries_do_not_mutate_or_downgrade(wire, monkeypatch, structured):
    bodies = wire(status=400)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    config = config_for("openai")
    config.retry_count = 2
    options = {"temperature": 0, "num_predict": 512}
    schema = copy.deepcopy(SYNTHESIS_RESPONSE_SCHEMA)
    before = copy.deepcopy((schema, options))
    messages = [{"role": "user", "content": "user"}] if structured else "user"
    with pytest.raises(httpx.HTTPStatusError):
        llm.call_llm("system", messages, config, format=schema, strict_schema=True, options=options)
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert_application_schema(bodies[0], "openai", schema)
    assert (schema, options) == before


def test_reviewer_still_rejects_semantically_invalid_support_after_strict_request(wire):
    from obsidian_llm_wiki.synth import review

    catalog, window, units, schema, response = review_fixture()
    response["reviews"]["u1"]["support_span_ids"] = ["foreign-span"]
    bodies = wire(json.dumps(response))
    with pytest.raises(review.SynthesisReviewError, match="invalid support span"):
        asyncio.run(review._review_units(units, catalog, window, config_for("openai")))
    assert_application_schema(bodies[0], "openai", schema)


def test_identity_still_fails_closed_after_strict_request(wire):
    from obsidian_llm_wiki.synth import identity

    synthesis, request, catalog, response = identity_fixture()
    response["assessments"]["a1"]["support_span_ids"] = ["foreign-span"]
    bodies = wire(json.dumps(response))
    records = asyncio.run(identity.assess_identity(
        synthesis, [request], catalog, config_for("openai"),
    ))
    assert len(records) == 1 and records[0]["verdict"] == "unassessed"
    assert "not offered" in records[0]["reason"]
    assert_application_schema(bodies[0], "openai", identity._schema([("a1", request)]))
