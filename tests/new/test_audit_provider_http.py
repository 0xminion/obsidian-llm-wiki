"""Exercise the provider wire contract through real httpx request serialization."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.providers import llm


@pytest.fixture
def http_transport(monkeypatch):
    """Replace only the transport; keep real clients, requests and status errors."""
    client_type = httpx.Client

    def install(handler):
        def client(**kwargs):
            return client_type(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(llm.httpx, "Client", client)

    return install


def completion(content: Any = "{}", finish_reason: str | None = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def test_structured_synthesis_translates_ollama_parameters_for_openai(http_transport):
    from obsidian_llm_wiki.core.pipeline import (
        _SYNTHESIS_RESPONSE_SCHEMA,
        _call_structured_synthesis,
    )

    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        if {"format", "options", "num_ctx"} & body.keys():
            return httpx.Response(400, json={"error": "unsupported Ollama parameters"})
        return httpx.Response(200, json=completion())

    http_transport(handle)
    config = Config(
        llm=LLMProviderConfig(provider="openai", host="https://fixture.invalid"), retry_count=1
    )
    messages = [{"role": "user", "content": "prompt"}]
    assert asyncio.run(_call_structured_synthesis("prompt", messages, config)) == "{}"
    assert len(bodies) == 1
    body = bodies[0]
    assert body["max_tokens"] == 16_384
    assert body["temperature"] == 0
    assert body["stream"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == _SYNTHESIS_RESPONSE_SCHEMA
    assert body["response_format"]["json_schema"]["name"]


@pytest.mark.parametrize("explicit", [False, True])
def test_openai_json_mode_and_portable_options_respect_native_overrides(http_transport, explicit):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion("answer"))

    http_transport(handle)
    options = {
        "num_predict": 512,
        "num_ctx": 2048,
        "temperature": 0,
        "top_p": 0.8,
        "seed": 42,
        "stop": ["END"],
        "top_k": 40,
    }
    original = copy.deepcopy(options)
    kwargs = {"format": "json", "options": options, "num_ctx": 4096}
    if explicit:
        kwargs.update(max_completion_tokens=256, temperature=0.2, response_format={"type": "text"})
    client = llm.OpenAICompatibleClient(LLMProviderConfig(host="https://fixture.invalid"))
    assert client.chat("system", "user", **kwargs) == "answer"
    body = bodies[0]
    assert not ({"options", "format", "num_ctx", "top_k"} & body.keys())
    assert body["response_format"] == {"type": "text" if explicit else "json_object"}
    assert body["temperature"] == (0.2 if explicit else 0)
    assert body["top_p"] == 0.8
    assert body["seed"] == 42
    assert body["stop"] == ["END"]
    if explicit:
        assert body["max_completion_tokens"] == 256
        assert "max_tokens" not in body
    else:
        assert body["max_tokens"] == 512
    assert options == original


@pytest.mark.parametrize("prefix", ["", "/gateway"])
@pytest.mark.parametrize("version", ["", "/v1", "/v1/"])
def test_versioned_base_url_is_shared_by_chat_discovery_and_preflight(
    http_transport,
    prefix,
    version,
):
    seen = []

    def handle(request):
        seen.append(request.url.path)
        if request.url.path == f"{prefix}/v1/chat/completions":
            return httpx.Response(200, json=completion("answer"))
        if request.url.path == f"{prefix}/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        return httpx.Response(404)

    http_transport(handle)
    config = LLMProviderConfig(
        provider="openai",
        model="model",
        api_key="fixture-key",
        host=f"https://fixture.invalid{prefix}{version}",
    )
    assert llm.OpenAICompatibleClient(config).chat("system", "user") == "answer"
    assert llm.list_provider_models(config) == ["model"]
    check = llm.check_provider(config)
    assert check["ok"] is True
    assert check["endpoint"]["url"] == f"https://fixture.invalid{prefix}/v1/models"
    assert seen == [f"{prefix}/v1/chat/completions", f"{prefix}/v1/models", f"{prefix}/v1/models"]


@pytest.mark.parametrize(
    "reason", ["length", "content_filter", "tool_calls", "function_call", None],
)
def test_openai_rejects_unsuccessful_finish_even_when_content_is_valid_json(http_transport, reason):
    http_transport(lambda _: httpx.Response(200, json=completion("{}", reason)))
    client = llm.OpenAICompatibleClient(LLMProviderConfig(host="https://fixture.invalid"))
    with pytest.raises(ValueError, match="response"):
        client.chat("system", "user")


@pytest.mark.parametrize(
    "payload",
    [
        completion(None),
        completion(""),
        completion(" \n"),
        completion([{"text": "partial"}]),
        {
            "choices": [
                {"message": {"content": "{}", "refusal": "refused"}, "finish_reason": "stop"}
            ]
        },
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": None},
        {"choices": []},
        [],
        {"error": "upstream failure"},
    ],
)
def test_openai_rejects_unusable_nonstreaming_responses(http_transport, payload):
    http_transport(lambda _: httpx.Response(200, json=payload))
    client = llm.OpenAICompatibleClient(LLMProviderConfig(host="https://fixture.invalid"))
    with pytest.raises(ValueError, match="response"):
        client.chat("system", "user")


@pytest.mark.parametrize(
    "payload",
    [
        {"done": True, "done_reason": "length", "message": {"content": "{}"}},
        {"done": False, "message": {"content": "{}"}},
        {"done": None, "message": {"content": "{}"}},
        {"done": True, "done_reason": None, "message": {"content": "{}"}},
        {"done": True, "message": {"content": None}},
        {"done": True, "message": {"content": " \n"}},
        {"done": True, "message": None},
        {"error": "upstream failure"},
        [],
    ],
)
def test_ollama_rejects_incomplete_or_unusable_responses(http_transport, payload):
    http_transport(lambda _: httpx.Response(200, json=payload))
    client = llm.OllamaClient(LLMProviderConfig(host="https://fixture.invalid"))
    with pytest.raises(ValueError, match="response"):
        client.chat("system", "user")


@pytest.mark.parametrize("provider", ["openai", "ollama"])
@pytest.mark.parametrize("metadata", [True, False])
def test_legacy_text_responses_remain_supported(http_transport, provider, metadata):
    payload = (
        completion("answer")
        if provider == "openai"
        else {
            "message": {"content": "answer"},
            "done": True,
            "done_reason": "stop",
        }
    )
    if not metadata:
        if provider == "openai":
            del payload["choices"][0]["finish_reason"]
        else:
            del payload["done"]
            del payload["done_reason"]
    http_transport(lambda _: httpx.Response(200, json=payload))
    config = LLMProviderConfig(provider=provider, host="https://fixture.invalid")
    assert llm.create_llm_client(config).chat("system", "user") == "answer"


@pytest.mark.parametrize("structured", [False, True])
def test_truncation_retries_but_never_returns_partial_success(
    http_transport, monkeypatch, structured
):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=completion("{}", "length"))

    http_transport(handle)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    config = Config(
        llm=LLMProviderConfig(provider="openai", host="https://fixture.invalid"), retry_count=2
    )
    messages = [{"role": "user", "content": "user"}] if structured else "user"
    with pytest.raises(ValueError, match="response"):
        llm.call_llm("system", messages, config)
    assert len(requests) == 2


@pytest.mark.parametrize("structured", [False, True])
def test_exponential_retry_delay_is_bounded(http_transport, monkeypatch, structured):
    http_transport(lambda _: httpx.Response(503))
    delays = []
    monkeypatch.setattr(llm.time, "sleep", delays.append)
    config = Config(
        llm=LLMProviderConfig(provider="openai", host="https://fixture.invalid"),
        retry_count=4,
        retry_base_ms=3_600_000,
        retry_multiplier=10,
    )
    messages = [{"role": "user", "content": "user"}] if structured else "user"
    with pytest.raises(httpx.HTTPStatusError):
        llm.call_llm("system", messages, config)
    assert delays == [3600, 3600, 3600]


def test_ollama_retains_native_schema_and_does_not_mutate_callers_options(http_transport):
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True, "message": {"content": "{}"}})

    http_transport(handle)
    options = {"temperature": 0}
    schema = {"type": "object"}
    client = llm.OllamaClient(
        LLMProviderConfig(host="https://fixture.invalid", context_window=4096)
    )
    assert client.chat("system", "user", format=schema, options=options) == "{}"
    assert bodies[0]["format"] == schema
    assert bodies[0]["options"] == {"temperature": 0, "num_predict": 16_384, "num_ctx": 4096}
    assert options == {"temperature": 0}
