"""HTTP-only doubles: real parser, review, queue, render and publication."""
import json

import httpx
import pytest

from obsidian_llm_wiki.config import Config, LLMProviderConfig
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.pipeline import recompile_single_source, run_pipeline
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir
from tests.new.grounding_doubles import cite, grounded_response, identity_answer, review_answer
from tests.new.immutable_semantic_helpers import file_state, published_registry


class ScriptedHTTP:
    """Synthetic responses prove lifecycle plumbing, not semantic accuracy."""

    def __init__(self, config):
        self.config = config
        self.fail_stage = None
        self.remove_empty_dirs = False
        self.calls = []

    def handle(self, request):
        assert request.method == "POST"
        assert str(request.url) == "http://offline.invalid/v1/chat/completions"
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        schema = body["response_format"]["json_schema"]["schema"]
        props = schema["properties"]
        stage = "review" if "reviews" in props else (
            "identity" if "assessments" in props else "producer"
        )
        self.calls.append(stage)
        if self.remove_empty_dirs:
            self.remove_empty_dirs = False
            root = self.config.wiki_dir / "OLW-Publications"
            for name in ("generations", "query-state"):
                (root / name).rmdir()  # empty-directory cleanup only
        if stage == self.fail_stage:
            return httpx.Response(502, json={"error": "Codex upstream request failed"})
        if stage == "review":
            text = json.dumps(review_answer(prompt, schema, {}))
        elif stage == "identity":
            text = json.dumps(identity_answer(prompt, schema, lambda _item: "same"))
        else:
            value = "29" if "buffers exactly 29" in prompt else "17"
            sentence = f"Amber Relay buffers exactly {value} messages."
            text = grounded_response(title="Amber", summary=sentence, concepts=[{
                "title": "Amber Relay", "slug": "amber-relay", "summary": sentence,
                "sections": [{"heading": "Operation", "prose": sentence}],
                "claims": [{"text": sentence, "evidence_span_ids": cite(prompt, sentence)}],
            }])
        return httpx.Response(200, json={"choices": [{
            "message": {"role": "assistant", "content": text}, "finish_reason": "stop",
        }]})


@pytest.fixture
def wire(tmp_path, monkeypatch):
    config = Config(
        vault_path=str(tmp_path), embeddings_enabled=False,
        llm=LLMProviderConfig(provider="openai", host="http://offline.invalid/v1",
                              model="fixture", api_key="not-a-secret"),
        retry_count=1, retry_base_ms=0, compile_concurrency=1,
        concept_min_body_chars=0, entry_min_body_chars=0,
    )
    config.sources_dir.mkdir(parents=True)
    wire = ScriptedHTTP(config)
    original = httpx.Client

    class OfflineClient(original):
        def __init__(self, *args, **kwargs):
            kwargs.update(transport=httpx.MockTransport(wire.handle), trust_env=False)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", OfflineClient)
    return wire


def source(config, value):
    (config.sources_dir / "amber.md").write_text(
        "---\ntitle: Amber\n---\nAmber Relay is device AR-100 made by Lab A. "
        f"Amber Relay buffers exactly {value} messages.\n"
    )
    return load_sources_from_dir(config.sources_dir)


def assert_provider_failure(result):
    assert result.compiled == 0
    assert any("502 Bad Gateway" in error for error in result.errors), result.errors
    assert not any("No such file" in error for error in result.errors)


@pytest.mark.parametrize("stage", ["producer", "review"])
@pytest.mark.parametrize("cleanup", [False, True])
async def test_provider_failure_is_diagnostic_not_source_admission(wire, stage, cleanup):
    config = wire.config
    sources = source(config, "17")
    before = file_state(config.sources_dir)
    wire.fail_stage = stage
    wire.remove_empty_dirs = cleanup
    result = await run_pipeline(config.vault, sources, config)
    assert_provider_failure(result)
    directory, manifest = current_publication(config.wiki_dir)
    assert manifest["errors"] == result.errors
    marker = json.loads((directory / ".llmwiki/identity-publication.json").read_text())
    assert marker["source_payloads"] == {}
    assert not (directory / "sources/amber.md").exists()
    assert not (directory / "concepts/amber-relay.md").exists()
    assert (directory / ".llmwiki/source-inputs/amber.md").read_bytes() == before["amber.md"][0]
    assert file_state(config.sources_dir) == before
    assert not (directory / ".llmwiki/publication-stage").exists()
    with published_registry(config) as registry:
        assert registry.source_filenames() == set()
        assert registry.load_syntheses() == {}
        pending = registry.list_dirty(state="pending")
        assert len(pending) == 1 and "502 Bad Gateway" in pending[0].error
    count = len(wire.calls)
    backoff = await run_pipeline(config.vault, sources, config)
    assert backoff.errors and backoff.compiled == 0
    assert len(wire.calls) == count
    frozen_diagnostic = file_state(directory)
    wire.fail_stage = None
    recovered = await recompile_single_source(config.vault, "amber.md", config)
    assert recovered.compiled == 1 and not recovered.errors
    latest, manifest = current_publication(config.wiki_dir)
    assert latest != directory and not manifest["errors"]
    assert (latest / "sources/amber.md").exists()
    assert "17 messages" in (latest / "concepts/amber-relay.md").read_text()
    assert file_state(directory) == frozen_diagnostic
    assert file_state(config.sources_dir) == before
    count = len(wire.calls)
    noop = await run_pipeline(config.vault, sources, config)
    assert noop.skipped == 1 and noop.compiled == 0 and not noop.errors
    assert len(wire.calls) == count


async def test_failed_replacement_preserves_prior_generation_and_source_until_retry(wire):
    config = wire.config
    first = await run_pipeline(config.vault, source(config, "17"), config)
    assert first.compiled == 1 and not first.errors
    old, _ = current_publication(config.wiki_dir)
    frozen = file_state(old)
    updated = source(config, "29")
    before = file_state(config.sources_dir)
    wire.fail_stage = "review"
    result = await run_pipeline(config.vault, updated, config)
    assert_provider_failure(result)
    diagnostic, manifest = current_publication(config.wiki_dir)
    assert manifest["errors"] == result.errors
    page = (diagnostic / "concepts/amber-relay.md").read_text()
    assert "17 messages" in page and "29 messages" not in page
    assert "Not current verified" in page
    source_view = (diagnostic / "sources/amber.md").read_text()
    assert "17 messages" in source_view and "29 messages" not in source_view
    assert file_state(old) == frozen
    assert file_state(config.sources_dir) == before
    wire.fail_stage = None
    recovered = await recompile_single_source(config.vault, "amber.md", config)
    assert recovered.compiled == 1 and not recovered.errors
    latest, _ = current_publication(config.wiki_dir)
    assert "29 messages" in (latest / "concepts/amber-relay.md").read_text()
    assert file_state(old) == frozen
    assert file_state(config.sources_dir) == before


@pytest.mark.parametrize("replacement", ["symlink", "file", "owner-marker"])
async def test_install_revalidation_does_not_adopt_unowned_paths(wire, monkeypatch, replacement):
    from obsidian_llm_wiki.core import immutable_publication as publication

    config = wire.config
    sources = source(config, "17")
    before = file_state(config.sources_dir)
    outside = config.vault / "unowned"
    outside.mkdir()
    sentinel = outside / "sentinel.md"
    sentinel.write_bytes(b"human-owned content\r\n")
    untouched = file_state(outside)

    def substitute(event, **context):
        if event != "before_install":
            return
        parent = context["target"].parent
        if replacement == "owner-marker":
            (parent.parent / "owner.json").write_text('{"owner":"not-the-publisher"}')
        else:
            parent.rmdir()
            if replacement == "symlink":
                parent.symlink_to(outside, target_is_directory=True)
            else:
                parent.write_bytes(b"human-owned file")

    monkeypatch.setattr(publication, "_checkpoint", substitute)
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        await run_pipeline(config.vault, sources, config)
    assert not (config.wiki_dir / "OLW-Publications/current.json").exists()
    assert file_state(outside) == untouched
    assert file_state(config.sources_dir) == before
    parent = config.wiki_dir / "OLW-Publications/generations"
    if replacement == "symlink":
        assert parent.is_symlink() and parent.resolve() == outside
    elif replacement == "file":
        assert parent.read_bytes() == b"human-owned file"
    else:
        assert json.loads((parent.parent / "owner.json").read_text()) == {
            "owner": "not-the-publisher"
        }
