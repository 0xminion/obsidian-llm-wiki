"""Grounded lifecycle checks use scripted producer/reviewer, not live semantics."""

import json

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc
from obsidian_llm_wiki.core.pipeline import run_pipeline
from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
from tests.new.test_immutable_publication import human_state

FACT = "Amber Relay is device AR-100 made by Lab A."


def setup(tmp_path, monkeypatch):
    config = Config(
        vault_path=str(tmp_path),
        retry_count=1,
        retry_base_ms=0,
        concept_min_body_chars=0,
        entry_min_body_chars=0,
    )
    config.sources_dir.mkdir(parents=True)

    def producer(prompt):
        value = "29" if "buffers exactly 29" in prompt else "17"
        sentence = f"Amber Relay buffers exactly {value} messages."
        return grounded_response(
            title="Amber",
            summary=sentence,
            concepts=[
                {
                    "title": "Amber Relay",
                    "slug": "amber-relay",
                    "summary": sentence,
                    "sections": [{"heading": "Operation", "prose": sentence}],
                    "claims": [{"text": sentence, "evidence_span_ids": cite(prompt, sentence)}],
                }
            ],
        )

    provider = GroundedProvider(producer, identity_verdicts=lambda item: "same")
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    return config, provider


def source(config, value):
    doc = SourceDoc("Amber", FACT + f" Amber Relay buffers exactly {value} messages.\n")
    (config.sources_dir / "amber.md").write_text(doc.content)
    return {"amber.md": doc}


def current(config):
    marker = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    return config.wiki_dir / "OLW-Publications/generations" / marker["generation"], marker


async def test_failed_revision_keeps_original_cited_source_and_warning_then_retries(
    tmp_path, monkeypatch
):
    config, provider = setup(tmp_path, monkeypatch)
    initial = source(config, "17")
    first = await run_pipeline(tmp_path, initial, config)
    assert not first.errors
    old, old_marker = current(config)
    frozen = {p: human_state(p) for p in old.rglob("*") if p.is_file()}
    updated = source(config, "29")
    producer = provider.producer

    def fail(prompt):
        raise RuntimeError("synthetic provider unavailable")

    provider.producer = fail
    result = await run_pipeline(tmp_path, updated, config)
    assert result.errors
    directory, marker = current(config)
    assert marker["errors"]
    page = (directory / "concepts/amber-relay.md").read_text()
    assert "17 messages" in page and "Not current verified" in page
    rendered_source = (directory / "sources/amber.md").read_text()
    assert "17 messages" in rendered_source and "29 messages" not in rendered_source
    assert FACT in rendered_source  # a title prefix is not a duplicate title line
    assert "29 messages" in (directory / ".llmwiki/source-inputs/amber.md").read_text()
    calls = len(provider.calls)
    again = await run_pipeline(tmp_path, updated, config)
    assert again.errors and len(provider.calls) == calls  # persisted retry/backoff, not clean noop
    provider.producer = producer
    recovered = await run_pipeline(tmp_path, updated, config, force=True)
    assert not recovered.errors
    latest, _ = current(config)
    assert "29 messages" in (latest / "concepts/amber-relay.md").read_text()
    calls = len(provider.calls)
    noop = await run_pipeline(tmp_path, updated, config)
    assert not noop.errors and noop.compiled == 0 and len(provider.calls) == calls
    assert frozen == {p: human_state(p) for p in frozen}


async def test_deletion_preserves_annotation_and_historical_generation(tmp_path, monkeypatch):
    config, provider = setup(tmp_path, monkeypatch)
    docs = source(config, "17")
    first = await run_pipeline(tmp_path, docs, config, annotation_templates=True)
    assert not first.errors
    old, marker = current(config)
    cid = next(iter(marker["entities"]))
    annotation = config.annotations_dir / "entities" / (cid + ".md")
    annotation.write_text("Human history survives source deletion\n")
    human = human_state(annotation)
    frozen = {p: human_state(p) for p in old.rglob("*") if p.is_file()}
    (config.sources_dir / "amber.md").unlink()  # explicit human source deletion
    result = await run_pipeline(tmp_path, {}, config)
    assert not result.errors and result.deleted == 1
    latest, _ = current(config)
    assert "orphan" in (latest / "concepts/amber-relay.md").read_text().lower()
    assert human_state(annotation) == human
    assert frozen == {p: human_state(p) for p in frozen}
    calls = len(provider.calls)
    noop = await run_pipeline(tmp_path, {}, config)
    assert not noop.errors and len(provider.calls) == calls


async def test_bilingual_reuse_keeps_identity_prose_and_annotation(tmp_path, monkeypatch):
    from obsidian_llm_wiki.render.frontmatter import extract_wikilinks, parse_frontmatter

    config, provider = setup(tmp_path, monkeypatch)
    docs = source(config, "17")
    await run_pipeline(tmp_path, docs, config, annotation_templates=True)
    old, marker = current(config)
    cid = next(iter(marker["entities"]))
    prior_producer = provider.producer
    zh = "琥珀继电器可以缓冲十七条消息。"

    def producer(prompt):
        if zh not in prompt:
            return prior_producer(prompt)
        return grounded_response(
            title="琥珀",
            summary=zh,
            language="zh",
            concepts=[
                {
                    "title": "琥珀继电器",
                    "slug": "琥珀继电器",
                    "aliases": ["Amber Relay"],
                    "summary": zh,
                    "sections": [{"heading": "操作", "prose": zh}],
                    "claims": [{"text": zh, "evidence_span_ids": cite(prompt, zh)}],
                }
            ],
        )

    provider.producer = producer
    docs["amber-zh.md"] = SourceDoc("琥珀", FACT + " " + zh + "\n")
    (config.sources_dir / "amber-zh.md").write_text(docs["amber-zh.md"].content)
    result = await run_pipeline(tmp_path, docs, config)
    assert not result.errors
    latest, manifest = current(config)
    assert list(manifest["entities"]) == [cid]
    page = latest / manifest["entities"][cid]
    text = page.read_text()
    assert parse_frontmatter(text)[0]["canonical_id"] == cid
    assert zh in text and "17 messages" in text
    assert f"[[03-Raw-Annotations/entities/{cid}|Annotations]]" in text
    # Exact vault-root file resolution, not a basename-only fake Obsidian resolver.
    for path in latest.rglob("*.md"):
        if ".llmwiki" in path.relative_to(latest).parts:
            continue
        for target, _ in extract_wikilinks(path.read_text()):
            target = target.split("#", 1)[0]
            if not target:
                continue
            assert target.startswith(("04-Wiki/", "03-Raw-Annotations/")), (path, target)
            assert (tmp_path / (target.removesuffix(".md") + ".md")).is_file(), (path, target)
