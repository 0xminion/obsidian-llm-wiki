"""Real CLI builds, scripted inference, exact vault-root navigation (not GUI QA)."""

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.cache import load_synthesis
from obsidian_llm_wiki.render.frontmatter import parse_frontmatter
from tests.new.grounding_doubles import GroundedProvider, cite, grounded_response
from tests.new.test_immutable_publication import human_state

QUOTE = "The router stores [[router]] literally, not a destination."
BODY = (
    "# Index\n\n" + QUOTE + "\n\n"
    "Use `[[router]]` and \\[[router]] literally.\n\n"
    "```text\n[[router]]\n```\n\n"
    "~~~text\n[[router]]\n~~~\n\n"
    "Human legacy reference [[04-Wiki/concepts/router|original]].\n"
)


def current(config):
    manifest = json.loads((config.wiki_dir / "OLW-Publications/current.json").read_text())
    generation = config.wiki_dir / "OLW-Publications/generations" / manifest["generation"]
    return generation, manifest


def test_real_cli_typed_collisions_preserve_literal_evidence_and_noop(tmp_path, monkeypatch):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False, min_source_chars=1)
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.build.resolve_vault", lambda vault: (tmp_path, config),
    )
    config.sources_dir.mkdir(parents=True)
    raw = config.sources_dir / "router.md"
    raw.write_text("---\ntitle: Index\n---\n" + BODY)
    legacy = config.concepts_dir / "router.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"# Human\r\n[[router]] and `[[router]]`\r\n")
    protected = {p: human_state(p) for p in (raw, legacy)}

    def producer(prompt):
        return grounded_response(
            title="Index", summary=QUOTE,
            concepts=[{
                "title": "Index", "slug": "router", "summary": QUOTE,
                "sections": [{"heading": "Notation", "prose": QUOTE}],
                "claims": [{"text": QUOTE, "evidence_span_ids": cite(prompt, QUOTE)}],
            }],
            maps=[{"title": "Index", "slug": "router", "summary": QUOTE,
                   "concept_slugs": ["router"]}],
        )

    provider = GroundedProvider(producer)
    monkeypatch.setattr("obsidian_llm_wiki.providers.llm.acall_llm", provider)
    cli = CliRunner()
    result = cli.invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 0, (result.output, result.exception)
    generation, manifest = current(config)
    prefix = generation.relative_to(tmp_path).as_posix()
    entry = (generation / "entries/router.md").read_text()
    assert f"[[{prefix}/concepts/router]]" in entry
    assert f"[[{prefix}/sources/router|Index]]" in entry
    assert f"[[{prefix}/concepts/router]]" in (generation / "mocs/router.md").read_text()
    for folder in ("sources", "entries", "concepts", "mocs"):
        assert f"[[{prefix}/{folder}/index|" in (generation / "index.md").read_text()
        nav = (generation / folder / "index.md").read_text()
        assert f"[[{prefix}/{folder}/router|Index]]" in nav
    for view, folder in (("concepts-by-confidence", "concepts"), ("mocs-by-count", "mocs")):
        assert f"[[{prefix}/{folder}/router|Index]]" in (
            generation / "views" / f"{view}.md"
        ).read_text()
    rendered_source = (generation / "sources/router.md").read_text()
    assert BODY.strip().encode() in rendered_source.encode()
    synthesis = load_synthesis(generation / ".llmwiki", "router.md")
    evidence = synthesis.concepts[0].claims[0].evidence
    assert evidence.verification == "verified"
    source_digest = hashlib.sha256(BODY.strip().encode()).hexdigest()
    assert synthesis.review_receipt["source_sha256"] == source_digest
    assert BODY[evidence.start_offset:evidence.end_offset] == evidence.quote
    assert QUOTE in evidence.quote
    page = (generation / "concepts/router.md").read_text()
    assert f'“{evidence.quote}”'.encode() in page.encode()
    assert f"@{evidence.start_offset}–{evidence.end_offset}" in page
    assert evidence.source_file == "router.md" and evidence.span_ids
    cid = parse_frontmatter(page)[0]["canonical_id"]
    assert f"[[03-Raw-Annotations/entities/{cid}|Annotations]]" in page
    assert "[[04-Wiki/concepts/router|Legacy page (retained in place)]]" in page
    assert protected == {p: human_state(p) for p in protected}
    frozen = {p: human_state(p) for p in generation.rglob("*") if p.is_file()}
    calls = len(provider.calls)
    second = cli.invoke(app, ["build", str(tmp_path)])
    assert second.exit_code == 0, second.output
    assert current(config)[1] == manifest and len(provider.calls) == calls
    assert frozen == {p: human_state(p) for p in frozen}
    refreshed = cli.invoke(app, ["build", str(tmp_path), "--refresh-render"])
    assert refreshed.exit_code == 0, (refreshed.output, refreshed.exception)
    new_generation, _ = current(config)
    assert new_generation != generation and len(provider.calls) == calls
    new_prefix = new_generation.relative_to(tmp_path).as_posix()
    refreshed_synthesis = load_synthesis(new_generation / ".llmwiki", "router.md")
    assert refreshed_synthesis.review_receipt == synthesis.review_receipt
    assert f"[[{new_prefix}/concepts/router]]" in (
        new_generation / "entries/router.md"
    ).read_text()
    assert frozen == {p: human_state(p) for p in frozen}
    assert protected == {p: human_state(p) for p in protected}
    # Resolve app-emitted targets, never mistake literal source/quote tokens for navigation.
    links = json.loads((new_generation / ".llmwiki/generated-links.json").read_text())
    for relative in links["targets"]:
        assert (tmp_path / links["prefix"] / Path(relative)).is_file()


def test_generation_refresh_removes_stale_navigation_shards(tmp_path):
    import shutil
    from types import SimpleNamespace

    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.render.obsidian import render_vault
    from tests.new.test_scalability_incremental_render import _bundle

    first = tmp_path / "04-Wiki/OLW-Publications/generations/first"
    second = first.with_name("second")
    source = {"incremental.md": SourceDoc("Incremental Source", "Original [[concept-0]].")}
    render_vault(first, _bundle(), source,
                 config=SimpleNamespace(index_shard_size=1, report_shard_size=1),
                 link_prefix=first.relative_to(tmp_path).as_posix(), cross_lingual_links={})
    old_shards = {p.relative_to(first) for p in first.rglob("*.md")
                  if p.name.startswith(("index-b", "concepts-by-confidence-b"))}
    assert old_shards
    frozen = {p: human_state(p) for p in first.rglob("*") if p.is_file()}
    shutil.copytree(first, second)
    render_vault(second, _bundle(), source,
                 link_prefix=second.relative_to(tmp_path).as_posix(), cross_lingual_links={})
    remaining = [str(p) for p in old_shards if (second / p).exists()]
    assert not remaining, remaining
    assert frozen == {p: human_state(p) for p in frozen}


async def test_missing_typed_inventory_blocks_generation_commit(tmp_path, monkeypatch):
    import pytest

    import obsidian_llm_wiki.core.immutable_publication as publication
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from tests.new.test_immutable_publication import fixture

    config, sources, _paths, _llm = fixture(tmp_path, monkeypatch)
    original = publication._qualify_links

    def remove_inventory(wiki, *args):
        (wiki / ".llmwiki/generated-links.json").unlink()
        original(wiki, *args)

    monkeypatch.setattr(publication, "_qualify_links", remove_inventory)
    with pytest.raises(RuntimeError, match="missing generated link inventory"):
        await run_pipeline(tmp_path, sources, config)
    assert not (config.wiki_dir / "OLW-Publications/current.json").exists()


def test_source_view_retains_all_captured_whitespace_and_title_bytes():
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.render.obsidian import render_source_page

    captured = "\n\r\n  [[router]]\r\n`[[router]]`  \t\r\n"
    assert captured.encode() in render_source_page(SourceDoc("Router", captured)).encode()
    plain_title = "Router\r\n\r\n  [[router]] \t\r\n"
    assert plain_title.encode() in render_source_page(SourceDoc("Router", plain_title)).encode()


def test_reserved_index_basename_resolves_by_entity_kind(tmp_path):
    from obsidian_llm_wiki.core.models import (
        ConceptNote,
        MapOfContent,
        SourceDoc,
        SourceSynthesis,
        SynthesisBundle,
    )
    from obsidian_llm_wiki.render.obsidian import render_vault

    concept = ConceptNote("Index", "index", "A literal [[index]].")
    bundle = SynthesisBundle(
        concepts=[concept],
        sources=[SourceSynthesis("Index", "Summary", source_file="index.md", concepts=[concept])],
        maps=[MapOfContent("Index", "index", "Summary", concept_slugs=["index"])],
    )
    generation = tmp_path / "04-Wiki/OLW-Publications/generations/reserved"
    prefix = generation.relative_to(tmp_path).as_posix()
    render_vault(generation, bundle, {"index.md": SourceDoc("Index", "Original [[index]].")},
                 link_prefix=prefix, cross_lingual_links={})
    for folder in ("sources", "entries", "concepts", "mocs"):
        assert (generation / folder / "index.md").exists()
        assert f"[[{prefix}/{folder}/__olw-index-1|" in (generation / "index.md").read_text()
        assert f"[[{prefix}/{folder}/index|Index]]" in (
            generation / folder / "__olw-index-1.md"
        ).read_text()
    assert "Original [[index]]." in (generation / "sources/index.md").read_text()


def test_missing_exact_typed_target_cannot_fall_back_to_same_basename(tmp_path):
    import pytest

    from obsidian_llm_wiki.render.links import (
        ACTIVE_LINK_PRESENTATION,
        LinkPresentation,
        generated_link,
        validate_link_targets,
    )

    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/router.md").write_text("# Not the concept\n")
    context = LinkPresentation("04-Wiki/OLW-Publications/generations/control")
    token = ACTIVE_LINK_PRESENTATION.set(context)
    try:
        assert generated_link("concepts", "router", "Display") == (
            f"[[{context.prefix}/concepts/router|Display]]"
        )
        with pytest.raises(ValueError, match="untyped"):
            generated_link("unknown", "router")
        with pytest.raises(ValueError, match="invalid"):
            generated_link("concepts", "../router")
    finally:
        ACTIVE_LINK_PRESENTATION.reset(token)
    with pytest.raises(RuntimeError, match="missing generated wikilink target: concepts/router.md"):
        validate_link_targets(tmp_path, context.payload(), prefix=context.prefix)
    with pytest.raises(RuntimeError, match="invalid generated link inventory"):
        validate_link_targets(tmp_path, context.payload(), prefix="different-generation")


async def test_orphan_history_links_stay_pinned_while_navigation_moves(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.pipeline import run_pipeline
    from tests.new.test_immutable_publication_lifecycle import setup, source

    config, _provider = setup(tmp_path, monkeypatch)
    legacy = config.concepts_dir / "amber-relay.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"Human [[amber-relay]]\r\n")
    protected = human_state(legacy)
    docs = source(config, "17")
    await run_pipeline(tmp_path, docs, config, annotation_templates=True)
    first, marker = current(config)
    cid = next(iter(marker["entities"]))
    annotation = config.annotations_dir / "entities" / f"{cid}.md"
    annotation.write_bytes(b"Human history\r\n")
    annotation_before = human_state(annotation)
    old_source = first.relative_to(tmp_path).as_posix() + "/sources/amber"
    frozen = {p: human_state(p) for p in first.rglob("*") if p.is_file()}
    (config.sources_dir / "amber.md").unlink()
    deleted = await run_pipeline(tmp_path, {}, config)
    assert not deleted.errors
    latest, _ = current(config)
    text = (latest / "concepts/amber-relay.md").read_text()
    assert f"[[{old_source}|amber]]" in text
    assert "[[04-Wiki/concepts/amber-relay|Legacy page (retained in place)]]" in text
    assert f"[[03-Raw-Annotations/entities/{cid}|Annotations]]" in text
    assert (tmp_path / (old_source + ".md")).is_file()
    new_prefix = latest.relative_to(tmp_path).as_posix()
    assert f"[[{new_prefix}/concepts/index|Concepts]]" in (latest / "index.md").read_text()
    assert protected == human_state(legacy) and annotation_before == human_state(annotation)
    assert frozen == {p: human_state(p) for p in frozen}
