"""Regression tests for April 2026 end-to-end code review findings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.config import Config, load_config
from pipeline.models import ExtractedSource, Manifest, Plan, SourceType
from pipeline.store import ContentStore


def _vault(tmp_path: Path) -> Config:
    vault = tmp_path / "vault"
    for d in [
        "01-Raw",
        "04-Wiki/sources",
        "04-Wiki/entries",
        "04-Wiki/concepts",
        "04-Wiki/mocs",
        "05-Outputs",
        "06-Config",
        "08-Archive-Raw",
        "Meta/Scripts",
    ]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    extract = tmp_path / "extract"
    extract.mkdir()
    return Config(vault_path=vault, extract_dir=extract, max_retries=1)


def test_validate_url_blocks_alternate_private_ip_forms():
    from pipeline.extractors._shared import _validate_url

    blocked = [
        "http://127.0.0.1/",
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://[::1]/",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://169.254.169.254/latest/meta-data",
    ]
    for url in blocked:
        assert not _validate_url(url), url

    assert _validate_url("https://example.com/article")


def test_web_fetch_fallbacks_validate_url_before_subprocess():
    from pipeline.extractors import web

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        raise AssertionError("network subprocess should not run for unsafe URL")

    with patch.object(web, "_run", fake_run):
        assert web._try_defuddle("http://127.0.0.1:1/admin") == ""
        assert web._try_defuddle_json("http://127.0.0.1:1/admin") == ""
        assert web._try_curl_extract("http://127.0.0.1:1/admin") == ""

    assert calls == []


def test_podcast_xml_parser_accepts_simple_rss():
    from pipeline.extractors.podcast import _safe_xml_parse

    root = _safe_xml_parse("<rss><channel><item><title>x</title></item></channel></rss>")
    assert root is not None
    assert root.tag == "rss"


def test_podcast_audio_download_validates_internal_url(tmp_path):
    from pipeline.extractors import podcast

    cfg = _vault(tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        raise AssertionError("yt-dlp should not run for unsafe URL")

    with patch.object(podcast, "_run", fake_run):
        assert podcast._transcribe_podcast_audio("http://127.0.0.1:1/secret.mp3", cfg) == ""

    assert calls == []


def test_extract_all_raises_when_all_requested_urls_fail(tmp_path):
    from pipeline import extract as extract_mod
    from pipeline.extract import ExtractionError, extract_all

    cfg = _vault(tmp_path)

    def fail(*args, **kwargs):
        raise ExtractionError("boom")

    with patch.object(extract_mod, "_extract_web", fail):
        with pytest.raises(ExtractionError, match="all extractions failed"):
            extract_all(["https://example.com/a"], cfg, parallel=1)


def test_extract_url_saves_artifact_before_registering_content(tmp_path):
    from pipeline import extract as extract_mod
    from pipeline.extract import ExtractionError, extract_url

    cfg = _vault(tmp_path)
    store = ContentStore.open(cfg.resolved_extract_dir)

    def source_for(url, cfg, source_type=SourceType.WEB):
        return ExtractedSource(url=url, title="T", content="valid content " * 30, type=SourceType.WEB)

    with patch.object(extract_mod, "_extract_web", source_for), patch.object(
        ExtractedSource,
        "save",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(ExtractionError):
            extract_url("https://example.com/a", cfg, store=store)

    assert store.get_content_duplicate("valid content " * 30) is None
    store.close()


def test_pipeline_tmpdir_loaded_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"PIPELINE_TMPDIR={tmp_path / 'custom-extract'}\nVAULT_PATH={tmp_path / 'vault'}\n")
    monkeypatch.delenv("PIPELINE_TMPDIR", raising=False)
    monkeypatch.delenv("VAULT_PATH", raising=False)

    cfg = load_config(env_file=env)

    assert cfg.extract_dir == tmp_path / "custom-extract"


def test_template_concept_links_use_canonical_stem_alias(tmp_path):
    from pipeline.create.templates import create_file_templates
    from pipeline.lint import run_lint

    cfg = _vault(tmp_path)
    plan = Plan(hash="abc123def456", title="Concept Link Test", concept_new=["My Concept"])
    (cfg.resolved_extract_dir / f"{plan.hash}.json").write_text(json.dumps({
        "url": "https://example.com/c",
        "title": plan.title,
        "content": "Paragraph about a concept. " * 60,
        "type": "web",
    }))

    create_file_templates([plan], cfg, use_agent_insights=False)

    entry = (cfg.entries_dir / "concept-link-test.md").read_text()
    assert "[[my-concept|My Concept]]" in entry
    lint = run_lint(cfg.vault_path, fix=False)
    assert not [i for i in lint.issues if i.check in {"orphaned_concepts", "orphaned_notes"}]


def test_review_approval_collision_does_not_rewrite_source_link_to_entry(tmp_path):
    from pipeline.review import approve_reviews, stage_for_review
    from pipeline.models import Plans

    cfg = _vault(tmp_path)
    plan = Plan(hash="abc123def456", title="Race Collision Article")
    (cfg.resolved_extract_dir / f"{plan.hash}.json").write_text(json.dumps({
        "url": "https://example.com/race",
        "title": plan.title,
        "content": "Long content. " * 50,
        "type": "web",
    }))
    stage_for_review(Plans([plan]), cfg, use_agent_insights=False)

    (cfg.entries_dir / "race-collision-article.md").write_text("---\ntitle: external\n---\n# external")
    approve_reviews(cfg)

    entry = (cfg.entries_dir / "race-collision-article-1.md").read_text()
    assert 'source: "[[race-collision-article]]"' in entry
    assert 'source: "[[race-collision-article-1]]"' not in entry


def test_approve_reviews_validates_before_archiving(tmp_path):
    from pipeline.review import approve_reviews

    cfg = _vault(tmp_path)
    store = ContentStore.open(cfg.resolved_extract_dir)
    source_url = "https://example.com/bad"
    url_file = cfg.inbox_dir / "bad.url"
    url_file.write_text(source_url)
    plan_hash = ExtractedSource(source_url, "Bad", "content", SourceType.WEB).hash
    store.review_add(plan_hash, {"hash": plan_hash, "title": "Bad"}, "entry", str(cfg.entries_dir / "bad.md"), "# Bad\n\nTODO")
    store.close()

    approve_reviews(cfg)

    assert url_file.exists(), "invalid approved output must not archive raw input"


def test_build_edges_rebuilds_generated_edges_and_drops_removed_links(tmp_path):
    from pipeline.compile import _build_edges

    cfg = _vault(tmp_path)
    (cfg.entries_dir / "A.md").write_text("---\ntitle: A\ntags: []\n---\n# A\n\n[[B]]\n")
    (cfg.concepts_dir / "B.md").write_text("---\ntitle: B\ntags: []\n---\n# B\n")

    _build_edges(cfg)
    assert "A\tB\trelates_to" in cfg.edges_file.read_text()

    (cfg.entries_dir / "A.md").write_text("---\ntitle: A\ntags: []\n---\n# A\n\n(no link)\n")
    _build_edges(cfg)

    assert "A\tB\trelates_to" not in cfg.edges_file.read_text()


def test_shared_tags_generate_symmetric_relates_to_not_directional_extends(tmp_path):
    from pipeline.compile import _build_edges

    cfg = _vault(tmp_path)
    for name in ["A", "B"]:
        (cfg.concepts_dir / f"{name}.md").write_text(
            f"---\ntitle: {name}\ntags:\n- x\n- y\n---\n# {name}\n"
        )

    _build_edges(cfg)
    edges = cfg.edges_file.read_text()

    assert "\textends\tshared tags" not in edges
    assert edges.count("\trelates_to\tshared tags") == 1


def test_merge_concepts_does_not_leave_broken_link_or_stale_edge(tmp_path):
    from pipeline.compile import NoteIndex, _merge_concepts
    from pipeline.lint import check_broken_wikilinks, check_edges_consistency

    cfg = _vault(tmp_path)
    (cfg.entries_dir / "Entry.md").write_text("---\ntitle: Entry\ntags: []\n---\n# Entry\n\n[[Dup]]\n")
    (cfg.concepts_dir / "Canon.md").write_text("---\ntitle: Canon\ntags: []\n---\n# Canon\n\n## Core concept\n\nCanon\n")
    (cfg.concepts_dir / "Dup.md").write_text("---\ntitle: Dup\ntags: []\n---\n# Dup\n\n## Core concept\n\nDup\n")
    cfg.edges_file.write_text("source\ttarget\ttype\tdescription\nEntry\tDup\trelates_to\tmanual\n")

    index = NoteIndex()
    index.load(cfg)

    assert _merge_concepts(cfg, "Canon", "Dup", index)
    assert "[[Dup]]" not in (cfg.concepts_dir / "Canon.md").read_text()
    assert "Entry\tCanon\trelates_to" in cfg.edges_file.read_text()
    assert not check_broken_wikilinks(cfg.vault_path)
    assert not check_edges_consistency(cfg.vault_path)


def test_lint_report_lists_stub_issues_under_stub_section(tmp_path):
    from pipeline.lint import LintChecker

    cfg = _vault(tmp_path)
    (cfg.entries_dir / "Stub.md").write_text(
        "---\ntitle: Stub\nstatus: draft\ntemplate: standard\ntags: []\n---\n"
        "# Stub\n\n## Summary\n\nTODO\n\n## Core insights\n\nreal content\n\n"
        "## Other takeaways\n\nreal\n\n## Diagrams\n\nnone\n\n"
        "## Open questions\n\nnone\n\n## Linked concepts\n\nnone\n"
    )
    checker = LintChecker(cfg.vault_path)
    result = checker.run_all()
    report = checker.write_report(result, cfg.vault_path / "report.md").read_text()
    stub_section = report.split("## 11. Stubs/Placeholders", 1)[1].split("## 12. Tag Quality", 1)[0]

    assert "TODO" in stub_section
    assert "All clear" not in stub_section


def test_qmd_cache_is_scoped_by_concepts_dir(tmp_path):
    import pipeline.qmd as qmd

    qmd._cache_loaded = False
    qmd._concept_embedding_cache = {}
    if hasattr(qmd, "_cache_key"):
        qmd._cache_key = None

    c1 = tmp_path / "c1"
    c2 = tmp_path / "c2"
    c1.mkdir()
    c2.mkdir()
    (c1 / "Alpha.md").write_text("# Alpha")
    (c2 / "Beta.md").write_text("# Beta")

    def fake_batch(files):
        return {p.stem: ([1.0, 0.0] if p.stem == "Alpha" else [0.0, 1.0]) for p in files}

    with patch.object(qmd, "_embed_concepts_batch", fake_batch), patch.object(qmd, "_ollama_embed", lambda text: [1.0, 0.0]):
        assert [m.concept for m in qmd.run_qmd_query("anything", "", "", concepts_dir=c1, min_score=0)] == ["Alpha"]
        assert [m.concept for m in qmd.run_qmd_query("anything", "", "", concepts_dir=c2, min_score=0)] == ["Beta"]


def test_archive_inbox_uses_collision_safe_names(tmp_path):
    from pipeline.vault import archive_inbox

    cfg = _vault(tmp_path)
    url = "https://example.com/new"
    existing = "https://example.com/old"
    (cfg.inbox_dir / "same.url").write_text(url)
    (cfg.archive_dir / "same.url").write_text(existing)
    file_hash = ExtractedSource(url, "T", "content", SourceType.WEB).hash

    assert archive_inbox(cfg, {file_hash}) == 1
    assert (cfg.archive_dir / "same.url").read_text() == existing
    assert any(p.read_text() == url for p in cfg.archive_dir.glob("same-*.url"))


def test_hermes_provider_uses_configured_agent_command(tmp_path):
    from pipeline.llm_client import get_llm_client

    cfg = _vault(tmp_path)
    cfg.llm_provider = "hermes"
    cfg.agent_cmd = "/bin/echo"

    client = get_llm_client(cfg)

    assert client.generate("hello", timeout=3).strip()


def test_cli_parallel_updates_template_creation_config(tmp_path):
    from typer.testing import CliRunner
    from pipeline import cli

    cfg = _vault(tmp_path)
    url = "https://example.com/a"
    (cfg.inbox_dir / "a.url").write_text(url)

    captured = {}

    def fake_load(vault):
        return cfg

    def fake_extract_all(urls, cfg_arg, parallel):
        return Manifest([ExtractedSource(url, "T", "content " * 30, SourceType.WEB)])

    def fake_plan_sources(manifest, cfg_arg):
        from pipeline.models import Plans
        return Plans([Plan(hash=manifest.entries[0].hash, title="T")])

    def fake_create_file_templates(plans, cfg_arg, use_agent_insights=True):
        captured["parallel"] = cfg_arg.parallel
        return {"created": 1, "failed": 0, "sources": 1, "entries": 1}

    with patch.object(cli, "_load_cfg", fake_load), patch.object(cli, "extract_all", fake_extract_all), patch.object(
        cli, "plan_sources", fake_plan_sources
    ), patch.object(cli, "create_file_templates", fake_create_file_templates):
        result = CliRunner().invoke(cli.app, ["ingest", str(cfg.vault_path), "--parallel", "7"])

    assert result.exit_code == 0, result.output
    assert captured["parallel"] == 7


def test_query_cli_outputs_do_not_overwrite(tmp_path):
    from typer.testing import CliRunner
    from pipeline import cli

    cfg = _vault(tmp_path)

    with patch.object(cli, "_load_cfg", lambda vault: cfg), patch.object(cli, "query_vault_fast", lambda cfg_arg, question: "answer"):
        runner = CliRunner()
        r1 = runner.invoke(cli.app, ["query", str(cfg.vault_path), "--ask", "one", "--fast"])
        r2 = runner.invoke(cli.app, ["query", str(cfg.vault_path), "--ask", "two", "--fast"])

    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    outputs = list((cfg.vault_path / "05-Outputs").glob("cli-query*.md"))
    assert len(outputs) == 2
