"""Public application callers, offline doubles and exact preservation controls."""

import asyncio
import json

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.pipeline import run_pipeline
from tests.new.test_immutable_publication import fixture, human_state


def snapshot(root):
    return {p.relative_to(root).as_posix(): human_state(p) for p in root.rglob("*") if p.is_file()}


def test_ingest_uses_separate_operational_state_and_duplicates(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    asyncio.run(run_pipeline(tmp_path, sources, config))
    generation = current_publication(config.wiki_dir)[0]
    before = snapshot(generation)
    human = {p: human_state(p) for p in paths}
    import obsidian_llm_wiki.cli.ingest as ingest
    from obsidian_llm_wiki.core.models import SourceDoc

    extracted = SourceDoc(
        "New source", "Distinct local source body. " * 20, url="https://example.test/new"
    )
    monkeypatch.setattr(ingest, "_extract_direct", lambda url: extracted)
    runner = CliRunner()
    command = ["ingest", str(tmp_path), "--url", extracted.url, "--skip-synthesis", "--json"]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert not (config.llmwiki_dir / "registry.db").exists(), (
        "ingest created a mutable legacy registry"
    )
    operations = config.wiki_dir / "OLW-Publications/operations"
    assert (operations / "registry.db").is_file()
    all_sources = snapshot(config.sources_dir)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert "duplicate_of" in result.output
    assert snapshot(config.sources_dir) == all_sources
    assert snapshot(generation) == before
    assert {p: human_state(p) for p in paths} == human


def test_new_only_preview_resolves_generation_without_writable_constructors(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.registry import VaultRegistry

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    built = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert built.exit_code == 0, built.output
    before = snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("preview called a writer constructor")

    monkeypatch.setattr(VaultRegistry, "__init__", forbidden)
    monkeypatch.setattr(Config, "__init__", forbidden)
    result = CliRunner().invoke(app, ["identity-migration", str(tmp_path), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["publication"]["verified"]
    assert not report["blockers"]
    assert report["summary"]["bindings"]["resolved"] > 0
    assert report["assessment_estimate"]["status"] == "unavailable"
    assert report["assessment_estimate"]["model_calls_made"] == 0
    assert snapshot(tmp_path) == before


def test_fix_apply_rebuilds_private_generation_never_legacy_notes(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    runner = CliRunner()
    assert runner.invoke(app, ["build", str(tmp_path)]).exit_code == 0
    generation, manifest = current_publication(config.wiki_dir)
    legacy = config.concepts_dir / "human.md"
    legacy.write_text('---\ntype: Concept\ntags: ["Human Tag"]\n---\n# Keep me\n')
    protected = {p: human_state(p) for p in [*paths, legacy]}
    before = snapshot(generation)
    result = runner.invoke(app, ["fix", str(tmp_path), "--apply", "--json"])
    assert result.exit_code == 0, result.output
    assert current_publication(config.wiki_dir)[1]["generation"] != manifest["generation"]
    assert snapshot(generation) == before
    assert {p: human_state(p) for p in protected} == protected
    assert json.loads(result.output)["mode"] == "publish"


@pytest.mark.parametrize("action", ["restore", "refresh", "answer"])
def test_destructive_options_refuse_before_any_writes(tmp_path, monkeypatch, action):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["build", str(tmp_path)]).exit_code == 0
    from obsidian_llm_wiki.core.backups import backup_file

    backup = backup_file(paths[1], config.llmwiki_dir / "backups")
    before = snapshot(tmp_path)
    args = {
        "restore": ["fix", str(tmp_path), "--restore", str(backup), "--json"],
        "refresh": [
            "ingest",
            str(tmp_path),
            "--refresh",
            "--url",
            "https://example.test/refresh",
            "--json",
        ],
        "answer": [
            "query",
            str(tmp_path),
            "--ask",
            "A1",
            "--save-answer",
            str(paths[0]),
            "--force",
            "--json",
        ],
    }[action]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0, (args, result.output)
    assert "refus" in result.output.lower(), (args, result.output)
    assert snapshot(tmp_path) == before, args


def test_ingest_build_resume_retains_clipping_and_existing_sources(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    clipping = config.clippings_dir / "queued.md"
    clipping.parent.mkdir(parents=True)
    clipping.write_text("---\ntitle: Clipped\n---\n" + "Unique clipping content. " * 40)
    human = {p: human_state(p) for p in [*paths, clipping]}
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    generation = current_publication(config.wiki_dir)[0]
    before = snapshot(generation)
    assert {p: human_state(p) for p in human} == human
    assert not (config.llmwiki_dir / "registry.db").exists()
    result = runner.invoke(app, ["ingest", str(tmp_path), "--resume", "--json"])
    assert result.exit_code == 0, result.output
    assert snapshot(generation) == before
    assert {p: human_state(p) for p in human} == human


def test_health_legacy_inspection_is_read_only(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    result = CliRunner().invoke(app, ["health", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    assert snapshot(tmp_path) == before


def test_query_legacy_unverifiable_content_is_refused_without_false_pending(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    paths[1].write_text(
        "---\ntitle: Subject\ntype: Concept\n---\nSubject explanation and supporting data. " * 5
    )
    import obsidian_llm_wiki.cli.query as query

    monkeypatch.setattr(query, "call_llm", lambda *a, **kw: "Subject explanation.")
    before = {p: human_state(p) for p in paths}
    for _ in range(2):
        result = CliRunner().invoke(app, ["query", str(tmp_path), "--ask", "Subject", "--json"])
        assert result.exit_code == 1, result.output
        assert json.loads(result.output)['freshness']['status'] == 'unverifiable'
        assert 'pending' not in result.output.lower()
    assert {p: human_state(p) for p in paths} == before
    assert not (config.llmwiki_dir / "query-sessions.json").exists()


def test_source_install_collision_keeps_concurrent_human_bytes(tmp_path, monkeypatch):
    import obsidian_llm_wiki.core.application as application
    import obsidian_llm_wiki.core.immutable_publication as publication
    from obsidian_llm_wiki.core.models import SourceDoc

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    install = publication._install_exclusive
    saved = {}

    def race(source, target):
        if target == config.sources_dir / "new.md":
            target.write_bytes(b"Human concurrent edit\r\n")
            saved[target] = human_state(target)
        return install(source, target)

    monkeypatch.setattr(publication, "_install_exclusive", race)
    written = application.install_source(config, SourceDoc("New", "Body"), "Body\n")
    assert saved and all(human_state(path) == state for path, state in saved.items())
    assert written.name == "new-1.md" and written.read_text() == "Body\n"


def test_build_does_not_depend_on_malformed_legacy_state(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["build", str(tmp_path)]).exit_code == 0
    config.state_file.write_text("Human legacy metadata, not a compiler input")
    before = human_state(config.state_file)
    result = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert human_state(config.state_file) == before


def test_pending_and_unknown_namespaces_fail_closed(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    root = config.wiki_dir / "OLW-Publications"
    root.mkdir()
    (root / "human.md").write_text("unowned namespace")
    before = snapshot(tmp_path)
    result = CliRunner().invoke(app, ["current", str(tmp_path), "--json"])
    assert result.exit_code != 0
    assert "collision" in result.output or "corrupt" in result.output
    assert snapshot(tmp_path) == before


def test_metrics_reads_current_generation(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.metrics import load_metrics

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["build", str(tmp_path)]).exit_code == 0
    before = snapshot(tmp_path)
    assert load_metrics(tmp_path)["summary"]["total_syntheses"] == 1
    result = CliRunner().invoke(app, ["metrics", str(tmp_path)])
    assert result.exit_code == 0 and "Metrics for run" in result.output, result.output
    assert snapshot(tmp_path) == before


def test_current_rejects_manifest_entrypoint_outside_generation(tmp_path, monkeypatch):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    assert CliRunner().invoke(app, ["build", str(tmp_path)]).exit_code == 0
    pointer = config.wiki_dir / "OLW-Publications/current.json"
    manifest = json.loads(pointer.read_text())
    manifest["entrypoint"] = "../../outside.md"
    pointer.write_text(json.dumps(manifest))
    before = snapshot(tmp_path)
    result = CliRunner().invoke(app, ["current", str(tmp_path), "--json"])
    assert result.exit_code != 0 and "entrypoint" in result.output, result.output
    assert snapshot(tmp_path) == before


def test_failed_ingest_resume_and_recompile_use_only_private_writable_sql(tmp_path, monkeypatch):
    import obsidian_llm_wiki.cli.ingest as ingest
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.registry import VaultRegistry

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    original = VaultRegistry.__init__
    opened = []

    def guarded(self, directory, *args, **kwargs):
        path = str(directory)
        assert "/generations/" not in path, "opened published SQLite writable"
        assert directory != config.llmwiki_dir, "opened legacy SQLite writable"
        opened.append(path)
        original(self, directory, *args, **kwargs)

    monkeypatch.setattr(VaultRegistry, "__init__", guarded)
    calls = []

    def extract(url):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("local scripted extraction failure")
        return SourceDoc("Recovered", "Recovered evidence and discussion. " * 30, url=url)

    monkeypatch.setattr(ingest, "_extract_direct", extract)
    protected = {p: human_state(p) for p in paths}
    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", str(tmp_path), "--url", "https://example.test/retry", "--json"]
    )
    assert result.exit_code == 1, result.output
    result = runner.invoke(app, ["ingest", str(tmp_path), "--resume", "--json"])
    assert result.exit_code == 0, result.output
    generation = current_publication(config.wiki_dir)[0]
    before = snapshot(generation)
    result = runner.invoke(app, ["recompile", str(tmp_path), "recovered.md"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 2 and opened
    assert {p: human_state(p) for p in paths} == protected
    assert snapshot(generation) == before


@pytest.mark.parametrize(
    "command", ["ingest", "fix", "health", "validate", "current", "query", "metrics"]
)
def test_all_callers_refuse_corruption_without_writes(tmp_path, monkeypatch, command):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    runner = CliRunner()
    assert runner.invoke(app, ["build", str(tmp_path)]).exit_code == 0
    generation = current_publication(config.wiki_dir)[0]
    (generation / "index.md").write_text("Human mistakenly edited immutable output")
    before = snapshot(tmp_path)
    args = [command, str(tmp_path)]
    if command == "ingest":
        args += ["--url", "https://example.test/new", "--json"]
    elif command == "fix":
        args += ["--apply", "--json"]
    elif command == "query":
        args += ["--ask", "A1", "--json"]
    result = runner.invoke(app, args)
    assert result.exit_code != 0, result.output
    assert "generation" in result.output or "corrupt" in result.output, result.output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["ingest-resume.json", "operations.json", "ingest.lock"])
def test_legacy_operational_state_requires_explicit_migration(tmp_path, monkeypatch, name):
    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    (config.llmwiki_dir / name).write_text("{}")
    before = snapshot(tmp_path)
    result = CliRunner().invoke(app, ["ingest", str(tmp_path), "--resume", "--skip-synthesis"])
    assert result.exit_code != 0 and "legacy" in result.output.lower(), result.output
    assert snapshot(tmp_path) == before


def test_operational_ledger_never_enters_legacy_readers(tmp_path, monkeypatch):
    from obsidian_llm_wiki.core.application import operational_directory
    from obsidian_llm_wiki.query.graph import build_graph_from_vault

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    operations = operational_directory(config, create=True)
    (operations / "failed_urls.md").write_text("SECRET operational diagnostics, not wiki evidence")
    graph = build_graph_from_vault(config.wiki_dir)
    assert all("OLW-Publications" not in name for name in graph.pages)
    result = CliRunner().invoke(app, ["validate", str(tmp_path)])
    assert "failed_urls.md" not in result.output
    result = CliRunner().invoke(app, ["health", str(tmp_path), "--json"])
    assert "failed_urls.md" not in result.output


@pytest.mark.parametrize("interruption", ["cancelled", "unclean"])
def test_operational_resume_after_interruption_preserves_old_generation(
    tmp_path,
    monkeypatch,
    interruption,
):
    import obsidian_llm_wiki.cli.ingest as ingest
    from obsidian_llm_wiki.core.application import operational_directory
    from obsidian_llm_wiki.core.models import SourceDoc
    from obsidian_llm_wiki.core.operations import OperationRecord, OperationStatus, OperationStore
    from obsidian_llm_wiki.core.registry import VaultRegistry

    config, sources, paths, llm = fixture(tmp_path, monkeypatch)
    runner = CliRunner()
    assert runner.invoke(app, ["build", str(tmp_path)]).exit_code == 0
    generation = current_publication(config.wiki_dir)[0]
    published = snapshot(generation)
    protected = {p: human_state(p) for p in paths}
    url = "https://example.test/interrupted"
    if interruption == "cancelled":

        def cancel(_url):
            raise KeyboardInterrupt()

        monkeypatch.setattr(ingest, "_extract_direct", cancel)
        result = runner.invoke(app, ["ingest", str(tmp_path), "--url", url, "--json"])
        assert result.exit_code == 130, result.output
    else:
        directory = operational_directory(config, create=True)
        store = OperationStore(directory)
        record = OperationRecord.create(run_id="stopped-worker", source=url)
        store.transition(record, OperationStatus.RUNNING)
        with VaultRegistry(directory) as registry:
            queued = registry.enqueue_dirty("ingest", url, "requested")
            assert registry.start_dirty(queued, worker_id="stopped-worker")
    monkeypatch.setattr(
        ingest,
        "_extract_direct",
        lambda value: SourceDoc(
            "Interrupted",
            "A complete local source containing evidence. " * 20,
            url=value,
        ),
    )
    result = runner.invoke(app, ["ingest", str(tmp_path), "--resume", "--json"])
    assert result.exit_code == 0, result.output
    store = OperationStore(operational_directory(config))
    assert store.latest_for_source(url).status is OperationStatus.SUCCEEDED
    assert not store.read_resume_marker()
    assert (config.sources_dir / "interrupted.md").is_file()
    assert snapshot(generation) == published
    assert {p: human_state(p) for p in paths} == protected
