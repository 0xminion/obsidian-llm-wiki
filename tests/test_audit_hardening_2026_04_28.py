from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.compile import _build_edges, _detect_duplicates, _run_semantic_compile
from pipeline.compile.core import CompileResult
from pipeline.compile.semantic import NoteIndex, _add_wikilink
from pipeline.config import Config
from pipeline.extract import detect_source_type
from pipeline.extractors._shared import _curl_get, transcribe_assemblyai
from pipeline.extractors.youtube import _extract_youtube_video_id, _try_youtube_transcript
from pipeline.fixtures import create_adversarial_vault
from pipeline.models import Edge, EdgeType, SourceType
from pipeline.qmd_mcp import QMDSearchResult, _qmd_results_to_concept_matches
from pipeline.review import approve_reviews
from pipeline.store import ContentStore
from pipeline.utils import safe_note_stem, smart_filename, title_to_filename
from pipeline.vault import update_moc, write_edge

runner = CliRunner()


def test_safe_note_stem_blocks_path_breakout_and_llm_filename(monkeypatch):
    assert title_to_filename("/tmp/pwn中文") == "tmp-pwn中文"
    assert safe_note_stem("../../evil") == "evil"
    assert safe_note_stem("C:\\temp\\evil") == "c-temp-evil"
    assert safe_note_stem("bad\nfrontmatter") == "bad-frontmatter"

    import pipeline.utils as utils

    monkeypatch.setattr(utils, "_llm_short_filename", lambda *a, **k: "../../llm-escape")
    assert smart_filename("中" * 120) == "llm-escape"


def test_review_approval_rejects_paths_outside_vault_and_writes_nothing(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        store.review_add("h", {}, "entry", "/tmp/outside-review.md", "# ok")
    finally:
        store.close()

    result = approve_reviews(cfg)

    assert result["approved"] == 0
    assert result["written"] == 0
    assert result["failed"] == 1
    assert not Path("/tmp/outside-review.md").exists()
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        assert store.review_get_pending()[0]["status"] == "pending"
    finally:
        store.close()


def test_review_approval_is_atomic_per_plan(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        store.review_add("h", {}, "source", str(cfg.sources_dir / "one.md"), "# ok")
        store.review_add("h", {}, "entry", str(cfg.entries_dir / "two.md"), "TODO invalid")
    finally:
        store.close()

    result = approve_reviews(cfg)

    assert result["approved"] == 0
    assert result["written"] == 0
    assert result["failed"] == 2
    assert not (cfg.sources_dir / "one.md").exists()
    assert not (cfg.entries_dir / "two.md").exists()


def test_review_approval_rolls_back_if_final_replace_fails(monkeypatch, tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    store = ContentStore.open(cfg.resolved_extract_dir)
    try:
        store.review_add("h", {}, "source", str(cfg.sources_dir / "one.md"), "# ok")
        store.review_add("h", {}, "entry", str(cfg.entries_dir / "two.md"), "# ok")
    finally:
        store.close()

    original_replace = Path.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    result = approve_reviews(cfg)

    assert result["approved"] == 0
    assert result["written"] == 0
    assert result["failed"] == 2
    assert not (cfg.sources_dir / "one.md").exists()
    assert not (cfg.entries_dir / "two.md").exists()


def test_youtube_detection_requires_real_youtube_hostname():
    url = "http://127.0.0.1:9/?x=youtube.com&v=AAAAAAAAAAA"
    assert detect_source_type(url) is SourceType.WEB
    assert _extract_youtube_video_id(url) == ""


def test_youtube_fallback_uses_canonical_url_not_raw_private_url(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, *a, **k):
        calls.append(args)
        class Result:
            returncode = 1
            stdout = ""
            stderr = ""
        return Result()

    monkeypatch.setattr("pipeline.extractors.youtube._run", fake_run)
    monkeypatch.setattr("pipeline.extractors.youtube.transcribe_with_whisper", lambda *a, **k: "")

    _try_youtube_transcript(
        "https://www.youtube.com/watch?v=AAAAAAAAAAA&feature=share",
        "AAAAAAAAAAA",
        Config(extract_timeout=5),
    )

    assert calls
    assert calls[-1][-1] == "https://www.youtube.com/watch?v=AAAAAAAAAAA"


def test_secret_headers_are_not_exposed_in_curl_argv(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(args, *a, **k):
        seen.append(args)
        class Result:
            stdout = "{}"
            returncode = 0
        return Result()

    monkeypatch.setattr("pipeline.extractors._shared._run", fake_run)
    _curl_get("https://example.com", headers={"Authorization": "Bearer SECRET_TOKEN"})

    flat = "\n".join(" ".join(call) for call in seen)
    assert "SECRET_TOKEN" not in flat
    assert any("--config" in call for call in seen)


def test_assemblyai_secret_headers_are_not_exposed_in_argv(monkeypatch, tmp_path: Path):
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"abc")
    calls: list[tuple[list[str], str]] = []

    def fake_run(args, *a, **kwargs):
        calls.append((args, kwargs.get("input_data", "")))
        class Result:
            returncode = 0
            stdout = "{}"
        if any("/v2/upload" in arg for arg in args):
            Result.stdout = json.dumps({"upload_url": "https://upload.example/audio"})
        elif any(arg.endswith("/v2/transcript") for arg in args):
            Result.stdout = json.dumps({"id": "abc"})
        else:
            Result.stdout = json.dumps({"status": "completed", "text": "done"})
        return Result()

    monkeypatch.setattr("pipeline.extractors._shared._run", fake_run)
    assert transcribe_assemblyai(str(audio), "ASSEMBLY_SECRET", timeout=5) == "done"
    argv = "\n".join(" ".join(args) for args, _input in calls)
    stdin_configs = "\n".join(input_data for _args, input_data in calls)
    assert "ASSEMBLY_SECRET" not in argv
    assert "ASSEMBLY_SECRET" in stdin_configs


def test_update_moc_sanitizes_frontmatter_and_uses_canonical_alias_link(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    update_moc(
        cfg,
        "Good\nstatus: owned\n---\n# injected",
        "my-great-paper",
        "Display Title",
        entry_display_title="My Great Paper",
    )

    moc = next(cfg.mocs_dir.glob("*.md"))
    text = moc.read_text(encoding="utf-8")
    assert "status: owned" not in text
    assert "# injected" not in text
    assert "[[my-great-paper|My Great Paper]]" in text

    before = moc.stat().st_mtime_ns
    time.sleep(0.01)
    update_moc(cfg, "Good", "my-great-paper", "Display Title", entry_display_title="My Great Paper")
    assert moc.stat().st_mtime_ns == before


def test_edge_cache_invalidated_after_build_edges_rewrite(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.entries_dir.mkdir(parents=True)
    (cfg.entries_dir / "a.md").write_text("# a", encoding="utf-8")
    (cfg.entries_dir / "b.md").write_text("# b", encoding="utf-8")

    write_edge(cfg, Edge("a", "b", EdgeType.SUPPORTS, "manual"))
    (cfg.entries_dir / "a.md").unlink()
    (cfg.entries_dir / "b.md").unlink()
    _build_edges(cfg)

    (cfg.entries_dir / "a.md").write_text("# a", encoding="utf-8")
    (cfg.entries_dir / "b.md").write_text("# b", encoding="utf-8")
    write_edge(cfg, Edge("a", "b", EdgeType.SUPPORTS, "manual"))

    assert "a\tb\tsupports\tmanual" in cfg.edges_file.read_text(encoding="utf-8")


def test_duplicate_report_is_cleared_when_duplicates_disappear(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.concepts_dir.mkdir(parents=True)
    (cfg.concepts_dir / "ai-safety.md").write_text("---\ntitle: AI Safety\n---\n# AI Safety\n")
    (cfg.concepts_dir / "ai-safety-2.md").write_text("---\ntitle: AI Safety\n---\n# AI Safety\n")
    assert _detect_duplicates(cfg) == 1
    report = cfg.vault_path / "Meta" / "Scripts" / "compile-duplicate-report.md"
    assert report.exists()

    (cfg.concepts_dir / "ai-safety-2.md").unlink()
    assert _detect_duplicates(cfg) == 0
    assert "Found 0 potential duplicate pairs" in report.read_text(encoding="utf-8")


def test_qmd_path_conversion_and_disable_env(monkeypatch):
    assert _qmd_results_to_concept_matches(
        [QMDSearchResult(file="04-Wiki/concepts/foo.md", score=0.9, collection="")],
        "concepts",
    )[0].concept == "foo"

    class FakeClient:
        def __init__(self, *a, **k):
            raise AssertionError("QMD client should not be constructed")

    monkeypatch.setenv("USE_QMD_MCP", "false")
    monkeypatch.setattr("pipeline.qmd.QMDMCPClient", FakeClient)
    from pipeline.qmd import _get_client

    assert _get_client() is None


def test_qmd_available_does_not_remove_semantic_similarity(monkeypatch, tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.concepts_dir.mkdir(parents=True)
    (cfg.concepts_dir / "a.md").write_text("---\ntitle: A\ntags: []\n---\n# A\nalpha")

    class FakeQMD:
        def embed_batch(self, texts):
            return {text: [1.0] for text in texts}

    class LocalClient:
        def embed_batch(self, texts):
            raise AssertionError("local embeddings should not be used when QMD has embeddings")

    monkeypatch.setattr("pipeline.qmd._get_client", lambda: FakeQMD())
    index = NoteIndex()
    index.load(cfg)
    index.embed_all(LocalClient())
    assert index.embeddings == {"a": [1.0]}


def test_empty_llm_response_with_candidates_marks_semantic_compile_degraded(monkeypatch, tmp_path: Path):
    cfg = Config(vault_path=tmp_path, llm_provider="ollama", agent_cmd="definitely-missing-hermes")
    cfg.concepts_dir.mkdir(parents=True)
    for name in ["ai-safety", "ai-safety-2"]:
        (cfg.concepts_dir / f"{name}.md").write_text("---\ntitle: AI Safety\ntags: []\n---\n# AI Safety\nbody")

    class FailingClient:
        def embed_batch(self, texts):
            return {text: [1.0] for text in texts}
        def generate(self, prompt, timeout=120):
            return ""

    monkeypatch.setattr("pipeline.qmd._get_client", lambda: None)
    monkeypatch.setattr("pipeline.llm_client.get_llm_client", lambda cfg: FailingClient())
    result = CompileResult()

    ok, output = _run_semantic_compile(cfg, result)

    assert ok is False
    assert result.agent_succeeded is False
    assert "empty" in output.lower() or result.semantic_status == "degraded"


def test_semantic_add_wikilink_sanitizes_llm_source_and_target(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.entries_dir.mkdir(parents=True)
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# outside\n", encoding="utf-8")
    (cfg.entries_dir / "safe-source.md").write_text("# safe\n", encoding="utf-8")

    assert _add_wikilink(cfg, "../../../outside", "../../../target", "bad") is False
    assert "target" not in outside.read_text(encoding="utf-8")
    assert _add_wikilink(cfg, "safe-source", "../target note", "ok") is True
    assert "[[target-note]]" in (cfg.entries_dir / "safe-source.md").read_text(encoding="utf-8")


def test_graph_doctor_cli_reports_unresolved_links_and_stale_edges(tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.entries_dir.mkdir(parents=True)
    cfg.config_dir.mkdir(parents=True)
    (cfg.entries_dir / "entry.md").write_text("# Entry\n[[missing-note]]\n", encoding="utf-8")
    cfg.edges_file.write_text("source\ttarget\ttype\tdescription\nentry\tghost\trelates_to\tstale\n")

    res = runner.invoke(app, ["graph-doctor", str(tmp_path), "--json"])

    assert res.exit_code == 1
    report = json.loads(res.stdout)
    assert report["ok"] is False
    assert report["unresolved_links"]
    assert report["stale_edges"]


def test_migrate_cli_writes_schema_version_and_is_idempotent(tmp_path: Path):
    res = runner.invoke(app, ["migrate", str(tmp_path), "--yes", "--json"])
    assert res.exit_code == 0, res.stdout
    first = json.loads(res.stdout)
    assert first["schema_version"] >= 1
    version_file = tmp_path / "06-Config" / "schema-version.json"
    assert version_file.exists()

    res2 = runner.invoke(app, ["migrate", str(tmp_path), "--yes", "--json"])
    assert res2.exit_code == 0, res2.stdout
    second = json.loads(res2.stdout)
    assert second["schema_version"] == first["schema_version"]


def test_adversarial_fixture_and_golden_compile_workflow(tmp_path: Path):
    summary = create_adversarial_vault(tmp_path, overwrite=True)
    assert summary["files_written"] > 0

    cfg = Config(vault_path=tmp_path)
    _build_edges(cfg)
    graph = runner.invoke(app, ["graph-doctor", str(tmp_path), "--json"])

    assert graph.exit_code == 0, graph.stdout
    report = json.loads(graph.stdout)
    assert report["ok"] is True
    assert not report["unresolved_links"]


def test_semantic_candidate_generation_uses_blocking_not_full_pairwise(monkeypatch, tmp_path: Path):
    cfg = Config(vault_path=tmp_path)
    cfg.concepts_dir.mkdir(parents=True)
    for i in range(80):
        tag = "shared" if i < 10 else f"tag-{i}"
        (cfg.concepts_dir / f"concept-{i:03d}.md").write_text(
            f"---\ntitle: Concept {i}\ntags:\n  - {tag}\n---\n# Concept {i}\nbody",
            encoding="utf-8",
        )
    index = NoteIndex()
    index.load(cfg)
    calls = {"count": 0}
    original = index.similarity

    def counted(a, b):
        calls["count"] += 1
        return original(a, b)

    index.similarity = counted  # type: ignore[method-assign]
    client = MagicMock()
    client.generate.return_value = ""
    from pipeline.compile.semantic import _semantic_crosslink

    _semantic_crosslink(cfg, client, index)
    assert calls["count"] < 800  # far below 80*79/2 = 3160
