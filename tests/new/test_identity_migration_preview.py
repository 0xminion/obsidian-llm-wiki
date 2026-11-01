"""``olw identity-migration --dry-run`` is a read-only legacy identity inventory.

Every test snapshots the complete vault tree (entries, bytes, inode metadata,
mtimes) before and after the preview and requires it unchanged: no lock,
metrics, WAL/SHM sidecars, cache or page may appear or change. Fixtures are
produced by the ordinary writer path, then explicitly downgraded to schema 6
with a unilateral legacy merge the new writer correctly refuses. This is a
historical fixture, NOT a schema-7 build. The preview must never use the writable
constructor. Busy/WAL cases use a *separate process* writer because SQLite
locks are per process.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.cache import save_synthesis
from obsidian_llm_wiki.core.models import (
    Claim,
    ConceptNote,
    EvidenceSpan,
    SourceDoc,
    SourceState,
    SourceSynthesis,
    WikiState,
    source_synthesis_to_dict,
)
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.core.state import write_state

LIMA = "The Delta observatory in Lima observes the southern sky from Peru."
QUITO = "The Delta observatory in Quito sits on the equator in Ecuador."
CUSCO = "The Cusco telescope array records radio bursts nightly."


def _snapshot(root: Path) -> dict[str, tuple]:
    """Recursive entry/bytes/inode/mtime snapshot; symlinks are not followed."""
    result: dict[str, tuple] = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(directory) / name
            info = path.lstat()
            digest = ""
            if stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            elif stat.S_ISLNK(info.st_mode):
                digest = os.readlink(path)
            key = path.relative_to(root).as_posix()
            result[key] = (
                stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns, info.st_ino, digest,
                stat.S_IMODE(info.st_mode), info.st_ctime_ns, info.st_uid, info.st_gid,
                tuple((key, os.getxattr(path, key, follow_symlinks=False).hex())
                      for key in sorted(os.listxattr(path, follow_symlinks=False))),
            )
    return result


def _receipt(source_file: str, content: str) -> dict:
    return {
        "version": "grounded-review-v2", "status": "approved", "source_file": source_file,
        "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "catalog_version": "span-catalog-v2", "reviewer_fingerprint": "fixture",
        "output_sha256": "", "units": {},
    }


def _synthesis(source_file: str, content: str, concept: ConceptNote) -> SourceSynthesis:
    return SourceSynthesis(
        source_title=source_file, source_summary=content[:40], source_file=source_file,
        concepts=[concept], review_receipt=_receipt(source_file, content),
    )


def _concept(title: str, slug: str, summary: str, source_file: str, quote: str,
             canonical_id: str = "") -> ConceptNote:
    evidence = EvidenceSpan(
        quote=quote, source_file=source_file,
        source_hash=hashlib.sha256(summary.encode()).hexdigest(), verification="verified",
        span_ids=(f"{source_file}#s1",),
    )
    return ConceptNote(
        title=title, slug=slug, summary=summary, canonical_id=canonical_id,
        claims=[Claim(f"{title} claim from {source_file}", slug, source_file, evidence)],
    )


@pytest.fixture
def legacy_vault(tmp_path: Path) -> Path:
    """Ambiguous names, a co-owner, a stale-revision citation, a human edit."""
    vault = tmp_path / "vault"
    wiki = vault / "04-Wiki"
    sources = wiki / "sources"
    llmwiki = wiki / ".llmwiki"
    concepts = wiki / "concepts"
    for d in (sources, llmwiki, concepts):
        d.mkdir(parents=True)
    bodies = {"lima.md": LIMA, "quito.md": QUITO, "cusco.md": CUSCO}
    for name, body in bodies.items():
        (sources / name).write_text(f"---\ntitle: {name}\n---\n{body}\n")
    versions = ArtifactVersions("extractor-v1", "prompt-v1", "schema-v1", "model-fixture")
    with VaultRegistry(llmwiki) as registry:
        lima = _synthesis("lima.md", LIMA, _concept(
            "Delta Observatory", "delta-observatory", "an astronomical observatory in Lima",
            "lima.md", "observes the southern sky",
        ))
        registry.upsert_synthesis("lima.md", SourceDoc("lima", LIMA), lima, versions)
        delta_id = lima.concepts[0].canonical_id
        quito = _synthesis("quito.md", QUITO, _concept(
            "Delta Observatory", "delta-observatory", "an astronomical observatory in Quito",
            "quito.md", "sits on the equator", canonical_id=delta_id,
        ))
        requests = registry.identity_requests(quito)
        assert len(requests) == 1
        historical_assessment = {
            "subject_sha256": requests[0]["subject_sha256"], "subject_title": "Delta Observatory",
            "candidate_id": delta_id, "candidate_sha256": requests[0]["candidate_sha256"],
            "candidate_title": "Delta Observatory", "anchor_source": "lima.md",
            "verdict": "same", "span_ids": ["quito.md#s1"], "assessor": "fixture",
            "source_sha256": quito.review_receipt["source_sha256"],
        }
        registry.upsert_synthesis("quito.md", SourceDoc("quito", QUITO), quito, versions)
        rejected_id = quito.concepts[0].canonical_id
        assert rejected_id != delta_id, "schema 7 must reject unilateral legacy evidence"
        # Reconstruct old stored rows without teaching the schema-7 writer to merge.
        quito.concepts[0].canonical_id = delta_id
        quito.concepts[0].slug = "delta-observatory"
        quito.review_receipt["identity"] = {
            "version": "identity-assessment-v1", "assessments": [historical_assessment],
        }
        registry.connection.execute(
            "UPDATE synthesis SET payload_json=? WHERE source_id=(SELECT id FROM source "
            "WHERE filename='quito.md')", (json.dumps(source_synthesis_to_dict(quito)),))
        registry.connection.execute(
            "UPDATE concept_source SET concept_id=? WHERE concept_id=?", (delta_id, rejected_id))
        registry.connection.execute("DELETE FROM concept WHERE id=?", (rejected_id,))
        registry.connection.commit()
        assert quito.concepts[0].canonical_id == delta_id
        cusco = _synthesis("cusco.md", CUSCO, _concept(
            "Cusco Array", "cusco-array", "a radio telescope array", "cusco.md", "radio bursts",
        ))
        registry.upsert_synthesis("cusco.md", SourceDoc("cusco", CUSCO), cusco, versions)
        for name, synth in (("lima.md", lima), ("quito.md", quito), ("cusco.md", cusco)):
            save_synthesis(synth, llmwiki, name, artifact_versions=versions.as_dict())
        registry.connection.execute("DROP TABLE identity_source_context")
        registry.connection.execute("DROP TABLE identity_entity_history")
        registry.connection.execute("UPDATE meta SET value='6' WHERE key='schema_version'")
        registry.connection.commit()
    # Old-revision citation: the body on disk moved on after the review.
    (sources / "cusco.md").write_text(f"---\ntitle: cusco.md\n---\n{CUSCO} Updated tonight.\n")
    write_state(llmwiki / "state.json", WikiState(sources={
        name: SourceState(hashlib.sha256(body.encode()).hexdigest(), [])
        for name, body in bodies.items()
    }))
    (llmwiki / "cache" / "broken.md.json").write_text("{not json")
    (concepts / "delta-observatory.md").write_text(
        "---\ntype: concept\nreviewed: true\n---\nHuman note: see [[cusco-array|the array]] "
        "and [[Lost Page]] and [[delta-observatory#History]].\n"
    )
    (concepts / "cusco-array.md").write_text("---\ntype: concept\n---\nGenerated body.\n")
    (llmwiki / "render-page-hashes.json").write_text(json.dumps({
        "concepts/cusco-array.md": hashlib.sha256(
            (concepts / "cusco-array.md").read_bytes()).hexdigest(),
    }))
    assert sorted(p.name for p in llmwiki.iterdir() if p.name.startswith("registry")) == [
        "registry.db"
    ], "fixture must start without WAL/SHM sidecars"
    return vault


def _run(*args: str):
    # Import lazily to avoid the shared-fixture module's collection-time cycle.
    from tests.new.test_identity_migration_schema7 import _guard

    with pytest.MonkeyPatch.context() as guard:
        attempts = _guard(guard)
        result = CliRunner().invoke(app, ["identity-migration", *args])
        assert attempts == [], "a refused result must not hide attempted mutation/network"
    return result


def _report(result) -> dict:
    assert "Traceback" not in result.output, result.output
    return json.loads(result.stdout)


def test_preview_inventories_bindings_without_touching_the_vault(legacy_vault: Path):
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert _snapshot(legacy_vault) == before, "preview modified or created vault entries"
    assert result.exit_code == 3, result.output  # completed, but blockers (malformed cache)
    report = _report(result)
    assert report["mode"] == "dry-run"
    assert report["status"] == "preview"
    assert report["apply"] == {"implemented": False, "handled_by": "future plan build"}
    assert report["vault_unchanged"] is True
    assert report["registry"]["strategy"] == "in-memory-deserialize"
    assert report["registry"]["schema_version"] == 6
    assert report["registry"]["sidecars"] == {"wal": None, "shm": None, "journal": None}
    by_key = {(b["source_file"], b["slug"]): b for b in report["bindings"]}
    assert set(by_key) == {
        ("lima.md", "delta-observatory"), ("quito.md", "delta-observatory"),
        ("cusco.md", "cusco-array"),
    }
    lima = by_key[("lima.md", "delta-observatory")]
    quito = by_key[("quito.md", "delta-observatory")]
    assert lima["status"] == quito["status"] == "needs_assessment"
    assert lima["canonical_id"] == quito["canonical_id"]
    assert "co-owned" in lima["reason"]
    assert quito["evidence"]["legacy_assessments"] == 1
    assert lima["evidence"]["legacy_assessments"] == 0
    assert lima["evidence"]["receipt_version"] == "grounded-review-v2"
    assert lima["evidence"]["source_revision_matches_receipt"] is True
    cusco = by_key[("cusco.md", "cusco-array")]
    assert cusco["status"] == "unresolved"
    assert cusco["evidence"]["source_revision_matches_receipt"] is False
    assert "revision" in cusco["reason"]
    assert cusco["evidence"]["cited_source_hash_matches_current"] is False
    assert cusco["evidence"]["claims"] == 1
    assert cusco["evidence"]["catalog_span_citations"] == 1
    assert cusco["provenance"]["registry_versions"] == {
        "prompt": "prompt-v1", "schema": "schema-v1", "model": "model-fixture",
    }
    assert cusco["provenance"]["cache_artifact"] == {
        "kind": "source_synthesis", "schema_version": 2,
    }
    assert report["summary"]["bindings"] == {
        "resolved": 0, "unresolved": 1, "needs_assessment": 2, "total": 3,
    }
    assert report["assessment_estimate"]["status"] == "unavailable"
    assert report["assessment_estimate"]["lower"] is None
    assert report["assessment_estimate"]["upper"] is None
    assert report["assessment_estimate"]["needs_assessment_bindings"] == 2
    assert report["assessment_estimate"]["model_calls_made"] == 0
    refs = {(r["page"], r["target"]): r for r in report["references"]}
    assert refs[("concepts/delta-observatory.md", "cusco-array")]["status"] == "resolved"
    assert refs[("concepts/delta-observatory.md", "cusco-array")]["display"] == "the array"
    assert refs[("concepts/delta-observatory.md", "Lost Page")]["status"] == "unresolved"
    heading = refs[("concepts/delta-observatory.md", "delta-observatory#History")]
    assert heading["status"] == "resolved"
    assert all(r["page_human_edited"] for r in refs.values())
    edits = {p["page"]: p for p in report["human_edited_pages"]}
    assert edits["concepts/delta-observatory.md"]["basis"] == "reviewed: true"
    assert "concepts/cusco-array.md" not in edits
    assert any("broken.md.json" in b["path"] and b["kind"] == "malformed_cache"
               for b in report["blockers"])
    assert [b["kind"] for b in report["blockers"]] == ["malformed_cache"]


def test_repeated_preview_is_byte_identical_and_still_read_only(legacy_vault: Path):
    before = _snapshot(legacy_vault)
    first, second = _run(str(legacy_vault), "--dry-run", "--json"), None
    second = _run(str(legacy_vault), "--dry-run", "--json")
    assert _snapshot(legacy_vault) == before
    assert first.exit_code == second.exit_code == 3
    a, b = _report(first), _report(second)
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def _hold_writer(db: Path, *, commit: bool) -> subprocess.Popen:
    code = textwrap.dedent(f"""
        import sqlite3, sys, time
        c = sqlite3.connect({str(db)!r})
        c.execute("PRAGMA wal_autocheckpoint=0")
        c.execute("BEGIN IMMEDIATE")
        c.execute("INSERT INTO meta(key, value) VALUES('fixture-busy', 'x')")
        if {commit!r}:
            c.commit()
        print("held", flush=True)
        sys.stdin.readline()
    """)
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    return proc


def _release(proc: subprocess.Popen) -> None:
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait(timeout=30)


@pytest.mark.parametrize("commit", [False, True])
def test_live_writer_or_uncheckpointed_wal_fails_closed(legacy_vault: Path, commit: bool):
    db = legacy_vault / "04-Wiki" / ".llmwiki" / "registry.db"
    proc = _hold_writer(db, commit=commit)
    try:
        wal = db.with_name("registry.db-wal")
        assert wal.exists()
        if commit:
            assert wal.stat().st_size > 0, "committed row must live in the WAL"
            # immutable=1 is NOT a consistent view: it misses the committed WAL row.
            # Demonstrate the missing committed WAL row using a CLOSED private
            # memory copy, never opening the fixture database through SQLite.
            raw = bytearray(db.read_bytes())
            raw[18] = raw[19] = 1
            ro = sqlite3.connect(":memory:")
            try:
                ro.deserialize(bytes(raw))
                query = "SELECT count(*) FROM meta WHERE key='fixture-busy'"
                assert ro.execute(query).fetchone()[0] == 0
            finally:
                ro.close()
        before = _snapshot(legacy_vault)
        result = _run(str(legacy_vault), "--dry-run", "--json")
        assert _snapshot(legacy_vault) == before
        assert result.exit_code == 2, result.output
        report = _report(result)
        assert report["status"] == "refused"
        assert report["bindings"] == []
        reasons = " ".join(report["refusal"]["reasons"])
        if commit:
            assert "WAL" in reasons and "immutable" in reasons
        assert "live SQLite connection" in reasons or "WAL" in reasons
        assert "stop" in report["refusal"]["action"].lower()
        assert report["registry"]["sidecars"]["wal"]["bytes"] == wal.stat().st_size
    finally:
        _release(proc)


def test_read_only_fixture_is_previewable(legacy_vault: Path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    changed: list[tuple[Path, int]] = []
    for directory, dirs, files in os.walk(legacy_vault, topdown=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            changed.append((path, mode))
            os.chmod(path, 0o555 if stat.S_ISDIR(mode) else 0o444)
    changed.append((legacy_vault, legacy_vault.stat().st_mode))
    os.chmod(legacy_vault, 0o555)
    try:
        before = _snapshot(legacy_vault)
        result = _run(str(legacy_vault), "--dry-run", "--json")
        assert _snapshot(legacy_vault) == before
        assert result.exit_code == 3, result.output
        assert _report(result)["summary"]["bindings"]["total"] == 3
    finally:
        for path, mode in reversed(changed):
            os.chmod(path, mode)


def test_missing_vault_and_missing_registry_are_reported_not_created(tmp_path: Path):
    missing = tmp_path / "nope"
    result = _run(str(missing), "--dry-run", "--json")
    assert result.exit_code == 2, result.output
    assert not missing.exists()
    assert _report(result)["refusal"]["reasons"][0].startswith("vault directory does not exist")

    vault = tmp_path / "vault"
    sources = vault / "04-Wiki" / "sources"
    sources.mkdir(parents=True)
    (sources / "only.md").write_text("---\ntitle: only\n---\nA lonely source body.\n")
    before = _snapshot(vault)
    result = _run(str(vault), "--dry-run", "--json")
    assert _snapshot(vault) == before
    assert not (vault / "04-Wiki" / ".llmwiki").exists()
    assert result.exit_code == 3, result.output
    report = _report(result)
    assert report["registry"]["strategy"] == "absent"
    assert report["sources"]["present"] == ["only.md"]
    assert [b["kind"] for b in report["blockers"]] == ["registry_absent"]
    assert report["summary"]["bindings"]["total"] == 0


def test_malformed_registry_file_is_refused(legacy_vault: Path):
    db = legacy_vault / "04-Wiki" / ".llmwiki" / "registry.db"
    db.write_bytes(b"this is not a sqlite database at all" * 4)
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert _snapshot(legacy_vault) == before
    assert result.exit_code == 2, result.output
    report = _report(result)
    assert report["status"] == "refused"
    assert "not an SQLite" in " ".join(report["refusal"]["reasons"])


def test_future_schema_version_is_an_unknown_version_blocker(legacy_vault: Path):
    db = legacy_vault / "04-Wiki" / ".llmwiki" / "registry.db"
    from tests.new.migration_fixture_helpers import closed_registry

    with closed_registry(db.parent, edit=True) as registry:
        registry.connection.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert _snapshot(legacy_vault) == before
    assert result.exit_code == 3, result.output
    report = _report(result)
    assert report["registry"]["schema_version"] == 99
    kinds = {b["kind"] for b in report["blockers"]}
    assert "unknown_schema_version" in kinds
    assert report["summary"]["bindings"]["resolved"] == 0


def test_symlink_inside_metadata_is_refused_before_reading(legacy_vault: Path, tmp_path: Path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (legacy_vault / "04-Wiki" / ".llmwiki" / "cache" / "zzz.md.json").symlink_to(outside)
    opened: list[str] = []

    def audit(event, args):
        if (event == "open" and isinstance(args[0], (str, bytes, os.PathLike))
                and os.path.realpath(os.fsdecode(args[0])) == str(outside)):
            opened.append(os.fsdecode(args[0]))

    sys.addaudithook(audit)
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert _snapshot(legacy_vault) == before
    assert result.exit_code == 2, result.output
    assert "symlink" in " ".join(_report(result)["refusal"]["reasons"])
    assert opened == []


def test_dry_run_is_required_and_apply_does_not_exist(legacy_vault: Path):
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--json")
    assert result.exit_code == 2, result.output
    assert "--dry-run" in result.output and "future plan build" in result.output
    result = _run(str(legacy_vault), "--apply")
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert _snapshot(legacy_vault) == before


def test_preview_makes_no_network_or_provider_calls(legacy_vault: Path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/provider access during preview")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    import httpx

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert result.exit_code == 3, result.output
    assert _report(result)["assessment_estimate"]["model_calls_made"] == 0


def test_scratch_report_outside_vault_only(legacy_vault: Path, tmp_path: Path):
    inside = legacy_vault / "scratch"
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json", "--scratch", str(inside))
    assert result.exit_code == 2, result.output
    assert not inside.exists()
    assert _snapshot(legacy_vault) == before
    outside = tmp_path / "scratch"
    result = _run(str(legacy_vault), "--dry-run", "--json", "--scratch", str(outside))
    assert result.exit_code == 3, result.output
    assert _snapshot(legacy_vault) == before
    written = json.loads((outside / "identity-migration-preview.json").read_text())
    assert written == _report(result)


def test_human_summary_mode_mentions_counts_and_apply_boundary(legacy_vault: Path):
    before = _snapshot(legacy_vault)
    # A spelled ``..`` component is ordinary CLI usage, not parent traversal.
    spelled = legacy_vault.parent / "detour" / ".." / legacy_vault.name
    (legacy_vault.parent / "detour").mkdir()
    result = _run(str(spelled), "--dry-run")
    assert _snapshot(legacy_vault) == before
    assert result.exit_code == 3, result.output
    assert "needs_assessment: 2" in result.output
    assert "unresolved: 1" in result.output
    assert "apply" in result.output and "not implemented" in result.output


def test_real_process_cli_json_and_exit_code(legacy_vault: Path, tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    # The venv's ``olw`` entry point imports whatever editable install it has;
    # pin the package under test like the inventory-boundary CLI test does.
    repo = Path(__file__).resolve().parents[2]
    guard_dir = tmp_path / "process-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        "import atexit, os, pytest\n"
        "from tests.new.test_identity_migration_schema7 import _guard\n"
        "guard = pytest.MonkeyPatch()\nattempts = _guard(guard)\n"
        "atexit.register(lambda: os._exit(97) if attempts else None)\n"
    )
    env = {
        "PATH": os.defpath, "HOME": str(home), "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_CONFIG_HOME": str(home / "config"), "XDG_DATA_HOME": str(home / "data"),
        "TMPDIR": str(home), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(guard_dir) + os.pathsep + str(repo),
    }
    before = _snapshot(legacy_vault)
    home_before = _snapshot(home)
    command = [
        str(Path(sys.executable).with_name("olw")), "identity-migration", str(legacy_vault),
        "--dry-run", "--json",
    ]
    started = time.monotonic()
    proc = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)
    assert _snapshot(legacy_vault) == before
    assert _snapshot(home) == home_before, "preview wrote under HOME/cache"
    assert proc.returncode == 3, proc.stderr
    report = json.loads(proc.stdout)
    assert report["summary"]["bindings"]["total"] == 3
    assert report["vault_unchanged"] is True
    assert "Traceback" not in proc.stderr and "sitecustomize" not in proc.stderr
    assert time.monotonic() - started < 120
