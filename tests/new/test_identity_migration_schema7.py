"""Read-only preview against actual schema-7 public build artifacts.

Only the HTTP transport is a deterministic, source-context-aware local double.
The preview must not construct a registry, recover a journal or call a provider.
"""

from __future__ import annotations

import json
import socket
import sqlite3

import pytest

from obsidian_llm_wiki.core.immutable_publication import current_publication
from obsidian_llm_wiki.core.registry import VaultRegistry
from tests.new import test_identity_migration_preview as legacy
from tests.new import test_identity_structural_cli as structural
from tests.new import test_publication_identity_cli as harness
from tests.new.migration_fixture_helpers import (
    closed_registry,
    edit_payload_and_reseal,
    edit_receipt_and_reseal,
    meta_for,
    pending_workspace,
    reseal_inner_generation,
    reseal_outer,
)
from tests.new.test_identity_migration_preview import _report, _run, _snapshot

cli = harness.cli
legacy_vault = legacy.legacy_vault


@pytest.fixture
def committed(cli, monkeypatch):
    monkeypatch.setattr(harness, "producer", structural._producer)
    endpoint = cli.use(harness.Endpoint(identity=structural._verdict))
    structural._write(cli, "owner.md", structural.KERNEL)
    structural._write(cli, "router.md", structural.LINKS)
    first = cli.build()
    assert first.exit_code == 0, first.output
    structural._write(cli, "reuse.md", structural.KERNEL)
    second = cli.build()
    assert second.exit_code == 0, second.output
    cli.generation, cli.manifest = current_publication(cli.wiki)
    meta = meta_for(cli)
    marker = json.loads((meta / "identity-publication.json").read_text())
    assert marker["status"] == "committed"
    plan = json.loads((meta / "publications" / (marker["plan_id"] + ".json")).read_text())
    return cli, endpoint, marker, plan


def _guard(monkeypatch):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("writer/provider/network")
        raise AssertionError("preview called a writer/provider/network")

    monkeypatch.setattr(VaultRegistry, "__init__", forbidden)
    from obsidian_llm_wiki.core import publication_transaction

    monkeypatch.setattr(publication_transaction, "recover_publication", forbidden)
    monkeypatch.setattr(publication_transaction.PublicationTransaction, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    from obsidian_llm_wiki.providers import llm

    for name in ("create_llm_client", "call_llm", "acall_llm", "list_provider_models"):
        monkeypatch.setattr(llm, name, forbidden)
    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        if database != ":memory:":
            attempts.append("filesystem SQLite connection")
            raise AssertionError("preview opened a filesystem SQLite connection")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    return attempts


def _preview(committed, monkeypatch, expected=0):
    cli, endpoint, _marker, _plan = committed
    before, calls = _snapshot(cli.wiki.parent), len(endpoint.requests)
    with monkeypatch.context() as guard:
        attempts = _guard(guard)
        result = _run(str(cli.wiki.parent), "--dry-run", "--json")
        assert attempts == []
    assert _snapshot(cli.wiki.parent) == before
    assert len(endpoint.requests) == calls
    detail = json.loads(result.stdout) if result.stdout else repr(result.exception)
    if isinstance(detail, dict) and "assessment_estimate" in detail:
        assert detail["assessment_estimate"]["model_calls_made"] == 0
    # Harness evidence is outside the vault and records actual CLI output even
    # when the acceptance assertion stays RED. It is not fabricated status data.
    evidence = cli.wiki.parent.parent / "preview-evidence"
    evidence.mkdir(exist_ok=True)
    number = len(list(evidence.glob("*.json")))
    (evidence / f"{number:02d}.json").write_text(
        json.dumps(
            {
                "expected_exit": expected,
                "actual_exit": result.exit_code,
                "stdout": result.stdout,
                "exception": repr(result.exception),
                "vault_unchanged": True,
                "model_calls_added": len(endpoint.requests) - calls,
                "forbidden_attempts": attempts,
            },
            indent=2,
        )
    )
    assert result.exit_code == expected, detail
    return _report(result)


def test_committed_preview_matches_actual_plan_and_citations(committed, monkeypatch):
    _cli, _endpoint, marker, plan = committed
    report = _preview(committed, monkeypatch)
    assert report["registry"]["schema_version"] == 7
    assert report["publication"]["status"] == "committed"
    assert report["publication"]["verified"] is True
    assert report["publication"]["plan_id"] == marker["plan_id"]
    expected = {}
    for name, raw in plan["sources"]:
        source = json.loads(raw)
        for concept in source["concepts"]:
            expected[(name, concept["canonical_id"])] = concept
    actual = {(b["source_file"], b["canonical_id"]): b for b in report["bindings"]}
    assert actual.keys() == expected.keys()
    assert {b["status"] for b in actual.values()} == {"resolved"}
    for key, concept in expected.items():
        evidence = actual[key]["evidence"]
        assert evidence["claims"] == len(concept["claims"])
        assert evidence["catalog_span_citations"] == sum(
            bool(c["evidence"]["span_ids"]) for c in concept["claims"]
        )
        assert evidence["historical_source_context"] is True
    assert report["sources"]["registered"] == sorted(name for name, _ in plan["sources"])
    expected_refs = {
        (name, key): record
        for name, raw in plan["sources"]
        for key, record in json.loads(raw)["review_receipt"].get("references", {}).items()
    }
    observed_refs = {(r["source_file"], r["occurrence"]): r for r in report["identity_references"]}
    assert observed_refs.keys() == expected_refs.keys()
    for key, record in expected_refs.items():
        assert observed_refs[key]["recorded_status"] == record["status"]
        assert observed_refs[key]["target_id"] == record["target_id"]
        assert observed_refs[key]["status"] == (
            "resolved" if record["status"] == "bound" else "unresolved"
        )
    assert report["summary"]["source_count"] == len(plan["sources"])
    assert report["summary"]["claims"] == sum(len(c["claims"]) for c in expected.values())
    meta = meta_for(_cli)
    assert len(report["cache_history"]) == len(list((meta / "cache-history").glob("*.json")))
    estimate = report["assessment_estimate"]
    assert estimate["status"] == "unavailable"
    assert estimate["lower"] is estimate["upper"] is None
    assert estimate["model_calls_made"] == 0
    again = _preview(committed, monkeypatch)
    report.pop("generated_at")
    again.pop("generated_at")
    assert again == report


@pytest.mark.parametrize(
    "stage_state",
    [
        "pending",
        "pending-marker",
        "interrupted-staging",
        "binary-stuck",
        "corrupt-journal",
        "committed-cleanup",
    ],
)
def test_real_pending_journal_never_looks_committed(committed, monkeypatch, stage_state):
    import shutil

    cli, _endpoint, _marker, raw = committed
    pointer = cli.wiki / "OLW-Publications" / "current.json"
    accepted = _snapshot(cli.generation)
    pointer_bytes = pointer.read_bytes()
    config = pending_workspace(cli, raw, stage_state)
    stage = config.llmwiki_dir / "publication-stage"
    assert (stage / "registry-before.db").exists()
    if stage_state == "pending":
        assert json.loads((stage / "manifest.json").read_text())["status"] == "pending"
        assert (
            json.loads((meta_for(cli) / "identity-publication.json").read_text())["status"]
            == "committed"
        )
    # A failed NEW private workspace cannot revoke the already committed pointer.
    report = _preview(committed, monkeypatch, expected=0)
    assert report["publication"]["verified"] is True
    assert report["publication"]["recovery_required"] is False
    assert report["summary"]["bindings"]["resolved"] == 3
    assert report["publication_workflow"]["current_generation"] == cli.generation.name
    assert _snapshot(cli.generation) == accepted
    assert pointer.read_bytes() == pointer_bytes

    # Different state: a journal INSIDE the accepted generation is corruption.
    # Reseal outer hashes to reach the journal/marker semantic gates, not inventory drift.
    shutil.copytree(stage, meta_for(cli) / "publication-stage")
    if stage_state == "pending-marker":
        shutil.copyfile(
            config.llmwiki_dir / "identity-publication.json",
            meta_for(cli) / "identity-publication.json",
        )
    reseal_outer(cli)
    report = _preview(committed, monkeypatch, expected=2 if stage_state == "pending-marker" else 3)
    if stage_state == "pending-marker":
        assert report["status"] == "refused" and report["bindings"] == []
        assert "no committed semantic plan" in " ".join(report["refusal"]["reasons"])
    else:
        assert report["publication"]["verified"] is False
        assert report["publication"]["recovery_required"] is True
        assert "pending_publication_recovery" in {b["kind"] for b in report["blockers"]}
        assert report["summary"]["bindings"]["resolved"] == 0
        expected_journal = {
            "interrupted-staging": "interrupted-or-corrupt",
            "corrupt-journal": "interrupted-or-corrupt",
            "committed-cleanup": "committed",
        }.get(stage_state, "pending")
        assert report["publication"]["journal_status"] == expected_journal
    assert (meta_for(cli) / "publication-stage" / "registry-before.db").exists()


def _edit_receipt_and_reseal(committed, edit):
    edit_receipt_and_reseal(committed, edit)


@pytest.mark.parametrize("unknown", ["identity", "review", "context", "basis"])
def test_unknown_receipt_context_or_binding_not_resolved(committed, monkeypatch, unknown):
    def edit(receipt):
        if unknown == "identity":
            receipt["identity"]["version"] = "identity-assessment-v999"
        elif unknown == "review":
            receipt["version"] = "grounded-review-v999"
        elif unknown == "basis":
            for binding in receipt["publication_binding"].values():
                binding["basis"] = "future magic"
        else:
            from obsidian_llm_wiki.core.identity_plan import digest

            record = receipt["identity"]["assessments"][0]
            record["subject_context"]["version"] = "identity-context-v999"
            record["subject_context_sha256"] = digest(record["subject_context"])

    _edit_receipt_and_reseal(committed, edit)
    report = _preview(committed, monkeypatch, expected=3)
    reuse = next(b for b in report["bindings"] if b["source_file"] == "reuse.md")
    assert reuse["status"] == "unresolved"
    assert report["publication"]["verified"] is True
    assert any(b["kind"] == "unsupported_identity_evidence" for b in report["blockers"])
    assert not any(b["kind"] == "publication_unverified" for b in report["blockers"])


def test_unknown_registry_schema_cannot_resolve_bindings(committed, monkeypatch):
    cli, *_ = committed
    with closed_registry(meta_for(cli), edit=True) as registry:
        registry.connection.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    reseal_outer(cli)
    report = _preview(committed, monkeypatch, expected=3)
    assert report["registry"]["schema_version"] == 99
    assert report["summary"]["bindings"]["resolved"] == 0
    assert "unknown_schema_version" in {b["kind"] for b in report["blockers"]}


@pytest.mark.parametrize(
    "damage",
    [
        "corrupt-marker",
        "stale-marker",
        "unknown-plan",
        "corrupt-plan",
        "absent-plan",
        "source-drift",
        "review-drift",
        "missing-history",
        "missing-history-table",
    ],
)
def test_damaged_committed_artifacts_fail_closed(committed, monkeypatch, damage):
    import os

    cli, _endpoint, marker, _plan = committed
    meta = meta_for(cli)
    marker_path = meta / "identity-publication.json"
    if damage == "corrupt-marker":
        marker_path.write_text("{broken")
    elif damage == "stale-marker":
        marker["generation"] = "0" * 64
        marker_path.write_text(json.dumps(marker))
    elif damage == "unknown-plan":
        marker["version"] = "identity-publication-plan-v999"
        marker_path.write_text(json.dumps(marker))
    elif damage in {"corrupt-plan", "absent-plan"}:
        path = meta / "publications" / (marker["plan_id"] + ".json")
        if damage == "corrupt-plan":
            path.write_text("{}")
        else:
            path.unlink()
    elif damage == "source-drift":
        # Human input, not the separately rendered immutable source view.
        path = cli.wiki / "sources" / "owner.md"
        before_generation = _snapshot(cli.generation)
        info = path.stat()
        path.write_text(path.read_text().replace("kernel", "postal"))
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        assert path.stat().st_size == info.st_size
        assert _snapshot(cli.generation) == before_generation
    else:
        with closed_registry(meta, edit=True) as registry:
            if damage == "missing-history":
                registry.connection.execute(
                    "DELETE FROM identity_source_context WHERE filename='owner.md'"
                )
            elif damage == "missing-history-table":
                registry.connection.execute("DROP TABLE identity_source_context")
            else:
                registry.connection.execute(
                    "UPDATE synthesis SET payload_json=json_set("
                    "payload_json,'$.review_receipt.reviewer_fingerprint','changed')"
                )
        if damage == "missing-history":
            reseal_inner_generation(cli)
    reseal_outer(cli)
    expected = 3 if damage in {"source-drift", "unknown-plan", "missing-history"} else 2
    report = _preview(committed, monkeypatch, expected=expected)
    if expected == 2:
        assert report["status"] == "refused" and report["bindings"] == []
        reasons = " ".join(report["refusal"]["reasons"])
        assert "inventory/digest mismatch" not in reasons
        gate = {
            "stale-marker": "registry digest",
            "corrupt-plan": "semantic plan digest",
            "review-drift": "registry digest",
            "missing-history-table": "identity_source_context",
            "absent-plan": "No such file",
            "corrupt-marker": "Expecting property",
        }[damage]
        assert gate in reasons
    else:
        if damage == "missing-history":
            # Byte consistency is valid but historical identity evidence is not.
            assert report["publication"]["verified"] is True
            assert "missing_historical_context" in {b["kind"] for b in report["blockers"]}
            assert (
                next(b for b in report["bindings"] if b["source_file"] == "owner.md")["status"]
                == "unresolved"
            )
            assert (
                next(b for b in report["bindings"] if b["source_file"] == "reuse.md")["status"]
                == "unresolved"
            )
        else:
            assert report["publication"]["verified"] is False
            assert report["summary"]["bindings"]["resolved"] == 0
            detail = " ".join(b["detail"] for b in report["blockers"])
            assert (
                "source or policy file drift"
                if damage == "source-drift"
                else "unknown publication plan version"
            ) in detail


@pytest.mark.parametrize("commit", [False, True])
def test_schema7_live_wal_or_busy_writer_is_refused(committed, monkeypatch, commit):
    from tests.new.test_identity_migration_preview import _hold_writer, _release

    cli, *_ = committed
    proc = _hold_writer(meta_for(cli) / "registry.db", commit=commit)
    try:
        wal = meta_for(cli) / "registry.db-wal"
        assert wal.exists()
        if commit:
            assert wal.stat().st_size > 0
        # Include the stable held sidecars in outer hashes so the refusal must
        # come from the closed-registry gate, not generic inventory inequality.
        reseal_outer(cli)
        report = _preview(committed, monkeypatch, expected=2)
        assert report["status"] == "refused"
        assert report["bindings"] == []
        assert "uncheckpointed sidecars" in " ".join(report["refusal"]["reasons"])
    finally:
        _release(proc)


@pytest.mark.parametrize("damage", ["incomplete", "unsupported-spans", "missing-context"])
def test_bilateral_proof_not_silently_trusted(committed, monkeypatch, damage):
    def edit(receipt):
        from obsidian_llm_wiki.core.identity_plan import digest

        record = receipt["identity"]["assessments"][0]
        if damage == "incomplete":
            record["candidate_context"]["complete"] = False
            record["candidate_context_sha256"] = digest(record["candidate_context"])
        elif damage == "missing-context":
            record.pop("subject_context")
        else:
            record["span_ids"] = ["subject:0:invented"]

    _edit_receipt_and_reseal(committed, edit)
    report = _preview(committed, monkeypatch, expected=3)
    assert (
        next(b for b in report["bindings"] if b["source_file"] == "reuse.md")["status"]
        == "unresolved"
    )
    assert report["publication"]["verified"] is True
    assert any(
        b["kind"] == "unsupported_identity_evidence"
        and "bilateral same proof missing, incomplete or invalid" in b["detail"]
        for b in report["blockers"]
    )


def test_missing_candidate_history_not_filled_from_current_source(committed, monkeypatch):
    cli, *_ = committed
    with closed_registry(meta_for(cli), edit=True) as registry:
        registry.connection.execute("DELETE FROM identity_source_context WHERE filename='owner.md'")
    reseal_inner_generation(cli)
    reseal_outer(cli)
    report = _preview(committed, monkeypatch, expected=3)
    assert report["publication"]["verified"] is True
    reuse = next(b for b in report["bindings"] if b["source_file"] == "reuse.md")
    assert reuse["status"] == "unresolved", "current same-text owner is not historical proof"
    assert any(
        b["kind"] == "unsupported_identity_evidence"
        and "historical bilateral source revision unavailable" in b["detail"]
        for b in report["blockers"]
    )


def test_schema7_human_summary_exposes_publication_boundary(committed, monkeypatch):
    cli, endpoint, *_ = committed
    before, calls = _snapshot(cli.wiki.parent), len(endpoint.requests)
    with monkeypatch.context() as guard:
        attempts = _guard(guard)
        result = _run(str(cli.wiki.parent), "--dry-run")
        assert attempts == []
    assert result.exit_code == 0, result.output
    assert _snapshot(cli.wiki.parent) == before
    assert len(endpoint.requests) == calls
    assert "Publication: committed; verified: True; recovery required: False" in result.output
    assert "resolved: 3" in result.output
    assert "Assessment estimate: unavailable" in result.output
    assert "not implemented" in result.output


def test_actual_olw_process_exit_codes_and_json(committed, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from tests.new.test_identity_migration_preview import _hold_writer, _release

    cli, _endpoint, marker, _plan = committed
    home = tmp_path / "process-home"
    home.mkdir()
    guard_dir = tmp_path / "process-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        "import pytest\n"
        "from tests.new.test_identity_migration_schema7 import _guard\n"
        "guard = pytest.MonkeyPatch()\nattempts = _guard(guard)\n"
        "import atexit, os\natexit.register(lambda: os._exit(97) if attempts else None)\n"
    )
    repo = Path(__file__).resolve().parents[2]
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_DATA_HOME": str(home / "data"),
        "TMPDIR": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(guard_dir) + os.pathsep + str(repo),
    }
    evidence = tmp_path / "process-evidence"
    evidence.mkdir()

    def run(label, expected, human=False):
        before, home_before = _snapshot(cli.wiki.parent), _snapshot(home)
        command = [
            str(Path(sys.executable).with_name("olw")),
            "identity-migration",
            str(cli.wiki.parent),
            "--dry-run",
            *([] if human else ["--json"]),
        ]
        proc = subprocess.run(
            command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60
        )
        assert proc.returncode == expected, proc.stderr
        assert "sitecustomize" not in proc.stderr and "Traceback" not in proc.stderr
        assert _snapshot(cli.wiki.parent) == before
        assert _snapshot(home) == home_before
        (evidence / (label + ".stdout")).write_text(proc.stdout)
        (evidence / (label + ".stderr")).write_text(proc.stderr)
        (evidence / (label + ".receipt.json")).write_text(
            json.dumps(
                {
                    "command": command,
                    "returncode": proc.returncode,
                    "vault_unchanged": True,
                    "home_unchanged": True,
                    "before": before,
                    "provider_calls": 0,
                },
                indent=2,
            )
        )
        if not human:
            data = json.loads(proc.stdout)
            assert data["mode"] == "dry-run" and data["apply"]["implemented"] is False
            assert isinstance(data["bindings"], list)
            assert data["status"] in {"preview", "refused"}
            return data
        assert "Publication:" in proc.stdout

    assert run("committed-rc0", 0)["summary"]["bindings"]["resolved"] == 3
    run("human-rc0", 0, human=True)
    proc = _hold_writer(meta_for(cli) / "registry.db", commit=True)
    try:
        reseal_outer(cli)
        refused = run("wal-rc2", 2)
        assert refused["status"] == "refused"
        assert "uncheckpointed sidecars" in " ".join(refused["refusal"]["reasons"])
    finally:
        _release(proc)
    # An archived journal inside the committed generation blocks readiness even
    # though the pointer and semantic marker remain committed. Reseal inventory.
    stage = meta_for(cli) / "publication-stage"
    stage.mkdir()
    (stage / "manifest.json").write_text('{"status":"pending"}')
    reseal_outer(cli)
    report = run("pending-journal-rc3", 3)
    assert report["publication"]["recovery_required"] is True
    assert report["summary"]["bindings"]["resolved"] == 0
    run("human-rc3", 3, human=True)


def test_process_interruption_preserves_real_journal_and_refuses_readiness(committed, monkeypatch):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    cli, endpoint, _marker, raw = committed
    workspace = cli.wiki / "OLW-Publications" / ".staging" / "process-interruption"
    shutil.copytree(cli.generation, workspace / "04-Wiki")
    pointer = cli.wiki / "OLW-Publications" / "current.json"
    pointer_bytes, accepted = pointer.read_bytes(), _snapshot(cli.generation)
    script = """import json,os,sys
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.registry import VaultRegistry
from obsidian_llm_wiki.core.identity_plan import ProspectivePublicationPlan
from obsidian_llm_wiki.core.publication_transaction import PublicationTransaction
config=Config(vault_path=sys.argv[1])
marker=json.loads((config.llmwiki_dir/'identity-publication.json').read_text())
raw=json.loads((config.llmwiki_dir/'publications'/(marker['plan_id']+'.json')).read_text())
registry=VaultRegistry(config.llmwiki_dir)
plan=ProspectivePublicationPlan(**raw)
transaction=PublicationTransaction(registry, config, plan)
plan.write_marker(registry, config, 'pending')
os._exit(72)
"""
    # Whitelist, not an inherited credential-bearing environment.
    env = {
        key: os.environ[key]
        for key in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR", "PATH")
    }
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    proc = subprocess.run(
        [sys.executable, "-c", script, str(workspace)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 72, proc.stderr
    stage = workspace / "04-Wiki" / ".llmwiki" / "publication-stage"
    assert json.loads((stage / "manifest.json").read_text())["status"] == "pending"
    assert (stage / "registry-before.db").exists()
    report = _preview(committed, monkeypatch, expected=0)
    assert report["publication"]["verified"] is True
    assert report["publication"]["recovery_required"] is False
    assert report["summary"]["bindings"]["resolved"] == 3
    assert _snapshot(cli.generation) == accepted
    assert pointer.read_bytes() == pointer_bytes
    # The dead workspace's WAL/journal is retained and never treated as current.
    assert json.loads((stage / "manifest.json").read_text())["status"] == "pending"
    # Missing authoritative pointer: installed/staged bytes alone are not ready.
    # Moving only the disposable pointer is explicit test tampering, not recovery
    # or a claim to reproduce the first-build lifecycle.
    pointer.rename(pointer.with_name("fixture-saved-current.json"))
    report = _preview(committed, monkeypatch, expected=2)
    assert report["status"] == "refused" and report["bindings"] == []
    assert "no committed generation" in " ".join(report["refusal"]["reasons"])


@pytest.mark.parametrize("shape", ["receipt-list", "concepts-null", "schema-columns"])
def test_malformed_schema_or_payload_refuses_without_writing(committed, monkeypatch, shape):
    cli, *_ = committed
    if shape == "schema-columns":
        with closed_registry(meta_for(cli), edit=True) as registry:
            registry.connection.execute(
                "ALTER TABLE synthesis RENAME COLUMN payload_json TO surprise"
            )
        reseal_outer(cli)
    else:

        def edit(payload):
            if shape == "receipt-list":
                payload["review_receipt"] = []
            else:
                payload["concepts"] = None

        edit_payload_and_reseal(committed, edit)
    # Payloads agree in SQL, raw cache and plan. Deserialization must reject
    # their malformed shape, not merely notice a checksum mismatch.
    report = _preview(committed, monkeypatch, expected=2)
    assert report["status"] == "refused" and report["bindings"] == []
    reasons = " ".join(report["refusal"]["reasons"])
    assert "inventory/digest mismatch" not in reasons
    assert "registry digest" not in reasons
    assert ("payload_json" if shape == "schema-columns" else "cache disagreement") in reasons


def test_pending_journal_is_observed_even_with_legacy_schema(legacy_vault):
    meta = legacy_vault / "04-Wiki" / ".llmwiki"
    (meta / "publication-stage").mkdir()
    (meta / "publication-stage" / "manifest.json").write_text('{"status":"pending"}')
    before = _snapshot(legacy_vault)
    result = _run(str(legacy_vault), "--dry-run", "--json")
    assert result.exit_code == 3, result.output
    assert _snapshot(legacy_vault) == before
    report = _report(result)
    assert report["publication"]["recovery_required"] is True
    assert report["summary"]["bindings"]["resolved"] == 0
    assert "pending_publication_recovery" in {b["kind"] for b in report["blockers"]}
