"""Exact relocation of both historical and explicit logical input inventories."""

import json
import shutil

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.publication_inputs import input_freshness, logical_publication_inputs
from tests.new import test_identity_migration_schema7 as migration
from tests.new.test_immutable_publication import human_state

cli = migration.cli
committed = migration.committed


@pytest.mark.parametrize("remove_original", [False, True])
def test_exact_copy_preview_uses_copy_not_historical_origin(committed, tmp_path, remove_original):
    cli, endpoint, _marker, _plan = committed
    original = cli.wiki.parent
    copy = tmp_path / "relocated"
    shutil.copytree(original, copy)
    before = {p: human_state(p) for p in copy.rglob("*") if p.is_file()}
    if remove_original:
        shutil.rmtree(original)
    else:
        # Keeping an edited origin must not poison or certify the exact copy.
        (original / ".env").write_text("GROUNDING_REVIEW=off\n")
        (original / "04-Wiki/sources/owner.md").write_text("Origin changed, not the copy")
    calls = len(endpoint.requests)
    result = CliRunner().invoke(app, ["identity-migration", str(copy), "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["publication"]["verified"]
    assert report["summary"]["bindings"]["resolved"] == 3
    assert len(endpoint.requests) == calls
    assert before == {p: human_state(p) for p in copy.rglob("*") if p.is_file()}


def test_new_publication_records_logical_inputs_and_copy_build_is_noop(committed, tmp_path):
    cli, endpoint, _marker, _plan = committed
    manifest = cli.manifest
    assert manifest["source_inputs_format"] == "vault-relative-v1"
    assert all(not name.startswith("/") for name, _ in manifest["source_inputs"])
    assert logical_publication_inputs(manifest) == tuple(map(tuple, manifest["source_inputs"]))
    copy = tmp_path / "rehearsal"
    shutil.copytree(cli.wiki.parent, copy)
    pointer = copy / "04-Wiki/OLW-Publications/current.json"
    before = human_state(pointer)
    calls = len(endpoint.requests)
    result = CliRunner().invoke(app, ["build", str(copy)])
    assert result.exit_code == 0, result.output
    assert len(endpoint.requests) == calls
    assert human_state(pointer) == before


def _inputs(origin=None):
    names = [".env", ".llmwiki/schema.yaml", "04-Wiki/.llmwiki/schema.yaml",
             "04-Wiki/.llmwiki/equivalences.json", "04-Wiki/sources/owner.md"]
    return {"format": "immutable-generation-v1", "status": "committed",
            **({"source_inputs_format": "vault-relative-v1"} if origin is None else {}),
            "source_inputs": [[str(origin / name) if origin else name, None] for name in names]}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("damage", [
    "external-same-basename", "mixed-roots", "traversal", "absolute", "backslash",
    "duplicate", "missing-policy", "bad-digest", "unknown-format", "namespace", "noncanonical",
])
def test_bad_input_mapping_is_unverifiable_before_any_content_read(
    tmp_path, monkeypatch, legacy, damage,
):
    from pathlib import Path

    manifest = _inputs(tmp_path / "origin" if legacy else None)
    rows = manifest["source_inputs"]
    if damage == "external-same-basename":
        rows[-1][0] = str(tmp_path / "evil/owner.md")
    elif damage == "mixed-roots":
        rows[1][0] = str(tmp_path / "elsewhere/.llmwiki/schema.yaml")
    elif damage == "traversal":
        rows[-1][0] = "04-Wiki/sources/../sources/owner.md"
    elif damage == "absolute":
        rows[-1][0] = "/04-Wiki/sources/owner.md"
    elif damage == "backslash":
        rows[-1][0] = r"04-Wiki\sources\owner.md"
    elif damage == "duplicate":
        rows.append(rows[-1].copy())
    elif damage == "missing-policy":
        rows.pop(1)
    elif damage == "bad-digest":
        rows[-1][1] = "not-a-digest"
    elif damage == "unknown-format":
        manifest["source_inputs_format"] = "future-v999"
    elif damage == "namespace":
        rows[-1][0] = "04-Wiki/other/owner.md"
    elif damage == "noncanonical":
        rows[-1][0] = "04-Wiki//sources/owner.md"

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mapping caused a content read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    state = input_freshness(tmp_path / "copy", manifest)
    assert state["status"] == "unverifiable" and state["reasons"]
    assert state["changed_paths"] == []


@pytest.mark.parametrize("legacy", [False, True])
def test_freshness_is_logical_observation_not_api_url_revalidation(tmp_path, legacy):
    import hashlib

    vault = tmp_path / "copy"
    vault.mkdir()
    manifest = _inputs(tmp_path / "gone-original" if legacy else None)
    # Absent API-only source and absent policies are known snapshots, not errors.
    assert input_freshness(vault, manifest)["status"] == "current"
    env = vault / ".env"
    env.write_text("GROUNDING_REVIEW=required\n")
    result = input_freshness(vault, manifest)
    assert result["status"] == "drifted" and result["changed_paths"] == [".env"]
    manifest["source_inputs"][0][1] = hashlib.sha256(env.read_bytes()).hexdigest()
    assert input_freshness(vault, manifest)["status"] == "current"
    env.unlink()
    assert input_freshness(vault, manifest)["status"] == "drifted"


@pytest.mark.parametrize("where", [
    ".env", ".llmwiki", "04-Wiki/sources", "04-Wiki/sources/owner.md",
])
def test_symlink_input_cannot_borrow_outside_freshness(tmp_path, where):
    vault, evil = tmp_path / "copy", tmp_path / "evil"
    vault.mkdir()
    evil.mkdir()
    target = vault / where
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(evil)
    outside = human_state(evil) if evil.is_file() else evil.stat()
    result = input_freshness(vault, _inputs())
    assert result["status"] == "unverifiable"
    assert evil.stat() == outside


def test_missing_vault_is_not_a_current_absent_input_snapshot(tmp_path):
    assert input_freshness(tmp_path / "missing", _inputs())["status"] == "unverifiable"


@pytest.mark.parametrize("damage", ["unknown-format", "external", "missing-source"])
def test_current_reader_rejects_bad_input_inventory(committed, damage):
    from obsidian_llm_wiki.core.immutable_publication import current_publication

    cli, *_ = committed
    manifest = json.loads(json.dumps(cli.manifest))
    if damage == "unknown-format":
        manifest["source_inputs_format"] = "future-999"
    elif damage == "external":
        manifest["source_inputs"][-1][0] = "/external/sources/owner.md"
    else:
        manifest["source_inputs"] = [row for row in manifest["source_inputs"]
                                     if not row[0].endswith("/owner.md")]
    (cli.wiki / "OLW-Publications/current.json").write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError, ValueError), match="input"):
        current_publication(cli.wiki)
