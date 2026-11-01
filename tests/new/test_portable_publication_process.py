"""Real olw process rehearsal; guards prohibit providers and filesystem SQLite."""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from obsidian_llm_wiki.cli._helpers import resolve_vault
from obsidian_llm_wiki.core.application import source_inventory
from obsidian_llm_wiki.core.identity_plan import digest
from tests.new import test_identity_migration_schema7 as migration
from tests.new.test_immutable_publication import human_state

cli = migration.cli
committed = migration.committed


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("remove_original", [False, True])
def test_real_cli_exact_copy_rehearsal(committed, tmp_path, legacy, remove_original):
    cli, _endpoint, _marker, _plan = committed
    original = cli.wiki.parent
    pointer = original / "04-Wiki/OLW-Publications/current.json"
    if legacy:
        # Serialize the exact pre-change v1 algorithm, not a new-format pointer
        # with a relabelled fingerprint. Archived generation bytes remain intact.
        manifest = json.loads(pointer.read_bytes())
        manifest.pop("source_inputs_format")
        manifest["source_inputs"] = [[str(original / name), sha]
                                     for name, sha in manifest["source_inputs"]]
        _vault, config = resolve_vault(str(original))
        sources = source_inventory(config).sources
        manifest["input_fingerprint"] = digest({
            "config": asdict(config), "inputs": manifest["source_inputs"],
            "sources": {name: asdict(source) for name, source in sources.items()},
        })
        pointer.write_text(json.dumps(manifest))
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    if remove_original:
        shutil.rmtree(original)
    else:
        (original / ".env").write_text("GROUNDING_REVIEW=off\n")
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import pytest\n"
        "from tests.new.test_identity_migration_schema7 import _guard\n"
        "guard=pytest.MonkeyPatch()\nattempts=_guard(guard)\n"
        "import atexit,os\natexit.register(lambda: os._exit(97) if attempts else None)\n"
    )
    env = {key: os.environ[key] for key in (
        "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR", "PATH"
    )}
    repo = Path(__file__).resolve().parents[2]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(guard) + os.pathsep + str(repo))
    evidence = tmp_path / "process-receipts"
    evidence.mkdir()

    def run(label, command, expected, preserve=True):
        before = {p: human_state(p) for p in copy.rglob("*") if p.is_file()}
        proc = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True,
                              text=True, timeout=60)
        (evidence / (label + ".json")).write_text(json.dumps({
            "command": command, "exit_code": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr,
        }, indent=2))
        assert proc.returncode == expected, (label, proc.stdout, proc.stderr)
        assert "sitecustomize" not in proc.stderr and "Traceback" not in proc.stderr
        if preserve:
            assert before == {p: human_state(p) for p in copy.rglob("*") if p.is_file()}
        return proc

    olw = str(Path(sys.executable).with_name("olw"))
    report = json.loads(run("preview", [olw, "identity-migration", str(copy),
                                       "--dry-run", "--json"], 0).stdout)
    assert report["summary"]["bindings"]["resolved"] == 3
    run("current", [olw, "current", str(copy), "--json"], 0)
    run("noop-build", [olw, "build", str(copy)], 0)
    inspect = """import json,sys
from pathlib import Path
from obsidian_llm_wiki.query.graph import build_graph_from_vault
from obsidian_llm_wiki.query.retrieval import retrieve
from obsidian_llm_wiki.query.context import build_context
root=Path(sys.argv[1]); graph=build_graph_from_vault(root/'04-Wiki')
results=retrieve('Packet Scheduler',graph,max_results=5)
context=build_context(graph,results.candidates,'Packet Scheduler')
assert context
print(json.dumps({'pages':sorted(graph.pages),'sections':len(context)}))
"""
    run("query-inspection", [sys.executable, "-c", inspect, str(copy)], 0)
    pointer = copy / "04-Wiki/OLW-Publications/current.json"
    manifest = json.loads(pointer.read_bytes())
    generation = pointer.parent / "generations" / manifest["generation"]
    db = generation / ".llmwiki/registry.db"
    connection = sqlite3.connect(db)
    try:
        connection.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        connection.commit()
    finally:
        connection.close()
    import hashlib

    manifest["files"][".llmwiki/registry.db"] = hashlib.sha256(db.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(manifest))
    run("unknown-query", [olw, "query", str(copy), "--ask", "Packet Scheduler", "--json"], 1)
    report = json.loads(run("unknown-preview", [olw, "identity-migration", str(copy),
                                               "--dry-run", "--json"], 3).stdout)
    assert not report["publication"]["verified"]
