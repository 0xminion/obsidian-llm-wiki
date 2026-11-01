"""Fresh interpreter CLI smoke: only explicitly local scripted provider doubles."""

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.new.test_immutable_callers import snapshot


def test_fresh_process_application_roundtrip(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    vault = tmp_path / "vault"
    vault.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONPATH": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    for key in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        path = tmp_path / key
        path.mkdir()
        env[key] = str(path)
    bootstrap = """import pathlib,sys,socket
import obsidian_llm_wiki
expected = pathlib.Path(sys.argv.pop(1))
assert pathlib.Path(obsidian_llm_wiki.__file__).resolve().is_relative_to(expected)
def offline(*a, **kw):
    raise AssertionError('network forbidden')
socket.socket.connect = offline
from unittest.mock import AsyncMock
from tests.new.grounding_doubles import legacy_fixture_llm
from tests.new.test_incremental_cache import FAKE_SYNTH_A
import obsidian_llm_wiki.providers.llm as llm
llm.acall_llm = AsyncMock(side_effect=legacy_fixture_llm(FAKE_SYNTH_A))
import obsidian_llm_wiki.cli.ingest as ingest
from obsidian_llm_wiki.core.models import SourceDoc
ingest._extract_direct = lambda url: SourceDoc(
    'Local source', 'The local evidence supports this precise finding. ' * 30, url=url)
import obsidian_llm_wiki.cli.query as query
query.call_llm = lambda *a, **kw: 'The available evidence supports the finding.'
from obsidian_llm_wiki.cli import app
app()
"""
    records = []

    def run(*args):
        process = subprocess.run(
            [sys.executable, "-c", bootstrap, str(repo), *args],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        records.append(
            {
                "args": args,
                "rc": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        (tmp_path / "commands.json").write_text(json.dumps(records, indent=2))
        assert process.returncode == 0, records[-1]
        return process.stdout

    run("ingest", str(vault), "--url", "https://example.test/local", "--json")
    current = json.loads(run("current", str(vault), "--json"))
    generation = (vault / current["entrypoint"]).parent
    before = snapshot(generation)
    human = snapshot(vault / "04-Wiki/sources")
    run("ingest", str(vault), "--resume", "--json")
    run("ingest", str(vault), "--url", "https://example.test/local", "--json")
    run("build", str(vault), "--dry-run")
    run("recompile", str(vault), "local-source.md")
    run("fix", str(vault), "--apply", "--json")
    run("identity-migration", str(vault), "--dry-run", "--json")
    run("health", str(vault), "--json")
    run("validate", str(vault))
    run("metrics", str(vault))
    run("query", str(vault), "--ask", "Concept A1", "--json", "--save-answer", "answer.md")
    assert (vault / "05-Queries/answer.md").is_file()
    assert not (vault / "04-Wiki/.llmwiki/registry.db").exists()
    assert snapshot(generation) == before
    assert snapshot(vault / "04-Wiki/sources") == human
