"""Resume state must be guarded before operation locks or marker contents are opened."""
import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.core.operations import OperationStore


@pytest.mark.parametrize("name", ["ingest-resume.json", "operations.lock"])
@pytest.mark.parametrize("public_cli", [True, False])
def test_resume_rejects_metadata_symlink_before_open(tmp_path, monkeypatch, name, public_cli):
    vault = tmp_path / "vault"
    metadata = vault / "04-Wiki" / ".llmwiki"
    metadata.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    original = json.dumps({"run_id": "external", "sources": []})
    outside.write_text(original)
    (metadata / name).symlink_to(outside)
    opened = []
    original_open = io.open

    def observed(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() == outside:
            opened.append(str(file))
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", observed)
    if public_cli:
        result = CliRunner().invoke(app, ["ingest", str(vault), "--resume", "--json",
                                         "--skip-synthesis"])
        assert result.exit_code == 1
        events = [json.loads(line) for line in result.stdout.splitlines()]
        terminal = [event for event in events if event.get("type") == "run"
                    and event.get("event") in {"failed", "partial_failure"}]
        assert len(terminal) == 1
        assert terminal[0]["inventory_failures"] == 1
    else:
        with pytest.raises(ValueError, match="Unsafe generated path"):
            OperationStore(metadata).read_resume_marker()
    assert opened == [], "resume followed an outside metadata link before validation"
    monkeypatch.setattr(io, "open", original_open)
    assert outside.read_text() == original
