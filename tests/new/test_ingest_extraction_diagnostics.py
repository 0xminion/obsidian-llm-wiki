"""Extraction caveats must survive the public CLI output boundary."""
import importlib
import json

import pytest
from typer.testing import CliRunner

from obsidian_llm_wiki.cli import app
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.core.models import SourceDoc, SourceProvenance

WARNING = "libreoffice-docx: 1 embedded OLE object(s) not extracted; content not verified"


@pytest.mark.parametrize("mode", ["--preview", "--skip-synthesis"])
@pytest.mark.parametrize("json_output", [True, False])
@pytest.mark.parametrize("duplicate", [False, True])
def test_extraction_diagnostics_visible(tmp_path, monkeypatch, mode, json_output, duplicate):
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    ingest = importlib.import_module("obsidian_llm_wiki.cli.ingest")
    monkeypatch.setattr(ingest, "resolve_vault", lambda _: (tmp_path, config))
    monkeypatch.setattr(
        "obsidian_llm_wiki.ingest.extractors.extract",
        lambda url: SourceDoc(
            "Word document", "The document contains readable body text. " * 40,
            url=url, provenance=SourceProvenance(diagnostics=(WARNING,)),
        ),
    )
    if duplicate and mode == "--skip-synthesis":
        seeded = CliRunner().invoke(
            app, ["ingest", str(tmp_path), "--url", "https://example.test/original.doc",
                  "--parallel", "0", "--skip-synthesis"],
        )
        assert seeded.exit_code == 0, seeded.output
    args = ["ingest", str(tmp_path), "--url", "https://example.test/sample.doc",
            "--parallel", "0", mode]
    if json_output:
        args.append("--json")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    if json_output:
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        source = next(event for event in events if event.get("type") == "source")
        assert source.get("extraction_diagnostics") == [WARNING]
        assert sum(e.get("type") == "source" for e in events) == 1
    else:
        assert WARNING in result.stdout
    if mode == "--preview":
        assert not list(tmp_path.iterdir())
