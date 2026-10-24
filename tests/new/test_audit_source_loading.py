from pathlib import Path

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_incremental


@pytest.mark.parametrize("damage", ["empty", "invalid_utf8", "denied", "missing_directory"])
def test_incremental_loader_never_treats_unreadable_source_as_deletion(
    tmp_path, monkeypatch, damage
):
    directory = tmp_path / "sources"
    directory.mkdir()
    path = directory / "known.md"
    path.write_text("body")
    with VaultRegistry(tmp_path / "index") as registry:
        registry.upsert_synthesis(
            "known.md",
            SourceDoc("Known", "body", source_file="known.md"),
            SourceSynthesis(source_title="Known", source_summary="summary"),
            ArtifactVersions("e", "p", "s", "m"),
        )
        if damage == "empty":
            path.write_text("")
        elif damage == "invalid_utf8":
            path.write_bytes(b"\xff\xfe")
        elif damage == "missing_directory":
            path.unlink()
            directory.rmdir()
        else:
            original = Path.read_text

            def denied(self, *args, **kwargs):
                if self == path:
                    raise PermissionError("fixture denied")
                return original(self, *args, **kwargs)

            monkeypatch.setattr(Path, "read_text", denied)
        with pytest.raises(OSError, match="source|Source"):
            load_sources_incremental(directory, registry)
        assert registry.source_filenames() == {"known.md"}


def test_generated_source_index_shards_are_not_source_documents(tmp_path):
    from obsidian_llm_wiki.ingest.sources import load_sources_from_dir

    directory = tmp_path / "sources"
    directory.mkdir()
    (directory / "index-bab-000.md").write_text("# Generated source index")
    (directory / "article.md").write_text("Real body")
    with VaultRegistry(tmp_path / "index") as registry:
        assert set(load_sources_incremental(directory, registry).sources) == {"article.md"}
    assert set(load_sources_from_dir(directory)) == {"article.md"}


def test_build_unreadable_source_exits_with_diagnostic(tmp_path, monkeypatch):
    import importlib

    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app
    from obsidian_llm_wiki.config import Config

    build_module = importlib.import_module("obsidian_llm_wiki.cli.build")
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    config.sources_dir.mkdir(parents=True)
    (config.sources_dir / "empty.md").write_text("")
    monkeypatch.setattr(build_module, "resolve_vault", lambda _vault: (tmp_path, config))
    result = CliRunner().invoke(app, ["build", str(tmp_path)])
    assert result.exit_code == 1
    assert "empty or unreadable" in result.output


def test_build_refresh_render_does_not_force_synthesis(tmp_path, monkeypatch):
    import importlib

    from typer.testing import CliRunner

    from obsidian_llm_wiki.cli import app
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core import pipeline
    from obsidian_llm_wiki.core.models import CompileResult

    module = importlib.import_module("obsidian_llm_wiki.cli.build")
    config = Config(vault_path=str(tmp_path), embeddings_enabled=False)
    monkeypatch.setattr(module, "resolve_vault", lambda _vault: (tmp_path, config))
    calls = []

    async def run(*args, **kwargs):
        calls.append(kwargs)
        return CompileResult()

    monkeypatch.setattr(pipeline, "run_pipeline", run)
    result = CliRunner().invoke(app, ["build", str(tmp_path), "--refresh-render"])
    assert result.exit_code == 0, result.output
    assert calls[0]["refresh_render"] is True
    assert calls[0]["force"] is False
