"""Discovery must not touch bodies outside the source root."""
import io
from pathlib import Path

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SourceSynthesis
from obsidian_llm_wiki.core.registry import ArtifactVersions, VaultRegistry
from obsidian_llm_wiki.ingest.sources import load_sources_from_dir, load_sources_incremental


@pytest.mark.parametrize("loader", ["incremental", "full"])
@pytest.mark.parametrize("link_kind", ["file", "directory", "directory_md", "chain"])
def test_source_discovery_rejects_escape_before_open(tmp_path, monkeypatch, loader, link_kind):
    root = tmp_path / "sources"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "private.md"
    sentinel.write_text("PRIVATE OUTSIDE SENTINEL", encoding="utf-8")
    (root / "ordinary.md").write_text("Permitted source body", encoding="utf-8")
    if link_kind == "file":
        (root / "borrowed.md").symlink_to(sentinel)
    elif link_kind in {"directory", "directory_md"}:
        name = "borrowed.md" if link_kind == "directory_md" else "borrowed"
        (root / name).symlink_to(outside, target_is_directory=True)
    else:
        (root / "borrowed").symlink_to(outside, target_is_directory=True)
        (root / "borrowed.md").symlink_to(root / "borrowed" / "private.md")

    opened = []
    original_open = io.open

    def audited_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(outside):
            opened.append(str(file))
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", audited_open)
    with (
        VaultRegistry(tmp_path / "registry") as registry,
        pytest.raises((ValueError, OSError), match="[Ss]ource.*(escap|outside)"),
    ):
        if loader == "incremental":
            load_sources_incremental(root, registry)
        else:
            load_sources_from_dir(root)
    assert opened == [], f"outside bodies were opened before rejection: {opened}"


@pytest.mark.parametrize("loader", ["incremental", "full"])
@pytest.mark.parametrize("kind", ["regular", "file_link", "directory_link"])
def test_contained_source_files_keep_logical_identity(tmp_path, loader, kind):
    root = tmp_path / "sources"
    root.mkdir()
    path = root / "source.md"
    if kind == "regular":
        path.write_text("Contained source body", encoding="utf-8")
    else:
        nested = root / "nested"
        nested.mkdir()
        target = nested / "body.txt"
        target.write_text("Contained source body", encoding="utf-8")
        if kind == "directory_link":
            (root / "nested-link").symlink_to(nested, target_is_directory=True)
            target = root / "nested-link" / "body.txt"
        path.symlink_to(target)
    with VaultRegistry(tmp_path / "registry") as registry:
        sources = (
            load_sources_incremental(root, registry).sources
            if loader == "incremental" else load_sources_from_dir(root)
        )
        assert list(sources) == ["source.md"]
        assert sources["source.md"].content == "Contained source body"
        assert sources["source.md"].title == "source"
        assert sources["source.md"].source_file == "source.md"


@pytest.mark.parametrize("force", [False, True])
def test_unchanged_signature_cannot_bypass_source_containment(tmp_path, monkeypatch, force):
    root = tmp_path / "sources"
    root.mkdir()
    sentinel = tmp_path / "outside.md"
    sentinel.write_text("PRIVATE OUTSIDE SENTINEL", encoding="utf-8")
    (root / "known.md").symlink_to(sentinel)
    with VaultRegistry(tmp_path / "registry") as registry:
        registry.upsert_synthesis(
            "known.md", SourceDoc("Known", "old body"),
            SourceSynthesis("Known", "old summary"), ArtifactVersions("e", "p", "s", "m"),
        )
        signature = sentinel.stat()
        registry.record_source_file_state(
            "known.md", mtime_ns=signature.st_mtime_ns, size_bytes=signature.st_size,
        )
        before = list(registry.connection.iterdump())

        def forbidden_read(*args, **kwargs):
            pytest.fail("source body reader was called before containment rejection")

        monkeypatch.setattr("obsidian_llm_wiki.ingest.sources.safe_read_file", forbidden_read)
        with pytest.raises(ValueError, match="Source path escapes"):
            load_sources_incremental(root, registry, force=force)
        assert list(registry.connection.iterdump()) == before


def test_contained_unchanged_link_uses_metadata_stub(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    target = root / "body.txt"
    target.write_text("Current body", encoding="utf-8")
    (root / "known.md").symlink_to(target)
    with VaultRegistry(tmp_path / "registry") as registry:
        registry.upsert_synthesis(
            "known.md", SourceDoc("Known", "Current body"),
            SourceSynthesis("Known", "Current summary"), ArtifactVersions("e", "p", "s", "m"),
        )
        signature = target.stat()
        registry.record_source_file_state(
            "known.md", mtime_ns=signature.st_mtime_ns, size_bytes=signature.st_size,
        )

        def forbidden_read(*args, **kwargs):
            pytest.fail("unchanged contained source body was needlessly read")

        monkeypatch.setattr("obsidian_llm_wiki.ingest.sources.safe_read_file", forbidden_read)
        loaded = load_sources_incremental(root, registry)
        assert loaded.unchanged_filenames == {"known.md"}
        assert loaded.sources["known.md"].title == "Known"
