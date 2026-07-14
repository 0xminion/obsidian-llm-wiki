"""Regression tests for source filename containment across vault boundaries."""

from __future__ import annotations

import asyncio

import pytest

from obsidian_llm_wiki.core.models import SourceDoc, SynthesisBundle
from obsidian_llm_wiki.core.source_files import source_file_path, validate_source_filename


@pytest.mark.parametrize(
    "filename", ["../secret.md", "nested/source.md", "..\\secret.md", "note.txt", ".hidden.md"],
)
def test_source_filename_policy_rejects_non_source_basenames(filename: str):
    with pytest.raises(ValueError):
        validate_source_filename(filename)


def test_source_file_path_stays_under_requested_directory(tmp_path):
    path = source_file_path(tmp_path, "article.md")
    assert path == tmp_path / "article.md"
    with pytest.raises(ValueError):
        source_file_path(tmp_path, "../../escaped.md")


def test_synthesis_cache_path_rejects_source_traversal(tmp_path):
    from obsidian_llm_wiki.core.cache import synthesis_cache_path

    with pytest.raises(ValueError):
        synthesis_cache_path(tmp_path, "../../escaped.md")


def test_renderer_rejects_source_key_escape_before_writing(tmp_path):
    from obsidian_llm_wiki.render.obsidian import render_vault

    escaped = tmp_path.parent / "escaped.md"
    with pytest.raises(ValueError):
        render_vault(
            tmp_path,
            SynthesisBundle(),
            {"../../escaped.md": SourceDoc(title="T", content="body")},
        )
    assert not escaped.exists()


def test_recompile_rejects_source_traversal_before_reading(tmp_path):
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.core.pipeline import recompile_single_source

    result = asyncio.run(
        recompile_single_source(tmp_path, "../../outside.md", Config(vault_path=str(tmp_path)))
    )

    assert result.compiled == 0
    assert result.errors and result.errors[0].startswith("Invalid source filename:")
