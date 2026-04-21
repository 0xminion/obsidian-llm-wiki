"""Tests for pipeline.config."""

import os
import tempfile
from pathlib import Path

import pytest

from pipeline.config import Config, hashlib_md5_short, load_config


class TestConfig:
    def test_default_vault_path(self):
        cfg = Config()
        assert cfg.vault_path == Path.home() / "MyVault"

    def test_resolved_extract_dir_default(self):
        cfg = Config(vault_path=Path("/tmp/test-vault"))
        expected_hash = hashlib_md5_short("/tmp/test-vault")
        assert cfg.resolved_extract_dir == Path(f"/tmp/obsidian-extracted-{expected_hash}")

    def test_resolved_extract_dir_override(self):
        cfg = Config(vault_path=Path("/v"), extract_dir=Path("/custom/dir"))
        assert cfg.resolved_extract_dir == Path("/custom/dir")

    def test_property_paths(self):
        cfg = Config(vault_path=Path("/vault"))
        assert cfg.sources_dir == Path("/vault/04-Wiki/sources")
        assert cfg.entries_dir == Path("/vault/04-Wiki/entries")
        assert cfg.concepts_dir == Path("/vault/04-Wiki/concepts")
        assert cfg.mocs_dir == Path("/vault/04-Wiki/mocs")
        assert cfg.inbox_dir == Path("/vault/01-Raw")
        assert cfg.edges_file == Path("/vault/06-Config/edges.tsv")

    def test_validate_missing_vault(self, tmp_path):
        cfg = Config(vault_path=tmp_path / "nonexistent")
        errors = cfg.validate()
        assert len(errors) > 0
        assert "does not exist" in errors[0]


class TestLoadConfig:
    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", "/tmp/env-vault")
        monkeypatch.setenv("PARALLEL", "5")
        monkeypatch.setenv("AGENT_CMD", "claude")
        cfg = load_config()
        assert cfg.vault_path == Path("/tmp/env-vault")
        assert cfg.parallel == 5
        assert cfg.agent_cmd == "claude"

    def test_load_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", "/tmp/env-vault")
        cfg = load_config(vault_path=Path("/explicit"))
        assert cfg.vault_path == Path("/explicit")

    def test_pipeline_tmpdir_override(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_TMPDIR", "/tmp/custom-extract")
        cfg = load_config()
        assert cfg.extract_dir == Path("/tmp/custom-extract")

    def test_load_from_dotenv(self, tmp_path, monkeypatch):
        dotenv = tmp_path / ".env"
        dotenv.write_text("TRANSCRIPT_API_KEY=test-key-abc\nPARALLEL=7\n")
        monkeypatch.delenv("VAULT_PATH", raising=False)
        monkeypatch.delenv("TRANSCRIPT_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL", raising=False)
        cfg = load_config(env_file=dotenv)
        assert cfg.transcript_api_key == "test-key-abc"
        assert cfg.parallel == 7


class TestHashlibMd5Short:
    def test_deterministic(self):
        assert hashlib_md5_short("test") == hashlib_md5_short("test")

    def test_length(self):
        assert len(hashlib_md5_short("anything")) == 12

    def test_matches_shell(self):
        """Must match: echo -n 'test' | md5sum | cut -c1-12"""
        import hashlib
        expected = hashlib.md5(b"test").hexdigest()[:12]
        assert hashlib_md5_short("test") == expected
