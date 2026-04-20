"""Tests for pipeline/vault_setup.py."""

import pytest
from pathlib import Path

from pipeline.vault_setup import (
    VaultState,
    detect_vault,
    setup_vault,
    migrate_vault,
    ensure_vault_ready,
    REQUIRED_DIRS,
    CRITICAL_DIRS,
    _SEED_FILES,
)


class TestVaultState:
    def test_new_vault(self, tmp_path):
        state = VaultState(tmp_path / "nonexistent")
        assert state.state == "new"

    def test_empty_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        state = VaultState(vault)
        assert state.state == "new"

    def test_existing_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in REQUIRED_DIRS:
            (vault / d).mkdir(parents=True, exist_ok=True)
        for f, content in _SEED_FILES.items():
            (vault / f).parent.mkdir(parents=True, exist_ok=True)
            (vault / f).write_text(content)
        state = VaultState(vault)
        assert state.state == "existing"
        assert not state.missing_dirs

    def test_incomplete_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        # Create some dirs but not all
        (vault / "04-Wiki").mkdir()
        (vault / "04-Wiki/sources").mkdir()
        (vault / "01-Raw").mkdir()
        state = VaultState(vault)
        assert state.state == "incomplete"
        assert "06-Config" in state.missing_dirs


class TestSetupVault:
    def test_creates_all_dirs(self, tmp_path):
        vault = tmp_path / "vault"
        actions = setup_vault(vault, repo_root=tmp_path)  # no repo files, but dirs get created
        for d in REQUIRED_DIRS:
            assert (vault / d).is_dir(), f"Missing: {d}"

    def test_creates_seed_files(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        for f in _SEED_FILES:
            assert (vault / f).exists(), f"Missing: {f}"

    def test_idempotent(self, tmp_path):
        vault = tmp_path / "vault"
        actions1 = setup_vault(vault, repo_root=tmp_path)
        actions2 = setup_vault(vault, repo_root=tmp_path)
        # Second run should do nothing (dirs/files already exist)
        assert len(actions2) == 0

    def test_creates_run_sh(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        run_sh = vault / "run.sh"
        assert run_sh.exists()
        assert run_sh.stat().st_mode & 0o111  # executable

    def test_creates_env(self, tmp_path):
        vault = tmp_path / "vault"
        setup_vault(vault, repo_root=tmp_path)
        env = vault / "Meta/Scripts/.env"
        assert env.exists()
        assert "TRANSCRIPT_API_KEY" in env.read_text()


class TestMigrateVault:
    def test_migrate_adds_missing_dirs(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        # Create partial structure
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "01-Raw").mkdir()
        (vault / "06-Config").mkdir()

        state = VaultState(vault)
        assert state.state == "incomplete"

        actions = migrate_vault(vault, state, repo_root=tmp_path)

        # All dirs should exist now
        for d in REQUIRED_DIRS:
            assert (vault / d).is_dir(), f"Missing after migration: {d}"

    def test_migrate_preserves_existing_files(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "06-Config").mkdir()
        # Create a seed file with custom content
        (vault / "06-Config/edges.tsv").write_text("custom\tcontent\n")

        state = VaultState(vault)
        migrate_vault(vault, state, repo_root=tmp_path)

        # Custom content should be preserved (seed files never overwritten)
        assert (vault / "06-Config/edges.tsv").read_text() == "custom\tcontent\n"


class TestEnsureVaultReady:
    def test_new_vault(self, tmp_path):
        vault = tmp_path / "vault"
        result = ensure_vault_ready(vault, repo_root=tmp_path, force=True)
        assert result == "new"
        assert (vault / "06-Config").is_dir()

    def test_existing_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        for d in REQUIRED_DIRS:
            (vault / d).mkdir(parents=True, exist_ok=True)
        for f, content in _SEED_FILES.items():
            (vault / f).parent.mkdir(parents=True, exist_ok=True)
            (vault / f).write_text(content)

        result = ensure_vault_ready(vault, repo_root=tmp_path)
        assert result == "existing"

    def test_incomplete_vault_force(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "04-Wiki/sources").mkdir(parents=True)
        (vault / "01-Raw").mkdir()

        result = ensure_vault_ready(vault, repo_root=tmp_path, force=True)
        assert result == "migrated"
        assert (vault / "06-Config").is_dir()
