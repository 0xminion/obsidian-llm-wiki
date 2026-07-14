"""Configuration-loading invariants across base and vault .env files."""

from __future__ import annotations

import os


def test_load_config_is_idempotent_and_does_not_mutate_process_environment(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import load_config

    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / ".env").write_text(
        f"VAULT_PATH={vault}\nLLM_MODEL=base-model\n", encoding="utf-8"
    )
    (vault / ".env").write_text("LLM_MODEL=vault-model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VAULT_PATH", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    first = load_config(env_file=str(tmp_path / ".env"))
    second = load_config(env_file=str(tmp_path / ".env"))

    assert first.vault == vault
    assert second.vault == vault
    assert first.llm.model == "vault-model"
    assert second.llm.model == "vault-model"
    assert "VAULT_PATH" not in os.environ
    assert "LLM_MODEL" not in os.environ


def test_implicit_config_uses_process_vault_for_vault_specific_settings(tmp_path, monkeypatch):
    from obsidian_llm_wiki.config import load_config

    vault_a = tmp_path / "vault-a"
    vault_b = tmp_path / "vault-b"
    vault_a.mkdir()
    vault_b.mkdir()
    (tmp_path / ".env").write_text(f"VAULT_PATH={vault_b}\n", encoding="utf-8")
    (vault_a / ".env").write_text("LLM_MODEL=from-vault-a\n", encoding="utf-8")
    (vault_b / ".env").write_text("LLM_MODEL=from-vault-b\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VAULT_PATH", str(vault_a))
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = load_config()

    assert config.vault == vault_a
    assert config.llm.model == "from-vault-a"
