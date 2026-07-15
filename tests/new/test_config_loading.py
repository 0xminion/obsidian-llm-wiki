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


def test_extraction_environment_scopes_vault_credentials_and_restores_process_state(
    tmp_path, monkeypatch
):
    from obsidian_llm_wiki.config import extraction_environment, load_config

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".env").write_text(
        "RESIDENTIAL_PROXY_URL=http://vault-proxy:8080\n"
        "ASSEMBLYAI_API_KEY=vault-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RESIDENTIAL_PROXY_URL", "http://process-proxy:8080")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "process-key")
    config = load_config(env_file=str(vault / ".env"), VAULT_PATH=str(vault))

    with extraction_environment(config):
        assert os.environ["VAULT_PATH"] == str(vault.resolve())
        assert os.environ["RESIDENTIAL_PROXY_URL"] == "http://vault-proxy:8080"
        assert os.environ["ASSEMBLYAI_API_KEY"] == "vault-key"

    assert "VAULT_PATH" not in os.environ
    assert os.environ["RESIDENTIAL_PROXY_URL"] == "http://process-proxy:8080"
    assert os.environ["ASSEMBLYAI_API_KEY"] == "process-key"

    with extraction_environment(config, use_residential_proxy=False):
        assert "RESIDENTIAL_PROXY_URL" not in os.environ
        assert os.environ["ASSEMBLYAI_API_KEY"] == "vault-key"


def test_resolve_vault_does_not_leak_a_cli_vault_into_process_environment(tmp_path, monkeypatch):
    from obsidian_llm_wiki.cli._helpers import resolve_vault

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.delenv("VAULT_PATH", raising=False)

    resolved, config = resolve_vault(str(vault))

    assert resolved == vault.resolve()
    assert config.vault == vault.resolve()
    assert "VAULT_PATH" not in os.environ
