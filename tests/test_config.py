"""Tests for pipeline.config — LLMProviderConfig, OKF bundle paths, load_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import LLMProviderConfig, load_config


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all llmwiki-related env vars so tests start from a clean slate."""
    keys = [
        "LLM_PROVIDER", "LLM_HOST", "LLM_MODEL", "LLM_API_KEY",
        "LLM_TIMEOUT_MS", "LLM_EMBED_MODEL",
        "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_EMBED_MODEL", "OLLAMA_TIMEOUT_MS",
        "LLMWIKI_PROVIDER", "LLMWIKI_OUTPUT_LANGUAGE",
        "VAULT_PATH", "OKF_VERSION",
        "MAX_SOURCE_CHARS", "MIN_SOURCE_CHARS", "PROMPT_BUDGET_CHARS",
        "COMPILE_CONCURRENCY", "CONCEPT_MIN_BODY_CHARS", "ENTRY_MIN_BODY_CHARS",
        "CLIPPING_MIN_BODY_CHARS", "RETRY_COUNT", "RETRY_BASE_MS", "RETRY_MULTIPLIER",
    ]
    for k in keys:
        monkeypatch.delenv(k, raising=False)


class TestLoadConfigOllama:
    """load_config with LLM_PROVIDER=ollama."""

    def test_provider_is_ollama(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(
            env_file=None,
            LLM_PROVIDER="ollama",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.llm.provider == "ollama"

    def test_ollama_host_default(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.llm.host == "http://localhost:11434"

    def test_backward_compat_ollama_model(self, _clean_env: None, tmp_path: Path) -> None:
        """config.ollama_model should delegate to config.llm.model."""
        cfg = load_config(
            env_file=None,
            LLM_MODEL="gpt-oss:120b",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.ollama_model == "gpt-oss:120b"
        assert cfg.llm.model == "gpt-oss:120b"

    def test_backward_compat_ollama_host(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(
            env_file=None,
            LLM_HOST="http://myhost:8080",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.ollama_host == "http://myhost:8080"

    def test_backward_compat_ollama_timeout_ms(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(
            env_file=None,
            LLM_TIMEOUT_MS="60000",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.ollama_timeout_ms == 60_000
        assert cfg.llm.timeout_ms == 60_000

    def test_api_key_none_by_default(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.llm.api_key is None

    def test_api_key_loaded_from_env(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(
            env_file=None,
            LLM_API_KEY="sk-test-123",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.llm.api_key == "sk-test-123"

    def test_okf_version_default(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.okf_version == "0.1"

    def test_okf_version_from_env(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(
            env_file=None,
            OKF_VERSION="0.2",
            VAULT_PATH=str(tmp_path),
        )
        assert cfg.okf_version == "0.2"


class TestBundleDir:
    """bundle_dir property (vault/04-Wiki)."""

    def test_bundle_dir_is_wiki_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.bundle_dir == cfg.vault / "04-Wiki"
        assert cfg.bundle_dir == cfg.wiki_dir

    def test_bundle_dir_name(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.bundle_dir.name == "04-Wiki"


class TestReferencesDir:
    """references_dir property (bundle_dir/references)."""

    def test_references_dir_path(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.references_dir == cfg.bundle_dir / "references"

    def test_references_dir_name(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.references_dir.name == "references"

    def test_references_dir_under_bundle(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.references_dir.parent == cfg.bundle_dir


class TestLogFile:
    """log_file property (bundle_dir/log.md)."""

    def test_log_file_path(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.log_file == cfg.bundle_dir / "log.md"
        assert cfg.log_file.name == "log.md"


class TestExistingPropertiesPreserved:
    """Ensure existing path properties still work."""

    def test_sources_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.sources_dir == cfg.wiki_dir / "sources"

    def test_entries_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.entries_dir == cfg.wiki_dir / "entries"

    def test_concepts_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.concepts_dir == cfg.wiki_dir / "concepts"

    def test_mocs_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.mocs_dir == cfg.wiki_dir / "mocs"

    def test_clippings_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.clippings_dir == cfg.vault / "02-Clippings"

    def test_state_file(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.state_file == cfg.llmwiki_dir / "state.json"

    def test_lock_file(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.lock_file == cfg.llmwiki_dir / "lock"

    def test_candidates_dir(self, _clean_env: None, tmp_path: Path) -> None:
        cfg = load_config(env_file=None, VAULT_PATH=str(tmp_path))
        assert cfg.candidates_dir == cfg.llmwiki_dir / "candidates"


class TestLLMProviderConfigDataclass:
    """Direct dataclass tests."""

    def test_defaults(self) -> None:
        c = LLMProviderConfig()
        assert c.provider == "ollama"
        assert c.host == "http://localhost:11434"
        assert c.model == "gemma4:31b-cloud"
        assert c.api_key is None
        assert c.timeout_ms == 1_800_000
        assert c.embed_model == "qwen3-embedding:0.6b"

    def test_custom_values(self) -> None:
        c = LLMProviderConfig(
            provider="openai",
            host="https://api.openai.com",
            model="gpt-4o",
            api_key="sk-xxx",
            timeout_ms=30_000,
            embed_model="text-embedding-3-small",
        )
        assert c.provider == "openai"
        assert c.host == "https://api.openai.com"
        assert c.model == "gpt-4o"
        assert c.api_key == "sk-xxx"
        assert c.timeout_ms == 30_000
        assert c.embed_model == "text-embedding-3-small"
