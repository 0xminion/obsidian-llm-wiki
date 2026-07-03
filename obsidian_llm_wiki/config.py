"""Configuration management — loads from environment and .env files.

Clean port of the legacy ``pipeline.config`` module.  Env vars use the
``LLM_*`` prefix (no more ``OLLAMA_*``/``LLMWIKI_*`` dual naming).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class LLMProviderConfig:
    """LLM provider configuration.

    Attributes:
        provider: 'ollama' or 'openai' (OpenAI-compatible).
        host: Base URL for the LLM API.
        model: Model name for chat completions.
        api_key: API key (None for local Ollama).
        timeout_ms: Request timeout in milliseconds.
    """

    provider: str = "ollama"
    host: str = "http://localhost:11434"
    model: str = "gemma3:27b"
    api_key: str | None = None
    timeout_ms: int = 1_800_000  # 30 minutes


@dataclass
class Config:
    """Pipeline configuration, loaded from environment."""

    # ── LLM ───────────────────────────────────────
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)

    # ── Vault ─────────────────────────────────────
    vault_path: str = ""

    # ── Content thresholds ──────────────────────────
    max_source_chars: int = 1_000_000
    min_source_chars: int = 50

    # ── Concurrency ─────────────────────────────────
    compile_concurrency: int = 3

    # ── Language ────────────────────────────────────
    output_language: str = ""

    # ── Quality gates ───────────────────────────────
    concept_min_body_chars: int = 800
    entry_min_body_chars: int = 500
    clipping_min_body_chars: int = 500

    # ── Retry ───────────────────────────────────────
    retry_count: int = 3
    retry_base_ms: int = 1_000
    retry_multiplier: int = 4

    # ── Derived paths (lazy) ──────────────────────
    _vault: Path | None = field(default=None, repr=False)

    # ── Path properties ──────────────────────────────

    @property
    def vault(self) -> Path:
        """Resolved vault path."""
        if self._vault is None:
            self._vault = Path(os.path.expandvars(self.vault_path)).expanduser().resolve()
        return self._vault

    @property
    def wiki_dir(self) -> Path:
        """The OKF/Obsidian bundle directory."""
        return self.vault / "04-Wiki"

    @property
    def bundle_dir(self) -> Path:
        """Alias for wiki_dir."""
        return self.wiki_dir

    @property
    def sources_dir(self) -> Path:
        return self.wiki_dir / "sources"

    @property
    def entries_dir(self) -> Path:
        return self.wiki_dir / "entries"

    @property
    def concepts_dir(self) -> Path:
        return self.wiki_dir / "concepts"

    @property
    def mocs_dir(self) -> Path:
        return self.wiki_dir / "mocs"

    @property
    def clippings_dir(self) -> Path:
        return self.vault / "02-Clippings"

    @property
    def llmwiki_dir(self) -> Path:
        """Internal pipeline state directory."""
        return self.wiki_dir / ".llmwiki"

    @property
    def state_file(self) -> Path:
        return self.llmwiki_dir / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.llmwiki_dir / "lock"

    @property
    def candidates_dir(self) -> Path:
        return self.llmwiki_dir / "candidates"


def load_config(env_file: str | None = None, **overrides: str) -> Config:
    """Load configuration from environment and optional .env file."""
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        vault_raw = os.getenv("VAULT_PATH", "")
        vault_dir = os.path.expandvars(vault_raw)
        if vault_dir:
            vault_env = os.path.join(vault_dir, ".env")
            if os.path.isfile(vault_env):
                load_dotenv(vault_env, override=True)
        load_dotenv(override=False)

    for key, val in overrides.items():
        os.environ[key.upper()] = val

    llm_config = LLMProviderConfig(
        provider=os.getenv("LLM_PROVIDER", "ollama"),
        host=os.getenv("LLM_HOST", "http://localhost:11434"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"),
        api_key=os.getenv("LLM_API_KEY"),
        timeout_ms=_int_env("LLM_TIMEOUT_MS", 1_800_000),
    )

    return Config(
        llm=llm_config,
        vault_path=os.getenv("VAULT_PATH", str(Path.home() / "MyVault")),
        max_source_chars=_int_env("MAX_SOURCE_CHARS", 1_000_000),
        min_source_chars=_int_env("MIN_SOURCE_CHARS", 50),
        compile_concurrency=_int_env("COMPILE_CONCURRENCY", 3),
        output_language=os.getenv("OUTPUT_LANGUAGE", ""),
        concept_min_body_chars=_int_env("CONCEPT_MIN_BODY_CHARS", 800),
        entry_min_body_chars=_int_env("ENTRY_MIN_BODY_CHARS", 500),
        clipping_min_body_chars=_int_env("CLIPPING_MIN_BODY_CHARS", 500),
        retry_count=_int_env("RETRY_COUNT", 3),
        retry_base_ms=_int_env("RETRY_BASE_MS", 1_000),
        retry_multiplier=_int_env("RETRY_MULTIPLIER", 4),
    )


def _int_env(key: str, default: int) -> int:
    """Parse an integer from environment with fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default
