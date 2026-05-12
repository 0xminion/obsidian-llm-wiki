"""Configuration management for the llmwiki pipeline.

Loads settings from environment variables and .env files.
Ported from llm-wiki-compiler/src/utils/constants.ts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Pipeline configuration, loaded from environment."""
    # ── Ollama ──────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:31b-cloud"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_ms: int = 1_800_000  # 30 minutes

    # ── Vault ───────────────────────────────────────
    vault_path: str = ""

    # ── Content thresholds ──────────────────────────
    max_source_chars: int = 1_000_000
    min_source_chars: int = 50
    prompt_budget_chars: int = 200_000

    # ── Concurrency ─────────────────────────────────
    compile_concurrency: int = 3

    # ── Language ────────────────────────────────────
    output_language: str = ""

    # ── Provider ────────────────────────────────────
    provider: str = "ollama"

    # ── Quality gates ───────────────────────────────
    concept_min_body_chars: int = 800
    entry_min_body_chars: int = 500
    clipping_min_body_chars: int = 500

    # ── Retry ───────────────────────────────────────
    retry_count: int = 3
    retry_base_ms: int = 1_000
    retry_multiplier: int = 4

    # ── Derived paths (set after load) ──────────────
    _vault: Path | None = field(default=None, repr=False)

    @property
    def vault(self) -> Path:
        """Resolved vault path."""
        if self._vault is None:
            self._vault = Path(os.path.expandvars(self.vault_path)).expanduser().resolve()
        return self._vault

    @property
    def wiki_dir(self) -> Path:
        return self.vault / "04-Wiki"

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
        return self.wiki_dir / ".llmwiki"

    @property
    def state_file(self) -> Path:
        return self.llmwiki_dir / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.llmwiki_dir / "lock"

    @property
    def index_file(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def moc_file(self) -> Path:
        return self.wiki_dir / "MOC.md"

    @property
    def candidates_dir(self) -> Path:
        return self.llmwiki_dir / "candidates"

    @property
    def candidates_archive_dir(self) -> Path:
        return self.candidates_dir / "archive"


def load_config(env_file: str | None = None, **overrides: str) -> Config:
    """Load configuration from environment and optional .env file.

    Args:
        env_file: Path to a .env file to load from.
        **overrides: Additional key-value overrides (e.g., from CLI).

    Returns:
        Config object with all fields populated.
    """
    # Load .env if specified, fall back to VAULT_PATH/.env then .env
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

    # Apply CLI overrides
    for key, val in overrides.items():
        os.environ[key.upper()] = val

    return Config(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        ollama_timeout_ms=_int_env("OLLAMA_TIMEOUT_MS", 1_800_000),
        vault_path=os.getenv("VAULT_PATH", str(Path.home() / "MyVault")),
        max_source_chars=_int_env("MAX_SOURCE_CHARS", 1_000_000),
        min_source_chars=_int_env("MIN_SOURCE_CHARS", 50),
        prompt_budget_chars=_int_env("PROMPT_BUDGET_CHARS", 200_000),
        compile_concurrency=_int_env("COMPILE_CONCURRENCY", 3),
        output_language=os.getenv("LLMWIKI_OUTPUT_LANGUAGE", ""),
        provider=os.getenv("LLMWIKI_PROVIDER", "ollama"),
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
