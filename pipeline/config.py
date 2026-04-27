"""Configuration and environment management.

Loads settings from:
  1. Environment variables
  2. .env file (if exists)
  3. Defaults
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None


def hashlib_md5_short(s: str) -> str:
    """Portable short MD5 hash (12 chars, consistent across modules)."""
    import hashlib
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()[:12]


def _find_env_file(vault_path: Optional[Path] = None) -> Optional[Path]:
    """Search for .env in common locations.

    When a vault_path is provided, prefer that vault's local .env first so
    non-default vaults load their own credentials/configuration.
    """
    candidates: list[Path] = []
    if vault_path is not None:
        candidates.append(Path(vault_path) / "Meta" / "Scripts" / ".env")
    candidates.extend([
        Path.cwd() / ".env",
        Path.home() / "MyVault" / "Meta" / "Scripts" / ".env",
        Path(__file__).parent.parent / ".env",
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


@dataclass
class Config:
    """Pipeline configuration."""

    # Paths
    vault_path: Path = field(default_factory=lambda: Path.home() / "MyVault")
    extract_dir: Optional[Path] = None  # auto-derived if None

    # Agent
    agent_cmd: str = "hermes"
    max_retries: int = 3

    # Parallelism
    parallel: int = 3

    # API keys
    transcript_api_key: str = ""
    supadata_api_key: str = ""
    assemblyai_api_key: str = ""

    # QMD
    qmd_cmd: str = "qmd"
    qmd_collection: str = "concepts"

    # LLM Provider (ollama | openrouter | hermes)
    llm_provider: str = "ollama"
    llm_model: str = ""  # e.g. "minimax-m2.7:cloud" or "qwen/qwen3-30b-a3b:free"
    llm_api_key: str = ""  # required for openrouter
    llm_base_url: str = ""  # override base URL for any provider
    llm_timeout: int = 60

    # Embedding (primarily Ollama; kept separate from generation provider)
    embed_model: str = "qwen3-embedding:0.6b"
    embed_base_url: str = ""  # falls back to ollama_host if empty

    # Legacy Ollama settings (backward compat — used when llm_provider=ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_insight_model: str = "minimax-m2.7:cloud"
    ollama_filename_model: str = "minimax-m2.7:cloud"

    # Structured output
    llm_structured_timeout: int = 90

    # Extraction
    extract_timeout: int = 45

    # Timeouts
    agent_timeout: int = 900
    plan_timeout: int = 600

    # Content limits (configurable for token optimization)
    max_content_per_source: int = 8000  # max chars per source in batch prompt
    max_total_content: int = 15000  # max total content chars in batch prompt
    max_content_insights: int = 6000  # max chars for insight agent

    # Quality thresholds
    min_quality: float = 0.0
    min_clipping_quality: float = 0.5

    # Whisper
    whisper_language: str = ""  # empty = auto-detect, "en", "zh", etc.

    # Staleness thresholds (days)
    default_staleness_days: int = 3 * 365  # default = 3 years
    high_volatility_tags: list[str] = field(default_factory=
        lambda: ["crypto", "ai", "blockchain"])
    medium_volatility_tags: list[str] = field(default_factory=
        lambda: ["tech", "technology", "science"])
    low_volatility_tags: list[str] = field(default_factory=
        lambda: ["history", "philosophy"])
    high_staleness_days: int = 365
    medium_staleness_days: int = 730
    low_staleness_days: int = 5 * 365

    @property
    def staleness_thresholds(self) -> dict[str, int]:
        """Return {volatility_tag: days} mapping for staleness scoring."""
        thresholds: dict[str, int] = {}
        for tag in self.high_volatility_tags:
            thresholds[tag] = self.high_staleness_days
        for tag in self.medium_volatility_tags:
            thresholds[tag] = self.medium_staleness_days
        for tag in self.low_volatility_tags:
            thresholds[tag] = self.low_staleness_days
        return thresholds

    @property
    def resolved_extract_dir(self) -> Path:
        if self.extract_dir:
            return self.extract_dir
        vault_hash = hashlib_md5_short(str(self.vault_path))
        return Path(f"/tmp/obsidian-extracted-{vault_hash}")

    @property
    def prompts_dir(self) -> Path:
        return self.vault_path / "Meta" / "prompts"

    @property
    def templates_dir(self) -> Path:
        return self.vault_path / "Meta" / "Templates"

    @property
    def lib_dir(self) -> Path:
        return self.vault_path / "Meta" / "lib"

    @property
    def scripts_dir(self) -> Path:
        return self.vault_path / "Meta" / "Scripts"

    @property
    def log_file(self) -> Path:
        return self.scripts_dir / "processing.log"

    @property
    def sources_dir(self) -> Path:
        return self.vault_path / "04-Wiki" / "sources"

    @property
    def entries_dir(self) -> Path:
        return self.vault_path / "04-Wiki" / "entries"

    @property
    def concepts_dir(self) -> Path:
        return self.vault_path / "04-Wiki" / "concepts"

    @property
    def mocs_dir(self) -> Path:
        return self.vault_path / "04-Wiki" / "mocs"

    @property
    def inbox_dir(self) -> Path:
        return self.vault_path / "01-Raw"

    @property
    def clippings_dir(self) -> Path:
        return self.vault_path / "02-Clippings"

    @property
    def clippings_archive_dir(self) -> Path:
        return self.vault_path / "10-Archive-Clippings"

    @property
    def archive_dir(self) -> Path:
        return self.vault_path / "08-Archive-Raw"

    @property
    def config_dir(self) -> Path:
        return self.vault_path / "06-Config"

    @property
    def edges_file(self) -> Path:
        return self.config_dir / "edges.tsv"

    @property
    def wiki_index(self) -> Path:
        return self.config_dir / "wiki-index.md"

    @property
    def url_index(self) -> Path:
        return self.config_dir / "url-index.tsv"

    @property
    def log_md(self) -> Path:
        return self.config_dir / "log.md"

    @property
    def telemetry_file(self) -> Path:
        return self.config_dir / "telemetry.jsonl"

    def validate(self) -> list[str]:
        """Check for missing required paths and invalid config. Returns list of errors."""
        errors = []
        if not self.vault_path.exists():
            errors.append(f"Vault path does not exist: {self.vault_path}")
        if not self.sources_dir.parent.exists():
            errors.append(f"04-Wiki directory missing: {self.sources_dir.parent}")
        # Numeric bounds
        if self.parallel < 1:
            errors.append(f"parallel must be >= 1, got {self.parallel}")
        if self.parallel > 20:
            errors.append(f"parallel > 20 is likely a mistake, got {self.parallel}")
        if self.max_retries < 1:
            errors.append(f"max_retries must be >= 1, got {self.max_retries}")
        if self.extract_timeout < 5:
            errors.append(f"extract_timeout too low (<5s), got {self.extract_timeout}")
        if self.agent_timeout < 30:
            errors.append(f"agent_timeout too low (<30s), got {self.agent_timeout}")
        # Provider validation
        if self.llm_provider not in {"ollama", "openrouter", "hermes"}:
            errors.append(
                f"llm_provider must be ollama/openrouter/hermes, got {self.llm_provider}"
            )
        # Quality bounds
        if not (0.0 <= self.min_quality <= 1.0):
            errors.append(f"min_quality must be in [0, 1], got {self.min_quality}")
        if not (0.0 <= self.min_clipping_quality <= 1.0):
            errors.append(
                f"min_clipping_quality must be in [0, 1], got {self.min_clipping_quality}"
            )
        return errors




def _int_env(key: str, default: int, env_values: Optional[dict[str, str]] = None) -> int:
    """Safely parse an integer from environment variables or .env values."""
    raw = os.environ.get(key)
    if raw is None and env_values is not None:
        raw = env_values.get(key)
    if raw is None:
        raw = str(default)
    try:
        return int(raw)
    except (ValueError, TypeError):
        log.warning("Invalid integer for %s, using default %d", key, default)
        return default


def load_config(
    vault_path: Optional[Path] = None,
    env_file: Optional[Path] = None,
) -> Config:
    """Load configuration from environment + .env file.

    Environment variables override .env values. Reading .env must not mutate the
    process environment, otherwise one test or one vault can leak configuration
    into later runs.
    """
    env_values: dict[str, str] = {}
    if dotenv_values is not None:
        env_path = env_file or _find_env_file(vault_path=vault_path)
        if env_path:
            env_values = {
                key: value
                for key, value in dotenv_values(env_path).items()
                if value is not None
            }

    def _env(key: str, default: str = "") -> str:
        return os.environ.get(key, env_values.get(key, default))

    cfg = Config(
        vault_path=Path(
            vault_path
            or os.environ.get("VAULT_PATH")
            or os.environ.get("OBSIDIAN_VAULT")
            or env_values.get("VAULT_PATH")
            or env_values.get("OBSIDIAN_VAULT")
            or str(Path.home() / "MyVault")
        ),
        agent_cmd=_env("AGENT_CMD", "hermes"),
        max_retries=_int_env("MAX_RETRIES", 3, env_values),
        parallel=_int_env("PARALLEL", 3, env_values),
        transcript_api_key=_env("TRANSCRIPT_API_KEY", ""),
        supadata_api_key=_env("SUPADATA_API_KEY", ""),
        assemblyai_api_key=_env("ASSEMBLYAI_API_KEY", ""),
        qmd_cmd=_env("QMD_CMD", "qmd"),
        qmd_collection=_env("QMD_COLLECTION", "concepts"),
        llm_provider=_env("LLM_PROVIDER", "ollama"),
        llm_model=_env("LLM_MODEL", ""),
        llm_api_key=_env("LLM_API_KEY", ""),
        llm_base_url=_env("LLM_BASE_URL", ""),
        llm_timeout=_int_env("LLM_TIMEOUT", 60, env_values),
        embed_model=_env("EMBED_MODEL", "qwen3-embedding:0.6b"),
        embed_base_url=_env("EMBED_BASE_URL", ""),
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
        ollama_insight_model=_env("OLLAMA_INSIGHT_MODEL", "minimax-m2.7:cloud"),
        ollama_filename_model=_env("OLLAMA_FILENAME_MODEL", "minimax-m2.7:cloud"),
        llm_structured_timeout=_int_env("LLM_STRUCTURED_TIMEOUT", 90, env_values),
        extract_timeout=_int_env("EXTRACT_TIMEOUT", 45, env_values),
        agent_timeout=_int_env("AGENT_TIMEOUT", 900, env_values),
        plan_timeout=_int_env("PLAN_TIMEOUT", 600, env_values),
        max_content_per_source=_int_env("MAX_CONTENT_PER_SOURCE", 8000, env_values),
        max_total_content=_int_env("MAX_TOTAL_CONTENT", 15000, env_values),
        max_content_insights=_int_env("MAX_CONTENT_INSIGHTS", 6000, env_values),
        whisper_language=_env("WHISPER_LANGUAGE", ""),
    )

    # Override extract dir if PIPELINE_TMPDIR is set in environment or .env.
    pipeline_tmpdir = _env("PIPELINE_TMPDIR", "")
    if pipeline_tmpdir:
        cfg.extract_dir = Path(pipeline_tmpdir)

    return cfg
