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
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def hashlib_md5_short(s: str) -> str:
    """Portable short MD5 hash (matches shell md5sum | cut -c1-8)."""
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _find_env_file() -> Optional[Path]:
    """Search for .env in common locations."""
    candidates = [
        Path.cwd() / ".env",
        Path.home() / "MyVault" / "Meta" / "Scripts" / ".env",
        Path(__file__).parent.parent / ".env",
    ]
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

    # Extraction
    extract_timeout: int = 45

    # Timeouts
    agent_timeout: int = 900
    plan_timeout: int = 600

    # Content limits (configurable for token optimization)
    max_content_per_source: int = 8000  # max chars per source in batch prompt
    max_total_content: int = 15000  # max total content chars in batch prompt
    max_content_insights: int = 6000  # max chars for insight agent

    # Whisper
    whisper_language: str = ""  # empty = auto-detect, "en", "zh", etc.

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

    def validate(self) -> list[str]:
        """Check for missing required paths. Returns list of errors."""
        errors = []
        if not self.vault_path.exists():
            errors.append(f"Vault path does not exist: {self.vault_path}")
        if not self.sources_dir.parent.exists():
            errors.append(f"04-Wiki directory missing: {self.sources_dir.parent}")
        return errors




def _int_env(key: str, default: int) -> int:
    """Safely parse an integer from an environment variable."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        log.warning("Invalid integer for %s, using default %d", key, default)
        return default


def load_config(
    vault_path: Optional[Path] = None,
    env_file: Optional[Path] = None,
) -> Config:
    """Load configuration from environment + .env file."""
    # Load .env if available
    if load_dotenv is not None:
        env_path = env_file or _find_env_file()
        if env_path:
            load_dotenv(env_path)

    cfg = Config(
        vault_path=Path(
            vault_path
            or os.environ.get("VAULT_PATH")
            or os.environ.get("OBSIDIAN_VAULT")
            or str(Path.home() / "MyVault")
        ),
        agent_cmd=os.environ.get("AGENT_CMD", "hermes"),
        max_retries=_int_env("MAX_RETRIES", 3),
        parallel=_int_env("PARALLEL", 3),
        transcript_api_key=os.environ.get("TRANSCRIPT_API_KEY", ""),
        supadata_api_key=os.environ.get("SUPADATA_API_KEY", ""),
        assemblyai_api_key=os.environ.get("ASSEMBLYAI_API_KEY", ""),
        qmd_cmd=os.environ.get("QMD_CMD", "qmd"),
        qmd_collection=os.environ.get("QMD_COLLECTION", "concepts"),
        extract_timeout=_int_env("EXTRACT_TIMEOUT", 45),
        agent_timeout=_int_env("AGENT_TIMEOUT", 900),
        plan_timeout=_int_env("PLAN_TIMEOUT", 600),
        max_content_per_source=_int_env("MAX_CONTENT_PER_SOURCE", 8000),
        max_total_content=_int_env("MAX_TOTAL_CONTENT", 15000),
        max_content_insights=_int_env("MAX_CONTENT_INSIGHTS", 6000),
        whisper_language=os.environ.get("WHISPER_LANGUAGE", ""),
    )

    # Override extract dir if PIPELINE_TMPDIR is set
    if os.environ.get("PIPELINE_TMPDIR"):
        cfg.extract_dir = Path(os.environ["PIPELINE_TMPDIR"])

    return cfg
