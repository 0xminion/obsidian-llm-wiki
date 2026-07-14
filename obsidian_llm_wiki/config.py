"""Configuration management — loads from environment and .env files.

Clean port of the legacy ``pipeline.config`` module.  Env vars use the
``LLM_*`` prefix (no more ``OLLAMA_*``/``LLMWIKI_*`` dual naming).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


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
    # Optional task-specific overrides.  When absent, each task uses ``model``
    # so existing LLM_MODEL-only environments retain their current behavior.
    ingest_model: str | None = None
    maintenance_model: str | None = None
    query_model: str | None = None
    # Pass 2 (expand) model override — for perspective diversity in two-pass
    # synthesis.  When unset, falls back to the default ``model``.
    expand_model: str | None = None
    api_key: str | None = None
    timeout_ms: int = 1_800_000  # 30 minutes
    context_window: int = 256_000  # 256K tokens for cloud models (e.g. gemma4:31b-cloud)


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

    # ── Chunking ─────────────────────────────────────
    # Sources above this size (chars) are split into chunks for Pass 1.
    chunk_size: int = 30_000

    # ── Concurrency ─────────────────────────────────
    compile_concurrency: int = 3

    # ── Language ────────────────────────────────────
    output_language: str = ""

    # ── Synthesis mode ──────────────────────────────
    # "single" = one LLM call per source (default, fast)
    # "two_pass" = extract skeleton + expand each concept (deep, slow)
    synthesis_mode: str = "single"

    # ── Quality gates ───────────────────────────────
    concept_min_body_chars: int = 1200
    entry_min_body_chars: int = 500
    clipping_min_body_chars: int = 500

    # ── Semantic dedup ──────────────────────────────────────
    similarity_dedup_threshold: float = 0.85

    # ── MoC orphan assignment ───────────────────────────────
    moc_assignment_threshold: float = 0.55

    # ── Retry ───────────────────────────────────────────────
    retry_count: int = 3
    retry_base_ms: int = 1_000
    retry_multiplier: int = 4

    # ── Document safety boundaries ────────────────────────────────
    # Limits apply to every direct/discovered binary document download and
    # optional LiteParse subprocess invocation.
    max_document_bytes: int = 50_000_000
    max_document_candidates: int = 10
    parser_timeout_seconds: int = 120
    max_parser_stdout_bytes: int = 1_000_000
    max_parser_stderr_bytes: int = 16_384

    # Generic HTML is read incrementally before article extraction. This cap
    # prevents an untrusted page from being fully buffered in memory.
    max_html_bytes: int = 10_000_000

    # ── Embeddings ───────────────────────────────────────────────────
    # These must travel in Config rather than relying on process-global env,
    # so a vault-local .env controls its own semantic-indexing behavior.
    embeddings_enabled: bool = False
    embedding_model: str = "embeddinggemma:300m"
    embedding_host: str = ""

    # ── Extraction fallbacks ─────────────────────────────────────
    # Residential proxy URL (socks5h:// or http://) for blocked sites.
    # Tailscale exit node: socks5h://<tailscale-ip>:1080
    residential_proxy_url: str = ""

    # Path to Netscape cookies file for YouTube transcript extraction.
    # Export from browser: https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/
    youtube_cookies_file: str = ""

    # Path to TranscriptAPI.com API key file or direct string.
    # Get your key at https://transcriptapi.com
    # Set as env TRANSCRIPT_API_KEY=sk_...
    transcript_api_key: str = ""

    # Supadata API key for YouTube/media transcripts (replaces TranscriptAPI).
    # Get your key at https://dash.supadata.ai/organizations/api-key
    # Set as env SUPADATA_API_KEY=sd_...
    supadata_api_key: str = ""

    # AssemblyAI is the primary remote-URL transcript provider. It
    # fetches public RSS enclosure URLs itself, avoiding local media download.
    # Set as env ASSEMBLYAI_API_KEY=...
    assemblyai_api_key: str = ""

    # Optional Podcast Index discovery credentials. The API finds canonical
    # publisher RSS feeds for cross-platform podcast links; it does not supply
    # transcript text. Set PODCAST_INDEX_API_KEY and PODCAST_INDEX_API_SECRET.
    podcast_index_api_key: str = ""
    podcast_index_api_secret: str = ""

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
    def annotations_dir(self) -> Path:
        """Manual notes and annotations directory."""
        return self.vault / "03-Raw-Annotations"

    @property
    def queries_dir(self) -> Path:
        """Saved query results and analyses directory."""
        return self.vault / "05-Queries"

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


def _dotenv_mapping(path: Path) -> dict[str, str]:
    """Read a dotenv file without leaking its values into ``os.environ``."""
    return {
        key: value
        for key, value in dotenv_values(path).items()
        if value is not None
    }


def _config_environment(env_file: str | None, overrides: dict[str, str]) -> dict[str, str]:
    """Merge process, base, vault, and explicit config without global mutation."""
    environment = dict(os.environ)
    base_file = Path(env_file) if env_file else Path.cwd() / ".env"
    if base_file.is_file():
        environment.update(_dotenv_mapping(base_file))

    # Resolve the vault selector before loading vault-specific settings. For
    # implicit discovery, the caller's process VAULT_PATH outranks CWD .env.
    vault_raw = overrides.get("VAULT_PATH")
    if vault_raw is None:
        vault_raw = (
            os.environ.get("VAULT_PATH", "")
            if env_file is None
            else environment.get("VAULT_PATH", "")
        )
    vault_dir = os.path.expandvars(vault_raw)
    if vault_dir:
        vault_file = Path(vault_dir).expanduser() / ".env"
        if vault_file.is_file() and vault_file != base_file:
            environment.update(_dotenv_mapping(vault_file))

    # Implicit discovery must respect explicit process configuration. An
    # explicitly supplied env_file keeps its established file-wins semantics.
    if env_file is None:
        environment.update(os.environ)
    environment.update(overrides)
    return environment


def load_config(env_file: str | None = None, **overrides: str) -> Config:
    """Load deterministic configuration without mutating process environment."""
    env = _config_environment(env_file, overrides)
    get = env.get

    llm_config = LLMProviderConfig(
        provider=get("LLM_PROVIDER", "ollama"),
        host=get("LLM_HOST", "http://localhost:11434"),
        model=get("LLM_MODEL", "gemma3:27b"),
        ingest_model=_optional_model_env("INGEST_MODEL", env),
        maintenance_model=_optional_model_env("MAINTENANCE_MODEL", env),
        query_model=_optional_model_env("QUERY_MODEL", env),
        expand_model=_optional_model_env("PASS2_MODEL", env),
        api_key=get("LLM_API_KEY"),
        timeout_ms=_int_env("LLM_TIMEOUT_MS", 1_800_000, env),
        context_window=_int_env("LLM_CONTEXT_WINDOW", 256_000, env),
    )

    return Config(
        llm=llm_config,
        vault_path=get("VAULT_PATH", str(Path.home() / "MyVault")),
        max_source_chars=_int_env("MAX_SOURCE_CHARS", 1_000_000, env),
        min_source_chars=_int_env("MIN_SOURCE_CHARS", 50, env),
        chunk_size=_int_env("CHUNK_SIZE", 30_000, env),
        compile_concurrency=_int_env("COMPILE_CONCURRENCY", 3, env),
        output_language=get("OUTPUT_LANGUAGE", ""),
        synthesis_mode=get("SYNTHESIS_MODE", "single"),
        concept_min_body_chars=_int_env("CONCEPT_MIN_BODY_CHARS", 1200, env),
        entry_min_body_chars=_int_env("ENTRY_MIN_BODY_CHARS", 500, env),
        clipping_min_body_chars=_int_env("CLIPPING_MIN_BODY_CHARS", 500, env),
        similarity_dedup_threshold=_float_env("SIMILARITY_DEDUP_THRESHOLD", 0.85, env),
        moc_assignment_threshold=_float_env("MOC_ASSIGNMENT_THRESHOLD", 0.55, env),
        retry_count=_int_env("RETRY_COUNT", 3, env),
        retry_base_ms=_int_env("RETRY_BASE_MS", 1_000, env),
        retry_multiplier=_int_env("RETRY_MULTIPLIER", 4, env),
        max_document_bytes=_int_env("MAX_DOCUMENT_BYTES", 50_000_000, env),
        max_document_candidates=_int_env("MAX_DOCUMENT_CANDIDATES", 10, env),
        parser_timeout_seconds=_int_env("PARSER_TIMEOUT_SECONDS", 120, env),
        max_parser_stdout_bytes=_int_env("MAX_PARSER_STDOUT_BYTES", 1_000_000, env),
        max_parser_stderr_bytes=_int_env("MAX_PARSER_STDERR_BYTES", 16_384, env),
        max_html_bytes=_int_env("MAX_HTML_BYTES", 10_000_000, env),
        embeddings_enabled=_bool_env("EMBEDDINGS_ENABLED", False, env),
        embedding_model=get("EMBEDDING_MODEL", "embeddinggemma:300m").strip(),
        embedding_host=get("EMBEDDING_HOST", get("LLM_HOST", "http://localhost:11434")).rstrip("/"),
        residential_proxy_url=get("RESIDENTIAL_PROXY_URL", ""),
        youtube_cookies_file=get("YOUTUBE_COOKIES_FILE", ""),
        transcript_api_key=get("TRANSCRIPT_API_KEY", ""),
        supadata_api_key=get("SUPADATA_API_KEY", ""),
        assemblyai_api_key=get("ASSEMBLYAI_API_KEY", ""),
        podcast_index_api_key=get("PODCAST_INDEX_API_KEY", ""),
        podcast_index_api_secret=get("PODCAST_INDEX_API_SECRET", ""),
    )


def _optional_model_env(key: str, env: dict[str, str]) -> str | None:
    """Return a configured task model, treating empty values as no override."""
    value = env.get(key)
    if value is None or not value.strip():
        return None
    return value.strip()


def _int_env(key: str, default: int, env: dict[str, str]) -> int:
    """Parse an integer from the resolved config environment with fallback."""
    val = env.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _bool_env(key: str, default: bool, env: dict[str, str]) -> bool:
    """Parse a boolean from the resolved config environment with fallback."""
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(key: str, default: float, env: dict[str, str]) -> float:
    """Parse a float from the resolved config environment with fallback."""
    val = env.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default
