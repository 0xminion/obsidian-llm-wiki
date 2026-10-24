"""SQLite registry for scalable vault state, search, queues, and vectors.

The registry is the authoritative incremental index for v2 vaults. Markdown and
JSON cache files remain portable artifacts, but normal builds use indexed rows
rather than reconstructing state by scanning the filesystem.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import islice
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit, urlunsplit

from obsidian_llm_wiki.core.models import (
    ConceptNote,
    SourceDoc,
    SourceProvenance,
    SourceSynthesis,
    concept_note_to_dict,
    moc_to_dict,
    source_synthesis_from_dict,
    source_synthesis_to_dict,
)

__all__ = [
    "ArtifactVersions",
    "AnnCandidate",
    "AnnSearchResult",
    "RenderDiff",
    "ConceptCandidate",
    "DirtyRecord",
    "IngestMatch",
    "VaultRegistry",
]

_SCHEMA_VERSION = 6
_LSH_TABLES = 8
_LSH_BITS = 12
_LSH_SPARSITY = 8
_COMPLETED_QUEUE_HISTORY = 10_000
_LIVE_OPERATION_HISTORY = 1_000
_DIRTY_LEASE_SECONDS = 600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    normalized_url TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    extractor_version TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS source_content_hash_idx ON source(content_hash);
CREATE INDEX IF NOT EXISTS source_canonical_url_idx ON source(canonical_url);

CREATE TABLE IF NOT EXISTS source_file_state (
    source_id INTEGER PRIMARY KEY REFERENCES source(id) ON DELETE CASCADE,
    mtime_ns INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_version (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    extractor_version TEXT NOT NULL DEFAULT '',
    requested_url TEXT NOT NULL DEFAULT '',
    canonical_url TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(source_id, content_hash, extractor_version)
);

CREATE TABLE IF NOT EXISTS synthesis (
    source_id INTEGER PRIMARY KEY REFERENCES source(id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    parse_error_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS synthesis_versions_idx
    ON synthesis(prompt_version, schema_version, model_id);

CREATE TABLE IF NOT EXISTS concept (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS concept_slug_idx ON concept(slug);

CREATE TABLE IF NOT EXISTS concept_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    PRIMARY KEY(source_id, concept_id)
);
CREATE INDEX IF NOT EXISTS concept_source_concept_idx
    ON concept_source(concept_id);

CREATE TABLE IF NOT EXISTS concept_alias (
    alias TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS concept_alias_id_idx ON concept_alias(concept_id);

CREATE VIRTUAL TABLE IF NOT EXISTS concept_fts USING fts5(
    concept_id UNINDEXED,
    slug,
    title,
    aliases,
    tags,
    summary,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS concept_fts_identity (
    concept_id TEXT PRIMARY KEY REFERENCES concept(id) ON DELETE CASCADE,
    fts_rowid INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS moc (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS moc_slug_idx ON moc(slug);

CREATE TABLE IF NOT EXISTS moc_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    PRIMARY KEY(source_id, moc_id)
);
CREATE INDEX IF NOT EXISTS moc_source_moc_idx ON moc_source(moc_id);

CREATE TABLE IF NOT EXISTS membership_source (
    source_id INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(source_id, moc_id, concept_id)
);
CREATE INDEX IF NOT EXISTS membership_source_moc_idx
    ON membership_source(moc_id);
CREATE INDEX IF NOT EXISTS membership_source_concept_idx
    ON membership_source(concept_id);

CREATE TABLE IF NOT EXISTS membership (
    moc_id TEXT NOT NULL REFERENCES moc(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concept(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(moc_id, concept_id)
);
CREATE INDEX IF NOT EXISTS membership_concept_idx ON membership(concept_id);

CREATE TABLE IF NOT EXISTS embedding (
    entity_kind TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_id, model_id)
);
CREATE INDEX IF NOT EXISTS embedding_lookup_idx
    ON embedding(entity_kind, model_id, dimensions);

CREATE TABLE IF NOT EXISTS ann_bucket (
    entity_kind TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    table_id INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(entity_kind, model_id, dimensions, table_id, bucket, entity_id)
);
CREATE INDEX IF NOT EXISTS ann_bucket_lookup_idx
    ON ann_bucket(entity_kind, model_id, dimensions, table_id, bucket);
CREATE INDEX IF NOT EXISTS ann_bucket_entity_idx
    ON ann_bucket(entity_kind, entity_id, model_id);

CREATE TABLE IF NOT EXISTS dirty_queue (
    id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    claimed_by TEXT NOT NULL DEFAULT '',
    lease_expires_at TEXT NOT NULL DEFAULT '',
    lease_token TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    pending_payload_json TEXT,
    pending_next_attempt_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dirty_queue_claim_idx
    ON dirty_queue(state, next_attempt_at, id);
CREATE UNIQUE INDEX IF NOT EXISTS dirty_queue_pending_unique
    ON dirty_queue(entity_type, entity_id, reason)
    WHERE state IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS operation (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operation_source_idx ON operation(source, updated_at);

CREATE TABLE IF NOT EXISTS operation_archive (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS operation_archive_source_idx
    ON operation_archive(source, updated_at);

CREATE TABLE IF NOT EXISTS metric_run (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metric_run_started_idx ON metric_run(started_at, id);

CREATE TABLE IF NOT EXISTS metric_run_archive (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS metric_run_archive_started_idx
    ON metric_run_archive(started_at, id);

CREATE TABLE IF NOT EXISTS cross_lingual_link (
    source_slug TEXT NOT NULL,
    target_slug TEXT NOT NULL,
    score REAL NOT NULL,
    display TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(source_slug, target_slug, model_id, schema_version)
);
CREATE INDEX IF NOT EXISTS cross_lingual_link_namespace_idx
    ON cross_lingual_link(model_id, schema_version, source_slug);

CREATE TABLE IF NOT EXISTS ingest_hash (
    normalized_url TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    output_file TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ingest_hash_content_idx ON ingest_hash(content_hash);
CREATE INDEX IF NOT EXISTS ingest_hash_canonical_idx ON ingest_hash(canonical_url);

CREATE TABLE IF NOT EXISTS render_state (
    entity_kind TEXT NOT NULL,
    entity_key TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(entity_kind, entity_key)
);
CREATE INDEX IF NOT EXISTS render_state_path_idx ON render_state(path);

CREATE TABLE IF NOT EXISTS health_finding (
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    reviewed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(path, kind, message)
);
CREATE INDEX IF NOT EXISTS health_finding_kind_idx ON health_finding(kind, severity);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _canonical_vector(vector: list[float]) -> list[float]:
    if not vector:
        raise ValueError("embedding vector must not be empty")
    result: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("embedding vector must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding vector must contain finite numbers")
        result.append(number)
    return result


def _lsh_bucket(vector: list[float], model_id: str, table_id: int) -> int:
    """Compute one deterministic sparse random-hyperplane signature."""
    dimensions = len(vector)
    bucket = 0
    for bit in range(_LSH_BITS):
        projection = 0.0
        for sample in range(_LSH_SPARSITY):
            seed = f"{model_id}\x00{dimensions}\x00{table_id}\x00{bit}\x00{sample}".encode()
            digest = hashlib.blake2b(seed, digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            projection += sign * vector[index]
        if projection >= 0:
            bucket |= 1 << bit
    return bucket


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _normalize_url(url: str) -> str:
    """Canonicalize a public URL for exact indexed identity matching."""
    if not url.strip():
        return ""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.casefold()
    host = (parts.hostname or "").casefold()
    if not scheme or not host:
        return url.strip()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    # Query syntax is origin-defined: repeated-value order and '+' vs '%20'
    # are not universally interchangeable. Normalize transport identity only.
    return urlunsplit((scheme, host, parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class ArtifactVersions:
    """Versions that determine whether persisted synthesis is reusable."""

    extractor: str
    prompt: str
    schema: str
    synthesis_model: str

    def as_dict(self) -> dict[str, str]:
        return {
            "extractor": self.extractor,
            "prompt": self.prompt,
            "schema": self.schema,
            "synthesis_model": self.synthesis_model,
        }


@dataclass(frozen=True)
class DirtyRecord:
    """One durable unit of invalidation or retry work."""

    id: int
    entity_type: str
    entity_id: str
    reason: str
    state: str
    attempts: int
    next_attempt_at: str
    payload: dict[str, object]
    error: str


@dataclass(frozen=True)
class IngestMatch:
    """Indexed duplicate match for an already persisted source."""

    normalized_url: str
    canonical_url: str
    content_hash: str
    output_file: str


@dataclass(frozen=True)
class ConceptCandidate:
    """One bounded lexical/semantic prompt candidate."""

    id: str
    slug: str
    title: str
    summary: str
    aliases: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class AnnCandidate:
    """One exact-reranked candidate from bounded ANN buckets."""

    entity_id: str
    score: float


@dataclass(frozen=True)
class AnnSearchResult:
    """ANN results plus the candidate-evaluation budget used."""

    candidates: list[AnnCandidate]
    evaluated: int
    bucket_hits: int


@dataclass(frozen=True)
class RenderDiff:
    changed_keys: frozenset[str]
    stale_paths: tuple[str, ...]
    stale_keys: frozenset[str]


def _source_metadata_payload(source: SourceDoc) -> dict[str, object]:
    """Serialize the metadata needed to render unchanged source indexes."""
    provenance = source.provenance
    return {
        "aliases": list(source.aliases),
        "tags": list(source.tags),
        "source_type": source.source_type,
        "provenance": {
            "requested_url": provenance.requested_url,
            "resolved_url": provenance.resolved_url,
            "extracted_url": provenance.extracted_url,
            "extractor_chain": list(provenance.extractor_chain),
            "content_type": provenance.content_type,
            "document_format": provenance.document_format,
            "retrieved_at": provenance.retrieved_at,
            "content_sha256": provenance.content_sha256,
            "diagnostics": list(provenance.diagnostics),
        },
    }


def _metadata_text(value: object) -> str:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _metadata_list(value: object) -> list[str]:
    return [str(item) for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _source_stub_from_metadata(
    *,
    filename: str,
    title: str,
    canonical_url: str,
    metadata: dict[str, object],
) -> SourceDoc:
    """Restore metadata without loading the source Markdown body."""
    raw_provenance = metadata.get("provenance")
    provenance_data = raw_provenance if isinstance(raw_provenance, dict) else metadata
    provenance = SourceProvenance(
        requested_url=_metadata_text(provenance_data.get("requested_url")),
        resolved_url=_metadata_text(provenance_data.get("resolved_url")),
        extracted_url=_metadata_text(provenance_data.get("extracted_url")),
        extractor_chain=tuple(_metadata_list(provenance_data.get("extractor_chain"))),
        content_type=_metadata_text(provenance_data.get("content_type")),
        document_format=_metadata_text(provenance_data.get("document_format")),
        retrieved_at=_metadata_text(provenance_data.get("retrieved_at")),
        content_sha256=_metadata_text(provenance_data.get("content_sha256")),
        diagnostics=tuple(_metadata_list(provenance_data.get("diagnostics"))),
    )
    return SourceDoc(
        title=title or Path(filename).stem,
        content="",
        url=canonical_url or None,
        source_file=filename,
        provenance=provenance,
        aliases=_metadata_list(metadata.get("aliases")),
        tags=_metadata_list(metadata.get("tags")),
        source_type=_metadata_text(metadata.get("source_type")),
    )


class VaultRegistry:
    """Own the vault-local SQLite registry and its transactional schema."""

    def __init__(self, llmwiki_dir: str | Path) -> None:
        from obsidian_llm_wiki.core.generated_paths import (
            checked_generated_path,
            checked_metadata_directory,
        )

        self.llmwiki_dir = checked_metadata_directory(Path(llmwiki_dir))
        self.path = self.llmwiki_dir / "registry.db"
        # SQLite may read, create, recover or unlink sidecars during connect or
        # WAL negotiation. Reject the entire family before even creating dirs.
        for suffix in ("", "-wal", "-shm", "-journal"):
            checked_generated_path(self.path.with_name(self.path.name + suffix))
        self.llmwiki_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        # Concurrent first opens can fail the rollback->WAL lock upgrade
        # immediately even with busy_timeout set. Retry this negotiation only,
        # with bounded backoff; unrelated I/O/schema failures must surface.
        deadline = time.monotonic() + 30
        try:
            for attempt in range(8):
                try:
                    self.connection.execute("PRAGMA journal_mode = WAL")
                    break
                except sqlite3.OperationalError as exc:
                    code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
                    remaining = deadline - time.monotonic()
                    if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                        raise
                    if attempt == 7 or remaining <= 0:
                        raise
                    time.sleep(min(0.01 * (2 ** attempt), remaining))
        except BaseException:
            self.connection.close()
            raise
        self.connection.execute("PRAGMA synchronous = NORMAL")
        previous_version = 0
        meta_exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if meta_exists is not None:
            version_row = self.connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if version_row is not None:
                try:
                    previous_version = int(version_row[0])
                except (TypeError, ValueError) as exc:
                    self.connection.close()
                    raise RuntimeError("Registry has an invalid schema version") from exc
                if previous_version > _SCHEMA_VERSION:
                    self.connection.close()
                    raise RuntimeError(
                        "Registry schema is newer than this binary "
                        f"({previous_version} > {_SCHEMA_VERSION})"
                    )
        self.connection.executescript(_SCHEMA)
        # Serialize column discovery with ALTER for concurrent first opens.
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            dirty_columns = {
                str(row[1])
                for row in self.connection.execute("PRAGMA table_info(dirty_queue)")
            }
            for name, definition in (
                ("lease_expires_at", "TEXT NOT NULL DEFAULT ''"),
                ("lease_token", "TEXT NOT NULL DEFAULT ''"),
                ("pending_payload_json", "TEXT"),
                ("pending_next_attempt_at", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in dirty_columns:
                    self.connection.execute(
                        f"ALTER TABLE dirty_queue ADD COLUMN {name} {definition}"
                    )
        if previous_version < 5:
            with self.connection:
                self.connection.execute(
                    "INSERT OR REPLACE INTO concept_fts_identity(concept_id, fts_rowid) "
                    "SELECT concept_id, MAX(rowid) FROM concept_fts "
                    "WHERE concept_id IN (SELECT id FROM concept) GROUP BY concept_id"
                )
                self.connection.execute(
                    "DELETE FROM concept_fts WHERE rowid NOT IN "
                    "(SELECT fts_rowid FROM concept_fts_identity)"
                )
                self._set_meta_unlocked("pipeline_checkpoint", "")
        if previous_version < 3:
            self._backfill_entity_sources()
        self._migrate_legacy_operations()
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(_SCHEMA_VERSION),),
        )
        self.connection.commit()

    def _backfill_entity_sources(self) -> None:
        """Build v3 source/entity ownership from existing synthesis payloads."""
        rows = self.connection.execute(
            "SELECT source_id, payload_json FROM synthesis"
        ).fetchall()
        with self.connection:
            self.connection.execute("DELETE FROM membership_source")
            self.connection.execute("DELETE FROM concept_source")
            self.connection.execute("DELETE FROM moc_source")
            self.connection.execute("DELETE FROM membership")
            for row in rows:
                try:
                    payload = json.loads(row[1])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                synthesis = source_synthesis_from_dict(payload)
                source_id = int(row[0])
                for concept in synthesis.concepts:
                    concept_row = self.connection.execute(
                        "SELECT id FROM concept WHERE slug=?", (concept.slug,)
                    ).fetchone()
                    if concept_row is not None:
                        self.connection.execute(
                            "INSERT OR IGNORE INTO concept_source(source_id, concept_id) "
                            "VALUES(?, ?)",
                            (source_id, str(concept_row[0])),
                        )
                for moc in synthesis.maps:
                    moc_row = self.connection.execute(
                        "SELECT id FROM moc WHERE slug=?", (moc.slug,)
                    ).fetchone()
                    if moc_row is not None:
                        moc_id = str(moc_row[0])
                        self.connection.execute(
                            "INSERT OR IGNORE INTO moc_source(source_id, moc_id) VALUES(?, ?)",
                            (source_id, moc_id),
                        )
                        for position, slug in enumerate(moc.concept_slugs):
                            concept_row = self.connection.execute(
                                "SELECT id FROM concept WHERE slug=?", (slug,)
                            ).fetchone()
                            if concept_row is not None:
                                self.connection.execute(
                                    """
                                    INSERT OR IGNORE INTO membership_source(
                                        source_id, moc_id, concept_id, position
                                    ) VALUES(?, ?, ?, ?)
                                    """,
                                    (source_id, moc_id, str(concept_row[0]), position),
                                )
            moc_ids = {
                str(row[0])
                for row in self.connection.execute("SELECT id FROM moc")
            }
            self._rebuild_moc_memberships(moc_ids)
            self._cleanup_unreferenced_entities()

    def _migrate_legacy_operations(self) -> bool:
        """Import the bounded JSON operation ledger exactly once."""
        completed = self.connection.execute(
            "SELECT 1 FROM meta WHERE key='legacy_operations_migration_complete'"
        ).fetchone()
        if completed is not None:
            return False
        path = self.llmwiki_dir / "operations.json"
        if not path.is_file():
            with self.connection:
                self._set_meta_unlocked("legacy_operations_migration_complete", "1")
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            with self.connection:
                self._set_meta_unlocked(
                    "legacy_operations_migration_complete", "invalid-json"
                )
            return False
        raw_operations = payload.get("operations", []) if isinstance(payload, dict) else []
        if not isinstance(raw_operations, list):
            with self.connection:
                self._set_meta_unlocked(
                    "legacy_operations_migration_complete", "invalid-payload"
                )
            return False
        with self.connection:
            for operation in raw_operations:
                if isinstance(operation, dict):
                    self._upsert_operation_unlocked(operation)
            self._set_meta_unlocked("legacy_operations_migration_complete", "1")
        return True

    def upsert_operation(self, operation: dict[str, object]) -> None:
        """Persist one operation and archive rows beyond the live-history cap."""
        with self.connection:
            self._upsert_operation_unlocked(operation)

    def _upsert_operation_unlocked(self, operation: dict[str, object]) -> None:
        operation_id = _metadata_text(operation.get("id"))
        if not operation_id:
            return
        raw_attempt = operation.get("attempt", 1)
        try:
            attempt = max(
                1,
                int(raw_attempt)
                if isinstance(raw_attempt, (str, int)) and not isinstance(raw_attempt, bool)
                else 1,
            )
        except (TypeError, ValueError):
            attempt = 1
        payload_json = json.dumps(operation, ensure_ascii=False, separators=(",", ":"))
        values = (
            operation_id,
            _metadata_text(operation.get("run_id")),
            _metadata_text(operation.get("source")),
            _metadata_text(operation.get("source_kind")) or "url",
            _metadata_text(operation.get("mode")) or "ingest",
            _metadata_text(operation.get("status")) or "planned",
            attempt,
            payload_json,
            _metadata_text(operation.get("error"))[:1_000],
            _metadata_text(operation.get("created_at")) or _now(),
            _metadata_text(operation.get("updated_at")) or _now(),
        )
        self.connection.execute("DELETE FROM operation_archive WHERE id=?", (operation_id,))
        self.connection.execute(
            """
            INSERT INTO operation(
                id, run_id, source, source_kind, mode, status, attempt,
                payload_json, error, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                run_id=excluded.run_id,
                source=excluded.source,
                source_kind=excluded.source_kind,
                mode=excluded.mode,
                status=excluded.status,
                attempt=excluded.attempt,
                payload_json=excluded.payload_json,
                error=excluded.error,
                updated_at=excluded.updated_at
            """,
            values,
        )
        count_row = self.connection.execute("SELECT count(*) FROM operation").fetchone()
        overflow = max(0, int(count_row[0]) - _LIVE_OPERATION_HISTORY)
        if not overflow:
            return
        self.connection.execute(
            """
            INSERT OR REPLACE INTO operation_archive
            SELECT * FROM operation ORDER BY updated_at, id LIMIT ?
            """,
            (overflow,),
        )
        self.connection.execute(
            """
            DELETE FROM operation WHERE id IN (
                SELECT id FROM operation ORDER BY updated_at, id LIMIT ?
            )
            """,
            (overflow,),
        )

    def latest_operation_for_source(self, source: str) -> dict[str, object] | None:
        """Read the latest live or archived operation for an exact source."""
        row = self.connection.execute(
            """
            SELECT payload_json FROM (
                SELECT source, updated_at, id, payload_json FROM operation
                UNION ALL
                SELECT source, updated_at, id, payload_json FROM operation_archive
            )
            WHERE source=? ORDER BY updated_at DESC, id DESC LIMIT 1
            """,
            (source,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def interrupted_ingest_operations(self) -> list[dict[str, object]]:
        """Return latest operations left running by an unclean process exit."""
        rows = self.connection.execute(
            """
            SELECT payload_json FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY source ORDER BY updated_at DESC, id DESC
                ) AS source_rank
                FROM (
                    SELECT * FROM operation
                    UNION ALL
                    SELECT * FROM operation_archive
                )
                WHERE mode='ingest'
            )
            WHERE source_rank=1 AND status IN ('running', 'retrying')
            ORDER BY updated_at, id
            """
        ).fetchall()
        operations: list[dict[str, object]] = []
        for row in rows:
            try:
                payload = json.loads(str(row[0]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                operations.append(payload)
        return operations

    def release_dirty_worker(self, worker_id: str) -> int:
        """Release all leases for a proven-dead/stopped worker, preserving successors.

        Administrative recovery only: the caller must ensure this worker cannot
        still publish results, and must not reuse its ID for a live worker.
        """
        return self._release_dirty_workers(worker_id)

    def release_abandoned_pipeline_workers(self) -> int:
        """Reclaim old pipeline claims AFTER acquiring the exclusive compile lock.

        Call before starting any claims in the new compile run. Possession of
        that lock, not lease age, proves previous pipeline workers are stopped.
        The case-sensitive SQL prefix deliberately excludes ingest workers.
        """
        return self._release_dirty_workers(None)

    def _release_dirty_workers(self, worker_id: str | None) -> int:
        predicate = "claimed_by=?" if worker_id is not None else "claimed_by GLOB 'pipeline:*'"
        parameters = (_now(), worker_id) if worker_id is not None else (_now(),)
        with self.connection:
            cursor = self.connection.execute(
                f"""
                UPDATE dirty_queue
                SET state='pending', claimed_by='', lease_expires_at='', lease_token='',
                    updated_at=?,
                    payload_json=COALESCE(pending_payload_json, payload_json),
                    next_attempt_at=CASE WHEN pending_payload_json IS NULL
                        THEN next_attempt_at ELSE pending_next_attempt_at END,
                    attempts=CASE WHEN pending_payload_json IS NULL THEN attempts ELSE 0 END,
                    error=CASE WHEN pending_payload_json IS NULL THEN error ELSE '' END,
                    pending_payload_json=NULL, pending_next_attempt_at=''
                WHERE state='running' AND {predicate}
                """,
                parameters,
            )
        return cursor.rowcount

    def upsert_metric_run(
        self,
        metrics: dict[str, object],
        *,
        live_limit: int = 50,
    ) -> None:
        """Keep a bounded live metrics window while archiving older runs."""
        payload_json = json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))
        run_id = _metadata_text(metrics.get("run_id"))
        started_at = _metadata_text(metrics.get("started_at"))
        identity = hashlib.sha256(
            f"{run_id}\x00{started_at}".encode()
        ).hexdigest()
        now = _now()
        with self.connection:
            self.connection.execute("DELETE FROM metric_run_archive WHERE id=?", (identity,))
            self.connection.execute(
                """
                INSERT INTO metric_run(id, run_id, payload_json, started_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (identity, run_id, payload_json, started_at, now),
            )
            count_row = self.connection.execute("SELECT count(*) FROM metric_run").fetchone()
            overflow = max(0, int(count_row[0]) - max(1, live_limit))
            if overflow:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO metric_run_archive
                    SELECT * FROM metric_run ORDER BY started_at, id LIMIT ?
                    """,
                    (overflow,),
                )
                self.connection.execute(
                    """
                    DELETE FROM metric_run WHERE id IN (
                        SELECT id FROM metric_run ORDER BY started_at, id LIMIT ?
                    )
                    """,
                    (overflow,),
                )

    def upsert_synthesis(
        self,
        filename: str,
        source: SourceDoc,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
    ) -> None:
        """Atomically persist one source revision and its synthesis payload."""
        now = _now()
        input_hash = _content_hash(source.content)
        self._assign_canonical_ids(synthesis)
        requested_url = source.provenance.requested_url or source.url or ""
        canonical_url = (
            source.provenance.extracted_url
            or source.provenance.resolved_url
            or source.url
            or ""
        )
        synthesis.source_file = filename
        metadata = json.dumps(
            _source_metadata_payload(source),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO source(
                    filename, normalized_url, canonical_url, content_hash,
                    extractor_version, title, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    normalized_url=excluded.normalized_url,
                    canonical_url=excluded.canonical_url,
                    content_hash=excluded.content_hash,
                    extractor_version=excluded.extractor_version,
                    title=excluded.title,
                    status='active',
                    updated_at=excluded.updated_at
                """,
                (
                    filename,
                    requested_url,
                    canonical_url,
                    input_hash,
                    versions.extractor,
                    source.title,
                    now,
                    now,
                ),
            )
            source_id = int(
                self.connection.execute(
                    "SELECT id FROM source WHERE filename = ?", (filename,)
                ).fetchone()[0]
            )
            self.connection.execute(
                """
                INSERT INTO source_version(
                    source_id, content_hash, extractor_version, requested_url,
                    canonical_url, metadata_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, content_hash, extractor_version) DO UPDATE SET
                    requested_url=excluded.requested_url,
                    canonical_url=excluded.canonical_url,
                    metadata_json=excluded.metadata_json
                """,
                (
                    source_id,
                    input_hash,
                    versions.extractor,
                    requested_url,
                    canonical_url,
                    metadata,
                    now,
                ),
            )
            self._upsert_synthesis_payload_unlocked(
                source_id, filename, input_hash, synthesis, versions, now
            )

    def update_synthesis(
        self,
        filename: str,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
    ) -> None:
        """Update synthesis ownership without rewriting its source revision."""
        row = self.connection.execute(
            "SELECT id, content_hash FROM source WHERE filename=?", (filename,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown source: {filename}")
        self._assign_canonical_ids(synthesis)
        now = _now()
        with self.connection:
            self._upsert_synthesis_payload_unlocked(
                int(row[0]), filename, str(row[1]), synthesis, versions, now
            )

    def _upsert_synthesis_payload_unlocked(
        self,
        source_id: int,
        filename: str,
        input_hash: str,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
        now: str,
    ) -> None:
        """Persist one synthesis and rebuild its canonical ownership rows."""
        self._set_meta_unlocked("pipeline_checkpoint", "")
        synthesis.source_file = filename
        payload = json.dumps(
            source_synthesis_to_dict(synthesis),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.connection.execute(
            """
            INSERT INTO synthesis(
                source_id, payload_json, input_hash, prompt_version,
                schema_version, model_id, status, parse_error_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, 'ready', '{}', ?)
            ON CONFLICT(source_id) DO UPDATE SET
                payload_json=excluded.payload_json,
                input_hash=excluded.input_hash,
                prompt_version=excluded.prompt_version,
                schema_version=excluded.schema_version,
                model_id=excluded.model_id,
                status='ready',
                parse_error_json='{}',
                updated_at=excluded.updated_at
            """,
            (
                source_id,
                payload,
                input_hash,
                versions.prompt,
                versions.schema,
                versions.synthesis_model,
                now,
            ),
        )
        touched_moc_ids = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT moc_id FROM moc_source WHERE source_id=?", (source_id,)
            )
        }
        touched_concept_ids = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT concept_id FROM concept_source WHERE source_id=?",
                (source_id,),
            )
        }
        touched_concept_ids.update(concept.canonical_id for concept in synthesis.concepts)
        previous_member_inputs = self._moc_member_inputs(touched_concept_ids)
        self.connection.execute(
            "DELETE FROM membership_source WHERE source_id=?", (source_id,)
        )
        self.connection.execute("DELETE FROM concept_source WHERE source_id=?", (source_id,))
        self.connection.execute("DELETE FROM moc_source WHERE source_id=?", (source_id,))
        self._upsert_synthesis_entities(source_id, synthesis, versions, now)
        touched_concept_ids.update(
            str(row[0])
            for row in self.connection.execute(
                "SELECT concept_id FROM concept_source WHERE source_id=?", (source_id,)
            )
        )
        touched_moc_ids.update(
            str(row[0])
            for row in self.connection.execute(
                "SELECT moc_id FROM moc_source WHERE source_id=?", (source_id,)
            )
        )
        touched_moc_ids.update(self._mocs_referencing_concepts(touched_concept_ids))
        self._cleanup_unreferenced_entities(
            concept_ids=touched_concept_ids,
            moc_ids=touched_moc_ids,
        )
        self._rebuild_concept_aggregates(touched_concept_ids, versions, now)
        current_member_inputs = self._moc_member_inputs(touched_concept_ids)
        changed_members = {
            cid for cid in touched_concept_ids
            if previous_member_inputs.get(cid) != current_member_inputs.get(cid)
        }
        self._rebuild_moc_aggregates(touched_moc_ids, now, changed_members)

    def _assign_canonical_ids(self, synthesis: SourceSynthesis) -> None:
        """Resolve model-provided IDs only when they already exist locally."""
        slug_map: dict[str, str] = {}
        reservations = getattr(self, "_concept_reservations", None)
        if reservations is None:
            reservations = self._concept_reservations = {}

        def definition_key(title: str, summary: str) -> str:
            text = "\x1f".join(" ".join(value.casefold().split()) for value in (title, summary))
            return hashlib.sha256(text.encode()).hexdigest()

        def existing_identity(slug: str):
            row = self.connection.execute(
                "SELECT id, slug, title, summary FROM concept WHERE slug=?", (slug,)
            ).fetchone()
            if row is not None:
                owned = self.connection.execute(
                    "SELECT 1 FROM concept_source cs JOIN source s ON s.id=cs.source_id "
                    "WHERE cs.concept_id=? AND s.filename=?",
                    (row[0], synthesis.source_file),
                ).fetchone() is not None
                return str(row[0]), str(row[1]), definition_key(str(row[2]), str(row[3])), owned
            pending = reservations.get(slug)
            if pending is not None:
                identity, fingerprint, owner = pending
                return identity, slug, fingerprint, owner == synthesis.source_file
            return None

        for concept in synthesis.concepts:
            original_slug = concept.slug
            fingerprint = definition_key(concept.title, concept.summary)
            row = None
            if concept.canonical_id:
                row = self.connection.execute(
                    "SELECT id, slug FROM concept WHERE id=?", (concept.canonical_id,)
                ).fetchone()
            if row is None and synthesis.source_file:
                # Revisions keep a previously disambiguated source-owned note.
                # Look only through this source's indexed ownership, not the vault.
                owned_rows = self.connection.execute(
                    "SELECT c.id, c.slug, c.title FROM concept c "
                    "JOIN concept_source cs ON cs.concept_id=c.id "
                    "JOIN source s ON s.id=cs.source_id WHERE s.filename=?",
                    (synthesis.source_file,),
                ).fetchall()
                matches = [
                    candidate for candidate in owned_rows
                    if " ".join(str(candidate[2]).casefold().split())
                    == " ".join(concept.title.casefold().split())
                ]
                if len(matches) == 1:
                    row = matches[0]
            if row is None:
                candidate = existing_identity(concept.slug)
                if candidate is not None and candidate[2] != fingerprint and not candidate[3]:
                    # Equal titles can still be homonyms (Python the language vs
                    # the snake). Only exact definitions or prior source ownership
                    # permit implicit reuse. Semantic reuse must name a known ID.
                    suffix = fingerprint[:16]
                    concept.slug = f"{original_slug[:80 - len(suffix) - 1]}-{suffix}"
                    candidate = existing_identity(concept.slug)
                    if candidate is not None and candidate[2] != fingerprint and not candidate[3]:
                        raise ValueError(f"Ambiguous canonical identity for {original_slug!r}")
                if candidate is not None:
                    row = candidate[:2]
            if row is not None:
                concept.canonical_id = str(row[0])
                concept.slug = str(row[1])
            else:
                concept.canonical_id = self._stable_id("concept", concept.slug)
            reservations.setdefault(
                concept.slug, (concept.canonical_id, fingerprint, synthesis.source_file)
            )
            slug_map[original_slug] = concept.slug

        for concept in synthesis.concepts:
            for link in concept.related:
                link.slug = slug_map.get(link.slug, link.slug)
        for moc in synthesis.maps:
            moc.concept_slugs = list(
                dict.fromkeys(slug_map.get(slug, slug) for slug in moc.concept_slugs)
            )

        concepts_by_slug: dict[str, ConceptNote] = {}
        for concept in synthesis.concepts:
            existing = concepts_by_slug.get(concept.slug)
            if existing is None:
                concepts_by_slug[concept.slug] = concept
                continue
            existing.tags = list(dict.fromkeys([*existing.tags, *concept.tags]))
            existing.aliases = list(
                dict.fromkeys([*existing.aliases, concept.title, *concept.aliases])
            )
            existing.sections.extend(concept.sections)
            existing.claims.extend(concept.claims)
            existing_links = {(link.slug, link.relation) for link in existing.related}
            for link in concept.related:
                identity = (link.slug, link.relation)
                if identity not in existing_links:
                    existing.related.append(link)
                    existing_links.add(identity)
            existing.confidence = max(existing.confidence, concept.confidence)
            existing.is_new = existing.is_new and concept.is_new
        synthesis.concepts = list(concepts_by_slug.values())
        self._assign_moc_identities(synthesis)

    def _assign_moc_identities(self, synthesis: SourceSynthesis) -> None:
        """Reuse topic labels only with ownership or shared canonical members."""
        from obsidian_llm_wiki.synth.language import detect_language

        reservations = getattr(self, "_moc_reservations", None)
        if reservations is None:
            reservations = self._moc_reservations = {}
        owned = self.connection.execute(
            "SELECT m.slug, m.title FROM moc m JOIN moc_source ms ON ms.moc_id=m.id "
            "JOIN source s ON s.id=ms.source_id WHERE s.filename=?",
            (synthesis.source_file,),
        ).fetchall() if synthesis.source_file else []

        def lookup(slug: str):
            row = self.connection.execute(
                "SELECT id, title, summary FROM moc WHERE slug=?", (slug,)
            ).fetchone()
            if row is None:
                return reservations.get(slug)
            members = {
                str(member[0]) for member in self.connection.execute(
                    "SELECT c.slug FROM membership mb JOIN concept c ON c.id=mb.concept_id "
                    "WHERE mb.moc_id=?", (row[0],),
                )
            }
            return (" ".join(str(row[1]).casefold().split()),
                    detect_language(str(row[2]) or str(row[1])), members, "")

        for moc in synthesis.maps:
            title_key = " ".join(moc.title.casefold().split())
            language = detect_language(moc.summary or moc.title)
            members = set(moc.concept_slugs)
            previous_owned = [str(row[0]) for row in owned
                              if " ".join(str(row[1]).casefold().split()) == title_key]
            if len(previous_owned) == 1:
                moc.slug = previous_owned[0]
            else:
                candidate = lookup(moc.slug)
                if candidate is not None and not (
                    candidate[0] == title_key and candidate[1] == language
                    and (bool(members & candidate[2])
                         or bool(synthesis.source_file and candidate[3] == synthesis.source_file))
                ):
                    material = f"{synthesis.source_file}\x1f{moc.slug}\x1f{title_key}\x1f{language}"
                    suffix = hashlib.sha256(material.encode()).hexdigest()[:16]
                    moc.slug = f"{moc.slug[:80 - len(suffix) - 1]}-{suffix}"
            reservations.setdefault(moc.slug, (title_key, language, members, synthesis.source_file))

    def assign_canonical_ids(self, synthesis: SourceSynthesis) -> None:
        """Populate trusted stable IDs without persisting synthesis state."""
        self._assign_canonical_ids(synthesis)

    @staticmethod
    def _stable_id(kind: str, slug: str) -> str:
        digest = hashlib.sha256(f"{kind}\x00{slug}".encode()).hexdigest()[:24]
        return f"{kind}-{digest}"

    def _replace_concept_fts_unlocked(self, concept: ConceptNote) -> None:
        """Replace a document by indexed FTS rowid, never scan UNINDEXED IDs."""
        self.connection.execute(
            "DELETE FROM concept_fts WHERE rowid=("
            "SELECT fts_rowid FROM concept_fts_identity WHERE concept_id=?)",
            (concept.canonical_id,),
        )
        cursor = self.connection.execute(
            "INSERT INTO concept_fts(concept_id, slug, title, aliases, tags, summary) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (concept.canonical_id, concept.slug, concept.title,
             " ".join(concept.aliases), " ".join(concept.tags), concept.summary),
        )
        self.connection.execute(
            "INSERT INTO concept_fts_identity(concept_id, fts_rowid) VALUES(?, ?) "
            "ON CONFLICT(concept_id) DO UPDATE SET fts_rowid=excluded.fts_rowid",
            (concept.canonical_id, cursor.lastrowid),
        )

    def _upsert_synthesis_entities(
        self,
        source_id: int,
        synthesis: SourceSynthesis,
        versions: ArtifactVersions,
        now: str,
    ) -> None:
        """Update canonical concept/MoC rows and FTS documents."""
        for concept in synthesis.concepts:
            payload = concept_note_to_dict(concept)
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            tags_json = json.dumps(concept.tags, ensure_ascii=False, separators=(",", ":"))
            self.connection.execute(
                """
                INSERT INTO concept(
                    id, slug, title, summary, tags_json, payload_json,
                    content_hash, prompt_version, schema_version, model_id, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    slug=excluded.slug,
                    title=excluded.title,
                    summary=excluded.summary,
                    tags_json=excluded.tags_json,
                    payload_json=excluded.payload_json,
                    content_hash=excluded.content_hash,
                    prompt_version=excluded.prompt_version,
                    schema_version=excluded.schema_version,
                    model_id=excluded.model_id,
                    updated_at=excluded.updated_at
                """,
                (
                    concept.canonical_id,
                    concept.slug,
                    concept.title,
                    concept.summary,
                    tags_json,
                    payload_json,
                    _content_hash(payload_json),
                    versions.prompt,
                    versions.schema,
                    versions.synthesis_model,
                    now,
                ),
            )
            self.connection.execute(
                "DELETE FROM concept_alias WHERE concept_id=?", (concept.canonical_id,)
            )
            for alias in dict.fromkeys([concept.slug, concept.title, *concept.aliases]):
                normalized = " ".join(alias.casefold().split())
                if normalized:
                    self.connection.execute(
                        """
                        INSERT INTO concept_alias(alias, concept_id) VALUES(?, ?)
                        ON CONFLICT(alias) DO UPDATE SET concept_id=excluded.concept_id
                        """,
                        (normalized, concept.canonical_id),
                    )
            self._replace_concept_fts_unlocked(concept)
            self.connection.execute(
                "INSERT OR IGNORE INTO concept_source(source_id, concept_id) VALUES(?, ?)",
                (source_id, concept.canonical_id),
            )

        for moc in synthesis.maps:
            moc_id = self._stable_id("moc", moc.slug)
            payload_json = json.dumps(moc_to_dict(moc), ensure_ascii=False, separators=(",", ":"))
            self.connection.execute(
                """
                INSERT INTO moc(
                    id, slug, title, summary, tags_json, payload_json,
                    content_hash, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    moc_id,
                    moc.slug,
                    moc.title,
                    moc.summary,
                    json.dumps(moc.tags, ensure_ascii=False, separators=(",", ":")),
                    payload_json,
                    _content_hash(payload_json),
                    now,
                ),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO moc_source(source_id, moc_id) VALUES(?, ?)",
                (source_id, moc_id),
            )
            for position, slug in enumerate(moc.concept_slugs):
                row = self.connection.execute(
                    "SELECT id FROM concept WHERE slug=?", (slug,)
                ).fetchone()
                if row is not None:
                    self.connection.execute(
                        """
                        INSERT OR IGNORE INTO membership_source(
                            source_id, moc_id, concept_id, position
                        ) VALUES(?, ?, ?, ?)
                        """,
                        (source_id, moc_id, str(row[0]), position),
                    )

    def _rebuild_concept_aggregates(
        self,
        concept_ids: set[str],
        versions: ArtifactVersions | None,
        now: str,
    ) -> None:
        """Rebuild only touched aggregates from current source-owned payloads.

        Deletions have no new artifact version: retain the deterministic first
        surviving owner's version rather than attributing it to the deleted one.
        The caller owns the transaction, including aliases and FTS replacement.
        """
        from obsidian_llm_wiki.synth.dedupe import merge_concepts

        for concept_id in sorted(concept_ids):
            rows = self.connection.execute(
                """
                SELECT synthesis.payload_json, synthesis.prompt_version,
                       synthesis.schema_version, synthesis.model_id
                FROM concept_source
                JOIN source ON source.id=concept_source.source_id
                JOIN synthesis ON synthesis.source_id=source.id
                WHERE concept_source.concept_id=? AND synthesis.status='ready'
                ORDER BY source.filename
                """,
                (concept_id,),
            ).fetchall()
            contributions: list[ConceptNote] = []
            for row in rows:
                try:
                    source_synthesis = source_synthesis_from_dict(json.loads(row[0]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                for concept in source_synthesis.concepts:
                    candidate_id = concept.canonical_id or self._stable_id(
                        "concept", concept.slug
                    )
                    if candidate_id == concept_id:
                        contributions.append(concept)
            if not contributions:
                continue
            titles = [concept.title for concept in contributions if concept.title]
            merged = merge_concepts(contributions)[0]
            merged.canonical_id = concept_id
            merged.aliases = list(
                dict.fromkeys(
                    [*merged.aliases, *(title for title in titles if title != merged.title)]
                )
            )
            merged.summary = max(
                (concept.summary for concept in contributions),
                key=len,
                default=merged.summary,
            )
            payload_json = json.dumps(
                concept_note_to_dict(merged), ensure_ascii=False, separators=(",", ":")
            )
            self.connection.execute(
                """
                UPDATE concept
                SET slug=?, title=?, summary=?, tags_json=?, payload_json=?,
                    content_hash=?, prompt_version=?, schema_version=?, model_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    merged.slug,
                    merged.title,
                    merged.summary,
                    json.dumps(merged.tags, ensure_ascii=False, separators=(",", ":")),
                    payload_json,
                    _content_hash(payload_json),
                    versions.prompt if versions is not None else rows[0][1],
                    versions.schema if versions is not None else rows[0][2],
                    versions.synthesis_model if versions is not None else rows[0][3],
                    now,
                    concept_id,
                ),
            )
            self.connection.execute(
                "DELETE FROM concept_alias WHERE concept_id=?", (concept_id,)
            )
            aliases = list(dict.fromkeys([merged.slug, merged.title, *merged.aliases]))
            for alias in aliases:
                normalized = " ".join(alias.casefold().split())
                if normalized:
                    self.connection.execute(
                        """
                        INSERT INTO concept_alias(alias, concept_id) VALUES(?, ?)
                        ON CONFLICT(alias) DO UPDATE SET concept_id=excluded.concept_id
                        """,
                        (normalized, concept_id),
                    )
            self._replace_concept_fts_unlocked(merged)
            # Embeddings are keyed by canonical ID but derived only from title
            # and summary. Keep vectors already computed from the surviving
            # bundle; remove stale model variants and their ANN entries together.
            embedding_hash = _content_hash(f"{merged.title}\n{merged.summary}")
            stale_models = self.connection.execute(
                "SELECT model_id FROM embedding WHERE entity_kind='concept' "
                "AND entity_id=? AND input_hash<>?",
                (concept_id, embedding_hash),
            ).fetchall()
            for (model_id,) in stale_models:
                self.connection.execute(
                    "DELETE FROM ann_bucket WHERE entity_kind='concept' "
                    "AND entity_id=? AND model_id=?",
                    (concept_id, model_id),
                )
                self.connection.execute(
                    "DELETE FROM embedding WHERE entity_kind='concept' "
                    "AND entity_id=? AND model_id=?",
                    (concept_id, model_id),
                )

    def _moc_member_inputs(self, concept_ids: set[str]) -> dict[str, tuple[str, str]]:
        """Snapshot indexed vector inputs before source upserts overwrite them."""
        return {
            concept_id: (str(row[0]), str(row[1]))
            for concept_id in sorted(concept_ids)
            for row in self.connection.execute(
                "SELECT title, summary FROM concept WHERE id=?", (concept_id,),
            )
        }

    def _mocs_referencing_concepts(self, concept_ids: set[str]) -> set[str]:
        """Follow the reverse membership index before concept deletion cascades."""
        return {
            str(row[0]) for concept_id in sorted(concept_ids)
            for row in self.connection.execute(
                "SELECT moc_id FROM membership WHERE concept_id=?", (concept_id,),
            )
        }

    def _rebuild_moc_aggregates(
        self, moc_ids: set[str], now: str, changed_members: set[str] | None = None,
    ) -> None:
        """Project touched MoCs from indexed owners, in render's first-owner order.

        Filename order, not insertion/last-writer order, selects prose and tags.
        Union live members in that order and retain each owner's local positions.
        Never write the aggregate back into a source's synthesis contribution.
        The caller owns the transaction, including membership replacement.
        """
        from obsidian_llm_wiki.synth.dedupe import normalise_tags

        owners: dict[int, SourceSynthesis] = {}
        for moc_id in sorted(moc_ids):
            rows = self.connection.execute(
                "SELECT source.id, synthesis.payload_json FROM moc_source "
                "JOIN source ON source.id=moc_source.source_id "
                "JOIN synthesis ON synthesis.source_id=source.id "
                "WHERE moc_source.moc_id=? AND synthesis.status='ready' "
                "ORDER BY source.filename", (moc_id,),
            ).fetchall()
            contributions = []
            for source_id, payload in rows:
                if source_id not in owners:
                    owners[source_id] = source_synthesis_from_dict(json.loads(payload))
                contributions.extend(
                    (source_id, moc) for moc in owners[source_id].maps
                    if self._stable_id("moc", moc.slug) == moc_id
                )
            if not contributions:
                continue
            # These are freshly decoded objects, never caller-owned syntheses.
            merged = contributions[0][1]
            payload = moc_to_dict(merged)
            payload["tags"] = normalise_tags(merged.tags)
            members: dict[str, str] = {}
            self.connection.execute("DELETE FROM membership_source WHERE moc_id=?", (moc_id,))
            for source_id, moc in contributions:
                for position, slug in enumerate(moc.concept_slugs):
                    row = self.connection.execute(
                        "SELECT id FROM concept WHERE slug=?", (slug,),
                    ).fetchone()
                    if row is None:
                        continue
                    members.setdefault(slug, str(row[0]))
                    self.connection.execute(
                        "INSERT OR IGNORE INTO membership_source("
                        "source_id, moc_id, concept_id, position) "
                        "VALUES(?, ?, ?, ?)", (source_id, moc_id, row[0], position),
                    )
            payload["concept_slugs"] = list(members)
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.connection.execute(
                "UPDATE moc SET title=?, summary=?, tags_json=?, payload_json=?, "
                "content_hash=?, updated_at=? "
                "WHERE id=? AND payload_json<>?",
                (merged.title, merged.summary,
                 json.dumps(payload["tags"], ensure_ascii=False, separators=(",", ":")),
                 payload_json, _content_hash(payload_json), now, moc_id, payload_json),
            )
            self.connection.execute("DELETE FROM membership WHERE moc_id=?", (moc_id,))
            self.connection.executemany(
                "INSERT INTO membership(moc_id, concept_id, position) VALUES(?, ?, ?)",
                [(moc_id, concept_id, position)
                 for position, concept_id in enumerate(members.values())],
            )
            # MoC vectors are member centroids, keyed by slug (not canonical ID).
            # The pipeline may have persisted a fresh post-change centroid before
            # committing the owners. Keep that vector and its ANN rows intact.
            centroid_hash = _content_hash("\x1f".join(sorted(members)))
            centroids = self.connection.execute(
                "SELECT * FROM embedding WHERE entity_kind='moc' AND entity_id=?",
                (merged.slug,),
            ).fetchall()
            member_inputs_changed = bool(set(members.values()) & (changed_members or set()))
            for centroid in centroids:
                if centroid["input_hash"] == centroid_hash and (
                    not member_inputs_changed
                    or self._moc_centroid_is_current(centroid, members.values())
                ):
                    continue
                for table in ("ann_bucket", "embedding"):
                    self.connection.execute(
                        f"DELETE FROM {table} WHERE entity_kind='moc' "
                        "AND entity_id=? AND model_id=?",
                        (merged.slug, centroid["model_id"]),
                    )

    def _moc_centroid_is_current(self, centroid: sqlite3.Row, member_ids: Iterable[str]) -> bool:
        """A membership hash alone cannot validate changed member definitions.

        Preserve a freshly computed centroid only when current same-model,
        same-version member vectors prove its value; otherwise recompute later.
        """
        schema = str(centroid["schema_version"])
        if not schema.startswith("moc-centroid-v2:"):
            return False
        member_schema = schema.replace("moc-centroid-v2:", "concept-embedding-v2:", 1)
        dimensions = int(centroid["dimensions"])
        vectors = []
        for concept_id in member_ids:
            row = self.connection.execute(
                "SELECT e.vector, e.dimensions, e.input_hash, c.title, c.summary, e.schema_version "
                "FROM embedding e JOIN concept c ON c.id=e.entity_id "
                "WHERE e.entity_kind='concept' AND e.entity_id=? AND e.model_id=?",
                (concept_id, centroid["model_id"]),
            ).fetchone()
            if (row is None or row[1] != dimensions or row[5] != member_schema
                    or row[2] != _content_hash(f"{row[3]}\n{row[4]}")
                    or len(row[0]) != dimensions * 4):
                return False
            vectors.append(struct.unpack(f"<{dimensions}f", row[0]))
        if not vectors or len(centroid["vector"]) != dimensions * 4:
            return False
        actual = struct.unpack(f"<{dimensions}f", centroid["vector"])
        return all(
            math.isclose(value, sum(vector[index] for vector in vectors) / len(vectors),
                         rel_tol=1e-6, abs_tol=1e-7)
            for index, value in enumerate(actual)
        )

    def _rebuild_moc_memberships(self, moc_ids: set[str]) -> None:
        ordered = sorted(moc_ids)
        for offset in range(0, len(ordered), 500):
            chunk = ordered[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            self.connection.execute(
                f"DELETE FROM membership WHERE moc_id IN ({placeholders})", chunk
            )
            self.connection.execute(
                f"""
                INSERT INTO membership(moc_id, concept_id, position)
                SELECT moc_id, concept_id, MIN(position)
                FROM membership_source
                WHERE moc_id IN ({placeholders})
                GROUP BY moc_id, concept_id
                """,
                chunk,
            )

    def _cleanup_unreferenced_entities(
        self,
        *,
        concept_ids: set[str] | None = None,
        moc_ids: set[str] | None = None,
    ) -> None:
        """Remove canonical rows that no live synthesis still references."""
        if concept_ids is not None or moc_ids is not None:
            for moc_id in sorted(moc_ids or set()):
                if self.connection.execute(
                    "SELECT 1 FROM moc_source WHERE moc_id=? LIMIT 1", (moc_id,)
                ).fetchone() is not None:
                    continue
                row = self.connection.execute(
                    "SELECT slug FROM moc WHERE id=?", (moc_id,)
                ).fetchone()
                if row is not None:
                    moc_slug = str(row[0])
                    self.connection.execute(
                        "DELETE FROM ann_bucket WHERE entity_kind='moc' AND entity_id=?",
                        (moc_slug,),
                    )
                    self.connection.execute(
                        "DELETE FROM embedding WHERE entity_kind='moc' AND entity_id=?",
                        (moc_slug,),
                    )
                    self.connection.execute("DELETE FROM moc WHERE id=?", (moc_id,))
            for concept_id in sorted(concept_ids or set()):
                if self.connection.execute(
                    "SELECT 1 FROM concept_source WHERE concept_id=? LIMIT 1",
                    (concept_id,),
                ).fetchone() is not None:
                    continue
                self.connection.execute(
                    "DELETE FROM ann_bucket WHERE entity_kind='concept' AND entity_id=?",
                    (concept_id,),
                )
                self.connection.execute(
                    "DELETE FROM embedding WHERE entity_kind='concept' AND entity_id=?",
                    (concept_id,),
                )
                self.connection.execute(
                    "DELETE FROM concept_fts WHERE rowid=(SELECT fts_rowid "
                    "FROM concept_fts_identity WHERE concept_id=?)", (concept_id,)
                )
                self.connection.execute("DELETE FROM concept WHERE id=?", (concept_id,))
            return

        stale_mocs = "SELECT id FROM moc WHERE id NOT IN (SELECT moc_id FROM moc_source)"
        stale_moc_slugs = (
            "SELECT slug FROM moc WHERE id NOT IN (SELECT moc_id FROM moc_source)"
        )
        self.connection.execute(
            f"DELETE FROM ann_bucket WHERE entity_kind='moc' "
            f"AND entity_id IN ({stale_moc_slugs})"
        )
        self.connection.execute(
            f"DELETE FROM embedding WHERE entity_kind='moc' "
            f"AND entity_id IN ({stale_moc_slugs})"
        )
        self.connection.execute(f"DELETE FROM moc WHERE id IN ({stale_mocs})")

        stale_concepts = (
            "SELECT id FROM concept "
            "WHERE id NOT IN (SELECT concept_id FROM concept_source)"
        )
        self.connection.execute(
            f"DELETE FROM ann_bucket WHERE entity_kind='concept' "
            f"AND entity_id IN ({stale_concepts})"
        )
        self.connection.execute(
            f"DELETE FROM embedding WHERE entity_kind='concept' "
            f"AND entity_id IN ({stale_concepts})"
        )
        self.connection.execute(
            "DELETE FROM concept_fts WHERE rowid IN (SELECT fts_rowid "
            f"FROM concept_fts_identity WHERE concept_id IN ({stale_concepts}))"
        )
        self.connection.execute(
            f"DELETE FROM concept WHERE id IN ({stale_concepts})"
        )

    def load_synthesis(self, filename: str) -> SourceSynthesis | None:
        """Load one synthesis by indexed source filename."""
        row = self.connection.execute(
            """
            SELECT synthesis.payload_json
            FROM synthesis
            JOIN source ON source.id = synthesis.source_id
            WHERE source.filename = ? AND synthesis.status = 'ready'
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        synthesis = source_synthesis_from_dict(payload)
        synthesis.source_file = filename
        return synthesis

    def load_syntheses(
        self, filenames: set[str] | None = None
    ) -> dict[str, SourceSynthesis]:
        """Load ready syntheses from SQLite, optionally for an indexed subset."""
        if filenames is not None and not filenames:
            return {}
        if filenames is None:
            rows = self.connection.execute(
                """
                SELECT source.filename, synthesis.payload_json
                FROM synthesis JOIN source ON source.id=synthesis.source_id
                WHERE synthesis.status='ready' ORDER BY source.filename
                """
            ).fetchall()
        else:
            rows = []
            ordered = sorted(filenames)
            for offset in range(0, len(ordered), 500):
                chunk = ordered[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    self.connection.execute(
                        f"""
                        SELECT source.filename, synthesis.payload_json
                        FROM synthesis JOIN source ON source.id=synthesis.source_id
                        WHERE synthesis.status='ready'
                          AND source.filename IN ({placeholders})
                        ORDER BY source.filename
                        """,
                        chunk,
                    ).fetchall()
                )
        syntheses: dict[str, SourceSynthesis] = {}
        for row in rows:
            try:
                payload = json.loads(row[1])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            filename = str(row[0])
            synthesis = source_synthesis_from_dict(payload)
            synthesis.source_file = filename
            syntheses[filename] = synthesis
        return syntheses

    def source_filenames(self) -> set[str]:
        """Return active source identities without scanning source Markdown."""
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT filename FROM source WHERE status='active'"
            )
        }

    def source_content_hashes(self) -> dict[str, str]:
        """Return authoritative source hashes for explicit full audits."""
        return {
            str(row[0]): str(row[1])
            for row in self.connection.execute(
                "SELECT filename, content_hash FROM source WHERE status='active'"
            )
        }

    def rendered_paths(self) -> set[str]:
        """Return renderer-owned paths for explicit stale-artifact audits."""
        return {
            str(row[0])
            for row in self.connection.execute("SELECT path FROM render_state")
        }

    def source_file_states(self) -> dict[str, tuple[int, int, SourceDoc]]:
        """Return indexed file signatures and metadata-only source stubs."""
        rows = self.connection.execute(
            """
            SELECT source.filename, source.title, source.canonical_url,
                   source_file_state.mtime_ns, source_file_state.size_bytes,
                   COALESCE((
                       SELECT source_version.metadata_json
                       FROM source_version
                       WHERE source_version.source_id=source.id
                         AND source_version.content_hash=source.content_hash
                         AND source_version.extractor_version=source.extractor_version
                       LIMIT 1
                   ), '{}') AS metadata_json
            FROM source
            JOIN source_file_state ON source_file_state.source_id=source.id
            WHERE source.status='active'
            ORDER BY source.filename
            """
        ).fetchall()
        states: dict[str, tuple[int, int, SourceDoc]] = {}
        for row in rows:
            filename = str(row["filename"])
            try:
                metadata = json.loads(str(row["metadata_json"]))
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            states[filename] = (
                int(row["mtime_ns"]),
                int(row["size_bytes"]),
                _source_stub_from_metadata(
                    filename=filename,
                    title=str(row["title"]),
                    canonical_url=str(row["canonical_url"]),
                    metadata=metadata if isinstance(metadata, dict) else {},
                ),
            )
        return states

    def record_source_file_state(
        self,
        filename: str,
        *,
        mtime_ns: int,
        size_bytes: int,
    ) -> bool:
        """Commit a verified source-file signature after pipeline success."""
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO source_file_state(source_id, mtime_ns, size_bytes, updated_at)
                SELECT id, ?, ?, ? FROM source WHERE filename=? AND status='active'
                ON CONFLICT(source_id) DO UPDATE SET
                    mtime_ns=excluded.mtime_ns,
                    size_bytes=excluded.size_bytes,
                    updated_at=excluded.updated_at
                """,
                (int(mtime_ns), int(size_bytes), _now(), filename),
            )
        return cursor.rowcount > 0

    def refresh_source_metadata(
        self,
        filename: str,
        source: SourceDoc,
        versions: ArtifactVersions,
    ) -> None:
        """Refresh provenance when the body hash and synthesis stay current."""
        metadata = json.dumps(
            _source_metadata_payload(source),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        content_hash = _content_hash(source.content)
        with self.connection:
            self._set_meta_unlocked("pipeline_checkpoint", "")
            self.connection.execute(
                """
                UPDATE source SET title=?, canonical_url=?, updated_at=?
                WHERE filename=? AND content_hash=? AND status='active'
                """,
                (source.title, source.url or "", _now(), filename, content_hash),
            )
            self.connection.execute(
                """
                UPDATE source_version SET
                    requested_url=?, canonical_url=?, metadata_json=?
                WHERE source_id=(SELECT id FROM source WHERE filename=?)
                  AND content_hash=? AND extractor_version=?
                """,
                (
                    source.provenance.requested_url,
                    source.url or "",
                    metadata,
                    filename,
                    content_hash,
                    versions.extractor,
                ),
            )

    def synthesis_versions_current(
        self,
        filename: str,
        versions: ArtifactVersions,
    ) -> bool:
        """Check non-content versions when an indexed file signature is unchanged."""
        row = self.connection.execute(
            """
            SELECT source.extractor_version, synthesis.prompt_version,
                   synthesis.schema_version, synthesis.model_id
            FROM synthesis
            JOIN source ON source.id=synthesis.source_id
            WHERE source.filename=? AND synthesis.status='ready'
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return False
        expected = (
            versions.extractor,
            versions.prompt,
            versions.schema,
            versions.synthesis_model,
        )
        actual = tuple(row)
        if actual == expected:
            return True
        reasons = (
            "extractor_version_changed",
            "prompt_version_changed",
            "schema_version_changed",
            "synthesis_model_changed",
        )
        reason = next(
            reason
            for reason, current, wanted in zip(reasons, actual, expected, strict=True)
            if current != wanted
        )
        self.enqueue_dirty("source", filename, reason)
        return False

    def delete_source(self, filename: str) -> None:
        """Delete one source and its versioned synthesis rows transactionally."""
        with self.connection:
            self._set_meta_unlocked("pipeline_checkpoint", "")
            moc_ids = {
                str(row[0])
                for row in self.connection.execute(
                    """
                    SELECT ms.moc_id
                    FROM moc_source AS ms
                    JOIN source AS s ON s.id=ms.source_id
                    WHERE s.filename=?
                    """,
                    (filename,),
                )
            }
            concept_ids = {
                str(row[0])
                for row in self.connection.execute(
                    """
                    SELECT cs.concept_id
                    FROM concept_source AS cs
                    JOIN source AS s ON s.id=cs.source_id
                    WHERE s.filename=?
                    """,
                    (filename,),
                )
            }
            moc_ids.update(self._mocs_referencing_concepts(concept_ids))
            previous_member_inputs = self._moc_member_inputs(concept_ids)
            self.connection.execute("DELETE FROM source WHERE filename=?", (filename,))
            self._cleanup_unreferenced_entities(
                concept_ids=concept_ids,
                moc_ids=moc_ids,
            )
            now = _now()
            self._rebuild_concept_aggregates(concept_ids, None, now)
            current_member_inputs = self._moc_member_inputs(concept_ids)
            changed_members = {
                cid for cid in concept_ids
                if previous_member_inputs.get(cid) != current_member_inputs.get(cid)
            }
            self._rebuild_moc_aggregates(moc_ids, now, changed_members)

    def render_diff(
        self,
        entity_kind: str,
        manifest: dict[str, tuple[str, str]],
    ) -> RenderDiff:
        """Compare entity render fingerprints without reading generated files."""
        rows = self.connection.execute(
            "SELECT entity_key, path, content_hash FROM render_state WHERE entity_kind=?",
            (entity_kind,),
        ).fetchall()
        existing = {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
        changed = {
            key
            for key, value in manifest.items()
            if existing.get(key) != value
        }
        stale_keys = set(existing) - set(manifest)
        return RenderDiff(
            changed_keys=frozenset(changed),
            stale_paths=tuple(existing[key][0] for key in sorted(stale_keys)),
            stale_keys=frozenset(stale_keys),
        )

    def render_manifest(self, entity_kind: str) -> dict[str, tuple[str, str]]:
        """Return one indexed render manifest without reading generated pages."""
        return {
            str(row["entity_key"]): (str(row["path"]), str(row["content_hash"]))
            for row in self.connection.execute(
                """
                SELECT entity_key, path, content_hash FROM render_state
                WHERE entity_kind=?
                """,
                (entity_kind,),
            )
        }

    def commit_render_manifest(
        self,
        entity_kind: str,
        manifest: dict[str, tuple[str, str]],
        diff: RenderDiff,
    ) -> None:
        """Commit one render manifest atomically after render success."""
        self.commit_render_manifests(((entity_kind, manifest, diff),))

    def commit_render_manifests(
        self,
        manifests: Iterable[
            tuple[str, dict[str, tuple[str, str]], RenderDiff]
        ],
    ) -> None:
        """Commit every render manifest in one SQLite transaction.

        The filesystem renderer rolls back all touched paths when any late
        render step fails.  Render-state must use the same all-or-nothing
        boundary; otherwise an early manifest can be committed before a later
        manifest fails, causing the retry to skip a file that was rolled back.
        """
        changed = False
        with self.connection:
            for entity_kind, manifest, diff in manifests:
                self._commit_render_manifest_unlocked(entity_kind, manifest, diff)
                changed = changed or bool(diff.changed_keys or diff.stale_keys)
            if changed:
                self.connection.execute(
                    """
                    INSERT INTO meta(key, value)
                    VALUES('maintenance_snapshot_complete', '0')
                    ON CONFLICT(key) DO UPDATE SET value='0'
                    """
                )

    def _commit_render_manifest_unlocked(
        self,
        entity_kind: str,
        manifest: dict[str, tuple[str, str]],
        diff: RenderDiff,
    ) -> None:
        """Apply one manifest while the caller owns the transaction."""
        now = _now()
        for key in diff.changed_keys:
            path, content_hash = manifest[key]
            self.connection.execute(
                """
                INSERT INTO render_state(
                    entity_kind, entity_key, path, content_hash, updated_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(entity_kind, entity_key) DO UPDATE SET
                    path=excluded.path,
                    content_hash=excluded.content_hash,
                    updated_at=excluded.updated_at
                """,
                (entity_kind, key, path, content_hash, now),
            )
        if diff.stale_keys:
            self.connection.executemany(
                "DELETE FROM render_state WHERE entity_kind=? AND entity_key=?",
                [(entity_kind, key) for key in diff.stale_keys],
            )
        if diff.stale_paths:
            self.connection.executemany(
                "DELETE FROM health_finding WHERE path=?",
                [(path,) for path in diff.stale_paths],
            )

    def next_maintenance_batch(self, *, limit: int) -> tuple[list[str], bool]:
        """Return a bounded, persistent round-robin batch of rendered pages."""
        if limit < 1:
            raise ValueError("maintenance batch limit must be positive")
        cursor_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='maintenance_cursor'"
        ).fetchone()
        cursor = str(cursor_row[0]) if cursor_row is not None else ""
        rows = self.connection.execute(
            """
            SELECT path FROM render_state
            WHERE path > ?
            ORDER BY path
            LIMIT ?
            """,
            (cursor, limit + 1),
        ).fetchall()
        cycle_complete = len(rows) <= limit
        paths = [str(row[0]) for row in rows[:limit]]
        return paths, cycle_complete

    def commit_maintenance_batch(
        self,
        scanned_paths: Iterable[str],
        findings: Iterable[dict[str, object]],
        *,
        cycle_complete: bool,
    ) -> None:
        """Persist findings and advance their cursor in one transaction."""
        paths = tuple(dict.fromkeys(str(path) for path in scanned_paths if path))
        rows = tuple(findings)
        next_cursor = "" if cycle_complete or not paths else paths[-1]
        with self.connection:
            if paths:
                self.connection.executemany(
                    "DELETE FROM health_finding WHERE path=?",
                    ((path,) for path in paths),
                )
            self._insert_health_findings_unlocked(rows, now=_now())
            self._set_meta_unlocked("maintenance_cursor", next_cursor)
            if cycle_complete:
                self._set_meta_unlocked("maintenance_snapshot_complete", "1")

    def maintenance_graph_sets(self) -> tuple[set[str], set[str]]:
        """Return current concept and MoC-membership slugs from indexed state."""
        concepts = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT entity_key FROM render_state WHERE entity_kind='concept-page'"
            )
        }
        moced = {
            str(row[0])
            for row in self.connection.execute(
                """
                SELECT c.slug
                FROM membership AS ms
                JOIN concept AS c ON c.id=ms.concept_id
                JOIN moc AS m ON m.id=ms.moc_id
                JOIN render_state AS rs
                  ON rs.entity_kind='moc-page' AND rs.entity_key=m.slug
                """
            )
        }
        return concepts, moced

    def maintenance_page_sets(
        self,
    ) -> tuple[set[str], set[str], set[str], set[str], dict[str, int]]:
        """Return indexed page/link sets needed by bounded health batches."""
        rows = self.connection.execute(
            "SELECT entity_kind, path FROM render_state"
        ).fetchall()
        all_stems = {Path(str(row[1])).stem for row in rows}
        all_stems.update({"index", "log"})
        concept_stems = {
            Path(str(row[1])).stem for row in rows if str(row[0]) == "concept-page"
        }
        source_stems = {
            Path(str(row[1])).stem for row in rows if str(row[0]) == "source-page"
        }
        _, moced_slugs = self.maintenance_graph_sets()
        moc_counts = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                """
                SELECT moc.slug, COUNT(DISTINCT membership.concept_id)
                FROM moc
                LEFT JOIN membership ON membership.moc_id=moc.id
                GROUP BY moc.id, moc.slug
                """
            )
        }
        return all_stems, concept_stems, source_stems, moced_slugs, moc_counts

    def replace_health_findings(
        self,
        scanned_paths: Iterable[str],
        findings: Iterable[dict[str, object]],
    ) -> None:
        """Replace findings for one maintenance batch without losing peers."""
        paths = tuple(dict.fromkeys(str(path) for path in scanned_paths if path))
        rows = tuple(findings)
        with self.connection:
            if paths:
                self.connection.executemany(
                    "DELETE FROM health_finding WHERE path=?",
                    ((path,) for path in paths),
                )
            self._insert_health_findings_unlocked(rows, now=_now())

    def replace_all_health_findings(
        self,
        findings: Iterable[dict[str, object]],
    ) -> None:
        """Replace the complete indexed health snapshot after a full audit."""
        rows = tuple(findings)
        with self.connection:
            self.connection.execute("DELETE FROM health_finding")
            self._insert_health_findings_unlocked(rows, now=_now())
            self._set_meta_unlocked("maintenance_snapshot_complete", "1")

    def _insert_health_findings_unlocked(
        self,
        findings: Iterable[dict[str, object]],
        *,
        now: str,
    ) -> None:
        self.connection.executemany(
            """
            INSERT INTO health_finding(
                path, kind, severity, message, details_json, reviewed, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path, kind, message) DO UPDATE SET
                severity=excluded.severity,
                details_json=excluded.details_json,
                reviewed=excluded.reviewed,
                updated_at=excluded.updated_at
            """,
            (
                (
                    str(finding.get("path", "")),
                    str(finding.get("kind", "")),
                    str(finding.get("severity", "warning")),
                    str(finding.get("message", "")),
                    json.dumps(
                        finding.get("details", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    int(bool(finding.get("reviewed", False))),
                    now,
                )
                for finding in findings
                if finding.get("path") and finding.get("kind") and finding.get("message")
            ),
        )

    def health_findings(self) -> list[dict[str, object]]:
        """Return the durable whole-vault finding snapshot in stable order."""
        rows = self.connection.execute(
            """
            SELECT path, kind, severity, message, details_json, reviewed
            FROM health_finding
            ORDER BY path, kind, message
            """
        ).fetchall()
        findings: list[dict[str, object]] = []
        for row in rows:
            try:
                details = json.loads(str(row["details_json"]))
            except (TypeError, json.JSONDecodeError):
                details = {}
            findings.append(
                {
                    "path": str(row["path"]),
                    "kind": str(row["kind"]),
                    "severity": str(row["severity"]),
                    "message": str(row["message"]),
                    "details": details if isinstance(details, dict) else {},
                    "reviewed": bool(row["reviewed"]),
                }
            )
        return findings

    def mark_maintenance_snapshot_complete(self) -> None:
        with self.connection:
            self._set_meta_unlocked("maintenance_snapshot_complete", "1")

    def maintenance_snapshot_complete(self) -> bool:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='maintenance_snapshot_complete'"
        ).fetchone()
        return bool(row and str(row["value"]) == "1")

    def get_pipeline_checkpoint(self) -> str:
        """Return the last successful pipeline/config fingerprint, or no checkpoint."""
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='pipeline_checkpoint'"
        ).fetchone()
        return str(row[0]) if row is not None else ""

    def set_pipeline_checkpoint(self, value: str) -> None:
        """Publish only after render/state/signatures commit under the compile lock.

        Source mutations invalidate this marker in their own transactions. The
        caller remains responsible for matching file signatures and versions.
        """
        with self.connection:
            self._set_meta_unlocked("pipeline_checkpoint", value)

    def _set_meta_unlocked(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )

    def search_concepts(self, query: str, *, limit: int) -> list[ConceptCandidate]:
        """Return a bounded FTS5 candidate set without scanning concept rows."""
        if limit <= 0:
            return []
        tokens = list(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(r"[^\W_]{2,}", query, flags=re.UNICODE)
            )
        )[:32]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        rows = self.connection.execute(
            """
            SELECT concept.id, concept.slug, concept.title, concept.summary,
                   COALESCE((
                       SELECT group_concat(alias, char(31))
                       FROM concept_alias WHERE concept_id=concept.id
                   ), ''),
                   bm25(concept_fts)
            FROM concept_fts
            JOIN concept ON concept.id=concept_fts.concept_id
            WHERE concept_fts MATCH ?
            ORDER BY bm25(concept_fts), concept.slug
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [
            ConceptCandidate(
                id=str(row[0]),
                slug=str(row[1]),
                title=str(row[2]),
                summary=str(row[3]),
                aliases=tuple(alias for alias in str(row[4]).split("\x1f") if alias),
                score=-float(row[5]),
            )
            for row in rows
        ]

    def concept_candidates_by_id(
        self, entity_ids: list[str]
    ) -> list[ConceptCandidate]:
        """Load a bounded ANN result set without scanning canonical concepts."""
        if not entity_ids:
            return []
        result_by_id: dict[str, ConceptCandidate] = {}
        for offset in range(0, len(entity_ids), 500):
            chunk = entity_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT id, slug, title, summary,
                       COALESCE((
                           SELECT group_concat(alias, char(31))
                           FROM concept_alias WHERE concept_id=concept.id
                       ), '')
                FROM concept WHERE id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                result_by_id[str(row[0])] = ConceptCandidate(
                    id=str(row[0]),
                    slug=str(row[1]),
                    title=str(row[2]),
                    summary=str(row[3]),
                    aliases=tuple(alias for alias in str(row[4]).split("\x1f") if alias),
                    score=0.0,
                )
        return [result_by_id[entity_id] for entity_id in entity_ids if entity_id in result_by_id]

    def upsert_embedding(
        self,
        entity_kind: str,
        entity_id: str,
        vector: list[float],
        *,
        model_id: str,
        input_hash: str,
        schema_version: str,
    ) -> None:
        """Persist one float32 vector and replace its ANN bucket entries."""
        self.upsert_embeddings(
            entity_kind,
            [(entity_id, vector, input_hash)],
            model_id=model_id,
            schema_version=schema_version,
            batch_size=1,
        )

    def upsert_embeddings(
        self,
        entity_kind: str,
        records: Iterable[tuple[str, list[float], str]],
        *,
        model_id: str,
        schema_version: str,
        batch_size: int = 256,
    ) -> int:
        """Persist vectors in bounded transactions instead of one commit per row."""
        if batch_size < 1:
            raise ValueError("embedding batch_size must be positive")
        iterator = iter(records)
        written = 0
        while batch := list(islice(iterator, batch_size)):
            now = _now()
            embedding_rows: list[tuple[object, ...]] = []
            bucket_rows: list[tuple[object, ...]] = []
            delete_rows: list[tuple[str, str, str]] = []
            for entity_id, vector, input_hash in batch:
                canonical = _canonical_vector(vector)
                dimensions = len(canonical)
                embedding_rows.append(
                    (
                        entity_kind,
                        entity_id,
                        model_id,
                        dimensions,
                        input_hash,
                        schema_version,
                        struct.pack(f"<{dimensions}f", *canonical),
                        now,
                    )
                )
                delete_rows.append((entity_kind, entity_id, model_id))
                bucket_rows.extend(
                    (
                        entity_kind,
                        model_id,
                        dimensions,
                        table_id,
                        _lsh_bucket(canonical, model_id, table_id),
                        entity_id,
                    )
                    for table_id in range(_LSH_TABLES)
                )
            with self.connection:
                self.connection.executemany(
                    """
                    INSERT INTO embedding(
                        entity_kind, entity_id, model_id, dimensions, input_hash,
                        schema_version, vector, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_kind, entity_id, model_id) DO UPDATE SET
                        dimensions=excluded.dimensions,
                        input_hash=excluded.input_hash,
                        schema_version=excluded.schema_version,
                        vector=excluded.vector,
                        updated_at=excluded.updated_at
                    """,
                    embedding_rows,
                )
                self.connection.executemany(
                    """
                    DELETE FROM ann_bucket
                    WHERE entity_kind=? AND entity_id=? AND model_id=?
                    """,
                    delete_rows,
                )
                self.connection.executemany(
                    """
                    INSERT INTO ann_bucket(
                        entity_kind, model_id, dimensions, table_id, bucket, entity_id
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    bucket_rows,
                )
            written += len(batch)
        return written

    def load_embedding(
        self,
        entity_kind: str,
        entity_id: str,
        *,
        model_id: str,
    ) -> list[float] | None:
        """Load one authoritative float32 vector by its versioned key."""
        row = self.connection.execute(
            """
            SELECT dimensions, vector FROM embedding
            WHERE entity_kind=? AND entity_id=? AND model_id=?
            """,
            (entity_kind, entity_id, model_id),
        ).fetchone()
        if row is None:
            return None
        dimensions = int(row[0])
        raw = bytes(row[1])
        if dimensions <= 0 or len(raw) != dimensions * 4:
            return None
        return list(struct.unpack(f"<{dimensions}f", raw))

    def load_embeddings(
        self,
        entity_kind: str,
        entity_ids: Iterable[str],
        *,
        model_id: str,
    ) -> dict[str, list[float]]:
        """Load a bounded ID subset without materializing the vector table."""
        result: dict[str, list[float]] = {}
        ordered = list(dict.fromkeys(entity_ids))
        for offset in range(0, len(ordered), 500):
            chunk = ordered[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT entity_id, dimensions, vector FROM embedding
                WHERE entity_kind=? AND model_id=?
                  AND entity_id IN ({placeholders})
                """,
                [entity_kind, model_id, *chunk],
            ).fetchall()
            for row in rows:
                dimensions = int(row[1])
                raw = bytes(row[2])
                if dimensions > 0 and len(raw) == dimensions * 4:
                    result[str(row[0])] = list(
                        struct.unpack(f"<{dimensions}f", raw)
                    )
        return result

    def has_embeddings(
        self,
        entity_kind: str,
        *,
        model_id: str,
        schema_version: str | None = None,
    ) -> bool:
        """Check whether a versioned ANN namespace has any persisted vectors."""
        if schema_version is None:
            row = self.connection.execute(
                """
                SELECT 1 FROM embedding
                WHERE entity_kind=? AND model_id=? LIMIT 1
                """,
                (entity_kind, model_id),
            ).fetchone()
        else:
            row = self.connection.execute(
                """
                SELECT 1 FROM embedding
                WHERE entity_kind=? AND model_id=? AND schema_version=? LIMIT 1
                """,
                (entity_kind, model_id, schema_version),
            ).fetchone()
        return row is not None

    def embedding_count(
        self,
        entity_kind: str,
        *,
        model_id: str,
        schema_version: str,
    ) -> int:
        """Count current vectors without loading float blobs into memory."""
        row = self.connection.execute(
            """
            SELECT count(*) FROM embedding
            WHERE entity_kind=? AND model_id=? AND schema_version=?
            """,
            (entity_kind, model_id, schema_version),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def embedding_input_hashes(
        self,
        entity_kind: str,
        *,
        model_id: str,
        schema_version: str,
        entity_ids: Iterable[str] | None = None,
    ) -> dict[str, str]:
        """Load only vector metadata needed to resume partial indexing."""
        if entity_ids is None:
            rows = self.connection.execute(
                """
                SELECT entity_id, input_hash FROM embedding
                WHERE entity_kind=? AND model_id=? AND schema_version=?
                """,
                (entity_kind, model_id, schema_version),
            ).fetchall()
            return {str(row[0]): str(row[1]) for row in rows}

        ordered = sorted(set(entity_ids))
        result: dict[str, str] = {}
        for offset in range(0, len(ordered), 500):
            chunk = ordered[offset:offset + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"""
                SELECT entity_id, input_hash FROM embedding
                WHERE entity_kind=? AND model_id=? AND schema_version=?
                  AND entity_id IN ({placeholders})
                """,
                (entity_kind, model_id, schema_version, *chunk),
            ).fetchall()
            result.update((str(row[0]), str(row[1])) for row in rows)
        return result

    @staticmethod
    def _semantic_namespace_key(model_id: str, schema_version: str) -> str:
        digest = hashlib.sha256(f"{model_id}\x00{schema_version}".encode()).hexdigest()
        return f"cross_lingual_complete:{digest}"

    def cross_lingual_namespace_complete(
        self,
        *,
        model_id: str,
        schema_version: str,
    ) -> bool:
        key = self._semantic_namespace_key(model_id, schema_version)
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?",
            (key,),
        ).fetchone()
        return row is not None and str(row[0]) == "1"

    def replace_cross_lingual_links(
        self,
        source_slugs: Iterable[str],
        links: dict[str, list[tuple[str, float, str]]],
        *,
        model_id: str,
        schema_version: str,
        namespace_complete: bool = False,
    ) -> None:
        """Replace directional links for a bounded set of comparison slugs."""
        slugs = sorted(set(source_slugs))
        now = _now()
        with self.connection:
            for offset in range(0, len(slugs), 500):
                chunk = slugs[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                self.connection.execute(
                    f"""
                    DELETE FROM cross_lingual_link
                    WHERE model_id=? AND schema_version=?
                      AND source_slug IN ({placeholders})
                    """,
                    (model_id, schema_version, *chunk),
                )
            self.connection.executemany(
                """
                INSERT INTO cross_lingual_link(
                    source_slug, target_slug, score, display,
                    model_id, schema_version, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_slug, target_slug, model_id, schema_version)
                DO UPDATE SET
                    score=excluded.score,
                    display=excluded.display,
                    updated_at=excluded.updated_at
                """,
                (
                    (source, target, float(score), display, model_id, schema_version, now)
                    for source in slugs
                    for target, score, display in links.get(source, ())
                    if source != target
                ),
            )
            if namespace_complete:
                self._set_meta_unlocked(
                    self._semantic_namespace_key(model_id, schema_version),
                    "1",
                )

    def load_cross_lingual_links(
        self,
        *,
        model_id: str,
        schema_version: str,
        live_slugs: set[str],
    ) -> dict[str, list[tuple[str, float, str]]]:
        """Load a durable whole-vault semantic-link snapshot."""
        links: dict[str, list[tuple[str, float, str]]] = {}
        rows = self.connection.execute(
            """
            SELECT source_slug, target_slug, score, display
            FROM cross_lingual_link
            WHERE model_id=? AND schema_version=?
            ORDER BY source_slug, score DESC, target_slug
            """,
            (model_id, schema_version),
        ).fetchall()
        for row in rows:
            source = str(row[0])
            target = str(row[1])
            if source not in live_slugs or target not in live_slugs:
                continue
            links.setdefault(source, []).append((target, float(row[2]), str(row[3])))
        return links

    def prune_cross_lingual_links(self) -> None:
        """Remove links whose canonical concept no longer exists."""
        with self.connection:
            self.connection.execute(
                """
                DELETE FROM cross_lingual_link
                WHERE source_slug NOT IN (SELECT slug FROM concept)
                   OR target_slug NOT IN (SELECT slug FROM concept)
                """
            )

    def search_ann(
        self,
        entity_kind: str,
        vector: list[float],
        *,
        model_id: str,
        limit: int,
        candidate_pool: int = 1_000,
        schema_version: str | None = None,
    ) -> AnnSearchResult:
        """Search bounded LSH buckets, then exact-rerank only that pool."""
        if limit <= 0 or candidate_pool <= 0:
            return AnnSearchResult([], 0, 0)
        canonical = _canonical_vector(vector)
        dimensions = len(canonical)
        collision_counts: dict[str, int] = {}
        bucket_hits = 0
        for table_id in range(_LSH_TABLES):
            bucket = _lsh_bucket(canonical, model_id, table_id)
            probes = [bucket, *(bucket ^ (1 << bit) for bit in range(_LSH_BITS))]
            placeholders = ",".join("?" for _ in probes)
            rows = self.connection.execute(
                f"""
                SELECT entity_id FROM ann_bucket
                WHERE entity_kind=? AND model_id=? AND dimensions=? AND table_id=?
                  AND bucket IN ({placeholders})
                ORDER BY rowid
                LIMIT ?
                """,
                (entity_kind, model_id, dimensions, table_id, *probes, candidate_pool),
            ).fetchall()
            bucket_hits += len(rows)
            for row in rows:
                entity_id = str(row[0])
                collision_counts[entity_id] = collision_counts.get(entity_id, 0) + 1

        selected_ids = [
            entity_id
            for entity_id, _hits in sorted(
                collision_counts.items(), key=lambda item: (-item[1], item[0])
            )[:candidate_pool]
        ]
        scored: list[AnnCandidate] = []
        for offset in range(0, len(selected_ids), 500):
            chunk = selected_ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            schema_filter = "" if schema_version is None else " AND schema_version=?"
            params: list[object] = [entity_kind, model_id, dimensions]
            if schema_version is not None:
                params.append(schema_version)
            params.extend(chunk)
            rows = self.connection.execute(
                f"""
                SELECT entity_id, vector FROM embedding
                WHERE entity_kind=? AND model_id=? AND dimensions=?
                  {schema_filter}
                  AND entity_id IN ({placeholders})
                """,
                params,
            ).fetchall()
            for row in rows:
                candidate = list(struct.unpack(f"<{dimensions}f", bytes(row[1])))
                scored.append(AnnCandidate(str(row[0]), _cosine(canonical, candidate)))
        scored.sort(key=lambda item: (-item.score, item.entity_id))
        return AnnSearchResult(scored[:limit], len(scored), bucket_hits)

    def migrate_legacy(self, versions: ArtifactVersions) -> bool:
        """Import legacy JSON state/cache exactly once.

        The migration marker is written last. Partial imports after an external
        interruption are safe to retry because every row uses deterministic
        keys and upsert semantics.
        """
        marker = self.connection.execute(
            "SELECT value FROM meta WHERE key='legacy_migration_complete'"
        ).fetchone()
        if marker is not None and str(marker[0]) == "1":
            return False

        state_hashes: dict[str, str] = {}
        state_path = self.llmwiki_dir / "state.json"
        try:
            raw_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_state = {}
        raw_sources = raw_state.get("sources", {}) if isinstance(raw_state, dict) else {}
        if isinstance(raw_sources, dict):
            for filename, raw in raw_sources.items():
                if isinstance(filename, str) and isinstance(raw, dict):
                    value = raw.get("hash", "")
                    if isinstance(value, str):
                        state_hashes[filename] = value

        cache_dir = self.llmwiki_dir / "cache"
        if cache_dir.is_dir():
            for path in sorted(cache_dir.glob("*.json")):
                if path.name == "_resynthesis_overlay.json":
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("_artifact"), dict) and isinstance(
                    payload.get("payload"), dict
                ):
                    payload = payload["payload"]
                try:
                    synthesis = source_synthesis_from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    continue
                filename = synthesis.source_file or path.name.removesuffix(".json")
                synthesis.source_file = filename
                source = SourceDoc(
                    title=synthesis.source_title or Path(filename).stem,
                    content=synthesis.source_summary,
                    source_file=filename,
                )
                self.upsert_synthesis(filename, source, synthesis, versions)
                legacy_hash = state_hashes.get(filename)
                if legacy_hash:
                    with self.connection:
                        self.connection.execute(
                            "UPDATE source SET content_hash=? WHERE filename=?",
                            (legacy_hash, filename),
                        )
                        self.connection.execute(
                            """
                            UPDATE synthesis SET input_hash=?
                            WHERE source_id=(SELECT id FROM source WHERE filename=?)
                            """,
                            (legacy_hash, filename),
                        )

        now = _now()
        with self.connection:
            for filename, legacy_hash in state_hashes.items():
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO source(
                        filename, content_hash, extractor_version, status,
                        created_at, updated_at
                    ) VALUES(?, ?, ?, 'active', ?, ?)
                    """,
                    (filename, legacy_hash, versions.extractor, now, now),
                )
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES('legacy_migration_complete', '1')
                ON CONFLICT(key) DO UPDATE SET value='1'
                """
            )
        return True

    def synthesis_is_current(
        self,
        filename: str,
        content: str,
        versions: ArtifactVersions,
    ) -> bool:
        """Check content and version compatibility, recording invalidation."""
        row = self.connection.execute(
            """
            SELECT synthesis.input_hash, source.extractor_version,
                   synthesis.prompt_version, synthesis.schema_version,
                   synthesis.model_id
            FROM synthesis
            JOIN source ON source.id = synthesis.source_id
            WHERE source.filename = ? AND synthesis.status = 'ready'
            """,
            (filename,),
        ).fetchone()
        if row is None:
            return False
        expected = (
            _content_hash(content),
            versions.extractor,
            versions.prompt,
            versions.schema,
            versions.synthesis_model,
        )
        actual = tuple(row)
        if actual == expected:
            return True
        reasons = (
            "content_hash_changed",
            "extractor_version_changed",
            "prompt_version_changed",
            "schema_version_changed",
            "synthesis_model_changed",
        )
        reason = next(
            reason
            for reason, current, wanted in zip(reasons, actual, expected, strict=True)
            if current != wanted
        )
        self.enqueue_dirty("source", filename, reason)
        return False

    def enqueue_dirty(
        self,
        entity_type: str,
        entity_id: str,
        reason: str,
        payload: dict[str, object] | None = None,
        *,
        next_attempt_at: str | None = None,
        reset_retry: bool = False,
    ) -> int:
        """Coalesce an invalidation without erasing an existing retry schedule.

        A running payload is immutable. Invalidations during its lease coalesce
        into a durable successor slot, promoted when that claim is retired.
        ``next_attempt_at`` schedules new work. Only ``reset_retry=True`` resets
        an existing retry's due time, attempts and error. It cannot steal a
        running claim; there it resets only the successor's due time.
        """
        now = _now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO dirty_queue(
                    entity_type, entity_id, reason, payload_json, state,
                    next_attempt_at, created_at, updated_at
                ) VALUES(?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(entity_type, entity_id, reason)
                    WHERE state IN ('pending', 'running')
                DO UPDATE SET
                    payload_json=CASE WHEN dirty_queue.state='running'
                        THEN dirty_queue.payload_json ELSE excluded.payload_json END,
                    pending_payload_json=CASE WHEN dirty_queue.state='running'
                        THEN excluded.payload_json ELSE NULL END,
                    pending_next_attempt_at=CASE WHEN dirty_queue.state='running'
                        THEN CASE WHEN ? OR dirty_queue.pending_payload_json IS NULL
                            THEN excluded.next_attempt_at
                            ELSE dirty_queue.pending_next_attempt_at END
                        ELSE '' END,
                    next_attempt_at=CASE WHEN ? AND dirty_queue.state='pending'
                        THEN excluded.next_attempt_at ELSE dirty_queue.next_attempt_at END,
                    attempts=CASE WHEN ? AND dirty_queue.state='pending'
                        THEN 0 ELSE dirty_queue.attempts END,
                    error=CASE WHEN ? AND dirty_queue.state='pending'
                        THEN '' ELSE dirty_queue.error END,
                    updated_at=excluded.updated_at
                """,
                (
                    entity_type,
                    entity_id,
                    reason,
                    json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")),
                    next_attempt_at or now,
                    now,
                    now,
                    reset_retry,
                    reset_retry,
                    reset_retry,
                    reset_retry,
                ),
            )
            row = self.connection.execute(
                """
                SELECT id FROM dirty_queue
                WHERE entity_type=? AND entity_id=? AND reason=?
                  AND state IN ('pending', 'running')
                """,
                (entity_type, entity_id, reason),
            ).fetchone()
        assert row is not None
        return int(row[0])

    def claim_dirty(
        self, *, limit: int, worker_id: str, lease_token: str = ""
    ) -> list[DirtyRecord]:
        """Atomically claim at most ``limit`` due items for one worker.

        Use a unique worker ID for each claim lifecycle, or supply a fresh
        ``lease_token`` and echo it to complete/fail when reusing a worker ID.
        """
        if limit <= 0:
            return []
        now = _now()
        lease_expires_at = (datetime.now(UTC) + timedelta(
            seconds=_DIRTY_LEASE_SECONDS
        )).isoformat()
        with self.connection:
            rows = self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='running',
                    attempts=CASE WHEN pending_payload_json IS NULL THEN attempts+1 ELSE 1 END,
                    payload_json=COALESCE(pending_payload_json, payload_json),
                    next_attempt_at=CASE WHEN pending_payload_json IS NULL
                        THEN next_attempt_at ELSE pending_next_attempt_at END,
                    error=CASE WHEN pending_payload_json IS NULL THEN error ELSE '' END,
                    pending_payload_json=NULL, pending_next_attempt_at='', claimed_by=?,
                    lease_expires_at=?, lease_token=?, updated_at=?
                WHERE id IN (
                    SELECT id FROM dirty_queue
                    WHERE (state='pending' AND next_attempt_at <= ?)
                       OR (state='running' AND lease_expires_at != ''
                           AND lease_expires_at <= ?
                           AND (pending_payload_json IS NULL OR pending_next_attempt_at <= ?))
                    ORDER BY next_attempt_at, id
                    LIMIT ?
                )
                RETURNING id, entity_type, entity_id, reason, state, attempts,
                          next_attempt_at, payload_json, error
                """,
                (worker_id, lease_expires_at, lease_token, now, now, now, now, limit),
            ).fetchall()
        return sorted(self._dirty_records(rows), key=lambda record: record.id)

    def start_dirty(self, record_id: int, *, worker_id: str, lease_token: str = "") -> bool:
        """Claim a known item; worker/token uniqueness follows :meth:`claim_dirty`."""
        now = datetime.now(UTC)
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='running',
                    attempts=CASE WHEN pending_payload_json IS NULL THEN attempts+1 ELSE 1 END,
                    payload_json=COALESCE(pending_payload_json, payload_json),
                    next_attempt_at=CASE WHEN pending_payload_json IS NULL
                        THEN next_attempt_at ELSE pending_next_attempt_at END,
                    error=CASE WHEN pending_payload_json IS NULL THEN error ELSE '' END,
                    pending_payload_json=NULL, pending_next_attempt_at='', claimed_by=?,
                    lease_expires_at=?, lease_token=?, updated_at=?
                WHERE id=? AND (
                    (state='pending' AND next_attempt_at<=?)
                    OR (state='running' AND lease_expires_at!='' AND lease_expires_at<=?
                        AND (pending_payload_json IS NULL OR pending_next_attempt_at<=?))
                )
                """,
                (
                    worker_id,
                    (now + timedelta(seconds=_DIRTY_LEASE_SECONDS)).isoformat(),
                    lease_token,
                    now.isoformat(),
                    record_id,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def renew_dirty_worker(self, worker_id: str) -> int:
        """Extend only still-live claims owned by this unique worker lifecycle."""
        now = datetime.now(UTC)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE dirty_queue SET lease_expires_at=?, updated_at=? "
                "WHERE state='running' AND claimed_by=? AND lease_expires_at>?",
                ((now + timedelta(seconds=_DIRTY_LEASE_SECONDS)).isoformat(),
                 now.isoformat(), worker_id, now.isoformat()),
            )
        return cursor.rowcount

    def complete_dirty(self, record_id: int, *, worker_id: str, lease_token: str = "") -> bool:
        """Complete an item only while the caller still owns its lease."""
        now = _now()
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE dirty_queue
                SET state=CASE WHEN pending_payload_json IS NULL
                        THEN 'completed' ELSE 'pending' END,
                    payload_json=COALESCE(pending_payload_json, payload_json),
                    next_attempt_at=CASE WHEN pending_payload_json IS NULL
                        THEN next_attempt_at ELSE pending_next_attempt_at END,
                    attempts=CASE WHEN pending_payload_json IS NULL THEN attempts ELSE 0 END,
                    pending_payload_json=NULL, pending_next_attempt_at='',
                    claimed_by='', lease_expires_at='', lease_token='', error='', updated_at=?
                WHERE id=? AND state='running' AND claimed_by=?
                    AND lease_expires_at>? AND lease_token=?
                """,
                (now, record_id, worker_id, now, lease_token),
            )
            if cursor.rowcount == 1:
                self._prune_completed_dirty_unlocked()
        return cursor.rowcount == 1

    def _prune_completed_dirty_unlocked(
        self, keep: int = _COMPLETED_QUEUE_HISTORY
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM dirty_queue
            WHERE state='completed'
              AND id NOT IN (
                  SELECT id FROM dirty_queue
                  WHERE state='completed'
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (keep,),
        )

    def fail_dirty(
        self,
        record_id: int,
        error: str,
        *,
        worker_id: str,
        next_attempt_at: str | None = None,
        lease_token: str = "",
    ) -> bool:
        """Return a failed claim to the retry queue with bounded diagnostics."""
        now = _now()
        if next_attempt_at is None:
            row = self.connection.execute(
                """
                SELECT attempts FROM dirty_queue
                WHERE id=? AND state='running' AND claimed_by=?
                    AND lease_expires_at>? AND lease_token=?
                """,
                (record_id, worker_id, now, lease_token),
            ).fetchone()
            if row is None:
                return False
            attempts = int(row[0])
            delay_seconds = min(86_400, 60 * (2 ** max(0, attempts - 1)))
            next_attempt_at = datetime.fromtimestamp(
                datetime.now(UTC).timestamp() + delay_seconds,
                tz=UTC,
            ).isoformat()
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE dirty_queue
                SET state='pending', claimed_by='', lease_expires_at='', lease_token='',
                    error=CASE WHEN pending_payload_json IS NULL THEN ? ELSE '' END,
                    next_attempt_at=CASE WHEN pending_payload_json IS NULL
                        THEN ? ELSE pending_next_attempt_at END,
                    attempts=CASE WHEN pending_payload_json IS NULL THEN attempts ELSE 0 END,
                    payload_json=COALESCE(pending_payload_json, payload_json),
                    pending_payload_json=NULL, pending_next_attempt_at='', updated_at=?
                WHERE id=? AND state='running' AND claimed_by=?
                    AND lease_expires_at>? AND lease_token=?
                """,
                (error[:1_000], next_attempt_at, now, record_id, worker_id, now, lease_token),
            )
        return cursor.rowcount == 1

    def upsert_ingest_hash(
        self,
        *,
        requested_url: str,
        canonical_url: str,
        content_hash: str,
        output_file: str,
    ) -> None:
        """Atomically index URL and content identities for a written source."""
        normalized = _normalize_url(requested_url or canonical_url)
        canonical = _normalize_url(canonical_url or requested_url)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO ingest_hash(
                    normalized_url, canonical_url, content_hash, output_file, updated_at
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(normalized_url) DO UPDATE SET
                    canonical_url=excluded.canonical_url,
                    content_hash=excluded.content_hash,
                    output_file=excluded.output_file,
                    updated_at=excluded.updated_at
                """,
                (normalized, canonical, content_hash, output_file, _now()),
            )

    def ingest_index_empty(self) -> bool:
        """Return whether legacy source hashes still need one-time backfill."""
        return self.connection.execute(
            "SELECT 1 FROM ingest_hash LIMIT 1"
        ).fetchone() is None

    def ingest_backfill_complete(self) -> bool:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='ingest_backfill_complete'"
        ).fetchone()
        return row is not None and str(row[0]) == "1"

    def mark_ingest_backfill_complete(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO meta(key, value) VALUES('ingest_backfill_complete', '1')
                ON CONFLICT(key) DO UPDATE SET value='1'
                """
            )

    def find_ingest_duplicate(
        self,
        *,
        requested_url: str = "",
        content_hash: str = "",
    ) -> IngestMatch | None:
        """Resolve a duplicate by canonicalized URL or exact body hash."""
        row = None
        normalized = _normalize_url(requested_url) if requested_url else ""
        if normalized:
            row = self.connection.execute(
                """
                SELECT normalized_url, canonical_url, content_hash, output_file
                FROM ingest_hash
                WHERE normalized_url=? OR canonical_url=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (normalized, normalized),
            ).fetchone()
        if row is None and content_hash:
            row = self.connection.execute(
                """
                SELECT normalized_url, canonical_url, content_hash, output_file
                FROM ingest_hash WHERE content_hash=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
        if row is None:
            return None
        return IngestMatch(*(str(value) for value in row))

    def list_dirty(self, *, limit: int = 100, state: str = "pending") -> list[DirtyRecord]:
        """Return bounded queue records in deterministic claim order."""
        rows = self.connection.execute(
            """
            SELECT id, entity_type, entity_id, reason, state, attempts,
                   next_attempt_at, payload_json, error
            FROM dirty_queue
            WHERE state=?
            ORDER BY next_attempt_at, id
            LIMIT ?
            """,
            (state, max(0, limit)),
        ).fetchall()
        return self._dirty_records(rows)

    @staticmethod
    def _dirty_records(rows: list[sqlite3.Row]) -> list[DirtyRecord]:
        result: list[DirtyRecord] = []
        for row in rows:
            try:
                payload = json.loads(row[7])
            except json.JSONDecodeError:
                payload = {}
            result.append(
                DirtyRecord(
                    id=int(row[0]),
                    entity_type=str(row[1]),
                    entity_id=str(row[2]),
                    reason=str(row[3]),
                    state=str(row[4]),
                    attempts=int(row[5]),
                    next_attempt_at=str(row[6]),
                    payload=payload if isinstance(payload, dict) else {},
                    error=str(row[8]),
                )
            )
        return result

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> VaultRegistry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self.connection.rollback()
        self.close()
